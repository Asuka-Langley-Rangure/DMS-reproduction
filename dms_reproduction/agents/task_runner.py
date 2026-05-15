from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Literal, Optional, Protocol

from dms_reproduction.agents.android_actor import ActorRequest, ActorRunResult, AndroidActor
from dms_reproduction.agents.planner import AndroidTaskPlanner, PlannerResult, PlannerSubtask


TaskRunStatus = Literal[
    "completed",
    "planner_error",
    "actor_error",
    "round_limit",
    "task_check_failed",
]


@dataclass
class TaskRunConfig:
    max_planner_rounds: int = 5
    max_total_actor_steps: int = 40
    max_subtasks_per_round: int | None = None
    stop_on_goal_complete: bool = True


@dataclass
class SubtaskRunRecord:
    subtask: Dict[str, Any]
    actor_result: Dict[str, Any]
    post_observation: Dict[str, Any] | None
    subtask_success_check: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subtask": self.subtask,
            "actor_result": self.actor_result,
            "post_observation": self.post_observation,
            "subtask_success_check": self.subtask_success_check,
        }


@dataclass
class PlannerRoundRecord:
    round_id: int
    input_observation: Dict[str, Any]
    planner_messages: List[Dict[str, Any]]
    planner_prompt: str
    planner_raw_response: str
    planner_result: Dict[str, Any]
    planner_parse_repair: Dict[str, Any] | None = None
    normalized_subtasks: List[Dict[str, Any]] = field(default_factory=list)
    planner_grounding_check: List[Dict[str, Any]] = field(default_factory=list)
    observation_transition_report: List[Dict[str, Any]] = field(default_factory=list)
    subtask_runs: List[SubtaskRunRecord] = field(default_factory=list)
    replan_reason: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "input_observation": self.input_observation,
            "planner_messages": self.planner_messages,
            "planner_prompt": self.planner_prompt,
            "planner_raw_response": self.planner_raw_response,
            "planner_result": self.planner_result,
            "planner_parse_repair": self.planner_parse_repair,
            "normalized_subtasks": self.normalized_subtasks,
            "planner_grounding_check": self.planner_grounding_check,
            "observation_transition_report": self.observation_transition_report,
            "subtask_runs": [run.to_dict() for run in self.subtask_runs],
            "replan_reason": self.replan_reason,
        }


@dataclass
class TaskRunResult:
    status: TaskRunStatus
    planner_rounds: List[PlannerRoundRecord] = field(default_factory=list)
    final_observation: Dict[str, Any] | None = None
    final_task_success: bool | None = None
    total_actor_steps: int = 0
    completion_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "planner_rounds": [round_record.to_dict() for round_record in self.planner_rounds],
            "final_observation": self.final_observation,
            "final_task_success": self.final_task_success,
            "total_actor_steps": self.total_actor_steps,
            "completion_message": self.completion_message,
        }


class ObservationAdapter(Protocol):
    def capture_observation(
        self,
        env: Any,
        goal: str,
        *,
        step_id: int = 0,
        include_screenshots: bool = True,
    ) -> Dict[str, Any]:
        """Capture a standard observation dict."""


class AndroidTaskRunner:
    """Minimal planner -> actor -> replanning loop for one AndroidWorld task."""

    def __init__(
        self,
        planner: AndroidTaskPlanner,
        actor: AndroidActor,
        observation_adapter: ObservationAdapter,
        config: Optional[TaskRunConfig] = None,
    ) -> None:
        self.planner = planner
        self.actor = actor
        self.observation_adapter = observation_adapter
        self.config = config or TaskRunConfig()

    def run_task(self, env: Any, task: Any, user_goal: str) -> TaskRunResult:
        task.initialize_task(env)
        observation = self.observation_adapter.capture_observation(
            env,
            goal=user_goal,
            step_id=0,
            include_screenshots=True,
        )
        task_history: list[dict[str, Any]] = []
        planner_rounds: list[PlannerRoundRecord] = []
        total_actor_steps = 0
        recent_completed_subtasks: dict[str, int] = {}
        normalized_subtask_progress: dict[str, int] = {}
        failure_pattern_counts = {
            "invalid_index": 0,
            "unstable_observation": 0,
            "planner_subtask_invalid": 0,
            "actor_overshoot_after_goal": 0,
            "planner_near_json_repaired": 0,
            "actor_action_alias_normalized": 0,
            "recoverable_schema_error": 0,
            "text_entry_success_detected": 0,
        }

        for round_id in range(1, self.config.max_planner_rounds + 1):
            planner_messages = self.planner.build_messages(
                user_goal=user_goal,
                observation=observation,
                task_history=task_history,
                memory_context="",
            )
            planner_prompt = self.planner.extract_user_text_prompt(planner_messages)
            planner_raw_response = self.planner.llm_client.generate(
                messages=planner_messages,
                temperature=self.planner.config.temperature,
            )
            planner_result = self.planner.parse_response(planner_raw_response)
            round_record = PlannerRoundRecord(
                round_id=round_id,
                input_observation=observation,
                planner_messages=self.planner.messages_to_jsonable(planner_messages),
                planner_prompt=planner_prompt,
                planner_raw_response=planner_raw_response,
                planner_result=planner_result.to_dict(),
                planner_parse_repair={
                    "repaired_parse": planner_result.repaired_parse,
                    "repair_reason": planner_result.repair_reason,
                    "raw_response": planner_raw_response,
                },
            )
            planner_rounds.append(round_record)
            if planner_result.repaired_parse:
                failure_pattern_counts["planner_near_json_repaired"] += 1

            if planner_result.parse_error:
                return TaskRunResult(
                    status="planner_error",
                    planner_rounds=planner_rounds,
                    final_observation=observation,
                    final_task_success=None,
                    total_actor_steps=total_actor_steps,
                    completion_message=f"Planner parse error in round {round_id}: {planner_result.parse_error}",
                )

            if planner_result.is_goal_complete:
                success = self._task_success(task, env)
                if not success and not self.config.stop_on_goal_complete:
                    round_record.replan_reason = "planner_complete_but_task_check_failed"
                    continue
                return TaskRunResult(
                    status="completed" if success else "task_check_failed",
                    planner_rounds=planner_rounds,
                    final_observation=observation,
                    final_task_success=success,
                    total_actor_steps=total_actor_steps,
                    completion_message=planner_result.completion_message or (
                        "Task check passed."
                        if success
                        else "Planner declared completion but task check failed."
                    ),
                )

            subtasks = planner_result.subtasks
            if self.config.max_subtasks_per_round is not None:
                subtasks = subtasks[: self.config.max_subtasks_per_round]
            normalized_subtasks, invalid_reason = self._normalize_planner_subtasks(subtasks)
            round_record.normalized_subtasks = [subtask.to_dict() for subtask in normalized_subtasks]
            if invalid_reason:
                round_record.replan_reason = "planner_subtask_invalid"
                failure_pattern_counts["planner_subtask_invalid"] += 1
                task_history.append(
                    {
                        "source": "planner",
                        "round_id": round_id,
                        "subtask": "",
                        "status": "planner_subtask_invalid",
                        "reason": invalid_reason,
                        "summary": invalid_reason,
                        "action": None,
                        "error": invalid_reason,
                        "step_id": None,
                    }
                )
                continue
            grounding_check, grounding_reason = self._validate_subtasks_against_observation(
                normalized_subtasks,
                observation,
            )
            round_record.planner_grounding_check = grounding_check
            if grounding_reason:
                round_record.replan_reason = grounding_reason
                task_history.append(
                    {
                        "source": "planner",
                        "round_id": round_id,
                        "subtask": "",
                        "status": grounding_reason,
                        "reason": "Planner referenced a visible UI target that is absent from the current observation.",
                        "summary": "Planner referenced a visible UI target that is absent from the current observation.",
                        "action": None,
                        "error": grounding_reason,
                        "step_id": None,
                        "observation_unreliable_context": self._is_unstable_observation(observation),
                    }
                )
                continue

            for subtask in normalized_subtasks:
                if total_actor_steps >= self.config.max_total_actor_steps:
                    round_record.replan_reason = "max_total_actor_steps_reached"
                    return TaskRunResult(
                        status="actor_error",
                        planner_rounds=planner_rounds,
                        final_observation=observation,
                        final_task_success=None,
                        total_actor_steps=total_actor_steps,
                        completion_message=(
                            f"Stopped in round {round_id} because max_total_actor_steps="
                            f"{self.config.max_total_actor_steps} was reached."
                        ),
                    )

                actor_result = self.actor.run_subtask(
                    env,
                    ActorRequest(
                        subtask=subtask.task,
                        observation=observation,
                        action_history=task_history,
                        memory_context="",
                    ),
                    self.observation_adapter,
                )
                if self._actor_used_action_normalization(actor_result):
                    failure_pattern_counts["actor_action_alias_normalized"] += 1
                retry_reason = self._classify_actor_failure(actor_result)
                if retry_reason == "observation_unstable_error":
                    failure_pattern_counts["unstable_observation"] += 1
                    refreshed_observation = self.observation_adapter.capture_observation(
                        env,
                        goal=user_goal,
                        step_id=total_actor_steps + len(actor_result.steps) + 1,
                        include_screenshots=True,
                    )
                    actor_result = self.actor.run_subtask(
                        env,
                        ActorRequest(
                            subtask=subtask.task,
                            observation=refreshed_observation,
                            action_history=task_history,
                            memory_context="",
                        ),
                        self.observation_adapter,
                    )
                actor_result, observation, unstable_persisted = self._recover_unstable_subtask_observation(
                    env=env,
                    user_goal=user_goal,
                    task_history=task_history,
                    subtask=subtask,
                    actor_result=actor_result,
                    current_observation=observation,
                )
                actor_result, success_check = self._apply_subtask_success_override(subtask, actor_result, observation)
                if retry_reason == "recoverable_schema_error":
                    failure_pattern_counts["recoverable_schema_error"] += 1
                total_actor_steps += len(actor_result.steps)
                observation = actor_result.final_observation or observation
                round_record.subtask_runs.append(
                    SubtaskRunRecord(
                        subtask=subtask.to_dict(),
                        actor_result=actor_result.to_dict(),
                        post_observation=observation,
                        subtask_success_check=success_check,
                    )
                )
                if success_check.get("runner_overrode_to_completed"):
                    failure_pattern_counts["actor_overshoot_after_goal"] += 1
                if str(success_check.get("success_rule") or "").startswith("text_entry_"):
                    failure_pattern_counts["text_entry_success_detected"] += 1
                round_record.observation_transition_report.extend(
                    self._build_observation_transition_entries(subtask, actor_result)
                )
                self._append_actor_history(task_history, round_id, subtask, actor_result)
                self._append_observation_warning_history(task_history, round_id, subtask, observation)

                if actor_result.status == "completed":
                    recent_completed_subtasks[subtask.task] = recent_completed_subtasks.get(subtask.task, 0) + 1
                    normalized_subtask_progress[subtask.task] = normalized_subtask_progress.get(subtask.task, 0) + 1
                else:
                    recent_completed_subtasks[subtask.task] = 0
                    normalized_subtask_progress[subtask.task] = normalized_subtask_progress.get(subtask.task, 0) + 1

                degraded_replan_reason = self._degraded_observation_reason(observation)
                suspicious_completion = self._suspicious_completion_reason(
                    subtask_task=subtask.task,
                    actor_result=actor_result,
                    completed_count=recent_completed_subtasks.get(subtask.task, 0),
                    observation=observation,
                )
                repeated_no_progress_reason = self._same_subtask_no_progress_reason(
                    subtask.task,
                    normalized_subtask_progress.get(subtask.task, 0),
                    actor_result,
                    observation,
                )
                failure_reason = self._classify_actor_failure(actor_result)

                if unstable_persisted:
                    failure_pattern_counts["unstable_observation"] += 1
                    round_record.replan_reason = "observation_unstable_persisted"
                    break
                if actor_result.status != "completed":
                    if failure_reason == "invalid_index_error":
                        failure_pattern_counts["invalid_index"] += 1
                    if failure_reason == "observation_unstable_error":
                        failure_pattern_counts["unstable_observation"] += 1
                    round_record.replan_reason = degraded_replan_reason or failure_reason or f"actor_{actor_result.status}"
                    break
                if suspicious_completion:
                    round_record.replan_reason = suspicious_completion
                    break
                if repeated_no_progress_reason:
                    round_record.replan_reason = repeated_no_progress_reason
                    break
                if degraded_replan_reason:
                    round_record.replan_reason = degraded_replan_reason
                    break

            else:
                round_record.replan_reason = "subtasks_exhausted"
                continue

            continue

        return TaskRunResult(
            status="round_limit",
            planner_rounds=planner_rounds,
            final_observation=observation,
            final_task_success=None,
            total_actor_steps=total_actor_steps,
            completion_message=self._build_round_limit_message(planner_rounds, failure_pattern_counts),
        )

    @staticmethod
    def _task_success(task: Any, env: Any) -> bool:
        return bool(task.is_successful(env) == 1)

    @staticmethod
    def _append_actor_history(
        task_history: list[dict[str, Any]],
        round_id: int,
        subtask: PlannerSubtask,
        actor_result: ActorRunResult,
    ) -> None:
        for step in actor_result.steps:
            task_history.append(
                {
                    "source": "actor",
                    "round_id": round_id,
                    "subtask": subtask.task,
                    "status": actor_result.status if step.done else "progress",
                    "reason": step.reason,
                    "summary": step.summary,
                    "action": step.action.to_payload() if step.action else None,
                    "error": step.parse_error or step.execution_error or "",
                    "step_id": step.step_id,
                    "observation_unreliable_context": (
                        (step.before_observation or {}).get("observation_consistency") == "unstable"
                        or ((step.after_observation or {}).get("observation_consistency") == "unstable")
                    ),
                }
            )

    @staticmethod
    def _append_observation_warning_history(
        task_history: list[dict[str, Any]],
        round_id: int,
        subtask: PlannerSubtask,
        observation: dict[str, Any],
    ) -> None:
        if observation.get("keyboard_active_context"):
            return
        warning = str(observation.get("observation_warning") or "").strip()
        if not warning:
            return
        task_history.append(
            {
                "source": "actor",
                "round_id": round_id,
                "subtask": subtask.task,
                "status": "warning",
                "reason": warning,
                "summary": warning,
                "action": None,
                "error": warning,
                "step_id": None,
                "observation_unreliable_context": True,
            }
        )

    @staticmethod
    def _degraded_observation_reason(observation: dict[str, Any]) -> str | None:
        if AndroidTaskRunner._is_unstable_observation(observation):
            return "observation_degraded_system_ui_only"
        return None

    @staticmethod
    def _is_unstable_observation(observation: dict[str, Any] | None) -> bool:
        if not observation:
            return False
        foreground_package = observation.get("foreground_package") or ""
        if observation.get("observation_consistency") == "unstable":
            return True
        if foreground_package and foreground_package != "com.android.systemui" and observation.get("non_system_ui_count", 0) == 0:
            return True
        if observation.get("keyboard_active_context"):
            return False
        if foreground_package and foreground_package != "com.android.systemui" and observation.get("clickable_ui_count", 0) == 0:
            warning = str(observation.get("observation_warning") or "").lower()
            if "foreground package" in warning or "only system ui elements" in warning:
                return True
        return False

    @staticmethod
    def _suspicious_completion_reason(
        *,
        subtask_task: str,
        actor_result: ActorRunResult,
        completed_count: int,
        observation: dict[str, Any],
    ) -> str | None:
        if actor_result.status != "completed":
            return None
        if completed_count < 2:
            return None
        goal_text = subtask_task.lower()
        if "navigate" not in goal_text and "go to" not in goal_text and "reach" not in goal_text:
            return None
        warning = str(observation.get("observation_warning") or "")
        if warning or observation.get("non_system_ui_count", 0) == 0:
            return "suspicious_subtask_completion_without_progress"
        return None

    @staticmethod
    def _same_subtask_no_progress_reason(
        subtask_task: str,
        count: int,
        actor_result: ActorRunResult,
        observation: dict[str, Any],
    ) -> str | None:
        if AndroidTaskRunner._is_unstable_observation(observation):
            return None
        if count < 2:
            return None
        if actor_result.status == "completed":
            return "same_subtask_no_progress"
        return None

    @staticmethod
    def _classify_actor_failure(actor_result: ActorRunResult) -> str | None:
        if actor_result.status != "parse_error":
            return None
        if not actor_result.steps:
            return "format_only_error"
        step = actor_result.steps[-1]
        parse_error = str(step.parse_error or "").lower()
        if step.action_normalization_applied:
            return "recoverable_schema_error"
        before_observation = step.before_observation or {}
        if before_observation.get("observation_consistency") == "unstable":
            return "observation_unstable_error"
        if "clickable" in parse_error or "editable" in parse_error or "valid_ui_indices" in parse_error:
            return "invalid_index_error"
        if "failed to parse actor action json" in parse_error:
            return "recoverable_schema_error"
        return "format_only_error"

    def _recover_unstable_subtask_observation(
        self,
        *,
        env: Any,
        user_goal: str,
        task_history: list[dict[str, Any]],
        subtask: PlannerSubtask,
        actor_result: ActorRunResult,
        current_observation: dict[str, Any],
    ) -> tuple[ActorRunResult, dict[str, Any], bool]:
        final_observation = actor_result.final_observation or current_observation
        if not self._is_unstable_observation(final_observation):
            return actor_result, final_observation, False

        refreshed_observation = self.observation_adapter.capture_observation(
            env,
            goal=user_goal,
            step_id=len(task_history) + len(actor_result.steps) + 1,
            include_screenshots=True,
        )
        if not self._is_unstable_observation(refreshed_observation):
            actor_result.final_observation = refreshed_observation
            return actor_result, refreshed_observation, False

        retried_actor_result = self.actor.run_subtask(
            env,
            ActorRequest(
                subtask=subtask.task,
                observation=refreshed_observation,
                action_history=task_history,
                memory_context="",
            ),
            self.observation_adapter,
        )
        retried_final_observation = retried_actor_result.final_observation or refreshed_observation
        if self._is_unstable_observation(retried_final_observation):
            return retried_actor_result, retried_final_observation, True
        return retried_actor_result, retried_final_observation, False

    @staticmethod
    def _apply_subtask_success_override(
        subtask: PlannerSubtask,
        actor_result: ActorRunResult,
        current_observation: dict[str, Any],
    ) -> tuple[ActorRunResult, dict[str, Any]]:
        success_check = {
            "original_actor_status": actor_result.status,
            "runner_overrode_to_completed": False,
            "success_rule": None,
            "evidence_observation": None,
            "progress_made": False,
        }
        matched = AndroidTaskRunner._match_subtask_success(subtask, actor_result, current_observation)
        if matched is None:
            return actor_result, success_check
        observation, success_rule = matched
        success_check["runner_overrode_to_completed"] = actor_result.status != "completed"
        success_check["success_rule"] = success_rule
        success_check["evidence_observation"] = {
            "current_activity": observation.get("current_activity"),
            "foreground_package": observation.get("foreground_package"),
            "app_name": observation.get("app_name"),
        }
        actor_result.status = "completed"
        actor_result.final_observation = observation
        if not actor_result.completion_message:
            actor_result.completion_message = f"Runner marked subtask completed via {success_rule}."
        return actor_result, success_check

    @staticmethod
    def _match_subtask_success(
        subtask: PlannerSubtask,
        actor_result: ActorRunResult,
        current_observation: dict[str, Any],
    ) -> tuple[dict[str, Any], str] | None:
        observations: list[dict[str, Any]] = []
        for step in actor_result.steps:
            if step.after_observation:
                observations.append(step.after_observation)
        if actor_result.final_observation:
            observations.append(actor_result.final_observation)
        if not observations:
            observations.append(current_observation)
        goal_lower = subtask.goal.lower()
        for observation in observations:
            if "open the phone app" in goal_lower:
                if (observation.get("foreground_package") or "").endswith("dialer"):
                    return observation, "foreground_package_matches_phone_app"
            if "open the contacts app" in goal_lower:
                foreground_package = observation.get("foreground_package") or ""
                if foreground_package.endswith("contacts") or "contact" in (observation.get("current_activity") or "").lower():
                    return observation, "foreground_package_matches_contacts_app"
            if "navigate to the contacts section" in goal_lower:
                if AndroidTaskRunner._contacts_section_active(observation):
                    return observation, "contacts_section_selected"
            if "start creating a new contact" in goal_lower or "create a new contact" == goal_lower.strip(". "):
                if "contacteditoractivity" in (observation.get("current_activity") or "").lower():
                    return observation, "contact_editor_activity_visible"
            text_entry_match = AndroidTaskRunner._match_text_entry_success(goal_lower, observation)
            if text_entry_match is not None:
                return observation, text_entry_match
        return None

    @staticmethod
    def _match_text_entry_success(goal_lower: str, observation: dict[str, Any]) -> str | None:
        field_label, expected_text = AndroidTaskRunner._parse_text_entry_goal(goal_lower)
        if not field_label or expected_text is None:
            return None
        for element in observation.get("ui_elements") or []:
            if not bool(element.get("is_editable")):
                continue
            label = " ".join(
                part.strip().lower()
                for part in (
                    str(element.get("text") or ""),
                    str(element.get("content_description") or ""),
                    str(element.get("resource_name") or ""),
                )
                if part and part.strip()
            )
            if field_label not in label:
                continue
            raw = element.get("raw") or {}
            candidates = [
                str(element.get("text") or ""),
                str(element.get("content_description") or ""),
                str(raw.get("text") or ""),
                str(raw.get("content_description") or ""),
                str(raw.get("value") or ""),
            ]
            if any(expected_text == candidate.strip().lower() for candidate in candidates if candidate.strip()):
                return f"text_entry_{field_label.replace(' ', '_')}_matches_expected_value"
        return None

    @staticmethod
    def _parse_text_entry_goal(goal_lower: str) -> tuple[str | None, str | None]:
        patterns = (
            (r"enter the first name ['\"](?P<value>.+?)['\"]", "first name"),
            (r"enter the last name ['\"](?P<value>.+?)['\"]", "last name"),
            (r"enter the phone number ['\"](?P<value>.+?)['\"]", "phone"),
        )
        for pattern, label in patterns:
            match = re.search(pattern, goal_lower)
            if match:
                return label, match.group("value")
        return None, None

    @staticmethod
    def _actor_used_action_normalization(actor_result: ActorRunResult) -> bool:
        return any(step.action_normalization_applied for step in actor_result.steps)

    @staticmethod
    def _contacts_section_active(observation: dict[str, Any]) -> bool:
        for element in observation.get("ui_elements") or []:
            label = str(element.get("content_description") or element.get("text") or "").lower()
            if label == "contacts" and bool((element.get("raw") or {}).get("is_selected")):
                return True
        return False

    @staticmethod
    def _normalize_planner_subtasks(subtasks: list[PlannerSubtask]) -> tuple[list[PlannerSubtask], str | None]:
        normalized: list[PlannerSubtask] = []
        for subtask in subtasks:
            goal_lower = subtask.goal.lower()
            precondition_lower = subtask.precondition.lower()
            if "open" in precondition_lower and "click on the phone app to open it" in goal_lower:
                normalized.append(
                    PlannerSubtask(
                        precondition=subtask.precondition,
                        goal="Navigate to the contacts section.",
                        reason=subtask.reason,
                        agent=subtask.agent,
                    )
                )
                continue
            if _is_contradictory_subtask(subtask.precondition, subtask.goal):
                return [], (
                    f"Planner produced contradictory subtask: Precondition={subtask.precondition!r}, Goal={subtask.goal!r}"
                )
            normalized.append(subtask)
        return normalized, None

    @staticmethod
    def _validate_subtasks_against_observation(
        subtasks: list[PlannerSubtask],
        observation: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], str | None]:
        ui_elements = observation.get("ui_elements") or []
        evidence_fields: list[str] = []
        for element in ui_elements:
            evidence_fields.extend(
                [
                    str(element.get("text") or "").strip().lower(),
                    str(element.get("content_description") or "").strip().lower(),
                    str(element.get("resource_name") or "").strip().lower(),
                ]
            )
        evidence_fields = [field for field in evidence_fields if field]
        high_risk_targets = ("create new contact", "add contact", "contacts", "phone")
        checks: list[dict[str, Any]] = []
        veto_reason: str | None = None
        for subtask in subtasks:
            combined_text = f"{subtask.goal} {subtask.reason}".lower()
            matched_targets = [target for target in high_risk_targets if target in combined_text]
            claims_current_visibility = any(
                marker in combined_text
                for marker in ("visible", "clearly visible", "shown", "available", "clickable", "appeared")
            )
            grounded = True
            missing_targets: list[str] = []
            for target in matched_targets:
                if not any(target in field or field in target for field in evidence_fields):
                    grounded = False
                    missing_targets.append(target)
            checks.append(
                {
                    "subtask": subtask.to_dict(),
                    "referenced_targets": matched_targets,
                    "claims_current_visibility": claims_current_visibility,
                    "grounded": grounded,
                    "missing_targets": missing_targets,
                }
            )
            if matched_targets and not grounded and (
                claims_current_visibility or any(target in {"create new contact", "add contact"} for target in matched_targets)
            ):
                veto_reason = "planner_subtask_not_grounded_in_observation"
        return checks, veto_reason

    @staticmethod
    def _build_observation_transition_entries(
        subtask: PlannerSubtask,
        actor_result: ActorRunResult,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for step in actor_result.steps:
            before = step.before_observation or {}
            after = step.after_observation or {}
            entries.append(
                {
                    "subtask": subtask.task,
                    "step_id": step.step_id,
                    "before": {
                        "current_activity": before.get("current_activity"),
                        "foreground_package": before.get("foreground_package"),
                        "dominant_ui_package": before.get("app_name"),
                        "non_system_ui_count": before.get("non_system_ui_count"),
                        "clickable_ui_count": before.get("clickable_ui_count"),
                        "observation_consistency": before.get("observation_consistency"),
                    },
                    "after": {
                        "current_activity": after.get("current_activity"),
                        "foreground_package": after.get("foreground_package"),
                        "dominant_ui_package": after.get("app_name"),
                        "non_system_ui_count": after.get("non_system_ui_count"),
                        "clickable_ui_count": after.get("clickable_ui_count"),
                        "observation_consistency": after.get("observation_consistency"),
                    } if after else None,
                }
            )
        return entries

    @staticmethod
    def _build_round_limit_message(
        planner_rounds: list[PlannerRoundRecord],
        failure_pattern_counts: dict[str, int],
    ) -> str:
        if not planner_rounds:
            return "Round limit reached."
        last_reason = planner_rounds[-1].replan_reason or "unknown_replan_reason"
        if last_reason == "suspicious_subtask_completion_without_progress":
            return (
                f"Round limit reached after {len(planner_rounds)} planner rounds; "
                "repeated actor completion on the same navigation subtask without observable progress."
            )
        if last_reason == "observation_degraded_system_ui_only":
            return (
                f"Round limit reached after {len(planner_rounds)} planner rounds; "
                "foreground activity remained in the business app while retained UI elements degraded to system UI only."
            )
        if last_reason == "same_subtask_no_progress":
            return (
                f"Round limit reached after {len(planner_rounds)} planner rounds; "
                "the same normalized subtask repeated without observable progress."
            )
        if last_reason == "planner_subtask_not_grounded_in_observation":
            return (
                f"Round limit reached after {len(planner_rounds)} planner rounds; "
                "planner repeatedly referenced UI targets that were absent from the current observation."
            )
        if last_reason == "observation_unstable_persisted":
            return (
                f"Round limit reached after {len(planner_rounds)} planner rounds; "
                "observation remained unstable after local recovery and one same-subtask retry."
            )
        failure_summary = ", ".join(
            f"{name}={count}" for name, count in failure_pattern_counts.items() if count
        )
        if failure_summary:
            return (
                f"Round limit reached after {len(planner_rounds)} planner rounds; "
                f"last replan reason: {last_reason}. Failure pattern: {failure_summary}."
            )
        return f"Round limit reached after {len(planner_rounds)} planner rounds; last replan reason: {last_reason}."


def _is_contradictory_subtask(precondition: str, goal: str) -> bool:
    precondition_lower = precondition.lower()
    goal_lower = goal.lower()
    open_match = re.search(r"the (.+?) app is open", precondition_lower)
    if open_match and f"open the {open_match.group(1)} app" in goal_lower:
        return True
    return False
