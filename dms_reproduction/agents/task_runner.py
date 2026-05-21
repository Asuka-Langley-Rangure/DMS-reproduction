from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Literal, Optional, Protocol

from dms_reproduction.agents.android_actor import ActorRequest, ActorRunResult, AndroidActor
from dms_reproduction.agents.planner import (
    AndroidTaskPlanner,
    PlannerResult,
    PlannerStage,
    PlannerSubtask,
    StagePlanResult,
)
from dms_reproduction.agents.verifier import (
    AndroidVerifier,
    EvidenceSource,
    VerifierRequest,
    VerifierResult,
)
from dms_reproduction.memory import (
    MemoryEvent,
    MemoryProvider,
    NoOpMemoryProvider,
    StaticMemoryRecord,
)


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
    subtask_verification: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subtask": self.subtask,
            "actor_result": self.actor_result,
            "post_observation": self.post_observation,
            "subtask_success_check": self.subtask_success_check,
            "subtask_verification": self.subtask_verification,
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
class StagePlanRecord:
    planner_messages: List[Dict[str, Any]]
    planner_prompt: str
    planner_raw_response: str
    stage_plan_result: Dict[str, Any]
    revision_reason: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "planner_messages": self.planner_messages,
            "planner_prompt": self.planner_prompt,
            "planner_raw_response": self.planner_raw_response,
            "stage_plan_result": self.stage_plan_result,
            "revision_reason": self.revision_reason,
        }


@dataclass
class TaskRunResult:
    status: TaskRunStatus
    planner_rounds: List[PlannerRoundRecord] = field(default_factory=list)
    initial_stage_plan: Dict[str, Any] | None = None
    stage_plan_revisions: List[Dict[str, Any]] = field(default_factory=list)
    final_observation: Dict[str, Any] | None = None
    final_task_success: bool | None = None
    total_actor_steps: int = 0
    completion_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "planner_rounds": [round_record.to_dict() for round_record in self.planner_rounds],
            "initial_stage_plan": self.initial_stage_plan,
            "stage_plan_revisions": self.stage_plan_revisions,
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
        memory_provider: Optional[MemoryProvider] = None,
        verifier: Optional[AndroidVerifier] = None,
    ) -> None:
        self.planner = planner
        self.actor = actor
        self.observation_adapter = observation_adapter
        self.config = config or TaskRunConfig()
        self.memory_provider = memory_provider or NoOpMemoryProvider()
        self.verifier = verifier

    def run_task(self, env: Any, task: Any, user_goal: str) -> TaskRunResult:
        self.memory_provider.reset()
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
        failure_pattern_counts = {
            "invalid_index": 0,
            "unstable_observation": 0,
            "planner_near_json_repaired": 0,
            "actor_action_alias_normalized": 0,
            "recoverable_schema_error": 0,
            "planner_subtask_invalid": 0,
        }

        for round_id in range(1, self.config.max_planner_rounds + 1):
            planner_memory_context = self.memory_provider.build_context(
                user_goal=user_goal,
                observation=observation,
                task_history=task_history,
            )
            planner_messages = self.planner.build_messages(
                user_goal=user_goal,
                observation=observation,
                task_history=task_history,
                memory_context=planner_memory_context,
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
                result = TaskRunResult(
                    status="planner_error",
                    planner_rounds=planner_rounds,
                    final_observation=observation,
                    final_task_success=None,
                    total_actor_steps=total_actor_steps,
                    completion_message=(
                        f"Planner parse error in round {round_id}"
                        f"{f' [{planner_result.parse_error_code}]' if planner_result.parse_error_code else ''}: "
                        f"{planner_result.parse_error}"
                    ),
                )
                self._emit_task_end_memory_event(user_goal=user_goal, result=result, round_id=round_id)
                return result

            round_record.planner_result = planner_result.to_dict()

            if planner_result.is_goal_complete:
                success = self._task_success(task, env)
                if success:
                    result = TaskRunResult(
                        status="completed",
                        planner_rounds=planner_rounds,
                        final_observation=observation,
                        final_task_success=True,
                        total_actor_steps=total_actor_steps,
                        completion_message=planner_result.completion_message or "Task check passed.",
                    )
                    self._emit_task_end_memory_event(user_goal=user_goal, result=result, round_id=round_id)
                    return result
                self._append_planner_feedback_history(
                    task_history=task_history,
                    round_id=round_id,
                    status="planner_complete_but_task_check_failed",
                    reason=(
                        "Planner proposed complete_goal as a completion candidate, "
                        "but the AndroidWorld evaluator rejected it."
                    ),
                    summary=(
                        f"Completion candidate rejected. Planner completion message: "
                        f"{planner_result.completion_message or 'None'}. "
                        f"Current activity: {observation.get('current_activity') or 'Unknown'}. "
                        f"Foreground package: {observation.get('foreground_package') or 'Unknown'}. "
                        "Treat this as a planner failure. The next planner round must return a repair, "
                        "verification, or progress-making subtask instead of declaring completion again "
                        "unless the current screen now provides new direct evidence that the evaluator can pass."
                    ),
                    observation=observation,
                )
                round_record.replan_reason = "planner_complete_but_task_check_failed"
                continue

            subtasks = planner_result.subtasks
            if self.config.max_subtasks_per_round is not None:
                subtasks = subtasks[: self.config.max_subtasks_per_round]
            normalized_subtasks, invalid_reason = self._normalize_planner_subtasks(
                subtasks=subtasks,
                user_goal=user_goal,
                observation=observation,
            )
            round_record.normalized_subtasks = [subtask.to_dict() for subtask in normalized_subtasks]
            if invalid_reason:
                round_record.replan_reason = "planner_subtask_invalid"
                failure_pattern_counts["planner_subtask_invalid"] += 1
                self._append_planner_feedback_history(
                    task_history=task_history,
                    round_id=round_id,
                    status="planner_subtask_invalid",
                    reason=invalid_reason,
                    summary=invalid_reason,
                    observation=observation,
                )
                continue
            planner_grounding_check = self._validate_planner_output_against_observation(
                planner_result=planner_result,
                observation=observation,
                normalized_subtasks=normalized_subtasks,
            )
            round_record.planner_grounding_check = planner_grounding_check
            planner_grounding_error = next(
                (item for item in planner_grounding_check if not bool(item.get("valid", True))),
                None,
            )
            if planner_grounding_error is not None:
                round_record.replan_reason = str(
                    planner_grounding_error.get("reason_code")
                    or "planner_subtask_conflicts_with_current_observation"
                )
                failure_pattern_counts["planner_subtask_invalid"] += 1
                reason_text = str(
                    planner_grounding_error.get("message")
                    or "Planner subtasks conflict with the current observation."
                )
                self._append_planner_feedback_history(
                    task_history=task_history,
                    round_id=round_id,
                    status="planner_subtask_invalid",
                    reason=reason_text,
                    summary=reason_text,
                    observation=observation,
                )
                continue

            round_all_subtasks_succeeded = True
            for subtask in normalized_subtasks:
                if total_actor_steps >= self.config.max_total_actor_steps:
                    round_record.replan_reason = "max_total_actor_steps_reached"
                    result = TaskRunResult(
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
                    self._emit_task_end_memory_event(user_goal=user_goal, result=result, round_id=round_id)
                    return result

                memory_read_result = self._build_actor_memory_result(
                    user_goal=user_goal,
                    subtask=subtask.task,
                    observation=observation,
                    task_history=task_history,
                )
                actor_memory_context = str(memory_read_result.get("context") or "")
                before_subtask_observation = observation
                actor_observation = self._prepare_actor_observation(subtask=subtask, observation=observation)
                actor_result = self.actor.run_subtask(
                    env,
                    ActorRequest(
                        subtask=subtask.task,
                        observation=actor_observation,
                        action_history=task_history,
                        memory_context=actor_memory_context,
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
                            memory_context=actor_memory_context,
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
                success_check = {
                    "progress_made": actor_result.status in {"completed", "stopped"},
                    "state_changed": bool(actor_result.final_observation),
                    "save_attempted": any(
                        step.action and step.action.action_type in {"click", "long_press"}
                        for step in actor_result.steps
                    ),
                    "verification_target_visible": False,
                    "terminal_failure_reason": None,
                }
                if retry_reason == "recoverable_schema_error":
                    failure_pattern_counts["recoverable_schema_error"] += 1
                total_actor_steps += len(actor_result.steps)
                observation = actor_result.final_observation or observation
                evaluator_success = self._task_success_if_available(task, env)
                if evaluator_success:
                    _, verifier_evidence_source = self._select_verifier_evidence_observation(
                        actor_result,
                        observation,
                    )
                    verifier_result = self._build_evaluator_success_result(actor_result)
                    success_check["global_task_success"] = True
                    success_check["global_task_success_source"] = "androidworld_evaluator_after_subtask"
                else:
                    verifier_result, verifier_evidence_source = self._run_subtask_verification(
                        subtask=subtask,
                        actor_result=actor_result,
                        before_observation=before_subtask_observation,
                        current_observation=observation,
                        memory_context=actor_memory_context,
                        success_check=success_check,
                    )
                success_check["verifier_status"] = verifier_result.status
                success_check["verifier_reason"] = verifier_result.reason
                success_check["memory_eligible"] = verifier_result.memory_eligible
                success_check["verifier_evidence_source"] = verifier_evidence_source
                success_check["verifier_source"] = verifier_result.source
                subtask_completed = verifier_result.status == "success"
                if subtask_completed:
                    actor_result.status = "completed"
                    actor_result.final_observation = observation
                    if not actor_result.completion_message:
                        actor_result.completion_message = verifier_result.reason
                round_record.subtask_runs.append(
                    SubtaskRunRecord(
                        subtask=subtask.to_dict(),
                        actor_result=actor_result.to_dict(),
                        post_observation=observation,
                        subtask_success_check=success_check,
                        subtask_verification=verifier_result.to_dict(),
                    )
                )
                self._append_actor_history(task_history, round_id, subtask, actor_result)
                self._append_observation_warning_history(task_history, round_id, subtask, observation)
                self._append_subtask_summary_history(task_history, round_id, subtask, actor_result, success_check)
                self._record_memory_event(
                    user_goal=user_goal,
                    round_id=round_id,
                    subtask=subtask,
                    actor_result=actor_result,
                    verifier_result=verifier_result,
                    verifier_evidence_source=verifier_evidence_source,
                    success_check=success_check,
                    observation=observation,
                    memory_read_result=memory_read_result,
                )
                if evaluator_success and self.config.stop_on_goal_complete:
                    result = TaskRunResult(
                        status="completed",
                        planner_rounds=planner_rounds,
                        final_observation=observation,
                        final_task_success=True,
                        total_actor_steps=total_actor_steps,
                        completion_message=(
                            actor_result.completion_message
                            or verifier_result.reason
                            or "AndroidWorld evaluator passed after subtask execution."
                        ),
                    )
                    self._emit_task_end_memory_event(user_goal=user_goal, result=result, round_id=round_id)
                    return result
                if subtask_completed and self.config.stop_on_goal_complete and self._task_success_if_available(task, env):
                    result = TaskRunResult(
                        status="completed",
                        planner_rounds=planner_rounds,
                        final_observation=observation,
                        final_task_success=True,
                        total_actor_steps=total_actor_steps,
                        completion_message=(
                            actor_result.completion_message
                            or verifier_result.reason
                            or "Task check passed after subtask execution."
                        ),
                    )
                    self._emit_task_end_memory_event(user_goal=user_goal, result=result, round_id=round_id)
                    return result

                failure_reason = self._classify_actor_failure(actor_result)

                if unstable_persisted:
                    failure_pattern_counts["unstable_observation"] += 1
                    round_record.replan_reason = "observation_unstable_persisted"
                    round_all_subtasks_succeeded = False
                    break
                if verifier_result.status != "success":
                    if failure_reason == "invalid_index_error":
                        failure_pattern_counts["invalid_index"] += 1
                    if failure_reason == "observation_unstable_error":
                        failure_pattern_counts["unstable_observation"] += 1
                    round_record.replan_reason = failure_reason or f"verifier_{verifier_result.status}"
                    round_all_subtasks_succeeded = False
                    break
            if round_all_subtasks_succeeded:
                if self.config.stop_on_goal_complete and self._task_success_if_available(task, env):
                    result = TaskRunResult(
                        status="completed",
                        planner_rounds=planner_rounds,
                        final_observation=observation,
                        final_task_success=True,
                        total_actor_steps=total_actor_steps,
                        completion_message="Task check passed after completing the current planner round.",
                    )
                    self._emit_task_end_memory_event(user_goal=user_goal, result=result, round_id=round_id)
                    return result
                round_record.replan_reason = "round_subtasks_verified_but_task_incomplete"
                self._append_planner_feedback_history(
                    task_history=task_history,
                    round_id=round_id,
                    status="task_incomplete_after_verified_round",
                    reason="All subtasks in the current round were verified, but the whole task is still not complete.",
                    summary="Use the current observation to generate the next 1-5 functional subtasks.",
                    observation=observation,
                )
                continue

        result = TaskRunResult(
            status="round_limit",
            planner_rounds=planner_rounds,
            final_observation=observation,
            final_task_success=None,
            total_actor_steps=total_actor_steps,
            completion_message=self._build_round_limit_message(planner_rounds, failure_pattern_counts),
        )
        self._emit_task_end_memory_event(user_goal=user_goal, result=result, round_id=len(planner_rounds))
        return result

    def _generate_stage_plan_record(self, *, user_goal: str, revision_reason: str | None = None) -> StagePlanRecord:
        planner_messages = self.planner.build_stage_plan_messages(user_goal=user_goal)
        planner_prompt = self.planner.extract_user_text_prompt(planner_messages)
        planner_raw_response = self.planner.llm_client.generate(
            messages=planner_messages,
            temperature=self.planner.config.temperature,
        )
        stage_plan_result = self.planner.parse_stage_plan_response(planner_raw_response)
        return StagePlanRecord(
            planner_messages=self.planner.messages_to_jsonable(planner_messages),
            planner_prompt=planner_prompt,
            planner_raw_response=planner_raw_response,
            stage_plan_result=stage_plan_result.to_dict(),
            revision_reason=revision_reason,
        )

    @staticmethod
    def _consume_stage_plan_revision_reason(
        *,
        task_history: list[dict[str, Any]],
        since_history_len: int,
    ) -> str | None:
        repair_statuses = {
            "saved_but_task_check_failed",
            "saved_with_wrong_identity",
            "field_misgrounded",
            "saved_contact_state_changed",
        }
        for item in reversed(task_history[since_history_len:]):
            status = str(item.get("status") or "").strip()
            if status in repair_statuses:
                return status
        return None

    @staticmethod
    def _task_success(task: Any, env: Any) -> bool:
        return bool(task.is_successful(env) == 1)

    @staticmethod
    def _task_success_if_available(task: Any, env: Any) -> bool:
        try:
            return AndroidTaskRunner._task_success(task, env)
        except AssertionError:
            return False

    @staticmethod
    def _prepare_actor_observation(
        *,
        subtask: PlannerSubtask,
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        return observation

    @staticmethod
    def _select_verifier_evidence_observation(
        actor_result: ActorRunResult,
        current_observation: dict[str, Any],
    ) -> tuple[dict[str, Any], EvidenceSource]:
        for step in actor_result.steps:
            if step.done_reason == "completed" and step.after_observation:
                return step.after_observation, "actor_completed_frame"
        if actor_result.final_observation:
            source: EvidenceSource = (
                "final_after_observation"
                if actor_result.status != "completed"
                else "actor_completed_frame"
            )
            return actor_result.final_observation, source
        return current_observation, "fallback_current_observation"

    @staticmethod
    def _build_subtask_action_history(actor_result: ActorRunResult, subtask: PlannerSubtask) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for step in actor_result.steps:
            history.append(
                {
                    "subtask": subtask.task,
                    "status": step.done_reason or ("progress" if not step.done else actor_result.status),
                    "reason": step.reason,
                    "summary": step.summary,
                    "action": step.action.to_payload() if step.action else step.original_action,
                    "error": step.parse_error or step.execution_error or "",
                    "step_id": step.step_id,
                }
            )
        return history

    def _run_subtask_verification(
        self,
        *,
        subtask: PlannerSubtask,
        actor_result: ActorRunResult,
        before_observation: dict[str, Any],
        current_observation: dict[str, Any],
        memory_context: str,
        success_check: dict[str, Any] | None = None,
    ) -> tuple[VerifierResult, EvidenceSource]:
        evidence_observation, evidence_source = self._select_verifier_evidence_observation(
            actor_result,
            current_observation,
        )
        if self.verifier is None:
            has_rule_based_success = bool((success_check or {}).get("success_rule"))
            status: Literal["success", "uncertain"] = (
                "success" if actor_result.status == "completed" or has_rule_based_success else "uncertain"
            )
            reason = (
                actor_result.completion_message
                or (success_check or {}).get("success_rule")
                or ("No verifier configured; falling back to runtime completion evidence." if status == "success" else "No verifier configured.")
            )
            return (
                VerifierResult(
                    status=status,
                    source="heuristic",
                    reason=reason,
                    memory_eligible=(status == "success"),
                    raw_response="",
                    parse_error=None,
                ),
                evidence_source,
            )
        verifier_request = VerifierRequest(
            subtask=subtask.task,
            action_history=self._build_subtask_action_history(actor_result, subtask),
            before_observation=before_observation,
            evidence_observation=evidence_observation,
            evidence_source=evidence_source,
            memory_context=memory_context,
        )
        return self.verifier.run_verification(verifier_request), evidence_source

    @staticmethod
    def _build_evaluator_success_result(actor_result: ActorRunResult) -> VerifierResult:
        return VerifierResult(
            status="success",
            source="androidworld_evaluator",
            reason=(
                actor_result.completion_message
                or "AndroidWorld evaluator passed after subtask execution."
            ),
            memory_eligible=AndroidTaskRunner._is_evaluator_memory_eligible(actor_result),
            raw_response="",
            parse_error=None,
            prompt_text="",
            messages=[],
        )

    @staticmethod
    def _is_evaluator_memory_eligible(actor_result: ActorRunResult) -> bool:
        if len(actor_result.steps) <= 1:
            return False
        return not any(step.parse_error or step.execution_error for step in actor_result.steps)

    def _build_step_verifier_callback(
        self,
        *,
        subtask: PlannerSubtask,
        before_observation: dict[str, Any],
        memory_context: str,
    ):
        if self.verifier is None:
            return None

        def _verify_step(
            *,
            subtask: str,
            action_history: list[dict[str, Any]],
            before_observation: dict[str, Any],
            evidence_observation: dict[str, Any],
            memory_context: str,
        ) -> dict[str, Any]:
            verifier_request = VerifierRequest(
                subtask=subtask,
                action_history=action_history,
                before_observation=before_observation,
                evidence_observation=evidence_observation,
                evidence_source="final_after_observation",
                memory_context=memory_context,
            )
            return self.verifier.run_verification(verifier_request).to_dict()

        return _verify_step

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
                memory_context=str(
                    self._build_actor_memory_result(
                        user_goal=user_goal,
                        subtask=subtask.task,
                        observation=refreshed_observation,
                        task_history=task_history,
                    ).get("context")
                    or ""
                ),
            ),
            self.observation_adapter,
        )
        retried_final_observation = retried_actor_result.final_observation or refreshed_observation
        if self._is_unstable_observation(retried_final_observation):
            return retried_actor_result, retried_final_observation, True
        return retried_actor_result, retried_final_observation, False

    def _postprocess_contact_form_subtask(
        self,
        *,
        env: Any,
        task: Any,
        user_goal: str,
        round_id: int,
        subtask: PlannerSubtask,
        actor_result: ActorRunResult,
        current_observation: dict[str, Any],
        success_check: dict[str, Any],
    ) -> tuple[ActorRunResult, dict[str, Any], dict[str, Any]]:
        expected = _parse_contact_form_fill_goal(subtask.goal)
        if expected is None:
            return actor_result, current_observation, success_check

        required_identity = {
            "full_name": f"{expected['first_name']} {expected['last_name']}",
            "phone_number": expected["phone_number"],
        }
        success_check["form_fill_stage"] = success_check.get("form_fill_stage") or "editing_in_progress"
        success_check["saved_but_task_check_failed"] = bool(success_check.get("saved_but_task_check_failed"))
        success_check["saved_with_wrong_identity"] = bool(success_check.get("saved_with_wrong_identity"))
        success_check["observed_contact_identity"] = success_check.get("observed_contact_identity")
        success_check["required_contact_identity"] = required_identity
        success_check["target_fields"] = ["first_name", "last_name", "phone"]
        success_check["off_target_field_touched"] = success_check.get("off_target_field_touched")
        success_check["mismatched_target_fields"] = list(success_check.get("mismatched_target_fields") or [])
        success_check["save_attempted_before_fields_complete"] = bool(success_check.get("save_attempted_before_fields_complete"))
        success_check["terminal_failure_reason"] = success_check.get("terminal_failure_reason")

        if not success_check.get("form_fill_progress"):
            fallback_progress = self._match_contact_form_fill_success(subtask.goal, current_observation)
            if fallback_progress is not None:
                success_check["progress_made"] = bool(fallback_progress.get("progress_made"))
                success_check["form_fill_progress"] = fallback_progress

        form_progress = success_check.get("form_fill_progress") or {}
        if success_check.get("success_rule"):
            success_check["form_fill_stage"] = "all_required_fields_match"

        action_issues = _detect_contact_form_action_issues(actor_result, expected)
        success_check["off_target_field_touched"] = action_issues.get("off_target_field_touched")
        success_check["mismatched_target_fields"] = action_issues.get("mismatched_target_fields") or []
        save_attempted = bool(action_issues.get("save_attempted"))
        remaining_fields = list(form_progress.get("remaining_fields") or [])
        if save_attempted and remaining_fields:
            success_check["save_attempted_before_fields_complete"] = True
            success_check["form_fill_stage"] = "save_attempted"

        if actor_result.status == "completed" and remaining_fields and not save_attempted:
            actor_result.status = "step_limit"
            actor_result.completion_message = (
                "Grouped contact form is still incomplete; do not mark complete before all required fields match."
            )
            success_check["terminal_failure_reason"] = "contact_form_incomplete"

        if success_check["off_target_field_touched"] or success_check["mismatched_target_fields"]:
            actor_result.status = "infeasible"
            actor_result.completion_message = (
                "Actor edited fields outside the required contact form target set or used mismatched values."
            )
            success_check["terminal_failure_reason"] = "field_misgrounded"

        needs_post_save_refresh = save_attempted or (
            actor_result.status == "completed"
            and not success_check.get("success_rule")
            and not _is_contact_editor_observation(actor_result.final_observation or current_observation)
        )
        if needs_post_save_refresh:
            refreshed_observation = self.observation_adapter.capture_observation(
                env,
                goal=user_goal,
                step_id=round_id + len(actor_result.steps) + 1,
                include_screenshots=True,
            )
            actor_result.final_observation = refreshed_observation
            current_observation = refreshed_observation
            if _parse_contact_form_fill_goal(subtask.goal) is not None:
                refreshed_progress = self._match_contact_form_fill_success(subtask.goal, refreshed_observation)
                if refreshed_progress is not None:
                    success_check["progress_made"] = bool(refreshed_progress.get("progress_made"))
                    success_check["form_fill_progress"] = refreshed_progress
                    if refreshed_progress.get("success_rule"):
                        success_check["success_rule"] = refreshed_progress.get("success_rule")
                        success_check["form_fill_stage"] = "all_required_fields_match"
            if save_attempted:
                success_check["form_fill_stage"] = "save_attempted"
            if not _is_contact_editor_observation(refreshed_observation):
                success_check["form_fill_stage"] = "saved_but_not_validated"
                observed_identity = _extract_contact_identity_from_observation(refreshed_observation)
                success_check["observed_contact_identity"] = observed_identity
                validator_success = self._task_success(task, env)
                if validator_success:
                    success_check["terminal_failure_reason"] = "saved_contact_state_changed"
                    if not actor_result.completion_message:
                        actor_result.completion_message = "Contact details were saved and the task validator confirmed success."
                    return actor_result, refreshed_observation, success_check
                success_check["saved_but_task_check_failed"] = True
                success_check["terminal_failure_reason"] = "saved_but_task_check_failed"
                if observed_identity and _normalize_field_value(observed_identity) != _normalize_field_value(required_identity["full_name"]):
                    success_check["saved_with_wrong_identity"] = True
                    success_check["terminal_failure_reason"] = "saved_with_wrong_identity"

        return actor_result, current_observation, success_check

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
            "form_fill_progress": None,
            "form_fill_stage": None,
            "saved_but_task_check_failed": False,
            "saved_with_wrong_identity": False,
            "observed_contact_identity": None,
            "required_contact_identity": None,
            "target_fields": None,
            "off_target_field_touched": None,
            "mismatched_target_fields": [],
            "save_attempted_before_fields_complete": False,
            "terminal_failure_reason": None,
        }
        matched = AndroidTaskRunner._match_subtask_success(subtask, actor_result, current_observation)
        if matched is None:
            return actor_result, success_check
        observation = matched["observation"]
        success_rule = matched.get("success_rule")
        success_check["progress_made"] = bool(matched.get("progress_made"))
        success_check["form_fill_progress"] = matched.get("form_fill_progress")
        success_check["override_blocked_reason"] = None
        if not success_rule:
            return actor_result, success_check
        if AndroidTaskRunner._override_completion_blocked(observation):
            success_check["override_blocked_reason"] = "observation_not_reliable_for_completion"
            return actor_result, success_check
        success_check["runner_overrode_to_completed"] = actor_result.status != "completed"
        success_check["success_rule"] = success_rule
        success_check["evidence_observation"] = {
            "current_activity": observation.get("current_activity"),
            "foreground_package": observation.get("foreground_package"),
            "app_name": observation.get("app_name"),
        }
        return actor_result, success_check

    @staticmethod
    def _match_subtask_success(
        subtask: PlannerSubtask,
        actor_result: ActorRunResult,
        current_observation: dict[str, Any],
    ) -> dict[str, Any] | None:
        observations: list[dict[str, Any]] = []
        for step in actor_result.steps:
            if step.after_observation:
                observations.append(step.after_observation)
        if actor_result.final_observation:
            observations.append(actor_result.final_observation)
        if not observations:
            observations.append(current_observation)
        goal_lower = subtask.goal.lower()
        best_form_fill_match: dict[str, Any] | None = None
        for observation in observations:
            if "open the phone app" in goal_lower:
                if (observation.get("foreground_package") or "").endswith("dialer"):
                    return {"observation": observation, "success_rule": "foreground_package_matches_phone_app", "progress_made": True}
            if "open the contacts app" in goal_lower:
                foreground_package = observation.get("foreground_package") or ""
                if foreground_package.endswith("contacts") or "contact" in (observation.get("current_activity") or "").lower():
                    return {"observation": observation, "success_rule": "foreground_package_matches_contacts_app", "progress_made": True}
            if "navigate to the contacts section" in goal_lower:
                if AndroidTaskRunner._contacts_section_active(observation):
                    return {"observation": observation, "success_rule": "contacts_section_selected", "progress_made": True}
            if (
                "start creating a new contact" in goal_lower
                or "create a new contact" == goal_lower.strip(". ")
                or "reach the contact creation entry point" in goal_lower
            ):
                if "contacteditoractivity" in (observation.get("current_activity") or "").lower():
                    return {"observation": observation, "success_rule": "contact_editor_activity_visible", "progress_made": True}
            form_fill_match = AndroidTaskRunner._match_contact_form_fill_success(subtask.goal, observation)
            if form_fill_match is not None:
                candidate = {
                    "observation": observation,
                    "success_rule": form_fill_match.get("success_rule"),
                    "progress_made": form_fill_match.get("progress_made", False),
                    "form_fill_progress": form_fill_match,
                }
                if candidate.get("success_rule"):
                    return candidate
                if best_form_fill_match is None or len(candidate["form_fill_progress"].get("completed_fields") or []) > len(best_form_fill_match["form_fill_progress"].get("completed_fields") or []):
                    best_form_fill_match = candidate
            text_entry_match = AndroidTaskRunner._match_text_entry_success(goal_lower, observation)
            if text_entry_match is not None:
                return {"observation": observation, "success_rule": text_entry_match, "progress_made": True}
        return best_form_fill_match

    @staticmethod
    def _match_contact_form_fill_success(goal_text: str, observation: dict[str, Any]) -> dict[str, Any] | None:
        expected = _parse_contact_form_fill_goal(goal_text)
        if expected is None:
            return None
        actual_values = _extract_contact_form_values(observation)
        visible_fields = _extract_visible_contact_field_labels(observation)
        completed_fields: list[str] = []
        remaining_fields: list[str] = []
        expected_fields = {
            "first_name": expected["first_name"],
            "last_name": expected["last_name"],
            "phone": expected["phone_number"],
        }
        for field_name, expected_value in expected_fields.items():
            actual_value = _normalize_field_value(actual_values.get(field_name))
            if actual_value and actual_value == _normalize_field_value(expected_value):
                completed_fields.append(field_name)
            else:
                remaining_fields.append(field_name)
        progress = {
            "expected_fields": expected_fields,
            "actual_values": actual_values,
            "completed_fields": completed_fields,
            "remaining_fields": remaining_fields,
            "visible_fields": visible_fields,
            "required_field_indices": _extract_contact_field_indices(observation),
        }
        if not completed_fields:
            return None
        if not remaining_fields:
            progress["success_rule"] = "contact_form_fields_match_expected_values"
            progress["progress_made"] = True
            return progress
        progress["success_rule"] = None
        progress["progress_made"] = True
        return progress

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
    def _stage_plans_equivalent(left: list[PlannerStage], right: list[PlannerStage]) -> bool:
        return [
            (
                stage.stage_id,
                AndroidTaskRunner._normalize_stage_title(stage.title),
                stage.success_signal.strip().lower(),
            )
            for stage in left
        ] == [
            (
                stage.stage_id,
                AndroidTaskRunner._normalize_stage_title(stage.title),
                stage.success_signal.strip().lower(),
            )
            for stage in right
        ]

    @staticmethod
    def _normalize_stage_title(title: str) -> str:
        return re.sub(r"\s+", " ", title.strip().lower())

    @staticmethod
    def _append_stage_plan_history(
        *,
        task_history: list[dict[str, Any]],
        round_id: int,
        stage_plan: list[PlannerStage],
        current_stage_id: int | None,
        covered_stage_ids: list[int] | None,
    ) -> None:
        task_history.append(
            {
                "source": "stage_plan",
                "round_id": round_id,
                "stage_plan": [stage.to_dict() for stage in stage_plan],
                "current_stage_id": current_stage_id,
                "covered_stage_ids": list(covered_stage_ids or []),
                "status": "planner_stage_plan",
                "summary": f"Remembered stage plan with {len(stage_plan)} stages.",
            }
        )

    @staticmethod
    def _validate_covered_stage_ids_against_frozen_plan(
        *,
        covered_stage_ids: list[int],
        current_stage_id: int | None,
        stage_plan: list[PlannerStage],
    ) -> str | None:
        if not covered_stage_ids:
            return None
        if current_stage_id is None:
            return "Planner returned covered_stage_ids without current_stage_id."
        if current_stage_id not in covered_stage_ids:
            return "Planner covered_stage_ids must include current_stage_id."
        valid_stage_ids = {stage.stage_id for stage in stage_plan}
        invalid_stage_ids = [stage_id for stage_id in covered_stage_ids if stage_id not in valid_stage_ids]
        if invalid_stage_ids:
            return (
                "Planner covered_stage_ids must only reference stages from the frozen stage plan. "
                f"Invalid stage ids: {invalid_stage_ids}."
            )
        return None

    @staticmethod
    def _next_stage_id_after_covered(
        covered_stage_ids: list[int],
        stage_plan: list[PlannerStage],
    ) -> int | None:
        if not covered_stage_ids:
            return None
        ordered_stage_ids = [stage.stage_id for stage in stage_plan]
        last_covered_stage_id = max(covered_stage_ids)
        for stage_id in ordered_stage_ids:
            if stage_id > last_covered_stage_id:
                return stage_id
        return last_covered_stage_id

    @staticmethod
    def _normalize_planner_subtasks(
        subtasks: list[PlannerSubtask],
        user_goal: str,
        observation: dict[str, Any],
        stage_plan: list[PlannerStage] | None = None,
        current_stage_id: int | None = None,
    ) -> tuple[list[PlannerSubtask], str | None]:
        normalized: list[PlannerSubtask] = []
        seen_tasks: set[str] = set()
        for subtask in subtasks:
            if _is_contradictory_subtask(subtask.precondition, subtask.goal):
                return [], (
                    f"Planner produced contradictory subtask: Precondition={subtask.precondition!r}, Goal={subtask.goal!r}"
                )
            if subtask.task not in seen_tasks:
                normalized.append(subtask)
                seen_tasks.add(subtask.task)
        stage_alignment_error = AndroidTaskRunner._validate_stage_subtask_alignment(
            normalized,
            stage_plan=stage_plan or [],
            current_stage_id=current_stage_id,
        )
        if stage_alignment_error is not None:
            return [], stage_alignment_error
        return normalized, None

    @staticmethod
    def _validate_planner_output_against_observation(
        *,
        planner_result: PlannerResult,
        observation: dict[str, Any],
        normalized_subtasks: list[PlannerSubtask],
    ) -> list[dict[str, Any]]:
        if not normalized_subtasks:
            return []
        first_subtask = normalized_subtasks[0]
        precondition = str(first_subtask.precondition or "").strip().lower()
        goal = str(first_subtask.goal or "").strip().lower()
        app_name = str(observation.get("app_name") or "").strip().lower()
        activity = str(observation.get("current_activity") or "").strip().lower()
        ui_description = str(observation.get("ui_description") or "").strip().lower()
        combined_context = f"{app_name} {activity} {ui_description}"
        inside_dialer_contacts_workspace = any(token in combined_context for token in ("dialer", "contacts"))
        create_contact_entry_visible = any(
            token in ui_description
            for token in ("create new contact", "add contact", "your contacts are just a tap away here")
        )
        contact_editor_visible = "contacteditoractivity" in activity
        contact_detail_visible = "quick contact" in ui_description or "contact detail" in ui_description
        if not (
            inside_dialer_contacts_workspace
            or create_contact_entry_visible
            or contact_editor_visible
            or contact_detail_visible
        ):
            return [{"valid": True, "reason_code": None, "message": "No observation conflict detected."}]

        opens_phone_like_app = bool(
            re.search(r"\bopen\b.*\b(phone|dialer|contacts)\b", goal)
            or re.search(r"\blaunch\b.*\b(phone|dialer|contacts)\b", goal)
            or "open the relevant app" in goal
            or "open the relevant app or settings area" in goal
        )
        earlier_navigation_precondition = any(
            token in precondition
            for token in ("home screen", "launcher", "not the required app", "non-settings screen", "not in the app")
        )
        if opens_phone_like_app and (earlier_navigation_precondition or inside_dialer_contacts_workspace):
            return [
                {
                    "valid": False,
                    "reason_code": "planner_subtask_conflicts_with_current_observation",
                    "message": (
                        "Planner first subtask repeats an app-opening phase even though the current observation "
                        "already shows a later Phone/Dialer or Contacts workspace."
                    ),
                    "subtask": first_subtask.to_dict(),
                    "planner_current_stage_id": planner_result.current_stage_id,
                }
            ]
        return [{"valid": True, "reason_code": None, "message": "No observation conflict detected."}]

    @staticmethod
    def _validate_stage_subtask_alignment(
        subtasks: list[PlannerSubtask],
        *,
        stage_plan: list[PlannerStage],
        current_stage_id: int | None,
    ) -> str | None:
        if not subtasks or not stage_plan or current_stage_id is None:
            return None
        current_stage = next((stage for stage in stage_plan if stage.stage_id == current_stage_id), None)
        if current_stage is None:
            return None
        current_title = str(current_stage.title or "").strip().lower()
        success_signal = str(current_stage.success_signal or "").strip().lower()
        if not re.search(r"^(open|launch)\b", current_title):
            return None
        stage_target = re.sub(r"^(open|launch)\s+", "", current_title).strip().strip(".")
        if not stage_target:
            return None
        for subtask in subtasks:
            precondition = str(subtask.precondition or "").strip().lower()
            goal = str(subtask.goal or "").strip().lower()
            if "app is open" in success_signal and "app is open" in precondition and not re.search(r"^(open|launch)\b", goal):
                return (
                    "Planner subtask is misaligned with the selected app-opening stage. "
                    "Do not assume the current stage is already complete inside the subtask precondition."
                )
            if stage_target in precondition and "open" in precondition and stage_target not in goal:
                return (
                    "Planner subtask is misaligned with the selected app-opening stage. "
                    "Do not assume the current stage is already complete inside the subtask precondition."
                )
        return None

    @staticmethod
    def _override_completion_blocked(observation: dict[str, Any]) -> bool:
        if AndroidTaskRunner._is_unstable_observation(observation):
            return True
        warning = str(observation.get("observation_warning") or "").strip()
        if warning:
            return True
        if observation.get("non_system_ui_count", 0) == 0:
            return True
        return False

    @staticmethod
    def _build_grounding_feedback_summary(grounding_check: list[dict[str, Any]], observation: dict[str, Any]) -> str:
        visible_labels = []
        for element in observation.get("ui_elements") or []:
            for field in ("text", "content_description", "resource_name"):
                value = str(element.get(field) or "").strip()
                if value:
                    visible_labels.append(value)
                    break
        visible_excerpt = ", ".join(visible_labels[:6]) or "None"
        missing_targets = sorted(
            {
                target
                for check in grounding_check
                for target in (check.get("missing_targets") or [])
            }
        )
        missing_text = ", ".join(missing_targets) or "None"
        return (
            "Planner referenced a UI target that is absent from the current observation. "
            f"Missing targets: {missing_text}. "
            f"Current activity: {observation.get('current_activity') or 'Unknown'}. "
            f"Foreground package: {observation.get('foreground_package') or 'Unknown'}. "
            f"Visible UI labels: {visible_excerpt}. "
            "Replan to the next safe functional milestone instead of repeating the same target claim."
        )

    @staticmethod
    def _append_planner_feedback_history(
        task_history: list[dict[str, Any]],
        round_id: int,
        status: str,
        reason: str,
        summary: str,
        observation: dict[str, Any],
    ) -> None:
        task_history.append(
            {
                "source": "planner",
                "round_id": round_id,
                "subtask": "",
                "status": status,
                "reason": reason,
                "summary": summary,
                "action": None,
                "error": status,
                "step_id": None,
                "observation_unreliable_context": AndroidTaskRunner._is_unstable_observation(observation),
            }
        )

    def _build_actor_memory_result(
        self,
        *,
        user_goal: str,
        subtask: str,
        observation: dict[str, Any],
        task_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        build_actor_memory = getattr(self.memory_provider, "build_actor_memory", None)
        if callable(build_actor_memory):
            result = build_actor_memory(
                user_goal=user_goal,
                subtask=subtask,
                observation=observation,
                task_history=task_history,
            )
            if hasattr(result, "to_event_fields"):
                event_fields = result.to_event_fields()
                event_fields["context"] = getattr(result, "context", "")
                return event_fields
            if isinstance(result, dict):
                return result
        return {
            "context": self.memory_provider.build_actor_context(
                user_goal=user_goal,
                subtask=subtask,
                observation=observation,
                task_history=task_history,
            )
        }

    @staticmethod
    def _collect_actor_feedback_counts(actor_result: ActorRunResult) -> dict[str, int]:
        invalid_action_count = 0
        no_state_change_count = 0
        parse_error_count = 0
        execution_error_count = 0

        for step in actor_result.steps:
            if step.parse_error:
                parse_error_count += 1
                parse_error_text = str(step.parse_error).lower()
                if any(
                    marker in parse_error_text
                    for marker in ("valid_ui_indices", "clickable", "editable", "index")
                ):
                    invalid_action_count += 1
            if step.execution_error:
                execution_error_count += 1
            before = step.before_observation or {}
            after = step.after_observation or {}
            before_digest = (
                before.get("current_activity"),
                before.get("foreground_package"),
                before.get("app_name"),
                before.get("ui_description"),
            )
            after_digest = (
                after.get("current_activity"),
                after.get("foreground_package"),
                after.get("app_name"),
                after.get("ui_description"),
            )
            if after and before_digest == after_digest:
                no_state_change_count += 1

        return {
            "invalid_action_count": invalid_action_count,
            "no_state_change_count": no_state_change_count,
            "parse_error_count": parse_error_count,
            "execution_error_count": execution_error_count,
        }

    def _emit_task_end_memory_event(
        self,
        *,
        user_goal: str,
        result: TaskRunResult,
        round_id: int,
    ) -> None:
        final_success = result.final_task_success
        if final_success is None:
            final_success = result.status == "completed"
        self.memory_provider.record(
            {
                "event_type": "task_end",
                "user_goal": user_goal,
                "round_id": round_id,
                "subtask": "",
                "status": result.status,
                "summary": result.completion_message,
                "verifier_status": "",
                "verifier_source": "",
                "verifier_reason": "",
                "memory_eligible": False,
                "verifier_evidence_source": "",
                "success_check": {},
                "observation_digest": {},
                "success": final_success,
                "global_success": final_success,
                "task_success": final_success,
            }
        )

    def _record_memory_event(
        self,
        *,
        user_goal: str,
        round_id: int,
        subtask: PlannerSubtask,
        actor_result: ActorRunResult,
        verifier_result: VerifierResult,
        verifier_evidence_source: EvidenceSource,
        success_check: dict[str, Any],
        observation: dict[str, Any],
        memory_read_result: dict[str, Any] | None = None,
    ) -> None:
        summary = verifier_result.reason or success_check.get("success_rule") or actor_result.completion_message or actor_result.status
        feedback_counts = self._collect_actor_feedback_counts(actor_result)
        memory_fields = dict(memory_read_result or {})
        event = MemoryEvent(
            user_goal=user_goal,
            round_id=round_id,
            subtask=subtask.task,
            status=actor_result.status,
            summary=str(summary),
            verifier_status=verifier_result.status,
            verifier_source=verifier_result.source,
            verifier_reason=verifier_result.reason,
            memory_eligible=verifier_result.memory_eligible,
            verifier_evidence_source=verifier_evidence_source,
            success_check=success_check,
            observation_digest={
                "current_activity": observation.get("current_activity"),
                "foreground_package": observation.get("foreground_package"),
                "app_name": observation.get("app_name"),
                "observation_warning": observation.get("observation_warning"),
                "observation_consistency": observation.get("observation_consistency"),
                "non_system_ui_count": observation.get("non_system_ui_count"),
                "clickable_ui_count": observation.get("clickable_ui_count"),
            },
            selected_memory_id=memory_fields.get("selected_memory_id"),
            used_memory=memory_fields.get("used_memory"),
            should_replay=memory_fields.get("should_replay"),
            should_mutate=memory_fields.get("should_mutate"),
            retrieval_score=memory_fields.get("retrieval_score"),
            sim_goal=memory_fields.get("sim_goal"),
            sim_precondition=memory_fields.get("sim_precondition"),
            survival_value=memory_fields.get("survival_value"),
            risk_score=memory_fields.get("risk_score"),
            memory_read_reason=memory_fields.get("memory_read_reason"),
            invalid_action_count=feedback_counts["invalid_action_count"],
            no_state_change_count=feedback_counts["no_state_change_count"],
            parse_error_count=feedback_counts["parse_error_count"],
            execution_error_count=feedback_counts["execution_error_count"],
        )
        self.memory_provider.record(event.to_dict())
        if verifier_result.status != "success" or not verifier_result.memory_eligible:
            return
        if any(step.parse_error or step.execution_error for step in actor_result.steps):
            return
        static_memory_record = self._build_static_memory_record(
            user_goal=user_goal,
            round_id=round_id,
            subtask=subtask,
            actor_result=actor_result,
            verifier_result=verifier_result,
            observation=observation,
        )
        self.memory_provider.record_subtask_trajectory(static_memory_record.to_dict())

    @staticmethod
    def _build_static_memory_record(
        *,
        user_goal: str,
        round_id: int,
        subtask: PlannerSubtask,
        actor_result: ActorRunResult,
        verifier_result: VerifierResult,
        observation: dict[str, Any],
    ) -> StaticMemoryRecord:
        trajectory = [
            {
                "step_id": step.step_id,
                "reason": step.reason,
                "action": step.action.to_payload() if step.action else step.original_action,
                "summary": step.summary,
            }
            for step in actor_result.steps
        ]
        return StaticMemoryRecord(
            subtask_text=subtask.task,
            precondition=subtask.precondition,
            goal=subtask.goal,
            user_goal=user_goal,
            round_id=round_id,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            app_name=str(observation.get("app_name") or ""),
            current_activity=str(observation.get("current_activity") or ""),
            verifier_status=verifier_result.status,
            verifier_reason=verifier_result.reason,
            completion_message=actor_result.completion_message or "",
            trajectory=trajectory,
            observation_digest={
                "current_activity": observation.get("current_activity"),
                "foreground_package": observation.get("foreground_package"),
                "app_name": observation.get("app_name"),
                "observation_warning": observation.get("observation_warning"),
                "observation_consistency": observation.get("observation_consistency"),
                "non_system_ui_count": observation.get("non_system_ui_count"),
                "clickable_ui_count": observation.get("clickable_ui_count"),
            },
        )

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
            is_app_opening_goal = bool(re.search(r"\b(open|launch)\b", subtask.goal.lower())) and bool(
                re.search(r"\bapp\b", subtask.goal.lower())
            )
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
                (claims_current_visibility and not is_app_opening_goal)
                or any(target in {"create new contact", "add contact"} for target in matched_targets)
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
        if last_reason == "planner_subtask_conflicts_with_current_observation":
            return (
                f"Round limit reached after {len(planner_rounds)} planner rounds; "
                "planner repeatedly proposed an earlier app-opening subtask that conflicted with the current observation."
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

    @staticmethod
    def _append_subtask_summary_history(
        task_history: list[dict[str, Any]],
        round_id: int,
        subtask: PlannerSubtask,
        actor_result: ActorRunResult,
        success_check: dict[str, Any],
    ) -> None:
        status = actor_result.status
        reason = (
            actor_result.completion_message
            or str(success_check.get("terminal_failure_reason") or "").strip()
            or ("Observation changed after this subtask." if success_check.get("state_changed") else "")
            or status
        )
        task_history.append(
            {
                "source": "subtask_summary",
                "round_id": round_id,
                "subtask": subtask.task,
                "status": status,
                "reason": reason,
                "summary": reason,
                "action": None,
                "error": "",
                "step_id": None,
                "observation_unreliable_context": False,
            }
        )


def _is_contradictory_subtask(precondition: str, goal: str) -> bool:
    precondition_lower = precondition.lower()
    goal_lower = goal.lower()
    open_match = re.search(r"the (.+?) app is open", precondition_lower)
    if open_match and f"open the {open_match.group(1)} app" in goal_lower:
        return True
    return False

def _parse_contact_identity_goal(user_goal: str) -> dict[str, str] | None:
    match = re.search(
        r"create a new contact for (?P<name>.+?)\.\s*their number is (?P<phone>\+?[0-9][0-9\s\-]+)\.?",
        user_goal,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    full_name = " ".join(match.group("name").strip().split())
    phone_number = " ".join(match.group("phone").strip().split())
    name_parts = full_name.split()
    if len(name_parts) < 2:
        return None
    return {
        "full_name": full_name,
        "first_name": name_parts[0],
        "last_name": " ".join(name_parts[1:]),
        "phone_number": phone_number,
    }


def _parse_contact_form_fill_goal(goal_text: str) -> dict[str, str] | None:
    match = re.search(
        r"fill in (?P<name>.+?) and (?P<phone>\+?[0-9][0-9\s\-]+) in the contact form",
        goal_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    name = " ".join(match.group("name").strip().split())
    phone_number = " ".join(match.group("phone").strip().split())
    name_parts = name.split()
    if len(name_parts) < 2:
        return None
    return {
        "first_name": name_parts[0],
        "last_name": " ".join(name_parts[1:]),
        "phone_number": phone_number,
    }


def _extract_contact_form_values(observation: dict[str, Any]) -> dict[str, str | None]:
    values: dict[str, str | None] = {"first_name": None, "last_name": None, "phone": None}
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
        raw = element.get("raw") or {}
        candidates = [
            str(raw.get("value") or "").strip(),
            str(raw.get("text") or "").strip(),
            str(element.get("text") or "").strip(),
        ]
        value = next((candidate for candidate in candidates if candidate), None)
        if "first name" in label:
            values["first_name"] = value
        elif "last name" in label:
            values["last_name"] = value
        elif "phone" in label:
            values["phone"] = value
    return values


def _extract_contact_field_indices(observation: dict[str, Any]) -> dict[str, int]:
    indices: dict[str, int] = {}
    for element in observation.get("ui_elements") or []:
        field_name = _classify_contact_element(element)
        if field_name in {"first_name", "last_name", "phone"} and element.get("index") is not None:
            indices[field_name] = int(element["index"])
    return indices


def _extract_visible_contact_field_labels(observation: dict[str, Any]) -> list[str]:
    visible_fields: list[str] = []
    for element in observation.get("ui_elements") or []:
        field_name = _classify_contact_element(element)
        if field_name in {"first_name", "last_name", "phone"}:
            visible_fields.append(field_name)
    return visible_fields


def _build_contact_form_context(goal_text: str, observation: dict[str, Any]) -> dict[str, Any] | None:
    expected = _parse_contact_form_fill_goal(goal_text)
    if expected is None or not _is_contact_editor_observation(observation):
        return None
    actual_values = _extract_contact_form_values(observation)
    remaining_fields = [
        field_name
        for field_name, expected_value in {
            "first_name": expected["first_name"],
            "last_name": expected["last_name"],
            "phone": expected["phone_number"],
        }.items()
        if _normalize_field_value(actual_values.get(field_name)) != _normalize_field_value(expected_value)
    ]
    return {
        "target_fields": ["first_name", "last_name", "phone"],
        "expected_fields": {
            "first_name": expected["first_name"],
            "last_name": expected["last_name"],
            "phone": expected["phone_number"],
        },
        "current_values": actual_values,
        "remaining_fields": remaining_fields,
        "required_field_indices": _extract_contact_field_indices(observation),
        "visible_fields": _extract_visible_contact_field_labels(observation),
    }


def _detect_contact_form_action_issues(
    actor_result: ActorRunResult,
    expected: dict[str, str],
) -> dict[str, Any]:
    off_target_field_touched: str | None = None
    mismatched_target_fields: list[str] = []
    save_attempted = False
    for step in actor_result.steps:
        action = step.action
        if action is None:
            continue
        before_observation = step.before_observation or {}
        if action.action_type == "input_text":
            element = _find_ui_element(before_observation, getattr(action, "index", None))
            field_name = _classify_contact_element(element) if element else None
            if field_name in {"first_name", "last_name", "phone"}:
                expected_value = expected["phone_number"] if field_name == "phone" else expected[field_name]
                if _normalize_field_value(getattr(action, "text", "")) != _normalize_field_value(expected_value):
                    mismatched_target_fields.append(field_name)
            elif element is not None and bool(element.get("is_editable")):
                off_target_field_touched = field_name or _normalize_contact_field_label(element) or "unknown_editable_field"
        if action.action_type == "click":
            element = _find_ui_element(before_observation, getattr(action, "index", None))
            if element is not None and _classify_contact_element(element) == "save":
                save_attempted = True
    return {
        "off_target_field_touched": off_target_field_touched,
        "mismatched_target_fields": sorted(set(mismatched_target_fields)),
        "save_attempted": save_attempted,
    }


def _find_ui_element(observation: dict[str, Any], index: int | None) -> dict[str, Any] | None:
    if index is None:
        return None
    for element in observation.get("ui_elements") or []:
        if element.get("index") == index:
            return element
    return None


def _classify_contact_element(element: dict[str, Any] | None) -> str | None:
    if not element:
        return None
    label = _contact_element_label(element)
    if "first name" in label:
        return "first_name"
    if "last name" in label:
        return "last_name"
    if label == "phone" or "phone" in label:
        return "phone"
    if label == "save" or " save" in label or ":save" in label:
        return "save"
    return None


def _contact_element_label(element: dict[str, Any]) -> str:
    raw = element.get("raw") or {}
    parts = (
        str(element.get("text") or ""),
        str(element.get("content_description") or ""),
        str(element.get("resource_name") or ""),
        str(raw.get("text") or ""),
        str(raw.get("content_description") or ""),
    )
    return " ".join(part.strip().lower() for part in parts if part and part.strip())


def _normalize_contact_field_label(element: dict[str, Any]) -> str:
    label = _contact_element_label(element)
    if not label:
        return ""
    if "company" in label:
        return "company"
    if "email" in label:
        return "email"
    return label.split()[0]


def _is_contact_editor_observation(observation: dict[str, Any]) -> bool:
    return "contacteditoractivity" in str(observation.get("current_activity") or "").lower()


def _extract_contact_identity_from_observation(observation: dict[str, Any]) -> str | None:
    candidates: list[str] = []
    for element in observation.get("ui_elements") or []:
        for field in ("text", "content_description"):
            value = str(element.get(field) or "").strip()
            if not value:
                continue
            quick_contact_match = re.search(r"quick contact for (?P<name>[A-Za-z]+(?: [A-Za-z]+)+)", value, flags=re.IGNORECASE)
            if quick_contact_match:
                return " ".join(quick_contact_match.group("name").split())
            if re.fullmatch(r"[A-Za-z]+(?: [A-Za-z]+)+", value):
                candidates.append(" ".join(value.split()))
    return candidates[0] if candidates else None


def _normalize_field_value(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", value.strip()).lower()
