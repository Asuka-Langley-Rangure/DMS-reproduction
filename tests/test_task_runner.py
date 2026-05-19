from __future__ import annotations

import unittest

from dms_reproduction.agents.android_actor import (
    ActorStepResult,
    ActorRunResult,
    ClickAction,
    InputTextAction,
    StatusAction,
)
from dms_reproduction.agents.planner import PlannerResult, PlannerStage, PlannerSubtask, StagePlanResult
from dms_reproduction.agents.task_runner import AndroidTaskRunner, TaskRunConfig
from dms_reproduction.memory import NoOpMemoryProvider


def build_observation(tag: str) -> dict:
    return {
        "goal": "Create a contact",
        "current_activity": f"{tag}/activity",
        "foreground_package": tag,
        "app_name": "com.android.contacts",
        "screen_size": {"width": 1080, "height": 2400},
        "ui_elements": [{"index": 3, "text": "Create contact"}],
        "ui_description": f"UI element 3: text='Create contact' ({tag})",
        "valid_ui_indices": [3],
        "visible_ui_count": 1,
        "clickable_ui_count": 1,
        "non_system_ui_count": 1,
        "editable_ui_count": 0,
        "keyboard_active_context": False,
        "observation_warning": None,
        "observation_consistency": "stable",
        "screenshot_b64": "AAA",
        "labeled_screenshot_b64": "BBB",
        "extra_state": {"tag": tag},
    }


def build_dialer_contacts_observation() -> dict:
    return {
        "goal": "Create a contact",
        "current_activity": "com.google.android.dialer/com.google.android.dialer.extensions.GoogleDialtactsActivity",
        "foreground_package": "com.google.android.dialer",
        "app_name": "com.google.android.dialer",
        "screen_size": {"width": 1080, "height": 2400},
        "ui_elements": [
            {
                "index": 6,
                "text": None,
                "content_description": "Contacts",
                "resource_name": "com.google.android.dialer:id/tab_contacts",
                "class_name": "android.widget.FrameLayout",
                "is_clickable": False,
                "is_editable": False,
                "package_name": "com.google.android.dialer",
                "raw": {"is_selected": True},
            }
        ],
        "ui_description": "UI element 6: content_description='Contacts'",
        "valid_ui_indices": [6],
        "visible_ui_count": 1,
        "clickable_ui_count": 0,
        "non_system_ui_count": 1,
        "editable_ui_count": 0,
        "keyboard_active_context": False,
        "observation_warning": None,
        "observation_consistency": "stable",
        "screenshot_b64": "AAA",
        "labeled_screenshot_b64": "BBB",
        "extra_state": {"tag": "dialer"},
    }


def build_contact_editor_observation(
    *,
    first_name: str = "",
    last_name: str = "",
    phone: str = "",
) -> dict:
    return {
        "goal": "Create a contact",
        "current_activity": "com.google.android.contacts/.activities.ContactEditorActivity",
        "foreground_package": "com.google.android.contacts",
        "app_name": "com.google.android.contacts",
        "screen_size": {"width": 1080, "height": 2400},
        "ui_elements": [
            {
                "index": 5,
                "text": "First name",
                "content_description": "First name",
                "resource_name": "com.google.android.contacts:id/first_name",
                "class_name": "android.widget.EditText",
                "is_clickable": True,
                "is_editable": True,
                "raw": {"value": first_name},
            },
            {
                "index": 6,
                "text": "Last name",
                "content_description": "Last name",
                "resource_name": "com.google.android.contacts:id/last_name",
                "class_name": "android.widget.EditText",
                "is_clickable": True,
                "is_editable": True,
                "raw": {"value": last_name},
            },
            {
                "index": 8,
                "text": "Phone",
                "content_description": "Phone",
                "resource_name": "com.google.android.contacts:id/phone",
                "class_name": "android.widget.EditText",
                "is_clickable": True,
                "is_editable": True,
                "raw": {"value": phone},
            },
        ],
        "ui_description": "Contact editor fields are visible.",
        "valid_ui_indices": [5, 6, 8],
        "visible_ui_count": 3,
        "clickable_ui_count": 3,
        "non_system_ui_count": 3,
        "editable_ui_count": 3,
        "keyboard_active_context": False,
        "observation_warning": None,
        "observation_consistency": "stable",
        "screenshot_b64": "AAA",
        "labeled_screenshot_b64": "BBB",
        "extra_state": {"tag": "contact-editor"},
    }


def build_contact_detail_observation(name: str) -> dict:
    return {
        "goal": "Create a contact",
        "current_activity": "com.google.android.dialer/com.google.android.dialer.extensions.GoogleDialtactsActivity",
        "foreground_package": "com.google.android.dialer",
        "app_name": "com.google.android.dialer",
        "screen_size": {"width": 1080, "height": 2400},
        "ui_elements": [
            {
                "index": 1,
                "text": name,
                "content_description": f"Quick contact for {name}",
                "resource_name": "com.google.android.dialer:id/contact_name",
                "class_name": "android.widget.TextView",
                "is_clickable": False,
                "is_editable": False,
                "package_name": "com.google.android.dialer",
                "raw": {},
            }
        ],
        "ui_description": f"Contact detail page for {name}.",
        "valid_ui_indices": [1],
        "visible_ui_count": 1,
        "clickable_ui_count": 0,
        "non_system_ui_count": 1,
        "editable_ui_count": 0,
        "keyboard_active_context": False,
        "observation_warning": None,
        "observation_consistency": "stable",
        "screenshot_b64": "AAA",
        "labeled_screenshot_b64": "BBB",
        "extra_state": {"tag": "contact-detail"},
    }


def build_actor_step(step_id: int, reason: str, action, status: str, done: bool) -> ActorStepResult:
    return ActorStepResult(
        step_id=step_id,
        reason=reason,
        action=action,
        original_action=action.to_payload() if action else None,
        normalized_action=None,
        action_normalization_applied=False,
        normalization_reason=None,
        corrected_action=None,
        correction_reason=None,
        messages=[],
        prompt_text="prompt",
        raw_response="raw",
        parse_error=None,
        execution_error=None,
        before_observation=build_observation(f"before-{step_id}"),
        after_observation=build_observation(f"after-{step_id}"),
        summary=f"summary-{status}-{step_id}",
        done=done,
        done_reason=status if done else None,
    )


class FakePlanner:
    def __init__(self, responses: list[PlannerResult], stage_plan_results: list[StagePlanResult] | None = None) -> None:
        self.responses = list(responses)
        default_stage_plan = self._default_stage_plan_result_from_responses(self.responses)
        self.stage_plan_results = list(stage_plan_results or [default_stage_plan])
        self.calls = []
        self.stage_plan_calls = []
        self.llm_client = self
        self._last_response = ""
        self._generate_mode = "current"

    class config:
        temperature = 0.0

    @staticmethod
    def _default_stage_plan_result() -> StagePlanResult:
        return StagePlanResult(
            stage_plan=[
                PlannerStage(1, "Open the target app", "The target app is open"),
                PlannerStage(2, "Reach the relevant working area", "The relevant screen or section is visible"),
                PlannerStage(3, "Complete the requested action and verify it", "The requested result is visible and verified"),
            ],
            raw_response='{"stage_plan":[{"stage_id":1,"title":"Open the target app","success_signal":"The target app is open"},{"stage_id":2,"title":"Reach the relevant working area","success_signal":"The relevant screen or section is visible"},{"stage_id":3,"title":"Complete the requested action and verify it","success_signal":"The requested result is visible and verified"}]}',
        )

    @classmethod
    def _default_stage_plan_result_from_responses(cls, responses: list[PlannerResult]) -> StagePlanResult:
        for response in responses:
            if response.stage_plan:
                return StagePlanResult(
                    stage_plan=list(response.stage_plan),
                    raw_response='{"stage_plan":[]}',
                )
        return cls._default_stage_plan_result()

    def build_stage_plan_messages(self, user_goal: str):
        self._generate_mode = "stage_plan"
        self.stage_plan_calls.append({"user_goal": user_goal})
        return [
            {"role": "system", "content": "stage-plan-system"},
            {"role": "user", "content": [{"type": "text", "text": f"goal_only={user_goal}"}]},
        ]

    def build_current_subtasks_messages(
        self,
        user_goal: str,
        stage_plan: list[PlannerStage],
        observation: dict,
        task_history: list[dict] | None = None,
        memory_context: str = "",
    ):
        self._generate_mode = "current"
        self.calls.append(
            {
                "user_goal": user_goal,
                "stage_plan": [stage.to_dict() for stage in stage_plan],
                "observation": observation,
                "task_history": list(task_history or []),
                "memory_context": memory_context,
            }
        )
        return [
            {"role": "system", "content": "planner-system"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"goal={user_goal}; activity={observation['current_activity']}; history={len(task_history or [])}; stages={len(stage_plan)}",
                    }
                ],
            },
        ]

    def build_messages(self, user_goal: str, observation: dict, task_history: list[dict] | None = None, memory_context: str = ""):
        return self.build_current_subtasks_messages(
            user_goal=user_goal,
            stage_plan=[],
            observation=observation,
            task_history=task_history,
            memory_context=memory_context,
        )

    @staticmethod
    def extract_user_text_prompt(messages) -> str:
        return messages[1]["content"][0]["text"]

    @staticmethod
    def messages_to_jsonable(messages):
        return messages

    def generate(self, messages, temperature: float = 0.0) -> str:
        if self._generate_mode == "stage_plan":
            if not self.stage_plan_results:
                response = self._default_stage_plan_result()
            else:
                response = self.stage_plan_results[0]
            self._last_response = response.raw_response or '{"stage_plan":[]}'
            return self._last_response
        if not self.responses:
            raise AssertionError("No planner response queued.")
        response = self.responses[0]
        self._last_response = response.raw_response or '{"tool":"stub"}'
        return self._last_response

    def parse_response(self, raw_response: str) -> PlannerResult:
        response = self.responses.pop(0)
        response.raw_response = raw_response
        return response

    def parse_stage_plan_response(self, raw_response: str) -> StagePlanResult:
        if self.stage_plan_results:
            response = self.stage_plan_results.pop(0)
        else:
            response = self._default_stage_plan_result()
        response.raw_response = raw_response
        return response

class FakeActor:
    def __init__(self, responses: list[ActorRunResult]) -> None:
        self.responses = list(responses)
        self.calls = []

    def run_subtask(self, env, request, observation_adapter) -> ActorRunResult:
        self.calls.append(
            {
                "subtask": request.subtask,
                "observation": request.observation,
                "action_history": list(request.action_history),
                "memory_context": request.memory_context,
            }
        )
        if not self.responses:
            raise AssertionError("No actor response queued.")
        return self.responses.pop(0)


class FakeObservationAdapter:
    def __init__(self, observations: list[dict]) -> None:
        self.observations = list(observations)
        self.calls = []

    def capture_observation(self, env, goal: str, *, step_id: int = 0, include_screenshots: bool = True) -> dict:
        self.calls.append({"goal": goal, "step_id": step_id, "include_screenshots": include_screenshots})
        if not self.observations:
            raise AssertionError("No observation queued.")
        return self.observations.pop(0)


class FakeMemoryProvider(NoOpMemoryProvider):
    def __init__(self) -> None:
        self.context_calls = []
        self.actor_context_calls = []
        self.recorded_events = []
        self.recorded_trajectories = []
        self.reset_calls = 0

    def build_context(self, user_goal: str, observation: dict, task_history: list[dict]) -> str:
        self.context_calls.append(
            {
                "user_goal": user_goal,
                "observation": observation,
                "task_history": list(task_history),
            }
        )
        return f"memory-context-{len(self.context_calls)}"

    def build_actor_context(self, user_goal: str, subtask: str, observation: dict, task_history: list[dict]) -> str:
        self.actor_context_calls.append(
            {
                "user_goal": user_goal,
                "subtask": subtask,
                "observation": observation,
                "task_history": list(task_history),
            }
        )
        return f"actor-memory-context-{len(self.actor_context_calls)}"

    def record(self, event: dict) -> None:
        self.recorded_events.append(event)

    def record_subtask_trajectory(self, record: dict) -> None:
        self.recorded_trajectories.append(record)

    def reset(self) -> None:
        self.reset_calls += 1


class FakeVerifier:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls = []

    def run_verification(self, request):
        from dms_reproduction.agents.verifier import VerifierResult

        self.calls.append(request)
        if not self.responses:
            raise AssertionError("No verifier response queued.")
        payload = self.responses.pop(0)
        return VerifierResult(
            status=payload["status"],
            reason=payload["reason"],
            memory_eligible=payload.get("memory_eligible", False),
            raw_response=payload.get("raw_response", '{"status":"stub"}'),
            parse_error=payload.get("parse_error"),
            prompt_text="verifier prompt",
            messages=[{"role": "user", "content": "verifier prompt"}],
        )


class FakeTask:
    def __init__(self, success_scores: list[float]) -> None:
        self.success_scores = list(success_scores)
        self.initialized = False

    def initialize_task(self, env) -> None:
        self.initialized = True

    def is_successful(self, env) -> float:
        if not self.success_scores:
            raise AssertionError("No success score queued.")
        return self.success_scores.pop(0)


class FakeEnv:
    pass


class TaskRunnerTest(unittest.TestCase):
    def test_complete_goal_with_success_returns_completed(self) -> None:
        planner = FakePlanner([PlannerResult(is_goal_complete=True, completion_message="done")])
        actor = FakeActor([])
        adapter = FakeObservationAdapter([build_observation("initial")])
        runner = AndroidTaskRunner(planner, actor, adapter)
        task = FakeTask([1.0])

        result = runner.run_task(FakeEnv(), task, "Create a contact")

        self.assertEqual(result.status, "completed")
        self.assertTrue(result.final_task_success)
        self.assertEqual(result.completion_message, "done")

    def test_planner_parse_error_returns_planner_error(self) -> None:
        planner = FakePlanner([PlannerResult(is_goal_complete=False, parse_error="bad json")])
        actor = FakeActor([])
        adapter = FakeObservationAdapter([build_observation("initial")])
        runner = AndroidTaskRunner(planner, actor, adapter)

        result = runner.run_task(FakeEnv(), FakeTask([1.0]), "Create a contact")

        self.assertEqual(result.status, "planner_error")
        self.assertIn("Planner parse error", result.completion_message)

    def test_multiple_subtasks_execute_in_order(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(
                    is_goal_complete=False,
                    subtasks=[
                        PlannerSubtask("None", "Open Contacts", "Need app"),
                        PlannerSubtask("Contacts open", "Tap create contact", "Need create flow"),
                    ],
                ),
                PlannerResult(is_goal_complete=True, completion_message="done"),
            ]
        )
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "open app", ClickAction(3), "completed", True)],
                    final_observation=build_observation("after-subtask-1"),
                    last_action={"action_type": "click", "index": 3},
                ),
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "tap create", ClickAction(3), "completed", True)],
                    final_observation=build_observation("after-subtask-2"),
                    last_action={"action_type": "click", "index": 3},
                ),
            ]
        )
        adapter = FakeObservationAdapter([build_observation("initial")])
        runner = AndroidTaskRunner(planner, actor, adapter)
        task = FakeTask([1.0])

        result = runner.run_task(FakeEnv(), task, "Create a contact")

        self.assertEqual(result.status, "completed")
        self.assertEqual(actor.calls[0]["subtask"], "Precondition: None Goal: Open Contacts")
        self.assertEqual(actor.calls[1]["subtask"], "Precondition: Contacts open Goal: Tap create contact")
        self.assertEqual(actor.calls[1]["observation"]["current_activity"], "after-subtask-1/activity")

    def test_actor_failure_replans_with_accumulated_history(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(
                    is_goal_complete=False,
                    subtasks=[PlannerSubtask("None", "Tap create contact", "Need create flow")],
                ),
                PlannerResult(is_goal_complete=True, completion_message="done"),
            ]
        )
        actor = FakeActor(
            [
                ActorRunResult(
                    status="infeasible",
                    steps=[build_actor_step(0, "blocked", StatusAction("infeasible", "blocked"), "infeasible", True)],
                    final_observation=build_observation("after-failure"),
                    completion_message="blocked",
                    last_action={"action_type": "status", "goal_status": "infeasible"},
                )
            ]
        )
        adapter = FakeObservationAdapter([build_observation("initial")])
        runner = AndroidTaskRunner(planner, actor, adapter)
        task = FakeTask([1.0])

        result = runner.run_task(FakeEnv(), task, "Create a contact")

        self.assertEqual(result.status, "completed")
        self.assertEqual(planner.calls[1]["observation"]["current_activity"], "after-failure/activity")
        self.assertGreaterEqual(len(planner.calls[1]["task_history"]), 2)
        self.assertEqual(planner.calls[1]["task_history"][0]["source"], "stage_plan")
        self.assertTrue(any(item.get("source") == "actor" for item in planner.calls[1]["task_history"]))

    def test_actor_error_variants_all_trigger_replan(self) -> None:
        for actor_status in ("parse_error", "execution_error", "step_limit"):
            planner = FakePlanner(
                [
                    PlannerResult(
                        is_goal_complete=False,
                        subtasks=[PlannerSubtask("None", "Tap create contact", "Need create flow")],
                    ),
                    PlannerResult(is_goal_complete=True, completion_message="done"),
                ]
            )
            actor = FakeActor(
                [
                    ActorRunResult(
                        status=actor_status,
                        steps=[build_actor_step(0, f"{actor_status}-reason", ClickAction(3), actor_status, True)],
                        final_observation=build_observation(f"after-{actor_status}"),
                        last_action={"action_type": "click", "index": 3},
                    )
                ]
            )
            adapter = FakeObservationAdapter([build_observation("initial")])
            runner = AndroidTaskRunner(planner, actor, adapter)
            task = FakeTask([1.0])

            result = runner.run_task(FakeEnv(), task, "Create a contact")

            self.assertEqual(result.status, "completed")
            self.assertEqual(planner.calls[1]["observation"]["current_activity"], f"after-{actor_status}/activity")

    def test_round_limit(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(is_goal_complete=False, subtasks=[PlannerSubtask("None", "A", "r")]),
                PlannerResult(is_goal_complete=False, subtasks=[PlannerSubtask("None", "B", "r")]),
            ]
        )
        actor = FakeActor(
            [
                ActorRunResult(status="completed", steps=[build_actor_step(0, "a", ClickAction(3), "completed", True)], final_observation=build_observation("a")),
                ActorRunResult(status="completed", steps=[build_actor_step(0, "b", ClickAction(3), "completed", True)], final_observation=build_observation("b")),
            ]
        )
        adapter = FakeObservationAdapter([build_observation("initial")])
        runner = AndroidTaskRunner(planner, actor, adapter, TaskRunConfig(max_planner_rounds=2))

        result = runner.run_task(FakeEnv(), FakeTask([1.0]), "Create a contact")

        self.assertEqual(result.status, "round_limit")
        self.assertIn("Round limit reached", result.completion_message)

    def test_total_actor_steps_accumulates(self) -> None:
        planner = FakePlanner([PlannerResult(is_goal_complete=False, subtasks=[PlannerSubtask("None", "A", "r")])])
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[
                        build_actor_step(0, "step-1", ClickAction(3), "progress", False),
                        build_actor_step(1, "step-2", StatusAction("complete", "done"), "completed", True),
                    ],
                    final_observation=build_observation("after"),
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                )
            ]
        )
        adapter = FakeObservationAdapter([build_observation("initial")])
        runner = AndroidTaskRunner(planner, actor, adapter, TaskRunConfig(max_planner_rounds=1))

        result = runner.run_task(FakeEnv(), FakeTask([]), "Create a contact")

        self.assertEqual(result.total_actor_steps, 2)

    def test_degraded_observation_sets_replan_reason(self) -> None:
        degraded = build_observation("com.google.android.dialer")
        degraded["foreground_package"] = "com.google.android.dialer"
        degraded["current_activity"] = "com.google.android.dialer/.DialtactsActivity"
        degraded["app_name"] = "com.android.systemui"
        degraded["non_system_ui_count"] = 0
        degraded["observation_warning"] = "Only system UI elements were retained while foreground activity is dialer."
        degraded["observation_consistency"] = "unstable"
        planner = FakePlanner([PlannerResult(is_goal_complete=False, subtasks=[PlannerSubtask("None", "Navigate to contacts", "r")])])
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "done", StatusAction("complete", "done"), "completed", True)],
                    final_observation=degraded,
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                ),
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "done", StatusAction("complete", "done"), "completed", True)],
                    final_observation=degraded,
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                )
            ]
        )
        adapter = FakeObservationAdapter([build_observation("initial"), degraded])
        runner = AndroidTaskRunner(planner, actor, adapter, TaskRunConfig(max_planner_rounds=1))

        result = runner.run_task(FakeEnv(), FakeTask([]), "Create a contact")

        self.assertEqual(result.status, "round_limit")
        self.assertEqual(result.planner_rounds[0].replan_reason, "observation_unstable_persisted")

    def test_suspicious_repeated_completion_sets_reason(self) -> None:
        degraded = build_observation("com.google.android.dialer")
        degraded["observation_warning"] = "UI may be stale after recent navigation."
        planner = FakePlanner(
            [
                PlannerResult(is_goal_complete=False, subtasks=[PlannerSubtask("Phone open", "Navigate to contacts section", "r")]),
                PlannerResult(is_goal_complete=False, subtasks=[PlannerSubtask("Phone open", "Navigate to contacts section", "r")]),
            ]
        )
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "done", StatusAction("complete", "done"), "completed", True)],
                    final_observation=build_observation("after-1"),
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                ),
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "done", StatusAction("complete", "done"), "completed", True)],
                    final_observation=degraded,
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                ),
            ]
        )
        adapter = FakeObservationAdapter([build_observation("initial")])
        runner = AndroidTaskRunner(planner, actor, adapter, TaskRunConfig(max_planner_rounds=2))

        result = runner.run_task(FakeEnv(), FakeTask([]), "Create a contact")

        self.assertEqual(result.status, "round_limit")
        self.assertEqual(result.planner_rounds[-1].replan_reason, "suspicious_subtask_completion_without_progress")
        self.assertIn("repeated actor completion", result.completion_message)

    def test_bad_open_app_subtask_is_not_contact_specific_rewritten(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(
                    is_goal_complete=False,
                    subtasks=[PlannerSubtask("The Phone app is open.", "Click on the Phone app to open it.", "Need app")],
                )
            ]
        )
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "go to contacts", StatusAction("complete", "done"), "completed", True)],
                    final_observation=build_observation("after"),
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                )
            ]
        )
        adapter = FakeObservationAdapter([build_observation("initial")])
        runner = AndroidTaskRunner(planner, actor, adapter, TaskRunConfig(max_planner_rounds=1))

        result = runner.run_task(FakeEnv(), FakeTask([]), "Create a contact")

        self.assertEqual(result.planner_rounds[0].normalized_subtasks[0]["goal"], "Click on the Phone app to open it.")

    def test_contact_entry_atomic_goal_is_not_rewritten_by_runner(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(
                    is_goal_complete=False,
                    subtasks=[PlannerSubtask("Contacts section open.", "Tap the Create new contact button.", "The Create new contact button is visible.")],
                )
            ]
        )
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "reach entry point", StatusAction("complete", "done"), "completed", True)],
                    final_observation=build_observation("after"),
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                )
            ]
        )
        adapter = FakeObservationAdapter([build_observation("initial")])
        runner = AndroidTaskRunner(planner, actor, adapter, TaskRunConfig(max_planner_rounds=1))

        result = runner.run_task(FakeEnv(), FakeTask([]), "Create a contact")

        self.assertEqual(
            result.planner_rounds[0].normalized_subtasks[0]["goal"],
            "Tap the Create new contact button.",
        )

    def test_contact_editor_field_goal_is_not_rewritten_by_runner(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(
                    is_goal_complete=False,
                    subtasks=[PlannerSubtask("The user is in the contact creation screen.", "Enter the first name 'Sara' into the First name field.", "The contact creation screen is ready for input, and the first name field is visible and editable.")],
                )
            ]
        )
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "fill form", StatusAction("complete", "done"), "completed", True)],
                    final_observation=build_contact_editor_observation(),
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                )
            ]
        )
        adapter = FakeObservationAdapter([build_contact_editor_observation()])
        runner = AndroidTaskRunner(planner, actor, adapter, TaskRunConfig(max_planner_rounds=1))

        result = runner.run_task(
            FakeEnv(),
            FakeTask([]),
            "Create a new contact for Sara Khan. Their number is +15102899176.",
        )

        self.assertEqual(
            result.planner_rounds[0].normalized_subtasks[0]["goal"],
            "Enter the first name 'Sara' into the First name field.",
        )

    def test_invalid_planner_subtask_marks_round_invalid(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(
                    is_goal_complete=False,
                    subtasks=[PlannerSubtask("The Contacts app is open.", "Open the Contacts app.", "Need app")],
                )
            ]
        )
        actor = FakeActor([])
        adapter = FakeObservationAdapter([build_observation("initial")])
        runner = AndroidTaskRunner(planner, actor, adapter, TaskRunConfig(max_planner_rounds=1))

        result = runner.run_task(FakeEnv(), FakeTask([]), "Create a contact")

        self.assertEqual(result.status, "round_limit")
        self.assertEqual(result.planner_rounds[0].replan_reason, "planner_subtask_invalid")
        self.assertIn("planner_subtask_invalid", result.completion_message)

    def test_unstable_final_observation_triggers_local_recovery_before_replan(self) -> None:
        unstable = build_observation("after-unstable")
        unstable["foreground_package"] = "com.google.android.contacts"
        unstable["app_name"] = "com.android.systemui"
        unstable["non_system_ui_count"] = 0
        unstable["clickable_ui_count"] = 0
        unstable["observation_warning"] = "Only system UI elements were retained while foreground activity is contacts."
        unstable["observation_consistency"] = "unstable"
        recovered = build_observation("after-recovered")
        planner = FakePlanner(
            [
                PlannerResult(is_goal_complete=False, subtasks=[PlannerSubtask("None", "Open Contacts", "Need app flow")]),
                PlannerResult(is_goal_complete=True, completion_message="done"),
            ]
        )
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "done", StatusAction("complete", "done"), "completed", True)],
                    final_observation=unstable,
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                )
            ]
        )
        adapter = FakeObservationAdapter([build_observation("initial"), recovered])
        runner = AndroidTaskRunner(planner, actor, adapter, TaskRunConfig(max_planner_rounds=2))

        result = runner.run_task(FakeEnv(), FakeTask([1.0]), "Create a contact")

        self.assertEqual(result.status, "completed")
        self.assertEqual(planner.calls[1]["observation"]["current_activity"], "after-recovered/activity")

    def test_planner_visible_claim_without_target_is_vetoed(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(
                    is_goal_complete=False,
                    subtasks=[PlannerSubtask("Contacts open", "Create new contact", "Create new contact is clearly visible on the screen")],
                )
            ]
        )
        actor = FakeActor([])
        adapter = FakeObservationAdapter([build_observation("initial")])
        runner = AndroidTaskRunner(planner, actor, adapter, TaskRunConfig(max_planner_rounds=1))

        result = runner.run_task(FakeEnv(), FakeTask([]), "Create a contact")

        self.assertEqual(result.status, "round_limit")
        self.assertEqual(result.planner_rounds[0].replan_reason, "planner_subtask_not_grounded_in_observation")

    def test_runner_marks_open_phone_app_completed_after_successful_first_step(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(is_goal_complete=False, subtasks=[PlannerSubtask("None", "Open the Phone app.", "Need phone app")]),
                PlannerResult(is_goal_complete=True, completion_message="done"),
            ]
        )
        step_one = build_actor_step(0, "tap phone", ClickAction(3), "progress", False)
        step_one.after_observation = build_dialer_contacts_observation()
        step_two = build_actor_step(1, "tap contacts", ClickAction(6), "parse_error", True)
        step_two.before_observation = build_dialer_contacts_observation()
        step_two.after_observation = None
        step_two.parse_error = "click.index must point to a clickable UI element; non-clickable element selected."
        actor = FakeActor(
            [
                ActorRunResult(
                    status="parse_error",
                    steps=[step_one, step_two],
                    final_observation=build_dialer_contacts_observation(),
                    last_action={"action_type": "click", "index": 6},
                )
            ]
        )
        adapter = FakeObservationAdapter([build_observation("initial")])
        runner = AndroidTaskRunner(planner, actor, adapter, TaskRunConfig(max_planner_rounds=2))

        result = runner.run_task(FakeEnv(), FakeTask([1.0]), "Create a contact")

        self.assertEqual(result.status, "completed")
        subtask_run = result.planner_rounds[0].subtask_runs[0]
        self.assertEqual(subtask_run.subtask_success_check["success_rule"], "foreground_package_matches_phone_app")
        self.assertTrue(subtask_run.subtask_success_check["runner_overrode_to_completed"])

    def test_success_override_is_blocked_on_unstable_observation(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(is_goal_complete=False, subtasks=[PlannerSubtask("None", "Open the Phone app.", "Need phone app")]),
            ]
        )
        unstable_dialer = build_dialer_contacts_observation()
        unstable_dialer["app_name"] = "com.android.systemui"
        unstable_dialer["non_system_ui_count"] = 0
        unstable_dialer["clickable_ui_count"] = 0
        unstable_dialer["observation_warning"] = "Only system UI elements were retained while foreground activity is dialer."
        unstable_dialer["observation_consistency"] = "unstable"
        actor = FakeActor(
            [
                ActorRunResult(
                    status="parse_error",
                    steps=[build_actor_step(0, "tap phone", ClickAction(3), "parse_error", True)],
                    final_observation=unstable_dialer,
                    last_action={"action_type": "click", "index": 3},
                ),
                ActorRunResult(
                    status="parse_error",
                    steps=[build_actor_step(0, "tap phone again", ClickAction(3), "parse_error", True)],
                    final_observation=unstable_dialer,
                    last_action={"action_type": "click", "index": 3},
                ),
            ]
        )
        adapter = FakeObservationAdapter([build_observation("initial"), unstable_dialer])
        runner = AndroidTaskRunner(planner, actor, adapter, TaskRunConfig(max_planner_rounds=1))

        result = runner.run_task(FakeEnv(), FakeTask([]), "Create a contact")

        self.assertEqual(result.status, "round_limit")
        subtask_run = result.planner_rounds[0].subtask_runs[0]
        self.assertIsNone(subtask_run.subtask_success_check["success_rule"])
        self.assertEqual(
            subtask_run.subtask_success_check["override_blocked_reason"],
            "observation_not_reliable_for_completion",
        )

    def test_runner_marks_contacts_section_completed_when_selected(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(is_goal_complete=False, subtasks=[PlannerSubtask("Phone open", "Navigate to the contacts section.", "Need contacts section")]),
                PlannerResult(is_goal_complete=True, completion_message="done"),
            ]
        )
        step_one = build_actor_step(0, "contacts selected", ClickAction(6), "parse_error", True)
        step_one.before_observation = build_dialer_contacts_observation()
        step_one.after_observation = None
        step_one.parse_error = "click.index must point to a clickable UI element; non-clickable element selected."
        actor = FakeActor(
            [
                ActorRunResult(
                    status="parse_error",
                    steps=[step_one],
                    final_observation=build_dialer_contacts_observation(),
                    last_action={"action_type": "click", "index": 6},
                )
            ]
        )
        adapter = FakeObservationAdapter([build_dialer_contacts_observation()])
        runner = AndroidTaskRunner(planner, actor, adapter, TaskRunConfig(max_planner_rounds=2))

        result = runner.run_task(FakeEnv(), FakeTask([1.0]), "Create a contact")

        self.assertEqual(result.status, "completed")
        self.assertEqual(
            result.planner_rounds[0].subtask_runs[0].subtask_success_check["success_rule"],
            "contacts_section_selected",
        )

    def test_open_phone_app_goal_is_not_rewritten_when_dialer_is_already_open(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(is_goal_complete=False, subtasks=[PlannerSubtask("None", "Open the Phone app.", "Need to begin contact creation")]),
            ]
        )
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "open phone", StatusAction("complete", "done"), "completed", True)],
                    final_observation=build_dialer_contacts_observation(),
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                )
            ]
        )
        dialer = build_dialer_contacts_observation()
        dialer["ui_elements"][0]["raw"]["is_selected"] = False
        adapter = FakeObservationAdapter([dialer])
        runner = AndroidTaskRunner(planner, actor, adapter, TaskRunConfig(max_planner_rounds=1))

        result = runner.run_task(FakeEnv(), FakeTask([]), "Create a contact")

        self.assertEqual(result.planner_rounds[0].normalized_subtasks[0]["goal"], "Open the Phone app.")

    def test_text_entry_goal_is_completed_when_after_observation_contains_value(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(is_goal_complete=False, subtasks=[PlannerSubtask("Create contact screen open", "Enter the first name 'Sara'.", "Need first name")]),
                PlannerResult(is_goal_complete=True, completion_message="done"),
            ]
        )
        before = build_observation("before-text")
        before["ui_elements"] = [
            {"index": 5, "text": "First name", "content_description": "First name", "is_editable": True, "is_clickable": True, "raw": {}}
        ]
        before["valid_ui_indices"] = [5]
        after = build_observation("after-text")
        after["ui_elements"] = [
            {"index": 5, "text": "First name", "content_description": "First name", "is_editable": True, "is_clickable": True, "raw": {"value": "Sara"}}
        ]
        after["valid_ui_indices"] = [5]
        step = build_actor_step(0, "enter text", ClickAction(3), "progress", False)
        step.before_observation = before
        step.after_observation = after
        actor = FakeActor(
            [
                ActorRunResult(
                    status="step_limit",
                    steps=[step],
                    final_observation=after,
                    last_action={"action_type": "input_text", "index": 5, "text": "Sara"},
                )
            ]
        )
        adapter = FakeObservationAdapter([before])
        runner = AndroidTaskRunner(planner, actor, adapter, TaskRunConfig(max_planner_rounds=2))

        result = runner.run_task(FakeEnv(), FakeTask([1.0]), "Create a contact")

        self.assertEqual(result.status, "completed")
        self.assertEqual(
            result.planner_rounds[0].subtask_runs[0].subtask_success_check["success_rule"],
            "text_entry_first_name_matches_expected_value",
        )

    def test_contact_form_fill_goal_is_completed_when_all_fields_match(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(
                    is_goal_complete=False,
                    subtasks=[PlannerSubtask("Create contact screen open", "Fill in Sara Khan and +15102899176 in the contact form.", "Need contact details")],
                ),
                PlannerResult(is_goal_complete=True, completion_message="done"),
            ]
        )
        before = build_contact_editor_observation()
        after = build_contact_editor_observation(first_name="Sara", last_name="Khan", phone="+15102899176")
        step = build_actor_step(0, "fill form", ClickAction(5), "progress", False)
        step.before_observation = before
        step.after_observation = after
        actor = FakeActor(
            [
                ActorRunResult(
                    status="step_limit",
                    steps=[step],
                    final_observation=after,
                    last_action={"action_type": "input_text", "index": 8, "text": "+15102899176"},
                )
            ]
        )
        adapter = FakeObservationAdapter([before])
        runner = AndroidTaskRunner(planner, actor, adapter, TaskRunConfig(max_planner_rounds=2))

        result = runner.run_task(FakeEnv(), FakeTask([1.0]), "Create a contact")

        self.assertEqual(result.status, "completed")
        success_check = result.planner_rounds[0].subtask_runs[0].subtask_success_check
        self.assertEqual(success_check["success_rule"], "contact_form_fields_match_expected_values")
        self.assertEqual(success_check["form_fill_progress"]["completed_fields"], ["first_name", "last_name", "phone"])

    def test_contact_form_fill_goal_records_partial_progress_summary(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(
                    is_goal_complete=False,
                    subtasks=[PlannerSubtask("Create contact screen open", "Fill in Sara Khan and +15102899176 in the contact form.", "Need contact details")],
                ),
                PlannerResult(is_goal_complete=True, completion_message="done"),
            ]
        )
        before = build_contact_editor_observation()
        after = build_contact_editor_observation(first_name="Sara", last_name="Khan", phone="")
        step = build_actor_step(0, "fill first and last name", ClickAction(5), "progress", False)
        step.before_observation = before
        step.after_observation = after
        actor = FakeActor(
            [
                ActorRunResult(
                    status="step_limit",
                    steps=[step],
                    final_observation=after,
                    last_action={"action_type": "input_text", "index": 6, "text": "Khan"},
                )
            ]
        )
        adapter = FakeObservationAdapter([before])
        runner = AndroidTaskRunner(planner, actor, adapter, TaskRunConfig(max_planner_rounds=2))

        result = runner.run_task(FakeEnv(), FakeTask([1.0]), "Create a contact")

        self.assertEqual(result.status, "completed")
        success_check = result.planner_rounds[0].subtask_runs[0].subtask_success_check
        self.assertTrue(success_check["progress_made"])
        self.assertIsNone(success_check["success_rule"])
        self.assertEqual(success_check["form_fill_progress"]["remaining_fields"], ["phone"])
        planner_history = planner.calls[1]["task_history"]
        self.assertTrue(
            any(
                item.get("source") == "subtask_summary"
                and item.get("status") == "partial_progress"
                and "remaining fields: phone" in str(item.get("reason") or "").lower()
                for item in planner_history
            )
        )

    def test_contact_form_fill_marks_off_target_field_touch_as_field_misgrounded(self) -> None:
        planner = FakePlanner(
            [PlannerResult(is_goal_complete=False, subtasks=[PlannerSubtask("Create contact screen open", "Fill in Sara Khan and +15102899176 in the contact form.", "Need contact details")])]
        )
        before = build_contact_editor_observation()
        before["ui_elements"].append(
            {
                "index": 7,
                "text": "Company",
                "content_description": "Company",
                "resource_name": "com.google.android.contacts:id/company",
                "class_name": "android.widget.EditText",
                "is_clickable": True,
                "is_editable": True,
                "raw": {"value": ""},
            }
        )
        before["valid_ui_indices"] = [5, 6, 7, 8]
        step = build_actor_step(0, "fill company", InputTextAction(7, "ABC Inc."), "progress", False)
        step.before_observation = before
        step.after_observation = before
        actor = FakeActor(
            [
                ActorRunResult(
                    status="step_limit",
                    steps=[step],
                    final_observation=before,
                    last_action={"action_type": "input_text", "index": 7, "text": "ABC Inc."},
                )
            ]
        )
        adapter = FakeObservationAdapter([before])
        runner = AndroidTaskRunner(planner, actor, adapter, TaskRunConfig(max_planner_rounds=1))

        result = runner.run_task(FakeEnv(), FakeTask([]), "Create a new contact for Sara Khan. their number is +15102899176.")

        self.assertEqual(result.status, "round_limit")
        self.assertEqual(result.planner_rounds[0].replan_reason, "field_misgrounded")
        success_check = result.planner_rounds[0].subtask_runs[0].subtask_success_check
        self.assertEqual(success_check["off_target_field_touched"], "company")
        self.assertEqual(success_check["terminal_failure_reason"], "field_misgrounded")

    def test_contact_form_save_without_validator_success_sets_saved_but_task_check_failed(self) -> None:
        planner = FakePlanner(
            [PlannerResult(is_goal_complete=False, subtasks=[PlannerSubtask("Create contact screen open", "Fill in Amir Garcia and +17839000076 in the contact form.", "Need contact details")])]
        )
        before = build_contact_editor_observation(first_name="Amir", last_name="Garcia", phone="+17839000076")
        before["ui_elements"].append(
            {
                "index": 2,
                "text": "Save",
                "content_description": "Save",
                "resource_name": "com.google.android.contacts:id/toolbar_button",
                "class_name": "android.widget.Button",
                "is_clickable": True,
                "is_editable": False,
                "raw": {},
            }
        )
        before["valid_ui_indices"] = [2, 5, 6, 8]
        saved_detail = build_contact_detail_observation("Amir Garcia")
        step = build_actor_step(0, "save contact", ClickAction(2), "completed", True)
        step.before_observation = before
        step.after_observation = saved_detail
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[step],
                    final_observation=saved_detail,
                    completion_message="saved",
                    last_action={"action_type": "click", "index": 2},
                )
            ]
        )
        adapter = FakeObservationAdapter([before, saved_detail])
        runner = AndroidTaskRunner(planner, actor, adapter, TaskRunConfig(max_planner_rounds=1))

        result = runner.run_task(FakeEnv(), FakeTask([0.0]), "Create a new contact for Amir Garcia. their number is +17839000076.")

        self.assertEqual(result.status, "round_limit")
        self.assertEqual(result.planner_rounds[0].replan_reason, "saved_but_task_check_failed")
        success_check = result.planner_rounds[0].subtask_runs[0].subtask_success_check
        self.assertTrue(success_check["saved_but_task_check_failed"])
        self.assertEqual(success_check["form_fill_stage"], "saved_but_not_validated")
        self.assertEqual(success_check["observed_contact_identity"], "Amir Garcia")

    def test_contact_form_save_with_wrong_identity_sets_saved_with_wrong_identity(self) -> None:
        planner = FakePlanner(
            [PlannerResult(is_goal_complete=False, subtasks=[PlannerSubtask("Create contact screen open", "Fill in Amir Garcia and +17839000076 in the contact form.", "Need contact details")])]
        )
        before = build_contact_editor_observation(first_name="Amir", last_name="Doe", phone="+17839000076")
        before["ui_elements"].append(
            {
                "index": 2,
                "text": "Save",
                "content_description": "Save",
                "resource_name": "com.google.android.contacts:id/toolbar_button",
                "class_name": "android.widget.Button",
                "is_clickable": True,
                "is_editable": False,
                "raw": {},
            }
        )
        before["valid_ui_indices"] = [2, 5, 6, 8]
        wrong_detail = build_contact_detail_observation("Amir Doe")
        step = build_actor_step(0, "save wrong contact", ClickAction(2), "completed", True)
        step.before_observation = before
        step.after_observation = wrong_detail
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[step],
                    final_observation=wrong_detail,
                    completion_message="saved",
                    last_action={"action_type": "click", "index": 2},
                )
            ]
        )
        adapter = FakeObservationAdapter([before, wrong_detail])
        runner = AndroidTaskRunner(planner, actor, adapter, TaskRunConfig(max_planner_rounds=1))

        result = runner.run_task(FakeEnv(), FakeTask([0.0]), "Create a new contact for Amir Garcia. their number is +17839000076.")

        self.assertEqual(result.status, "round_limit")
        self.assertEqual(result.planner_rounds[0].replan_reason, "saved_with_wrong_identity")
        success_check = result.planner_rounds[0].subtask_runs[0].subtask_success_check
        self.assertTrue(success_check["saved_with_wrong_identity"])
        self.assertEqual(success_check["observed_contact_identity"], "Amir Doe")

    def test_memory_provider_is_called_and_records_structured_events(self) -> None:
        memory = FakeMemoryProvider()
        planner = FakePlanner(
            [
                PlannerResult(is_goal_complete=False, subtasks=[PlannerSubtask("None", "Open the Phone app.", "Need phone app")]),
                PlannerResult(is_goal_complete=True, completion_message="done"),
            ]
        )
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "open phone", StatusAction("complete", "done"), "completed", True)],
                    final_observation=build_dialer_contacts_observation(),
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                )
            ]
        )
        adapter = FakeObservationAdapter([build_observation("initial")])
        runner = AndroidTaskRunner(planner, actor, adapter, memory_provider=memory, config=TaskRunConfig(max_planner_rounds=2))

        result = runner.run_task(FakeEnv(), FakeTask([1.0]), "Create a contact")

        self.assertEqual(result.status, "completed")
        self.assertEqual(memory.reset_calls, 1)
        self.assertGreaterEqual(len(memory.context_calls), 1)
        self.assertEqual(len(memory.actor_context_calls), 1)
        self.assertEqual(planner.calls[0]["memory_context"], "memory-context-1")
        self.assertEqual(actor.calls[0]["memory_context"], "actor-memory-context-1")
        self.assertEqual(len(memory.recorded_events), 1)
        event = memory.recorded_events[0]
        self.assertEqual(event["user_goal"], "Create a contact")
        self.assertEqual(event["round_id"], 1)
        self.assertEqual(event["subtask"], "Precondition: None Goal: Open the Phone app.")
        self.assertIn("observation_digest", event)
        self.assertEqual(len(memory.recorded_trajectories), 1)
        trajectory_record = memory.recorded_trajectories[0]
        self.assertEqual(trajectory_record["subtask_text"], "Precondition: None Goal: Open the Phone app.")
        self.assertEqual(trajectory_record["goal"], "Open the Phone app.")
        self.assertEqual(trajectory_record["verifier_status"], "success")
        self.assertEqual(len(trajectory_record["trajectory"]), 1)

    def test_planner_complete_goal_with_failed_task_check_replans_instead_of_stopping(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(is_goal_complete=True, completion_message="created"),
                PlannerResult(is_goal_complete=True, completion_message="created for real"),
            ]
        )
        actor = FakeActor([])
        adapter = FakeObservationAdapter([build_observation("initial")])
        runner = AndroidTaskRunner(planner, actor, adapter, TaskRunConfig(max_planner_rounds=2))

        result = runner.run_task(FakeEnv(), FakeTask([0.0, 1.0]), "Create a contact")

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(planner.calls), 2)
        self.assertEqual(len(planner.stage_plan_calls), 1)
        self.assertTrue(
            any(
                item.get("status") == "planner_complete_but_task_check_failed"
                for item in planner.calls[1]["task_history"]
            )
        )
        self.assertIn("repair, verification, or progress-making subtask", planner.calls[1]["task_history"][-1].get("summary", ""))

    def test_second_pass_current_stage_id_is_preserved(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(
                    is_goal_complete=False,
                    subtasks=[PlannerSubtask("None", "Open the Phone app.", "Need app")],
                ),
                PlannerResult(
                    is_goal_complete=False,
                    subtasks=[PlannerSubtask("Phone app is open.", "Reach the contact creation entry point.", "Need new contact flow")],
                    current_stage_id=2,
                ),
                PlannerResult(is_goal_complete=True, completion_message="done"),
            ]
        )
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "open phone", StatusAction("complete", "done"), "completed", True)],
                    final_observation=build_dialer_contacts_observation(),
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                ),
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "reach entry", StatusAction("complete", "done"), "completed", True)],
                    final_observation=build_observation("after-entry"),
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                ),
            ]
        )
        adapter = FakeObservationAdapter([build_observation("initial")])
        runner = AndroidTaskRunner(planner, actor, adapter, config=TaskRunConfig(max_planner_rounds=3))

        result = runner.run_task(FakeEnv(), FakeTask([1.0]), "Create a contact")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.planner_rounds[1].planner_result["current_stage_id"], 2)

    def test_covered_stage_ids_advance_remembered_stage_cursor_after_success(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(
                    is_goal_complete=False,
                    stage_plan=[
                        PlannerStage(1, "Open the Phone app", "Phone app is foreground"),
                        PlannerStage(2, "Reach the contact creation entry point", "Create contact UI is visible"),
                        PlannerStage(3, "Fill required contact fields", "Required fields are populated"),
                        PlannerStage(4, "Save and verify the created contact", "Saved contact matches the request"),
                    ],
                    current_stage_id=2,
                    covered_stage_ids=[2, 3],
                    subtasks=[PlannerSubtask("Phone app is open.", "Reach and use the contact creation flow.", "Need entry and form")],
                ),
                PlannerResult(is_goal_complete=True, completion_message="done"),
            ]
        )
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "merged progress", StatusAction("complete", "done"), "completed", True)],
                    final_observation=build_contact_editor_observation(first_name="Sara", last_name="Khan", phone="+15102899176"),
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                )
            ]
        )
        verifier = FakeVerifier([{"status": "success", "reason": "Merged stages succeeded.", "memory_eligible": True}])
        adapter = FakeObservationAdapter([build_dialer_contacts_observation()])
        runner = AndroidTaskRunner(planner, actor, adapter, verifier=verifier, config=TaskRunConfig(max_planner_rounds=2))

        result = runner.run_task(FakeEnv(), FakeTask([1.0]), "Create a contact")

        self.assertEqual(result.status, "completed")
        stage_entry = next(item for item in reversed(planner.calls[1]["task_history"]) if item.get("source") == "stage_plan")
        self.assertEqual(stage_entry["covered_stage_ids"], [2, 3])
        self.assertEqual(stage_entry["current_stage_id"], 4)

    def test_invalid_covered_stage_ids_replans_instead_of_accepting_merge(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(
                    is_goal_complete=False,
                    subtasks=[PlannerSubtask("None", "Open the Phone app.", "Need app")],
                ),
                PlannerResult(
                    is_goal_complete=False,
                    current_stage_id=2,
                    covered_stage_ids=[2, 99],
                    subtasks=[PlannerSubtask("Phone app is open.", "Reach and use the contact creation flow.", "Need entry and form")],
                ),
                PlannerResult(is_goal_complete=True, completion_message="done"),
            ]
        )
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "open phone", StatusAction("complete", "done"), "completed", True)],
                    final_observation=build_dialer_contacts_observation(),
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                )
            ]
        )
        adapter = FakeObservationAdapter([build_observation("initial")])
        runner = AndroidTaskRunner(planner, actor, adapter, config=TaskRunConfig(max_planner_rounds=3))

        result = runner.run_task(FakeEnv(), FakeTask([1.0]), "Create a contact")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.planner_rounds[1].replan_reason, "planner_covered_stage_ids_invalid")
        self.assertTrue(
            any(
                item.get("status") == "planner_covered_stage_ids_invalid"
                for item in planner.calls[2]["task_history"]
            )
        )

    def test_invalid_second_pass_current_stage_id_replans_instead_of_falling_back(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(
                    is_goal_complete=False,
                    subtasks=[PlannerSubtask("None", "Open the Phone app.", "Need app")],
                ),
                PlannerResult(
                    is_goal_complete=False,
                    subtasks=[PlannerSubtask("Phone app is open.", "Reach the contact creation entry point.", "Need new contact flow")],
                    current_stage_id=99,
                ),
                PlannerResult(is_goal_complete=True, completion_message="done"),
            ]
        )
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "open phone", StatusAction("complete", "done"), "completed", True)],
                    final_observation=build_dialer_contacts_observation(),
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                )
            ]
        )
        adapter = FakeObservationAdapter([build_observation("initial")])
        runner = AndroidTaskRunner(planner, actor, adapter, config=TaskRunConfig(max_planner_rounds=3))

        result = runner.run_task(FakeEnv(), FakeTask([1.0]), "Create a contact")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.planner_rounds[1].replan_reason, "planner_current_stage_invalid")
        self.assertTrue(
            any(
                item.get("status") == "planner_current_stage_invalid"
                for item in planner.calls[2]["task_history"]
            )
        )

    def test_keyboard_active_context_is_not_added_to_warning_history(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(is_goal_complete=False, subtasks=[PlannerSubtask("Create contact screen open", "Enter the first name 'Sara'.", "Need first name")]),
                PlannerResult(is_goal_complete=True, completion_message="done"),
            ]
        )
        observation = build_observation("ime")
        observation["keyboard_active_context"] = True
        observation["observation_warning"] = "Soft keyboard is active while editing fields in com.google.android.contacts."
        actor = FakeActor(
            [
                ActorRunResult(
                    status="execution_error",
                    steps=[build_actor_step(0, "enter text", ClickAction(3), "execution_error", True)],
                    final_observation=observation,
                    last_action={"action_type": "click", "index": 3},
                )
            ]
        )
        adapter = FakeObservationAdapter([observation])
        runner = AndroidTaskRunner(planner, actor, adapter, TaskRunConfig(max_planner_rounds=2))

        result = runner.run_task(FakeEnv(), FakeTask([1.0]), "Create a contact")

        planner_history = planner.calls[1]["task_history"]
        self.assertTrue(planner_history)
        self.assertFalse(any(item.get("status") == "warning" for item in planner_history))
        self.assertEqual(result.status, "completed")

    def test_verifier_success_marks_subtask_completed(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(is_goal_complete=False, subtasks=[PlannerSubtask("None", "Open Contacts", "Need app")]),
                PlannerResult(is_goal_complete=True, completion_message="done"),
            ]
        )
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "done", StatusAction("complete", "done"), "completed", True)],
                    final_observation=build_observation("after"),
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                )
            ]
        )
        verifier = FakeVerifier([{"status": "success", "reason": "Contacts is visible.", "memory_eligible": True}])
        adapter = FakeObservationAdapter([build_observation("initial")])
        runner = AndroidTaskRunner(planner, actor, adapter, verifier=verifier, config=TaskRunConfig(max_planner_rounds=2))

        result = runner.run_task(FakeEnv(), FakeTask([1.0]), "Create a contact")

        self.assertEqual(result.status, "completed")
        subtask_run = result.planner_rounds[0].subtask_runs[0]
        self.assertEqual(subtask_run.subtask_verification["status"], "success")
        self.assertTrue(subtask_run.subtask_verification["memory_eligible"])

    def test_verifier_uncertain_replans_even_if_actor_completed(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(is_goal_complete=False, subtasks=[PlannerSubtask("None", "Open Contacts", "Need app")]),
                PlannerResult(is_goal_complete=True, completion_message="done"),
            ]
        )
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "done", StatusAction("complete", "done"), "completed", True)],
                    final_observation=build_observation("after"),
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                )
            ]
        )
        verifier = FakeVerifier([{"status": "uncertain", "reason": "Need stronger evidence.", "memory_eligible": False}])
        adapter = FakeObservationAdapter([build_observation("initial")])
        runner = AndroidTaskRunner(planner, actor, adapter, verifier=verifier, config=TaskRunConfig(max_planner_rounds=2))

        result = runner.run_task(FakeEnv(), FakeTask([1.0]), "Create a contact")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.planner_rounds[0].replan_reason, "verifier_uncertain")
        self.assertEqual(planner.calls[1]["task_history"][0]["source"], "stage_plan")
        self.assertTrue(any(item.get("source") == "actor" for item in planner.calls[1]["task_history"]))

    def test_verifier_uses_final_frame_when_actor_never_completed(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(is_goal_complete=False, subtasks=[PlannerSubtask("None", "Open Contacts", "Need app")]),
                PlannerResult(is_goal_complete=True, completion_message="done"),
            ]
        )
        after = build_observation("after")
        actor = FakeActor(
            [
                ActorRunResult(
                    status="step_limit",
                    steps=[build_actor_step(0, "tap", ClickAction(3), "progress", False)],
                    final_observation=after,
                    completion_message="",
                    last_action={"action_type": "click", "index": 3},
                )
            ]
        )
        verifier = FakeVerifier([{"status": "success", "reason": "Final frame shows success.", "memory_eligible": True}])
        adapter = FakeObservationAdapter([build_observation("initial")])
        runner = AndroidTaskRunner(planner, actor, adapter, verifier=verifier, config=TaskRunConfig(max_planner_rounds=2))

        result = runner.run_task(FakeEnv(), FakeTask([1.0]), "Create a contact")

        self.assertEqual(result.status, "completed")
        self.assertEqual(verifier.calls[0].evidence_source, "final_after_observation")

    def test_memory_event_records_verifier_fields(self) -> None:
        memory = FakeMemoryProvider()
        planner = FakePlanner(
            [
                PlannerResult(is_goal_complete=False, subtasks=[PlannerSubtask("None", "Open Contacts", "Need app")]),
                PlannerResult(is_goal_complete=True, completion_message="done"),
            ]
        )
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "done", StatusAction("complete", "done"), "completed", True)],
                    final_observation=build_observation("after"),
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                )
            ]
        )
        verifier = FakeVerifier([{"status": "success", "reason": "Verified", "memory_eligible": True}])
        adapter = FakeObservationAdapter([build_observation("initial")])
        runner = AndroidTaskRunner(
            planner,
            actor,
            adapter,
            verifier=verifier,
            memory_provider=memory,
            config=TaskRunConfig(max_planner_rounds=2),
        )

        result = runner.run_task(FakeEnv(), FakeTask([1.0]), "Create a contact")

        self.assertEqual(result.status, "completed")
        event = memory.recorded_events[0]
        self.assertEqual(event["verifier_status"], "success")
        self.assertEqual(event["verifier_reason"], "Verified")
        self.assertTrue(event["memory_eligible"])
        self.assertEqual(event["verifier_evidence_source"], "actor_completed_frame")

    def test_stage_plan_is_persisted_into_next_planner_history(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(
                    is_goal_complete=False,
                    stage_plan=[
                        PlannerStage(1, "Open the Phone app", "Phone app is foreground"),
                        PlannerStage(2, "Reach the contact creation entry point", "Create new contact is visible"),
                    ],
                    current_stage_id=2,
                    subtasks=[PlannerSubtask("None", "Reach the contact creation entry point.", "Need the entry point")],
                ),
                PlannerResult(is_goal_complete=True, completion_message="done"),
            ]
        )
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "entry reached", StatusAction("complete", "done"), "completed", True)],
                    final_observation=build_observation("after"),
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                )
            ]
        )
        verifier = FakeVerifier([{"status": "success", "reason": "Entry is visible.", "memory_eligible": True}])
        adapter = FakeObservationAdapter([build_observation("initial")])
        runner = AndroidTaskRunner(planner, actor, adapter, verifier=verifier, config=TaskRunConfig(max_planner_rounds=2))

        result = runner.run_task(FakeEnv(), FakeTask([1.0]), "Create a contact")

        self.assertEqual(result.status, "completed")
        self.assertTrue(
            any(item.get("source") == "stage_plan" for item in planner.calls[1]["task_history"])
        )
        stage_entry = next(item for item in reversed(planner.calls[1]["task_history"]) if item.get("source") == "stage_plan")
        self.assertEqual(stage_entry["current_stage_id"], 2)
        self.assertEqual(len(stage_entry["stage_plan"]), 2)

    def test_contact_stage_plan_is_not_rewritten_by_runner(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(
                    is_goal_complete=False,
                    stage_plan=[
                        PlannerStage(1, "Open the Phone app", "Phone app is foreground"),
                        PlannerStage(2, "Reach the contact creation entry point", "Create contact UI is visible"),
                        PlannerStage(3, "Fill in required contact fields", "Required fields are populated"),
                    ],
                    current_stage_id=1,
                    subtasks=[PlannerSubtask("Phone app open", "Navigate to the contacts section.", "Need contacts first")],
                ),
                PlannerResult(is_goal_complete=True, completion_message="done"),
            ]
        )
        contact_editor = build_contact_editor_observation()
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "fill fields", StatusAction("complete", "done"), "completed", True)],
                    final_observation=contact_editor,
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                )
            ]
        )
        verifier = FakeVerifier([{"status": "success", "reason": "Fields were filled.", "memory_eligible": True}])
        adapter = FakeObservationAdapter([contact_editor])
        runner = AndroidTaskRunner(planner, actor, adapter, verifier=verifier, config=TaskRunConfig(max_planner_rounds=2))

        result = runner.run_task(
            FakeEnv(),
            FakeTask([1.0]),
            "Create a new contact for Sara Khan. their number is +15102899176.",
        )

        self.assertEqual(result.status, "completed")
        planner_result = result.planner_rounds[0].planner_result
        self.assertEqual(planner_result["current_stage_id"], 1)
        self.assertEqual(len(planner_result["stage_plan"]), 3)
        self.assertEqual(planner_result["subtasks"][0]["goal"], "Navigate to the contacts section.")
        self.assertEqual(result.planner_rounds[0].normalized_subtasks[0]["goal"], "Navigate to the contacts section.")

    def test_contact_stage_one_subtask_is_not_promoted_to_entry_point_from_reason_only(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(
                    is_goal_complete=False,
                    stage_plan=[
                        PlannerStage(1, "Open the Phone app", "Phone app is foreground"),
                        PlannerStage(2, "Reach the contact creation entry point", "Create new contact is visible"),
                        PlannerStage(3, "Fill in required contact fields", "Required fields are populated"),
                        PlannerStage(4, "Save and verify the created contact", "Saved contact matches the request"),
                    ],
                    current_stage_id=1,
                    subtasks=[PlannerSubtask("None", "Open the Phone app.", "The Phone app must be open before contact creation can continue.")],
                ),
                PlannerResult(is_goal_complete=True, completion_message="done"),
            ]
        )
        launcher_observation = build_observation("initial")
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "open app", StatusAction("complete", "done"), "completed", True)],
                    final_observation=launcher_observation,
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                )
            ]
        )
        verifier = FakeVerifier([{"status": "success", "reason": "The app was opened.", "memory_eligible": True}])
        adapter = FakeObservationAdapter([launcher_observation])
        runner = AndroidTaskRunner(planner, actor, adapter, verifier=verifier, config=TaskRunConfig(max_planner_rounds=2))

        result = runner.run_task(
            FakeEnv(),
            FakeTask([1.0]),
            "Create a new contact for Sara Khan. their number is +15102899176.",
        )

        self.assertEqual(result.planner_rounds[0].planner_result["current_stage_id"], 1)
        self.assertEqual(result.planner_rounds[0].planner_result["subtasks"][0]["goal"], "Open the Phone app.")
        self.assertEqual(result.planner_rounds[0].normalized_subtasks[0]["goal"], "Open the Phone app.")

    def test_contact_observation_with_creation_entry_does_not_force_stage_two_planner_result(self) -> None:
        planner = FakePlanner(
            [
                PlannerResult(
                    is_goal_complete=False,
                    stage_plan=[PlannerStage(1, "Reach the contact creation entry point", "Contact creation entry point is visible")],
                    current_stage_id=1,
                    subtasks=[PlannerSubtask("None", "Reach the contact creation entry point.", "The create new contact option is visible.")],
                ),
                PlannerResult(is_goal_complete=True, completion_message="done"),
            ]
        )
        dialer_contacts = build_dialer_contacts_observation()
        dialer_contacts["ui_elements"] = [
            {
                "index": 0,
                "text": "Create new contact",
                "content_description": "",
                "resource_name": "com.google.android.dialer:id/empty_content_view_action",
                "is_clickable": True,
                "is_editable": False,
                "raw": {"is_selected": False},
            },
            {
                "index": 1,
                "text": "",
                "content_description": "Contacts",
                "resource_name": "com.google.android.dialer:id/tab_contacts",
                "is_clickable": False,
                "is_editable": False,
                "raw": {"is_selected": True},
            },
        ]
        dialer_contacts["ui_description"] = "UI element 0: text='Create new contact'; UI element 1: content_description='Contacts'"
        dialer_contacts["valid_ui_indices"] = [0, 1]
        dialer_contacts["visible_ui_count"] = 2
        dialer_contacts["clickable_ui_count"] = 1
        dialer_contacts["non_system_ui_count"] = 2
        actor = FakeActor(
            [
                ActorRunResult(
                    status="completed",
                    steps=[build_actor_step(0, "entry reached", StatusAction("complete", "done"), "completed", True)],
                    final_observation=dialer_contacts,
                    completion_message="done",
                    last_action={"action_type": "status", "goal_status": "complete"},
                )
            ]
        )
        verifier = FakeVerifier([{"status": "success", "reason": "Entry is visible.", "memory_eligible": True}])
        adapter = FakeObservationAdapter([dialer_contacts])
        runner = AndroidTaskRunner(planner, actor, adapter, verifier=verifier, config=TaskRunConfig(max_planner_rounds=2))

        result = runner.run_task(
            FakeEnv(),
            FakeTask([1.0]),
            "Create a new contact for Sara Khan. their number is +15102899176.",
        )

        self.assertEqual(result.planner_rounds[0].planner_result["current_stage_id"], 1)
        self.assertEqual(result.planner_rounds[0].planner_result["subtasks"][0]["goal"], "Reach the contact creation entry point.")
        self.assertEqual(result.planner_rounds[0].normalized_subtasks[0]["goal"], "Reach the contact creation entry point.")


if __name__ == "__main__":
    unittest.main()
