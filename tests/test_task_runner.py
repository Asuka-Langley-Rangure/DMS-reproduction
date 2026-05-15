from __future__ import annotations

import unittest

from dms_reproduction.agents.android_actor import (
    ActorStepResult,
    ActorRunResult,
    ClickAction,
    StatusAction,
)
from dms_reproduction.agents.planner import PlannerResult, PlannerSubtask
from dms_reproduction.agents.task_runner import AndroidTaskRunner, TaskRunConfig


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
    def __init__(self, responses: list[PlannerResult]) -> None:
        self.responses = list(responses)
        self.calls = []
        self.llm_client = self
        self._last_response = ""

    class config:
        temperature = 0.0

    def build_messages(self, user_goal: str, observation: dict, task_history: list[dict] | None = None, memory_context: str = ""):
        self.calls.append(
            {
                "user_goal": user_goal,
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
                        "text": f"goal={user_goal}; activity={observation['current_activity']}; history={len(task_history or [])}",
                    }
                ],
            },
        ]

    @staticmethod
    def extract_user_text_prompt(messages) -> str:
        return messages[1]["content"][0]["text"]

    @staticmethod
    def messages_to_jsonable(messages):
        return messages

    def generate(self, messages, temperature: float = 0.0) -> str:
        if not self.responses:
            raise AssertionError("No planner response queued.")
        response = self.responses[0]
        self._last_response = response.raw_response or '{"tool":"stub"}'
        return self._last_response

    def parse_response(self, raw_response: str) -> PlannerResult:
        response = self.responses.pop(0)
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
        self.assertEqual(len(planner.calls[1]["task_history"]), 1)
        self.assertEqual(planner.calls[1]["task_history"][0]["source"], "actor")

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

    def test_bad_open_app_subtask_is_normalized(self) -> None:
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

        self.assertEqual(result.planner_rounds[0].normalized_subtasks[0]["goal"], "Navigate to the contacts section.")

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


if __name__ == "__main__":
    unittest.main()
