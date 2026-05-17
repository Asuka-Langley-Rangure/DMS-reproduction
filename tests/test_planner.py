from __future__ import annotations

import unittest

from dms_reproduction.agents.planner import (
    AndroidTaskPlanner,
    PlannerConfig,
    PlannerResult,
    PlannerStage,
    extract_json_object,
    parse_precondition_goal,
)


class FakeLLMClient:
    def __init__(self, response: str = "") -> None:
        self.response = response
        self.messages = None
        self.temperature = None

    def generate(self, messages, temperature: float = 0.0) -> str:
        self.messages = messages
        self.temperature = temperature
        return self.response


def extract_user_text(messages) -> str:
    content = messages[1]["content"]
    for item in content:
        if item["type"] == "text":
            return item["text"]
    raise AssertionError("No text content found.")


def extract_image_items(messages):
    content = messages[1]["content"]
    return [item for item in content if item["type"] == "image_url"]


class PlannerPromptTest(unittest.TestCase):
    def test_stage_plan_prompt_is_goal_only(self) -> None:
        llm = FakeLLMClient(response='{"stage_plan":[{"stage_id":1,"title":"Open the target app","success_signal":"App is open"},{"stage_id":2,"title":"Reach the working area","success_signal":"Working area is visible"},{"stage_id":3,"title":"Complete the requested action","success_signal":"Requested action is completed"}]}')
        planner = AndroidTaskPlanner(llm, PlannerConfig(max_subtasks=5))

        result = planner.plan_stage_milestones("Delete target_file.mp3 from the device")

        self.assertIsNone(result.parse_error)
        self.assertEqual(len(result.stage_plan), 3)
        system_prompt = llm.messages[0]["content"]
        user_prompt = extract_user_text(llm.messages)
        self.assertIn("You do not see the current device state", system_prompt)
        self.assertIn("Do not include current_stage_id", system_prompt)
        self.assertIn("Do not return tasks or subtasks", user_prompt)
        self.assertNotIn("Current device state", user_prompt)
        self.assertNotIn("Visible UI elements JSON", user_prompt)
        self.assertNotIn("Complete task history", user_prompt)

    def test_prompt_contains_required_sections_and_omits_agent_list(self) -> None:
        llm = FakeLLMClient(response='{"tool":"complete_goal","message":"done"}')
        planner = AndroidTaskPlanner(llm, PlannerConfig(max_subtasks=5))
        observation = {
            "current_activity": "com.google.android.documentsui.files.FilesActivity",
            "app_name": "Files",
            "screen_size": {"width": 1080, "height": 2400},
            "ui_elements": [{"index": 0, "text": "Audio"}],
            "ui_description": "UI element 0: text='Audio'",
            "screenshot_b64": "AAA",
            "labeled_screenshot_b64": "BBB",
        }
        history = [
            {
                "task": "Precondition: None Goal: Open Contacts",
                "status": "completed",
                "reason": "App is open",
            },
            {
                "task": "",
                "status": "planner_complete_but_task_check_failed",
                "reason": "Planner declared completion too early.",
            },
        ]

        planner.plan(
            user_goal="Delete target_file.mp3 from the device",
            observation=observation,
            task_history=history,
            memory_context="A prior successful trial used the create button.",
        )

        system_prompt = llm.messages[0]["content"]
        user_prompt = extract_user_text(llm.messages)
        self.assertIn("You are an Android Task Planner", system_prompt)
        self.assertIn("Delete target_file.mp3 from the device", user_prompt)
        self.assertIn("Current activity", user_prompt)
        self.assertIn("Visible UI elements JSON", user_prompt)
        self.assertIn("Complete task history", user_prompt)
        self.assertIn("Retrieved memory context", user_prompt)
        self.assertIn("The frozen stage plan already describes the whole task", user_prompt)
        self.assertIn("Precondition: ... Goal: ...", system_prompt)
        self.assertIn("checkable state change after completion", user_prompt)
        self.assertIn("2-6 atomic actions", system_prompt)
        self.assertIn("Use task history to avoid repeating failed or no-progress strategies", system_prompt)
        self.assertIn("Return complete_goal only if the current screen directly shows the final requested result", user_prompt)
        self.assertIn("Consider complete_goal only as an exception after the previous three checks", user_prompt)
        self.assertIn("Then choose the earliest existing stage that is not yet directly satisfied", user_prompt)
        self.assertIn("a frozen whole-task stage plan already exists", system_prompt)
        self.assertIn("current_stage_id", system_prompt)
        self.assertIn("Frozen stage plan", user_prompt)
        self.assertIn("Contextual planning hints", user_prompt)
        self.assertIn("Use only the provided stage plan when choosing current_stage_id", system_prompt)
        self.assertIn("Choose the earliest stage that is not yet directly satisfied", system_prompt)
        self.assertIn("Default to one main subtask", system_prompt)
        self.assertIn("Do not repeat the same task wording multiple times in one response", system_prompt)
        self.assertIn("rewrite it as the state change or functional result", system_prompt)
        self.assertIn("Ensure current_stage_id and the returned tasks describe the same milestone", user_prompt)
        self.assertIn("must return current_stage_id and at least one task", user_prompt)
        self.assertIn("Do not return duplicate or near-duplicate tasks", user_prompt)
        self.assertIn("rewrite it as the milestone state", user_prompt)
        self.assertIn("planner_complete_but_task_check_failed", user_prompt)
        self.assertIn("complete_goal is only a completion candidate", system_prompt)
        self.assertNotIn("Contact creation:", system_prompt)
        self.assertNotIn("Mia Garcia", system_prompt)
        self.assertNotIn("contact creation entry point", user_prompt.lower())
        self.assertNotIn("Available Specialized Agents", system_prompt)
        image_items = extract_image_items(llm.messages)
        self.assertEqual(len(image_items), 1)
        self.assertIn("data:image/png;base64,BBB", image_items[0]["image_url"]["url"])

    def test_generic_prompt_uses_structure_only_hints(self) -> None:
        llm = FakeLLMClient(response='{"tool":"complete_goal","message":"done"}')
        planner = AndroidTaskPlanner(llm, PlannerConfig(max_subtasks=5))
        observation = {
            "current_activity": "com.google.android.documentsui.files.FilesActivity",
            "app_name": "Files",
            "screen_size": {"width": 1080, "height": 2400},
            "ui_elements": [{"index": 0, "text": "Delete"}],
            "ui_description": "Dialog with Delete and Cancel buttons.",
            "screenshot_b64": "AAA",
            "labeled_screenshot_b64": "BBB",
        }

        planner.plan(
            user_goal="Delete a file from storage",
            observation=observation,
            task_history=[],
            memory_context="",
        )

        user_prompt = extract_user_text(llm.messages)
        self.assertIn("confirmation or finalization", user_prompt)
        self.assertNotIn("contact editor", user_prompt.lower())
        self.assertNotIn("create-contact", user_prompt.lower())
        self.assertNotIn("fill in <name>", user_prompt.lower())

    def test_legacy_contact_tuned_profile_keeps_contact_specific_prompt(self) -> None:
        llm = FakeLLMClient(response='{"tool":"complete_goal","message":"done"}')
        planner = AndroidTaskPlanner(llm, PlannerConfig(max_subtasks=5, prompt_profile="legacy_contact_tuned"))
        observation = {
            "current_activity": "com.android.contacts/.PeopleActivity",
            "app_name": "com.android.contacts",
            "screen_size": {"width": 1080, "height": 2400},
            "ui_elements": [{"index": 0, "text": "Create contact"}],
            "ui_description": "UI element 0: text='Create contact'",
            "screenshot_b64": "AAA",
            "labeled_screenshot_b64": "BBB",
        }

        planner.plan(
            user_goal="Add a contact named Alice",
            observation=observation,
            task_history=[
                {
                    "task": "",
                    "status": "planner_complete_but_task_check_failed",
                    "reason": "Planner declared completion too early.",
                }
            ],
            memory_context="",
        )

        system_prompt = llm.messages[0]["content"]
        user_prompt = extract_user_text(llm.messages)
        self.assertIn("Canonical contact-creation milestones", system_prompt)
        self.assertIn("Reach the contact creation entry point.", system_prompt)
        self.assertIn("Fill in Mia Garcia and +18856139998 in the contact form.", system_prompt)
        self.assertIn("saved_but_task_check_failed", system_prompt)
        self.assertIn("On a contact editor screen, prefer one subtask", user_prompt)
        self.assertIn("Remembered stage plan", user_prompt)
        self.assertIn("Contextual planning hints", user_prompt)
        self.assertIn("prefer milestone wording over button-click wording", user_prompt)
        self.assertIn("do not return a one-stage plan", system_prompt)
        self.assertIn("do not return complete_goal", system_prompt)
        self.assertIn("planner_complete_but_task_check_failed", user_prompt)
        self.assertIn("repair, verification, or progress-making subtask", user_prompt)
        self.assertIn("Ensure current_stage_id and the returned subtask describe the same milestone", user_prompt)


class PlannerParsingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = AndroidTaskPlanner(FakeLLMClient())

    def test_parse_complete_goal(self) -> None:
        result = self.planner.parse_response('{"tool":"complete_goal","message":"done"}')
        self.assertTrue(result.is_goal_complete)
        self.assertEqual(result.completion_message, "done")

    def test_parse_set_tasks(self) -> None:
        raw = (
            '{"tool":"set_tasks","stage_plan":[{"stage_id":1,"title":"Open the Phone app","success_signal":"Phone app is foreground"},'
            '{"stage_id":2,"title":"Reach the contact creation entry point","success_signal":"Create contact UI is visible"}],'
            '"current_stage_id":2,"tasks":['
            '{"task":"Precondition: Contacts app is open Goal: Tap create contact flow",'
            '"reason":"Need to begin contact creation"}]}'
        )
        result = self.planner.parse_response(raw)
        self.assertFalse(result.is_goal_complete)
        self.assertEqual(len(result.subtasks), 1)
        self.assertEqual(len(result.stage_plan), 2)
        self.assertEqual(result.current_stage_id, 2)
        self.assertEqual(result.subtasks[0].precondition, "Contacts app is open")
        self.assertEqual(result.subtasks[0].goal, "Tap create contact flow")
        self.assertEqual(result.subtasks[0].agent, "android_actor")

    def test_parse_set_tasks_without_stage_plan_remains_compatible(self) -> None:
        raw = (
            '{"tool":"set_tasks","tasks":['
            '{"task":"Precondition: Contacts app is open Goal: Reach the creation entry point",'
            '"reason":"Need to begin contact creation"}]}'
        )
        result = self.planner.parse_response(raw)
        self.assertFalse(result.is_goal_complete)
        self.assertEqual(result.stage_plan, [])
        self.assertIsNone(result.current_stage_id)

    def test_parse_set_tasks_preserves_current_stage_id_without_stage_plan(self) -> None:
        raw = (
            '{"tool":"set_tasks","current_stage_id":2,"tasks":['
            '{"task":"Precondition: Contacts app is open Goal: Reach the creation entry point",'
            '"reason":"Need to begin contact creation"}]}'
        )
        result = self.planner.parse_response(raw)
        self.assertFalse(result.is_goal_complete)
        self.assertEqual(result.stage_plan, [])
        self.assertEqual(result.current_stage_id, 2)

    def test_parse_invalid_current_stage_id(self) -> None:
        raw = (
            '{"tool":"set_tasks","stage_plan":[{"stage_id":1,"title":"Open the Phone app","success_signal":"Phone app is foreground"}],'
            '"current_stage_id":3,"tasks":[{"task":"Precondition: None Goal: Open the Phone app","reason":"Need app"}]}'
        )
        result = self.planner.parse_response(raw)
        self.assertIn("current_stage_id 3", result.parse_error or "")

    def test_parse_task_format_failure_is_explicitly_classified(self) -> None:
        raw = (
            '{"tool":"set_tasks","current_stage_id":2,"tasks":['
            '{"task":"Goal: Reach the creation entry point","reason":"Need to begin contact creation"}]}'
        )
        result = self.planner.parse_response(raw)
        self.assertIn("Precondition: ... Goal: ...", result.parse_error or "")
        self.assertEqual(result.parse_error_code, "planner_task_format_invalid")

    def test_parse_stage_plan_response_requires_three_to_five_stages(self) -> None:
        raw = '{"stage_plan":[{"stage_id":1,"title":"Open the target app","success_signal":"App is open"}]}'
        result = self.planner.parse_stage_plan_response(raw)
        self.assertIn("3-5 milestones", result.parse_error or "")

    def test_parse_stage_plan_response_accepts_top_level_array(self) -> None:
        raw = (
            '[{"stage_id":1,"title":"Open the target app","success_signal":"App is open"},'
            '{"stage_id":2,"title":"Reach the working area","success_signal":"Working area is visible"},'
            '{"stage_id":3,"title":"Complete the requested action","success_signal":"Requested action is completed"}]'
        )
        result = self.planner.parse_stage_plan_response(raw)
        self.assertIsNone(result.parse_error)
        self.assertEqual(len(result.stage_plan), 3)

    def test_parse_stage_plan_rejects_low_level_stage_title(self) -> None:
        raw = (
            '{"tool":"set_tasks","stage_plan":[{"stage_id":1,"title":"Click the Contacts tab","success_signal":"Contacts tab is selected"}],'
            '"current_stage_id":1,"tasks":[{"task":"Precondition: None Goal: Reach the contacts section","reason":"Need contacts"}]}'
        )
        result = self.planner.parse_response(raw)
        self.assertIn("high-level milestone", result.parse_error or "")

    def test_parse_stage_plan_without_tasks_synthesizes_current_stage_subtask(self) -> None:
        raw = (
            '{"tool":"set_tasks","stage_plan":[{"stage_id":1,"title":"Open the Phone app","success_signal":"Phone app is open"}],'
            '"current_stage_id":1}'
        )
        result = self.planner.parse_response(raw)
        self.assertIsNone(result.parse_error)
        self.assertEqual(len(result.subtasks), 1)
        self.assertEqual(result.subtasks[0].goal, "Open the Phone app")

    def test_parse_invalid_json(self) -> None:
        result = self.planner.parse_response("not json")
        self.assertIsInstance(result, PlannerResult)
        self.assertIsNotNone(result.parse_error)

    def test_parse_missing_reason(self) -> None:
        raw = '{"tool":"set_tasks","tasks":[{"task":"Precondition: None Goal: Open app"}]}'
        result = self.planner.parse_response(raw)
        self.assertIn("missing 'reason'", result.parse_error or "")

    def test_parse_missing_precondition_goal_format(self) -> None:
        raw = '{"tool":"set_tasks","tasks":[{"task":"Open app","reason":"Need to start"}]}'
        result = self.planner.parse_response(raw)
        self.assertIn("Precondition: ... Goal: ...", result.parse_error or "")

    def test_extract_json_object_handles_markdown_fence(self) -> None:
        raw = '```json\n{"tool":"complete_goal","message":"ok"}\n```'
        self.assertEqual(extract_json_object(raw), {"tool": "complete_goal", "message": "ok"})

    def test_extract_json_object_handles_trailing_quote_after_nested_object(self) -> None:
        raw = (
            '{"tool":"set_tasks","tasks":[{"task":"Precondition: None. Goal: Open the Phone app.",'
            ' "reason":"To access the contacts list where we can add a new contact."}]}"'
        )
        self.assertEqual(
            extract_json_object(raw),
            {
                "tool": "set_tasks",
                "tasks": [
                    {
                        "task": "Precondition: None. Goal: Open the Phone app.",
                        "reason": "To access the contacts list where we can add a new contact.",
                    }
                ],
            },
        )

    def test_extract_json_object_handles_json_encoded_string(self) -> None:
        raw = '"{\\"tool\\":\\"complete_goal\\",\\"message\\":\\"ok\\"}"'
        self.assertEqual(extract_json_object(raw), {"tool": "complete_goal", "message": "ok"})

    def test_parse_precondition_goal(self) -> None:
        self.assertEqual(
            parse_precondition_goal("Precondition: None Goal: Open Contacts"),
            ("None", "Open Contacts"),
        )

    def test_parse_response_repairs_near_json_task_reason_boundary(self) -> None:
        raw = (
            '{"tool":"set_tasks","tasks":[{"task":"Precondition: The user wants to create a new contact.; '
            'Goal: Tap on the Create new contact button.; reason":"The Create new contact button is visible."}]}'
        )
        result = self.planner.parse_response(raw)
        self.assertIsNone(result.parse_error)
        self.assertTrue(result.repaired_parse)
        self.assertEqual(len(result.subtasks), 1)
        self.assertIn("Tap on the Create new contact button.", result.subtasks[0].goal)


if __name__ == "__main__":
    unittest.main()
