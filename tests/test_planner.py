from __future__ import annotations

import unittest

from dms_reproduction.agents.planner import (
    AndroidTaskPlanner,
    PlannerConfig,
    PlannerResult,
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
    def test_prompt_contains_required_sections_and_omits_agent_list(self) -> None:
        llm = FakeLLMClient(response='{"tool":"complete_goal","message":"done"}')
        planner = AndroidTaskPlanner(llm, PlannerConfig(max_subtasks=5))
        observation = {
            "current_activity": "com.android.contacts/.PeopleActivity",
            "app_name": "com.android.contacts",
            "screen_size": {"width": 1080, "height": 2400},
            "ui_elements": [{"index": 0, "text": "Create contact"}],
            "ui_description": "UI element 0: text='Create contact'",
            "screenshot_b64": "AAA",
            "labeled_screenshot_b64": "BBB",
        }
        history = [
            {
                "task": "Precondition: None Goal: Open Contacts",
                "status": "completed",
                "reason": "App is open",
            }
        ]

        planner.plan(
            user_goal="Add a contact named Alice",
            observation=observation,
            task_history=history,
            memory_context="A prior successful trial used the create button.",
        )

        system_prompt = llm.messages[0]["content"]
        user_prompt = extract_user_text(llm.messages)
        self.assertIn("You are an Android Task Planner", system_prompt)
        self.assertIn("Add a contact named Alice", user_prompt)
        self.assertIn("Current activity", user_prompt)
        self.assertIn("Visible UI elements JSON", user_prompt)
        self.assertIn("Complete task history", user_prompt)
        self.assertIn("Retrieved memory context", user_prompt)
        self.assertIn("1-5 functional steps", user_prompt)
        self.assertIn("Precondition: ... Goal: ...", system_prompt)
        self.assertIn("one short natural-language description of a small objective", system_prompt)
        self.assertIn("verifiable UI state change", system_prompt)
        self.assertIn("2-6 atomic actions", system_prompt)
        self.assertIn("Bad: 'Tap the Phone app icon.'", system_prompt)
        self.assertNotIn("Available Specialized Agents", system_prompt)
        image_items = extract_image_items(llm.messages)
        self.assertEqual(len(image_items), 1)
        self.assertIn("data:image/png;base64,BBB", image_items[0]["image_url"]["url"])


class PlannerParsingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = AndroidTaskPlanner(FakeLLMClient())

    def test_parse_complete_goal(self) -> None:
        result = self.planner.parse_response('{"tool":"complete_goal","message":"done"}')
        self.assertTrue(result.is_goal_complete)
        self.assertEqual(result.completion_message, "done")

    def test_parse_set_tasks(self) -> None:
        raw = (
            '{"tool":"set_tasks","tasks":['
            '{"task":"Precondition: Contacts app is open Goal: Tap create contact flow",'
            '"reason":"Need to begin contact creation"}]}'
        )
        result = self.planner.parse_response(raw)
        self.assertFalse(result.is_goal_complete)
        self.assertEqual(len(result.subtasks), 1)
        self.assertEqual(result.subtasks[0].precondition, "Contacts app is open")
        self.assertEqual(result.subtasks[0].goal, "Tap create contact flow")
        self.assertEqual(result.subtasks[0].agent, "android_actor")

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
