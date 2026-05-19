from __future__ import annotations

import unittest

from dms_reproduction.agents.verifier import AndroidVerifier, VerifierConfig, VerifierRequest


def build_observation(tag: str) -> dict:
    return {
        "goal": "Create a contact",
        "current_activity": f"{tag}/activity",
        "foreground_package": f"{tag}.package",
        "app_name": f"{tag}.package",
        "ui_description": f"UI description for {tag}",
        "ui_elements": [{"index": 1, "text": f"{tag} button", "is_clickable": True}],
        "visible_ui_count": 1,
        "clickable_ui_count": 1,
        "non_system_ui_count": 1,
        "observation_warning": None,
        "observation_consistency": "stable",
    }


class FakeLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = []

    def generate(self, messages, temperature: float = 0.0) -> str:
        self.calls.append({"messages": messages, "temperature": temperature})
        if not self.responses:
            raise AssertionError("No verifier response queued.")
        return self.responses.pop(0)


class VerifierTest(unittest.TestCase):
    def test_parses_success_response(self) -> None:
        llm = FakeLLMClient(['{"status":"success","reason":"goal visible","memory_eligible":true}'])
        verifier = AndroidVerifier(llm)

        result = verifier.run_verification(
            VerifierRequest(
                subtask="Precondition: None Goal: Open Contacts",
                action_history=[{"status": "completed", "reason": "done", "action": {"action_type": "click", "index": 1}}],
                before_observation=build_observation("before"),
                evidence_observation=build_observation("after"),
                evidence_source="actor_completed_frame",
            )
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.reason, "goal visible")
        self.assertTrue(result.memory_eligible)
        self.assertFalse(result.parse_error)

    def test_parses_failure_response(self) -> None:
        llm = FakeLLMClient(['{"status":"failure","reason":"wrong screen","memory_eligible":false}'])
        verifier = AndroidVerifier(llm)

        result = verifier.run_verification(
            VerifierRequest(
                subtask="Goal: Open Contacts",
                action_history=[],
                before_observation=build_observation("before"),
                evidence_observation=build_observation("after"),
                evidence_source="final_after_observation",
            )
        )

        self.assertEqual(result.status, "failure")
        self.assertEqual(result.reason, "wrong screen")
        self.assertFalse(result.memory_eligible)

    def test_parses_uncertain_response(self) -> None:
        llm = FakeLLMClient(['{"status":"uncertain","reason":"not enough evidence","memory_eligible":false}'])
        verifier = AndroidVerifier(llm)

        result = verifier.run_verification(
            VerifierRequest(
                subtask="Goal: Open Contacts",
                action_history=[],
                before_observation=build_observation("before"),
                evidence_observation=build_observation("after"),
                evidence_source="fallback_current_observation",
            )
        )

        self.assertEqual(result.status, "uncertain")
        self.assertEqual(result.reason, "not enough evidence")

    def test_invalid_json_returns_uncertain_parse_error(self) -> None:
        llm = FakeLLMClient(["not-json"])
        verifier = AndroidVerifier(llm)

        result = verifier.run_verification(
            VerifierRequest(
                subtask="Goal: Open Contacts",
                action_history=[],
                before_observation=build_observation("before"),
                evidence_observation=build_observation("after"),
                evidence_source="fallback_current_observation",
            )
        )

        self.assertEqual(result.status, "uncertain")
        self.assertEqual(result.memory_eligible, False)
        self.assertTrue(result.parse_error)

    def test_build_messages_contains_required_context(self) -> None:
        llm = FakeLLMClient(['{"status":"success","reason":"ok","memory_eligible":true}'])
        verifier = AndroidVerifier(llm, VerifierConfig(max_history_items=4, prompt_profile="paper_history_first"))
        request = VerifierRequest(
            subtask="Precondition: None Goal: Open Contacts",
            action_history=[{"status": "progress", "reason": "tap", "action": {"action_type": "click", "index": 1}, "summary": "moved", "error": ""}],
            before_observation=build_observation("before"),
            evidence_observation=build_observation("after"),
            evidence_source="actor_completed_frame",
            memory_context="memory text",
        )

        messages = verifier.build_messages(request)
        prompt_text = verifier.extract_user_text_prompt(messages)

        self.assertIn("Precondition: None Goal: Open Contacts", prompt_text)
        self.assertIn("Evidence Source:\nactor_completed_frame", prompt_text)
        self.assertIn("Before Observation:", prompt_text)
        self.assertIn("Evidence Observation:", prompt_text)
        self.assertIn("Execution History:", prompt_text)
        self.assertIn("memory text", prompt_text)

    def test_self_written_json_is_default_and_uses_old_prompt_shape(self) -> None:
        llm = FakeLLMClient(['{"status":"success","reason":"ok","memory_eligible":true}'])
        verifier = AndroidVerifier(llm)
        request = VerifierRequest(
            subtask="Precondition: None Goal: Open Contacts",
            action_history=[],
            before_observation=build_observation("before"),
            evidence_observation=build_observation("after"),
            evidence_source="actor_completed_frame",
            memory_context="memory text",
        )

        messages = verifier.build_messages(request)
        self.assertIn("You are an Android subtask verifier", messages[0]["content"])
        prompt_text = verifier.extract_user_text_prompt(messages)
        self.assertIn("Subtask:", prompt_text)
        self.assertIn("Before observation:", prompt_text)
        self.assertIn("Evidence observation:", prompt_text)
        self.assertIn("Action history for this subtask:", prompt_text)
        self.assertNotIn("Current Subtask:", prompt_text)

    def test_local_veto_rejects_app_launch_success_without_launch_evidence(self) -> None:
        llm = FakeLLMClient(['{"verified_success":true,"reason":"history looks okay"}'])
        verifier = AndroidVerifier(llm, VerifierConfig(prompt_profile="paper_history_first"))

        before = build_observation("launcher")
        before["foreground_package"] = "com.android.launcher"
        before["app_name"] = "com.android.launcher"
        after = build_observation("launcher")
        after["foreground_package"] = "com.android.launcher"
        after["app_name"] = "com.android.launcher"

        result = verifier.run_verification(
            VerifierRequest(
                subtask="Precondition: None Goal: Open the Contacts app.",
                action_history=[
                    {"status": "progress", "reason": "waiting", "action": {"action_type": "wait"}, "summary": "waited", "error": ""}
                ],
                before_observation=before,
                evidence_observation=after,
                evidence_source="final_after_observation",
            )
        )

        self.assertEqual(result.status, "failure")
        self.assertFalse(result.memory_eligible)
        self.assertIn("does not show the target app", result.reason)

    def test_local_veto_keeps_app_launch_success_with_open_app_evidence(self) -> None:
        llm = FakeLLMClient(['{"verified_success":true,"reason":"history looks okay"}'])
        verifier = AndroidVerifier(llm, VerifierConfig(prompt_profile="paper_history_first"))

        after = build_observation("after")
        after["foreground_package"] = "com.android.contacts"
        after["app_name"] = "com.android.contacts"

        result = verifier.run_verification(
            VerifierRequest(
                subtask="Precondition: None Goal: Open the Contacts app.",
                action_history=[
                    {"status": "progress", "reason": "open contacts", "action": {"action_type": "open_app", "app_name": "Contacts"}, "summary": "opened contacts", "error": ""}
                ],
                before_observation=build_observation("before"),
                evidence_observation=after,
                evidence_source="actor_completed_frame",
            )
        )

        self.assertEqual(result.status, "success")
        self.assertTrue(result.memory_eligible)

    def test_local_veto_rejects_navigation_success_when_target_is_only_visible(self) -> None:
        llm = FakeLLMClient(['{"verified_success":true,"reason":"target visible"}'])
        verifier = AndroidVerifier(llm, VerifierConfig(prompt_profile="paper_history_first"))

        before = build_observation("settings")
        before["ui_description"] = "Network & internet is visible"
        after = build_observation("settings")
        after["ui_description"] = "Network & internet is visible"

        result = verifier.run_verification(
            VerifierRequest(
                subtask="Precondition: The Settings app is open. Goal: Click on 'Network & internet' to access the network settings.",
                action_history=[
                    {
                        "status": "parse_error",
                        "reason": "tap network",
                        "action": {"action_type": "click", "index": 5},
                        "summary": "non-clickable label selected",
                        "error": "click.index must point to a clickable UI element; non-clickable element selected.",
                    }
                ],
                before_observation=before,
                evidence_observation=after,
                evidence_source="final_after_observation",
            )
        )

        self.assertEqual(result.status, "failure")
        self.assertFalse(result.memory_eligible)
        self.assertIn("no post-action evidence", result.reason)

    def test_local_veto_rejects_toggle_success_without_checked_state_change(self) -> None:
        llm = FakeLLMClient(['{"verified_success":true,"reason":"the switch was clicked","memory_eligible":true}'])
        verifier = AndroidVerifier(llm, VerifierConfig(prompt_profile="paper_history_first"))

        before = build_observation("settings")
        before["ui_elements"] = [
            {
                "index": 10,
                "text": None,
                "content_description": None,
                "resource_name": "android:id/switch_widget",
                "class_name": "android.widget.Switch",
                "is_clickable": False,
                "raw": {"is_checkable": True, "is_checked": True},
            }
        ]
        after = build_observation("settings")
        after["ui_elements"] = [
            {
                "index": 10,
                "text": None,
                "content_description": None,
                "resource_name": "android:id/switch_widget",
                "class_name": "android.widget.Switch",
                "is_clickable": False,
                "raw": {"is_checkable": True, "is_checked": True},
            }
        ]

        result = verifier.run_verification(
            VerifierRequest(
                subtask="Precondition: The Wi-Fi toggle switch is visible and accessible. Goal: Toggle the Wi-Fi off.",
                action_history=[
                    {
                        "status": "progress",
                        "reason": "toggle wifi",
                        "action": {"action_type": "click", "x": 965, "y": 889},
                        "summary": "Clicked the Wi-Fi switch.",
                        "error": "",
                    }
                ],
                before_observation=before,
                evidence_observation=after,
                evidence_source="final_after_observation",
            )
        )

        self.assertEqual(result.status, "failure")
        self.assertFalse(result.memory_eligible)
        self.assertIn("control state actually changed", result.reason)

    def test_local_veto_keeps_toggle_success_when_checked_state_changes(self) -> None:
        llm = FakeLLMClient(['{"verified_success":true,"reason":"the switch was clicked"}'])
        verifier = AndroidVerifier(llm, VerifierConfig(prompt_profile="paper_history_first"))

        before = build_observation("settings")
        before["ui_elements"] = [
            {
                "index": 10,
                "text": None,
                "content_description": None,
                "resource_name": "android:id/switch_widget",
                "class_name": "android.widget.Switch",
                "is_clickable": False,
                "raw": {"is_checkable": True, "is_checked": True},
            }
        ]
        after = build_observation("settings")
        after["ui_elements"] = [
            {
                "index": 10,
                "text": None,
                "content_description": None,
                "resource_name": "android:id/switch_widget",
                "class_name": "android.widget.Switch",
                "is_clickable": False,
                "raw": {"is_checkable": True, "is_checked": False},
            }
        ]

        result = verifier.run_verification(
            VerifierRequest(
                subtask="Precondition: The Wi-Fi toggle switch is visible and accessible. Goal: Toggle the Wi-Fi off.",
                action_history=[
                    {
                        "status": "progress",
                        "reason": "toggle wifi",
                        "action": {"action_type": "click", "x": 965, "y": 889},
                        "summary": "Clicked the Wi-Fi switch.",
                        "error": "",
                    }
                ],
                before_observation=before,
                evidence_observation=after,
                evidence_source="actor_completed_frame",
            )
        )

        self.assertEqual(result.status, "success")
        self.assertTrue(result.memory_eligible)


if __name__ == "__main__":
    unittest.main()
