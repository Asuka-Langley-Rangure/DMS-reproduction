from __future__ import annotations

import unittest

from dms_reproduction.agents.android_actor import (
    ActorConfig,
    ActorRequest,
    AndroidActor,
    AnswerAction,
    ClickAction,
    InputTextAction,
    ScrollAction,
    StatusAction,
    _should_stop_for_external_verification,
    parse_actor_action,
)


class FakeLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.messages = []
        self.temperature = None

    def generate(self, messages, temperature: float = 0.0) -> str:
        self.messages.append(messages)
        self.temperature = temperature
        if not self.responses:
            raise AssertionError("No fake response queued.")
        return self.responses.pop(0)


class FakeEnv:
    def __init__(self) -> None:
        self.actions = []
        self.fail_on_execute = False

    def execute_action(self, action) -> None:
        if self.fail_on_execute:
            raise RuntimeError("boom")
        self.actions.append(action)


class FakeObservationAdapter:
    def __init__(self) -> None:
        self.calls = []

    def capture_observation(
        self,
        env,
        goal: str,
        *,
        step_id: int = 0,
        include_screenshots: bool = True,
    ):
        self.calls.append(
            {
                "goal": goal,
                "step_id": step_id,
                "include_screenshots": include_screenshots,
                "action_count": len(env.actions),
            }
        )
        return build_observation(
            step_id=step_id,
            activity=f"activity-{step_id}",
            app_name="com.android.contacts",
        )


class SequencedObservationAdapter(FakeObservationAdapter):
    def __init__(self, observations: list[dict]) -> None:
        super().__init__()
        self.observations = list(observations)

    def capture_observation(
        self,
        env,
        goal: str,
        *,
        step_id: int = 0,
        include_screenshots: bool = True,
    ):
        self.calls.append(
            {
                "goal": goal,
                "step_id": step_id,
                "include_screenshots": include_screenshots,
                "action_count": len(env.actions),
            }
        )
        if not self.observations:
            raise AssertionError("No fake observation queued.")
        observation = dict(self.observations.pop(0))
        extra_state = dict(observation.get("extra_state") or {})
        extra_state.setdefault("step_id", step_id)
        extra_state.setdefault("orientation", 0)
        extra_state.setdefault("observation_attempt", 1)
        extra_state.setdefault("observation_resampled", False)
        extra_state.setdefault("final_after_resample", False)
        observation["extra_state"] = extra_state
        return observation


def build_observation(
    *,
    step_id: int = 0,
    activity: str = "com.android.contacts/.PeopleActivity",
    app_name: str = "com.android.contacts",
    foreground_package: str = "com.android.contacts",
    warning: str | None = None,
    non_system_ui_count: int = 3,
    observation_consistency: str = "stable",
):
    return {
        "goal": "Create a contact",
        "current_activity": activity,
        "foreground_package": foreground_package,
        "app_name": app_name,
        "screen_size": {"width": 1080, "height": 2400},
        "ui_elements": [
            {"index": 0, "text": "Phone", "is_clickable": True},
            {"index": 3, "text": "Create contact", "is_clickable": True, "is_editable": False},
            {"index": 4, "text": "Name", "is_clickable": True, "is_editable": True},
        ],
        "ui_description": (
            "UI element 0: text='Phone'\n"
            "UI element 3: text='Create contact'\n"
            "UI element 4: text='Name'"
        ),
        "valid_ui_indices": [0, 3, 4],
        "visible_ui_count": 3,
        "clickable_ui_count": 3,
        "non_system_ui_count": non_system_ui_count,
        "observation_warning": warning,
        "observation_consistency": observation_consistency,
        "screenshot_b64": "AAA",
        "labeled_screenshot_b64": "BBB",
        "actor_labeled_screenshot_b64": "CCC",
        "extra_state": {"step_id": step_id, "orientation": 0},
    }


def extract_user_text(messages) -> str:
    content = messages[1]["content"]
    for item in content:
        if item["type"] == "text":
            return item["text"]
    raise AssertionError("No text content found.")


class ActorPromptTest(unittest.TestCase):
    def test_prompt_contains_required_sections_and_actorcode_constraints(self) -> None:
        llm = FakeLLMClient(['Reason: complete.\nAction: {"action_type":"status","goal_status":"complete"}'])
        actor = AndroidActor(llm, ActorConfig(max_history_items=4, prompt_profile="generic_paper"))
        request = ActorRequest(
            subtask="Precondition: Contacts is open. Goal: Tap create contact.",
            observation=build_observation(),
            action_history=[
                {
                    "step_id": 1,
                    "subtask": "Open Contacts",
                    "reason": "Need the app",
                    "action": {"action_type": "open_app", "app_name": "Contacts"},
                    "summary": "Opened Contacts successfully.",
                    "status": "progress",
                    "error": "",
                }
            ],
            memory_context="On previous trials, the create button was visible on the main page.",
        )

        messages = actor.build_messages(request)
        system_prompt = messages[0]["content"]
        user_prompt = extract_user_text(messages)

        self.assertIn("You are an Android Actor", system_prompt)
        self.assertIn("Do not re-plan the overall task.", system_prompt)
        self.assertIn("Current subtask:", user_prompt)
        self.assertIn("Current device state", user_prompt)
        self.assertIn("Foreground package", user_prompt)
        self.assertIn("Dominant visible UI package", user_prompt)
        self.assertIn("Visible UI index table", user_prompt)
        self.assertIn("Visible UI elements summary", user_prompt)
        self.assertIn("[#0]", user_prompt)
        self.assertIn("Recent execution history", user_prompt)
        self.assertIn("Retrieved memory context", user_prompt)
        self.assertIn("Execute exactly one GUI action at a time", user_prompt)
        self.assertIn("Treat the Goal as the primary grounding target", user_prompt)
        self.assertIn("Only use indices that appear in the current Visible UI index table", user_prompt)
        self.assertIn("Do not copy or invent numbers from the screenshot", user_prompt)
        self.assertIn("Do not output action_type swipe; use scroll instead", user_prompt)
        self.assertIn("Return exactly one action in the required JSON action format", user_prompt)
        self.assertEqual(system_prompt.count("Never output action_type 'type'. Use action_type 'input_text' for entering text."), 1)
        self.assertEqual(system_prompt.count("Do not only click the field unless the Goal is only to focus/select the field."), 1)
        self.assertEqual(system_prompt.count("If the Goal mentions +, add, create, or new, ground to the visible create-entry control, FAB, or button whose label or content description indicates creation."), 1)
        self.assertIn("Do not infer an index for a + icon from the screenshot if the visible UI table already exposes a create-entry control.", system_prompt)
        self.assertNotIn("Visible UI elements JSON", user_prompt)

    def test_prompt_includes_observation_warning(self) -> None:
        llm = FakeLLMClient(['Reason: wait.\nAction: {"action_type":"wait"}'])
        actor = AndroidActor(llm, ActorConfig(prompt_profile="generic_paper"))
        request = ActorRequest(
            subtask="Precondition: Phone app is open. Goal: Navigate to contacts tab.",
            observation=build_observation(
                activity="com.google.android.dialer/.DialtactsActivity",
                app_name="com.android.systemui",
                foreground_package="com.google.android.dialer",
                warning="Only system UI elements were retained while foreground activity is dialer.",
                non_system_ui_count=0,
            ),
        )

        user_prompt = extract_user_text(actor.build_messages(request))
        self.assertIn("Observation warning", user_prompt)
        self.assertIn("Only system UI elements were retained", user_prompt)
        self.assertIn("prefer a cautious recovery action such as wait or navigate_back", user_prompt)
        self.assertIn("use a coordinate click on the visible target region", user_prompt)
        self.assertIn("use a coordinate click on the control itself", user_prompt)
        self.assertIn("do not continue tapping nearby UI after you have acted on the target control once", user_prompt)
        self.assertIn("prefer to stop and let the next observation verify the state change", user_prompt)

    def test_prompt_includes_grouped_form_fill_constraints(self) -> None:
        llm = FakeLLMClient(['Reason: fill first name.\nAction: {"action_type":"input_text","index":5,"text":"Mia"}'])
        actor = AndroidActor(llm, ActorConfig(prompt_profile="legacy_contact_tuned"))
        observation = build_observation(activity="com.google.android.contacts/.activities.ContactEditorActivity")
        observation["form_fill_context"] = {
            "target_fields": ["first_name", "last_name", "phone"],
            "expected_fields": {"first_name": "Mia", "last_name": "Fernandez", "phone": "+13268155334"},
            "current_values": {"first_name": "", "last_name": "", "phone": ""},
            "remaining_fields": ["first_name", "last_name", "phone"],
            "required_field_indices": {"first_name": 5, "last_name": 6, "phone": 8},
        }
        request = ActorRequest(
            subtask="Precondition: Contact editor is open. Goal: Fill in Mia Fernandez and +13268155334 in the contact form.",
            observation=observation,
        )

        user_prompt = extract_user_text(actor.build_messages(request))
        system_prompt = actor.build_messages(request)[0]["content"]
        self.assertIn("Grouped form-fill constraints", user_prompt)
        self.assertIn("Only edit those required fields", user_prompt)
        self.assertIn("Do not click Save, Done, Apply, or Submit until every required field matches the expected value", user_prompt)
        self.assertIn("\"first_name\": 5", user_prompt)
        self.assertIn("Never output action_type 'type'. Use action_type 'input_text' for entering text.", system_prompt)
        self.assertIn("If the Goal mentions +, add, create, or new, ground to the visible create-entry control, FAB, or button whose label or content description indicates creation.", system_prompt)

    def test_generic_prompt_ignores_contact_form_specific_block(self) -> None:
        llm = FakeLLMClient(['Reason: fill a field.\nAction: {"action_type":"input_text","index":5,"text":"Mia"}'])
        actor = AndroidActor(llm, ActorConfig(prompt_profile="generic_paper"))
        observation = build_observation(activity="com.google.android.contacts/.activities.ContactEditorActivity")
        observation["form_fill_context"] = {
            "target_fields": ["first_name", "last_name", "phone"],
            "expected_fields": {"first_name": "Mia", "last_name": "Fernandez", "phone": "+13268155334"},
            "current_values": {"first_name": "", "last_name": "", "phone": ""},
            "remaining_fields": ["first_name", "last_name", "phone"],
            "required_field_indices": {"first_name": 5, "last_name": 6, "phone": 8},
        }

        user_prompt = extract_user_text(actor.build_messages(ActorRequest(subtask="Fill a form", observation=observation)))
        self.assertNotIn("Grouped contact form constraints", user_prompt)
        self.assertNotIn("Do not click Save", user_prompt)
        self.assertIn("Do not click a non-clickable label", user_prompt)
        self.assertIn("choose that actionable row or container", user_prompt)

    def test_generic_self_written_prompt_uses_old_sections(self) -> None:
        llm = FakeLLMClient(['Reason: tap.\nAction: {"action_type":"click","index":3}'])
        actor = AndroidActor(llm, ActorConfig(prompt_profile="generic_self_written"))
        request = ActorRequest(
            subtask="Precondition: Contacts is open. Goal: Tap create contact.",
            observation=build_observation(),
            action_history=[],
            memory_context="history memory",
        )

        user_prompt = extract_user_text(actor.build_messages(request))
        self.assertIn("Subtask:", user_prompt)
        self.assertIn("Current screen state:", user_prompt)
        self.assertIn("Visible UI elements:", user_prompt)
        self.assertIn("Visible UI elements JSON:", user_prompt)
        self.assertIn("Decision policy:", user_prompt)
        self.assertIn("prefer open_app instead of wait", user_prompt)
        self.assertIn("Only use indices that appear in the current Visible UI index table", user_prompt)
        self.assertIn("Do not copy or invent numbers from the screenshot", user_prompt)
        self.assertIn("Do not output action_type swipe; use scroll instead", user_prompt)
        self.assertIn("do not click the non-clickable label", user_prompt.lower())
        self.assertIn("use x/y coordinates for that visible target region instead", user_prompt)
        self.assertIn("use a coordinate click on the control itself", user_prompt)
        self.assertIn("do not continue tapping nearby UI after you have acted on the target control once", user_prompt)
        self.assertIn("Return exactly one action per turn.", user_prompt)
        self.assertNotIn("Current subtask:", user_prompt)
        self.assertNotIn("Current device state:", user_prompt)
        self.assertNotIn("Acting rules:", user_prompt)

    def test_build_messages_prefers_actor_specific_box_only_screenshot(self) -> None:
        llm = FakeLLMClient(['Reason: tap.\nAction: {"action_type":"click","index":3}'])
        actor = AndroidActor(llm, ActorConfig(prompt_profile="generic_self_written"))
        request = ActorRequest(
            subtask="Precondition: Contacts is open. Goal: Tap create contact.",
            observation=build_observation(),
        )

        messages = actor.build_messages(request)
        content = messages[1]["content"]
        image_items = [item for item in content if item["type"] == "image_url"]
        self.assertEqual(len(image_items), 1)
        self.assertIn("CCC", image_items[0]["image_url"]["url"])


class ActorActionParsingTest(unittest.TestCase):
    def test_parse_valid_click_action(self) -> None:
        action, normalized, normalization_applied, normalization_reason, corrected, correction_reason = parse_actor_action({"action_type": "click", "index": 3}, build_observation())
        self.assertIsInstance(action, ClickAction)
        self.assertEqual(action.index, 3)
        self.assertIsNone(action.x)
        self.assertIsNone(action.y)
        self.assertIsNone(normalized)
        self.assertFalse(normalization_applied)
        self.assertIsNone(normalization_reason)
        self.assertIsNone(corrected)
        self.assertIsNone(correction_reason)

    def test_parse_valid_input_text_action(self) -> None:
        action, _, _, _, _, _ = parse_actor_action(
            {"action_type": "input_text", "index": 4, "text": "Alice"},
            build_observation(),
        )
        self.assertIsInstance(action, InputTextAction)
        self.assertEqual(action.text, "Alice")

    def test_parse_type_alias_normalizes_to_input_text(self) -> None:
        action, normalized, normalization_applied, normalization_reason, _, _ = parse_actor_action(
            {"action_type": "type", "index": 4, "text": "Alice"},
            build_observation(),
            subtask="Enter the first name 'Alice'.",
        )
        self.assertIsInstance(action, InputTextAction)
        self.assertTrue(normalization_applied)
        self.assertEqual(normalized, {"action_type": "input_text", "index": 4, "text": "Alice"})
        self.assertIn("Normalized action_type", normalization_reason or "")

    def test_parse_type_without_index_recovers_to_unique_editable_input_text(self) -> None:
        observation = build_observation()
        observation["ui_elements"] = [
            {"index": 4, "text": "First name", "is_clickable": True, "is_editable": True},
        ]
        observation["valid_ui_indices"] = [4]
        action, normalized, normalization_applied, normalization_reason, _, _ = parse_actor_action(
            {"action_type": "type", "text": "Ahmed"},
            observation,
            subtask="Enter the first name 'Ahmed' in the contact form.",
        )
        self.assertIsInstance(action, InputTextAction)
        self.assertEqual(action.to_payload(), {"action_type": "input_text", "index": 4, "text": "Ahmed", "clear_text": True})
        self.assertTrue(normalization_applied)
        self.assertEqual(
            normalized,
            {"action_type": "input_text", "text": "Ahmed", "index": 4, "clear_text": True},
        )
        self.assertIn("Recovered input_text.index", normalization_reason or "")

    def test_parse_input_text_without_index_recovers_from_recent_editable_index(self) -> None:
        observation = build_observation()
        observation["ui_elements"] = [
            {"index": 4, "text": "First name", "is_clickable": True, "is_editable": True},
            {"index": 6, "text": "Last name", "is_clickable": True, "is_editable": True},
        ]
        observation["valid_ui_indices"] = [4, 6]
        action, normalized, normalization_applied, normalization_reason, _, _ = parse_actor_action(
            {"action_type": "input_text", "text": "Ahmed"},
            observation,
            subtask="Enter the name 'Ahmed Lopez' in the contact form.",
            recent_editable_index=4,
        )
        self.assertIsInstance(action, InputTextAction)
        self.assertEqual(action.to_payload(), {"action_type": "input_text", "index": 4, "text": "Ahmed", "clear_text": True})
        self.assertTrue(normalization_applied)
        self.assertEqual(
            normalized,
            {"action_type": "input_text", "text": "Ahmed", "index": 4, "clear_text": True},
        )
        self.assertIn("recently focused editable field", normalization_reason or "")

    def test_parse_type_without_index_does_not_recover_when_multiple_editable_targets_are_ambiguous(self) -> None:
        observation = build_observation()
        observation["ui_elements"] = [
            {"index": 4, "text": "First name", "is_clickable": True, "is_editable": True},
            {"index": 6, "text": "Last name", "is_clickable": True, "is_editable": True},
        ]
        observation["valid_ui_indices"] = [4, 6]
        with self.assertRaisesRegex(ValueError, "input_text.index is required"):
            parse_actor_action(
                {"action_type": "type", "text": "Ahmed"},
                observation,
                subtask="Enter the name 'Ahmed Lopez' in the contact form.",
            )

    def test_parse_valid_scroll_action_without_index(self) -> None:
        action, _, _, _, _, _ = parse_actor_action({"action_type": "scroll", "direction": "down"}, build_observation())
        self.assertIsInstance(action, ScrollAction)
        self.assertIsNone(action.index)

    def test_parse_swipe_alias_normalizes_to_scroll(self) -> None:
        action, normalized, normalization_applied, normalization_reason, _, _ = parse_actor_action(
            {"action_type": "swipe", "direction": "up"},
            build_observation(),
        )
        self.assertIsInstance(action, ScrollAction)
        self.assertEqual(action.direction, "up")
        self.assertTrue(normalization_applied)
        self.assertEqual(normalized, {"action_type": "scroll", "direction": "up"})
        self.assertIn("Normalized action_type", normalization_reason or "")

    def test_parse_valid_click_coordinates_action(self) -> None:
        action, _, _, _, _, _ = parse_actor_action(
            {"action_type": "click", "x": 120, "y": 340},
            build_observation(),
        )
        self.assertIsInstance(action, ClickAction)
        self.assertIsNone(action.index)
        self.assertEqual(action.x, 120)
        self.assertEqual(action.y, 340)

    def test_reject_click_with_both_index_and_coordinates(self) -> None:
        with self.assertRaisesRegex(ValueError, "either index or x/y"):
            parse_actor_action(
                {"action_type": "click", "index": 3, "x": 120, "y": 340},
                build_observation(),
            )

    def test_reject_invalid_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid_ui_indices"):
            parse_actor_action({"action_type": "click", "index": 99}, build_observation())

    def test_reject_non_clickable_click_target(self) -> None:
        observation = build_observation()
        observation["ui_elements"][1]["is_clickable"] = False
        with self.assertRaisesRegex(ValueError, "non-clickable"):
            parse_actor_action({"action_type": "click", "index": 3}, observation)

    def test_reject_non_editable_input_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-editable"):
            parse_actor_action({"action_type": "input_text", "index": 3, "text": "Alice"}, build_observation())

    def test_reject_status_complete_on_degraded_observation(self) -> None:
        observation = build_observation(
            activity="com.google.android.dialer/.DialtactsActivity",
            app_name="com.android.systemui",
            foreground_package="com.google.android.dialer",
            warning="Only system UI elements were retained while foreground activity is dialer.",
            non_system_ui_count=0,
        )
        with self.assertRaisesRegex(ValueError, "degraded"):
            parse_actor_action(
                {"action_type": "status", "goal_status": "complete", "message": "done"},
                observation,
            )

    def test_corrects_unique_phone_index_from_reason(self) -> None:
        observation = build_observation()
        observation["ui_elements"] = [
            {"index": 1, "text": "Home", "content_description": "Home", "is_clickable": False, "is_editable": False},
            {"index": 2, "text": "Phone", "content_description": "Phone", "is_clickable": True, "is_editable": False},
        ]
        observation["valid_ui_indices"] = [1, 2]
        action, normalized, normalization_applied, normalization_reason, corrected, correction_reason = parse_actor_action(
            {"action_type": "click", "index": 1},
            observation,
            reason="The Phone app icon is visible and clickable on the screen.",
        )
        self.assertEqual(action.index, 2)
        self.assertIsNone(normalized)
        self.assertFalse(normalization_applied)
        self.assertIsNone(normalization_reason)
        self.assertEqual(corrected, {"action_type": "click", "index": 2})
        self.assertIn("Corrected click target", correction_reason or "")

    def test_corrects_valid_but_reason_mismatched_contacts_index(self) -> None:
        observation = build_observation()
        observation["ui_elements"] = [
            {"index": 4, "text": None, "content_description": "Contacts", "is_clickable": True, "is_editable": False},
            {"index": 6, "text": None, "content_description": "Start voice search", "is_clickable": True, "is_editable": False},
        ]
        observation["valid_ui_indices"] = [4, 6]
        action, normalized, normalization_applied, normalization_reason, corrected, correction_reason = parse_actor_action(
            {"action_type": "click", "index": 6},
            observation,
            reason="The Contacts tab is clearly visible and clickable in the current screen state.",
        )
        self.assertEqual(action.index, 4)
        self.assertIsNone(normalized)
        self.assertFalse(normalization_applied)
        self.assertIsNone(normalization_reason)
        self.assertEqual(corrected, {"action_type": "click", "index": 4})
        self.assertIn("Corrected click target", correction_reason or "")

    def test_falls_back_to_coordinates_for_non_clickable_target_without_clickable_match(self) -> None:
        observation = build_observation()
        observation["ui_elements"] = [
            {
                "index": 5,
                "text": "Network & internet",
                "content_description": None,
                "resource_name": "android:id/title",
                "is_clickable": False,
                "is_editable": False,
                "bbox": [189, 824, 625, 895],
            },
            {
                "index": 6,
                "text": "Mobile, Wi-Fi, hotspot",
                "content_description": None,
                "resource_name": "android:id/summary",
                "is_clickable": False,
                "is_editable": False,
                "bbox": [189, 895, 540, 946],
            },
        ]
        observation["valid_ui_indices"] = [5, 6]
        action, normalized, normalization_applied, normalization_reason, corrected, correction_reason = parse_actor_action(
            {"action_type": "click", "index": 5},
            observation,
            reason="The 'Network & internet' option is visible and needs to be selected to advance the subtask.",
            subtask="Precondition: The Settings app is open. Goal: Click on 'Network & Internet' to navigate to the network settings.",
        )
        self.assertIsInstance(action, ClickAction)
        self.assertIsNone(action.index)
        self.assertEqual(action.x, 320)
        self.assertEqual(action.y, 860)
        self.assertIsNone(normalized)
        self.assertFalse(normalization_applied)
        self.assertIsNone(normalization_reason)
        self.assertEqual(corrected, {"action_type": "click", "x": 320, "y": 860})
        self.assertIn("bbox-based coordinate click", correction_reason or "")

    def test_corrects_multiple_semantic_matches_to_unique_clickable_index(self) -> None:
        observation = build_observation()
        observation["ui_elements"] = [
            {
                "index": 5,
                "text": "Network & internet",
                "content_description": None,
                "resource_name": "android:id/title",
                "is_clickable": False,
                "is_editable": False,
                "bbox": [189, 824, 625, 895],
            },
            {
                "index": 7,
                "text": "Network & internet",
                "content_description": "Network & internet",
                "resource_name": "android:id/row",
                "is_clickable": True,
                "is_editable": False,
                "bbox": [120, 800, 900, 950],
            },
        ]
        observation["valid_ui_indices"] = [5, 7]
        action, _, _, _, corrected, correction_reason = parse_actor_action(
            {"action_type": "click", "index": 99},
            observation,
            reason="The Network & internet option is visible and should be opened.",
            subtask="Precondition: Settings is open. Goal: Open Network & internet.",
        )
        self.assertIsInstance(action, ClickAction)
        self.assertEqual(action.index, 7)
        self.assertEqual(corrected, {"action_type": "click", "index": 7})
        self.assertIn("unique clickable match among multiple semantic matches", correction_reason or "")

    def test_corrects_multiple_non_clickable_matches_to_unique_exact_label_bbox(self) -> None:
        observation = build_observation()
        observation["ui_elements"] = [
            {
                "index": 5,
                "text": "Network & internet",
                "content_description": None,
                "resource_name": "android:id/title",
                "is_clickable": False,
                "is_editable": False,
                "bbox": [189, 824, 625, 895],
            },
            {
                "index": 6,
                "text": "Mobile, Wi-Fi, hotspot",
                "content_description": "Network & internet settings summary",
                "resource_name": "android:id/summary",
                "is_clickable": False,
                "is_editable": False,
                "bbox": [189, 895, 540, 946],
            },
        ]
        observation["valid_ui_indices"] = [5, 6]
        action, _, _, _, corrected, correction_reason = parse_actor_action(
            {"action_type": "click", "index": 99},
            observation,
            reason="The Network & internet option is visible and should be opened.",
            subtask="Precondition: Settings is open. Goal: Open Network & internet.",
        )
        self.assertIsInstance(action, ClickAction)
        self.assertIsNone(action.index)
        self.assertEqual(action.x, 320)
        self.assertEqual(action.y, 860)
        self.assertEqual(corrected, {"action_type": "click", "x": 320, "y": 860})
        self.assertIn("unique exact visible label match among multiple non-clickable matches", correction_reason or "")

    def test_corrects_plus_button_goal_to_unique_clickable_create_entry(self) -> None:
        observation = build_observation()
        observation["ui_elements"] = [
            {
                "index": 1,
                "text": None,
                "content_description": "Create a new file or folder",
                "resource_name": "net.gsantner.markor:id/fab_add_new_item",
                "is_clickable": True,
                "is_editable": False,
            },
            {
                "index": 4,
                "text": "Search",
                "content_description": "Search",
                "resource_name": "net.gsantner.markor:id/action_search",
                "is_clickable": True,
                "is_editable": False,
            },
        ]
        observation["valid_ui_indices"] = [1, 4]
        action, _, _, _, corrected, correction_reason = parse_actor_action(
            {"action_type": "click", "index": 16},
            observation,
            reason="The + button should open the create flow for a new note.",
            subtask="Precondition: Markor is open. Goal: Click the '+' button to create a new note.",
        )
        self.assertIsInstance(action, ClickAction)
        self.assertEqual(action.index, 1)
        self.assertEqual(corrected, {"action_type": "click", "index": 1})
        self.assertIn("Corrected click target", correction_reason or "")

    def test_create_entry_aliases_match_content_description_and_resource_name(self) -> None:
        observation = build_observation()
        observation["ui_elements"] = [
            {
                "index": 1,
                "text": None,
                "content_description": "Create a new file or folder",
                "resource_name": "net.gsantner.markor:id/fab_add_new_item",
                "is_clickable": True,
                "is_editable": False,
            }
        ]
        observation["valid_ui_indices"] = [1]
        for token in ("+", "create", "add", "new", "fab"):
            action, _, _, _, corrected, _ = parse_actor_action(
                {"action_type": "click", "index": 99},
                observation,
                reason=f"The {token} control is visible and should be used.",
                subtask=f"Precondition: Markor is open. Goal: Tap the {token} button.",
            )
            self.assertEqual(action.index, 1)
            self.assertEqual(corrected, {"action_type": "click", "index": 1})

    def test_multiple_create_like_entries_without_unique_safe_target_are_not_corrected(self) -> None:
        observation = build_observation()
        observation["ui_elements"] = [
            {
                "index": 1,
                "text": None,
                "content_description": "Create a new file or folder",
                "resource_name": "net.gsantner.markor:id/fab_add_new_item",
                "is_clickable": True,
                "is_editable": False,
            },
            {
                "index": 2,
                "text": "Create",
                "content_description": "Create",
                "resource_name": "net.gsantner.markor:id/action_create",
                "is_clickable": True,
                "is_editable": False,
            },
        ]
        observation["valid_ui_indices"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "valid_ui_indices"):
            parse_actor_action(
                {"action_type": "click", "index": 99},
                observation,
                reason="The + button should open the create flow for a new note.",
                subtask="Precondition: Markor is open. Goal: Click the '+' button to create a new note.",
            )

    def test_multiple_semantic_matches_without_unique_safe_target_are_not_corrected(self) -> None:
        observation = build_observation()
        observation["ui_elements"] = [
            {
                "index": 5,
                "text": "Network & internet",
                "content_description": None,
                "resource_name": "android:id/title",
                "is_clickable": False,
                "is_editable": False,
                "bbox": [189, 824, 625, 895],
            },
            {
                "index": 7,
                "text": "Network & internet",
                "content_description": "Network & internet",
                "resource_name": "android:id/row",
                "is_clickable": True,
                "is_editable": False,
                "bbox": [120, 800, 900, 950],
            },
            {
                "index": 8,
                "text": "Network & internet",
                "content_description": "Network & internet",
                "resource_name": "android:id/row_alt",
                "is_clickable": True,
                "is_editable": False,
                "bbox": [120, 960, 900, 1110],
            },
        ]
        observation["valid_ui_indices"] = [5, 7, 8]
        with self.assertRaisesRegex(ValueError, "valid_ui_indices"):
            parse_actor_action(
                {"action_type": "click", "index": 99},
                observation,
                reason="The Network & internet option is visible and should be opened.",
                subtask="Precondition: Settings is open. Goal: Open Network & internet.",
            )

    def test_non_clickable_target_without_bbox_still_fails(self) -> None:
        observation = build_observation()
        observation["ui_elements"] = [
            {
                "index": 5,
                "text": "Network & internet",
                "content_description": None,
                "resource_name": "android:id/title",
                "is_clickable": False,
                "is_editable": False,
            }
        ]
        observation["valid_ui_indices"] = [5]
        with self.assertRaisesRegex(ValueError, "non-clickable"):
            parse_actor_action(
                {"action_type": "click", "index": 5},
                observation,
                reason="The 'Network & internet' option is visible and needs to be selected to advance the subtask.",
                subtask="Precondition: The Settings app is open. Goal: Click on 'Network & Internet' to navigate to the network settings.",
            )

    def test_falls_back_to_coordinates_for_non_clickable_switch_widget(self) -> None:
        observation = build_observation()
        observation["ui_elements"] = [
            {
                "index": 9,
                "text": "Wi-Fi",
                "content_description": None,
                "resource_name": "android:id/title",
                "class_name": "android.widget.TextView",
                "is_clickable": False,
                "is_editable": False,
                "bbox": [63, 854, 180, 925],
            },
            {
                "index": 10,
                "text": None,
                "content_description": None,
                "resource_name": "android:id/switch_widget",
                "class_name": "android.widget.Switch",
                "is_clickable": False,
                "is_editable": False,
                "bbox": [892, 826, 1038, 952],
                "raw": {"is_checkable": True, "is_checked": True},
            },
        ]
        observation["valid_ui_indices"] = [9, 10]
        action, normalized, normalization_applied, normalization_reason, corrected, correction_reason = parse_actor_action(
            {"action_type": "click", "index": 10},
            observation,
            reason="The Wi-Fi toggle switch is visible and needs to be toggled off.",
            subtask="Precondition: The Wi-Fi toggle switch is visible and accessible. Goal: Toggle the Wi-Fi off.",
        )
        self.assertIsInstance(action, ClickAction)
        self.assertIsNone(action.index)
        self.assertEqual(action.x, 965)
        self.assertEqual(action.y, 889)
        self.assertIsNone(normalized)
        self.assertFalse(normalization_applied)
        self.assertIsNone(normalization_reason)
        self.assertEqual(corrected, {"action_type": "click", "x": 965, "y": 889})
        self.assertIn("visible control widget", correction_reason or "")

    def test_falls_back_to_coordinates_for_non_clickable_checkbox_widget(self) -> None:
        observation = build_observation()
        observation["ui_elements"] = [
            {
                "index": 12,
                "text": None,
                "content_description": None,
                "resource_name": "android:id/checkbox",
                "class_name": "android.widget.CheckBox",
                "is_clickable": False,
                "is_editable": False,
                "bbox": [900, 700, 980, 780],
                "raw": {"is_checkable": True, "is_checked": False},
            },
        ]
        observation["valid_ui_indices"] = [12]
        action, _, _, _, corrected, correction_reason = parse_actor_action(
            {"action_type": "click", "index": 12},
            observation,
            reason="The checkbox is visible and should be enabled.",
            subtask="Precondition: The checkbox control is visible. Goal: Enable the checkbox.",
        )
        self.assertIsInstance(action, ClickAction)
        self.assertEqual((action.x, action.y), (940, 740))
        self.assertEqual(corrected, {"action_type": "click", "x": 940, "y": 740})
        self.assertIn("visible control widget", correction_reason or "")

    def test_control_widget_without_bbox_still_fails(self) -> None:
        observation = build_observation()
        observation["ui_elements"] = [
            {
                "index": 10,
                "text": None,
                "content_description": None,
                "resource_name": "android:id/switch_widget",
                "class_name": "android.widget.Switch",
                "is_clickable": False,
                "is_editable": False,
                "raw": {"is_checkable": True, "is_checked": True},
            },
        ]
        observation["valid_ui_indices"] = [10]
        with self.assertRaisesRegex(ValueError, "non-clickable"):
            parse_actor_action(
                {"action_type": "click", "index": 10},
                observation,
                reason="The Wi-Fi toggle switch is visible and needs to be toggled off.",
                subtask="Precondition: The Wi-Fi toggle switch is visible and accessible. Goal: Toggle the Wi-Fi off.",
            )

    def test_control_widget_prefers_unique_clickable_index_match_over_coordinates(self) -> None:
        observation = build_observation()
        observation["ui_elements"] = [
            {
                "index": 7,
                "text": None,
                "content_description": "Wi-Fi switch",
                "resource_name": "android:id/switch_row",
                "class_name": "android.widget.LinearLayout",
                "is_clickable": True,
                "is_editable": False,
                "bbox": [780, 800, 1040, 980],
            },
            {
                "index": 10,
                "text": None,
                "content_description": None,
                "resource_name": "android:id/switch_widget",
                "class_name": "android.widget.Switch",
                "is_clickable": False,
                "is_editable": False,
                "bbox": [892, 826, 1038, 952],
                "raw": {"is_checkable": True, "is_checked": True},
            },
        ]
        observation["valid_ui_indices"] = [7, 10]
        action, _, _, _, corrected, correction_reason = parse_actor_action(
            {"action_type": "click", "index": 10},
            observation,
            reason="The Wi-Fi toggle switch is visible and needs to be toggled off.",
            subtask="Precondition: The Wi-Fi toggle switch is visible and accessible. Goal: Toggle the Wi-Fi off.",
        )
        self.assertEqual(action.index, 7)
        self.assertEqual(corrected, {"action_type": "click", "index": 7})
        self.assertIn("Corrected click target", correction_reason or "")

    def test_non_toggle_goal_does_not_trigger_control_fallback(self) -> None:
        observation = build_observation()
        observation["ui_elements"] = [
            {
                "index": 10,
                "text": None,
                "content_description": None,
                "resource_name": "android:id/switch_widget",
                "class_name": "android.widget.Switch",
                "is_clickable": False,
                "is_editable": False,
                "bbox": [892, 826, 1038, 952],
                "raw": {"is_checkable": True, "is_checked": True},
            },
        ]
        observation["valid_ui_indices"] = [10]
        with self.assertRaisesRegex(ValueError, "non-clickable"):
            parse_actor_action(
                {"action_type": "click", "index": 10},
                observation,
                reason="The switch is visible on screen.",
                subtask="Precondition: The switch is visible. Goal: Inspect the Wi-Fi settings page.",
            )


class ActorExecutionTest(unittest.TestCase):
    def test_click_on_editable_field_does_not_trigger_external_verification_stop(self) -> None:
        before_observation = build_observation()
        after_observation = build_observation(step_id=1, activity="activity-after-click")

        reason = _should_stop_for_external_verification(
            subtask="Precondition: The contact creation form is open. Goal: Enter the name 'Noa Zhang' in the contact form.",
            action=ClickAction(index=4),
            before_observation=before_observation,
            after_observation=after_observation,
        )

        self.assertIsNone(reason)

    def test_input_text_on_editable_field_still_triggers_external_verification_stop(self) -> None:
        before_observation = build_observation()
        after_observation = build_observation(step_id=1, activity="activity-after-input")

        reason = _should_stop_for_external_verification(
            subtask="Precondition: The contact creation form is open. Goal: Enter the name 'Noa Zhang' in the contact form.",
            action=InputTextAction(index=4, text="Noa"),
            before_observation=before_observation,
            after_observation=after_observation,
        )

        self.assertIn("Form-editing action executed", reason or "")

    def test_high_impact_click_stopped_path_resamples_observation_before_return(self) -> None:
        llm = FakeLLMClient(
            ['Reason: tap create.\nAction: {"action_type":"click","index":3}']
        )
        actor = AndroidActor(llm, ActorConfig(max_steps=1))
        env = FakeEnv()
        first_observation = build_observation(
            step_id=1,
            activity="activity-first",
            app_name="com.android.contacts",
        )
        second_observation = build_observation(
            step_id=1,
            activity="activity-resampled",
            app_name="com.android.contacts",
        )
        adapter = SequencedObservationAdapter([first_observation, second_observation])

        result = actor.run_subtask(
            env,
            ActorRequest(subtask="Precondition: Contacts is open. Goal: Tap create contact.", observation=build_observation()),
            adapter,
        )

        self.assertEqual(result.status, "stopped")
        self.assertEqual(len(adapter.calls), 2)
        self.assertEqual(result.steps[0].after_observation["current_activity"], "activity-resampled")
        self.assertEqual(result.final_observation["current_activity"], "activity-resampled")
        extra_state = result.final_observation.get("extra_state") or {}
        self.assertTrue(extra_state.get("observation_resampled"))
        self.assertTrue(extra_state.get("final_after_resample"))
        self.assertEqual(extra_state.get("observation_attempt"), 2)

    def test_text_entry_subtask_clicks_editable_field_then_inputs_text_before_stop(self) -> None:
        llm = FakeLLMClient(
            [
                'Reason: The First name field is visible and actionable.\nAction: {"action_type":"click","index":4}',
                'Reason: Enter the requested first name.\nAction: {"action_type":"input_text","index":4,"text":"Noa"}',
            ]
        )
        actor = AndroidActor(llm, ActorConfig(max_steps=2))
        env = FakeEnv()
        first_after_click = build_observation(step_id=1, activity="contact-editor-focused")
        second_after_input = build_observation(step_id=2, activity="contact-editor-input")
        third_resampled = build_observation(step_id=2, activity="contact-editor-input-stable")
        adapter = SequencedObservationAdapter([first_after_click, second_after_input, third_resampled])

        result = actor.run_subtask(
            env,
            ActorRequest(
                subtask="Precondition: The contact creation form is open. Goal: Enter the name 'Noa Zhang' in the contact form.",
                observation=build_observation(),
            ),
            adapter,
        )

        self.assertEqual(result.status, "stopped")
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(result.steps[0].action.to_payload(), {"action_type": "click", "index": 4})
        self.assertFalse(result.steps[0].done)
        self.assertIsNone(result.steps[0].done_reason)
        self.assertEqual(
            result.steps[1].action.to_payload(),
            {"action_type": "input_text", "index": 4, "text": "Noa"},
        )
        self.assertEqual(result.steps[1].done_reason, "heuristic_stop_for_external_verification")
        self.assertEqual(result.steps[1].after_observation["current_activity"], "contact-editor-input-stable")
        self.assertEqual(len(adapter.calls), 3)

    def test_text_entry_subtask_recovers_type_without_index_from_recent_editable_click(self) -> None:
        llm = FakeLLMClient(
            [
                'Reason: The First name field is visible and actionable.\nAction: {"action_type":"click","index":4}',
                'Reason: Enter the requested first name.\nAction: {"action_type":"type","text":"Ahmed"}',
            ]
        )
        actor = AndroidActor(llm, ActorConfig(max_steps=2))
        env = FakeEnv()
        first_after_click = build_observation(step_id=1, activity="contact-editor-focused")
        first_after_click["ui_elements"] = [
            {"index": 4, "text": "First name", "is_clickable": True, "is_editable": True},
            {"index": 6, "text": "Last name", "is_clickable": True, "is_editable": True},
        ]
        first_after_click["valid_ui_indices"] = [4, 6]
        second_after_input = build_observation(step_id=2, activity="contact-editor-input")
        third_resampled = build_observation(step_id=2, activity="contact-editor-input-stable")
        adapter = SequencedObservationAdapter([first_after_click, second_after_input, third_resampled])

        result = actor.run_subtask(
            env,
            ActorRequest(
                subtask="Precondition: The contact creation form is open. Goal: Enter the name 'Ahmed Lopez' in the contact form.",
                observation=build_observation(),
            ),
            adapter,
        )

        self.assertEqual(result.status, "stopped")
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(
            result.steps[1].action.to_payload(),
            {"action_type": "input_text", "index": 4, "text": "Ahmed", "clear_text": True},
        )
        self.assertTrue(result.steps[1].action_normalization_applied)
        self.assertEqual(
            result.steps[1].normalized_action,
            {"action_type": "input_text", "text": "Ahmed", "index": 4, "clear_text": True},
        )
        self.assertIn("Recovered input_text.index", result.steps[1].normalization_reason or "")

    def test_status_complete_path_does_not_resample_observation(self) -> None:
        llm = FakeLLMClient(['Reason: done.\nAction: {"action_type":"status","goal_status":"complete","message":"ok"}'])
        actor = AndroidActor(llm)
        env = FakeEnv()
        adapter = FakeObservationAdapter()

        result = actor.run_subtask(
            env,
            ActorRequest(subtask="Finish contact creation", observation=build_observation()),
            adapter,
        )

        self.assertEqual(result.status, "stopped")
        self.assertEqual(len(adapter.calls), 0)

    def test_direct_open_app_stopped_path_resamples_observation_before_return(self) -> None:
        llm = FakeLLMClient([])
        actor = AndroidActor(llm)
        env = FakeEnv()
        first_observation = build_observation(
            step_id=1,
            activity="com.android.settings/.Transitional",
            app_name="com.android.settings",
            foreground_package="com.android.settings",
        )
        second_observation = build_observation(
            step_id=1,
            activity="com.android.settings/.Stable",
            app_name="com.android.settings",
            foreground_package="com.android.settings",
        )
        adapter = SequencedObservationAdapter([first_observation, second_observation])
        observation = build_observation(
            foreground_package="com.google.android.apps.nexuslauncher",
            app_name="com.google.android.apps.nexuslauncher",
            activity="com.google.android.apps.nexuslauncher/.NexusLauncherActivity",
        )

        result = actor.run_subtask(
            env,
            ActorRequest(
                subtask="Precondition: The launcher is visible. Goal: Open the Settings app.",
                observation=observation,
            ),
            adapter,
        )

        self.assertEqual(result.status, "stopped")
        self.assertEqual(len(adapter.calls), 2)
        self.assertEqual(result.steps[0].after_observation["current_activity"], "com.android.settings/.Stable")
        extra_state = result.steps[0].after_observation.get("extra_state") or {}
        self.assertTrue(extra_state.get("observation_resampled"))
        self.assertEqual(extra_state.get("observation_attempt"), 2)

    def test_toggle_control_click_stops_after_first_meaningful_action(self) -> None:
        llm = FakeLLMClient(
            ['Reason: The grounded target element is the Wi-Fi switch.\nAction: {"action_type":"click","index":10}']
        )
        actor = AndroidActor(llm, ActorConfig(max_steps=4))
        env = FakeEnv()

        class ToggleObservationAdapter(FakeObservationAdapter):
            def capture_observation(self, env, goal: str, *, step_id: int = 0, include_screenshots: bool = True):
                self.calls.append(
                    {
                        "goal": goal,
                        "step_id": step_id,
                        "include_screenshots": include_screenshots,
                        "action_count": len(env.actions),
                    }
                )
                return {
                    **build_observation(step_id=step_id, activity=f"activity-{step_id}", app_name="com.android.settings"),
                    "ui_elements": [
                        {
                            "index": 10,
                            "text": None,
                            "content_description": None,
                            "resource_name": "android:id/switch_widget",
                            "class_name": "android.widget.Switch",
                            "is_clickable": False,
                            "is_editable": False,
                            "bbox": [892, 826, 1038, 952],
                            "raw": {"is_checkable": True, "is_checked": False},
                        }
                    ],
                    "valid_ui_indices": [10],
                    "ui_description": "Wi-Fi switch is visible",
                }

        adapter = ToggleObservationAdapter()
        start_observation = {
            **build_observation(app_name="com.android.settings", foreground_package="com.android.settings"),
            "ui_elements": [
                {
                    "index": 10,
                    "text": None,
                    "content_description": None,
                    "resource_name": "android:id/switch_widget",
                    "class_name": "android.widget.Switch",
                    "is_clickable": False,
                    "is_editable": False,
                    "bbox": [892, 826, 1038, 952],
                    "raw": {"is_checkable": True, "is_checked": True},
                }
            ],
            "valid_ui_indices": [10],
            "ui_description": "Wi-Fi switch is visible",
        }

        result = actor.run_subtask(
            env,
            ActorRequest(
                subtask="Precondition: The Wi-Fi toggle switch is visible and accessible. Goal: Toggle the Wi-Fi off.",
                observation=start_observation,
            ),
            adapter,
        )

        self.assertEqual(result.status, "stopped")
        self.assertEqual(len(result.steps), 1)
        self.assertIn("Meaningful control action executed", result.steps[0].summary)
        self.assertEqual(len(env.actions), 1)

    def test_step_verifier_success_stops_after_high_value_action(self) -> None:
        llm = FakeLLMClient(
            [
                'Reason: tap create.\nAction: {"action_type":"click","index":3}',
                'Reason: should not be needed.\nAction: {"action_type":"status","goal_status":"complete","message":"late"}',
            ]
        )
        actor = AndroidActor(llm, ActorConfig(max_steps=2))
        env = FakeEnv()
        adapter = FakeObservationAdapter()
        step_verifier_calls = []

        def step_verifier(**kwargs):
            step_verifier_calls.append(kwargs)
            return {
                "status": "success",
                "reason": "The subtask is already complete after this action.",
                "memory_eligible": True,
            }

        result = actor.run_subtask(
            env,
            ActorRequest(subtask="Tap create contact", observation=build_observation()),
            adapter,
            step_verifier=step_verifier,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.steps), 1)
        self.assertTrue(result.steps[0].step_verification_triggered)
        self.assertEqual(result.steps[0].step_verification["status"], "success")
        self.assertEqual(result.steps[0].done_reason, "step_verifier_subtask_complete")
        self.assertEqual(result.completion_message, "The subtask is already complete after this action.")
        self.assertEqual(len(step_verifier_calls), 1)
        self.assertEqual(len(llm.messages), 1)

    def test_step_verifier_failure_allows_actor_to_continue(self) -> None:
        llm = FakeLLMClient(
            [
                'Reason: tap create.\nAction: {"action_type":"click","index":3}',
                'Reason: now complete.\nAction: {"action_type":"status","goal_status":"complete","message":"done"}',
            ]
        )
        actor = AndroidActor(llm, ActorConfig(max_steps=2))
        env = FakeEnv()
        adapter = FakeObservationAdapter()
        step_verifier_calls = []

        def step_verifier(**kwargs):
            step_verifier_calls.append(kwargs)
            return {
                "status": "failure",
                "reason": "Not complete yet.",
                "memory_eligible": False,
            }

        result = actor.run_subtask(
            env,
            ActorRequest(subtask="Tap create contact", observation=build_observation()),
            adapter,
            step_verifier=step_verifier,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.steps), 2)
        self.assertTrue(result.steps[0].step_verification_triggered)
        self.assertEqual(result.steps[0].step_verification["status"], "failure")
        self.assertEqual(len(step_verifier_calls), 1)

    def test_wait_does_not_trigger_step_verifier(self) -> None:
        llm = FakeLLMClient(
            [
                'Reason: wait.\nAction: {"action_type":"wait"}',
                'Reason: complete.\nAction: {"action_type":"status","goal_status":"complete","message":"done"}',
            ]
        )
        actor = AndroidActor(llm, ActorConfig(max_steps=2))
        env = FakeEnv()
        adapter = FakeObservationAdapter()
        step_verifier_calls = []

        def step_verifier(**kwargs):
            step_verifier_calls.append(kwargs)
            return {"status": "success", "reason": "should not run", "memory_eligible": True}

        result = actor.run_subtask(
            env,
            ActorRequest(subtask="Wait until page settles", observation=build_observation()),
            adapter,
            step_verifier=step_verifier,
        )

        self.assertEqual(result.status, "completed")
        self.assertFalse(result.steps[0].step_verification_triggered)
        self.assertEqual(step_verifier_calls, [])

    def test_status_complete_terminates_immediately(self) -> None:
        llm = FakeLLMClient(['Reason: done.\nAction: {"action_type":"status","goal_status":"complete","message":"ok"}'])
        actor = AndroidActor(llm)
        env = FakeEnv()
        adapter = FakeObservationAdapter()

        result = actor.run_subtask(
            env,
            ActorRequest(subtask="Finish contact creation", observation=build_observation()),
            adapter,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.completion_message, "ok")
        self.assertEqual(len(env.actions), 0)

    def test_status_infeasible_terminates_immediately(self) -> None:
        llm = FakeLLMClient(['Reason: blocked.\nAction: {"action_type":"status","goal_status":"infeasible","message":"blocked"}'])
        actor = AndroidActor(llm)
        env = FakeEnv()
        adapter = FakeObservationAdapter()

        result = actor.run_subtask(
            env,
            ActorRequest(subtask="Finish contact creation", observation=build_observation()),
            adapter,
        )

        self.assertEqual(result.status, "infeasible")
        self.assertEqual(result.completion_message, "blocked")

    def test_answer_does_not_terminate_and_next_step_can_complete(self) -> None:
        llm = FakeLLMClient(
            [
                'Reason: inform the user.\nAction: {"action_type":"answer","text":"I found the number."}',
                'Reason: subtask is done.\nAction: {"action_type":"status","goal_status":"complete","message":"done"}',
            ]
        )
        actor = AndroidActor(llm)
        env = FakeEnv()
        adapter = FakeObservationAdapter()

        result = actor.run_subtask(
            env,
            ActorRequest(subtask="Read the number and report it", observation=build_observation()),
            adapter,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer_text, "I found the number.")
        self.assertEqual(len(result.steps), 2)
        self.assertIsInstance(result.steps[0].action, AnswerAction)
        self.assertEqual(adapter.calls[0]["step_id"], 1)

    def test_action_only_response_is_accepted(self) -> None:
        llm = FakeLLMClient(['Action: {"action_type":"click","index":3}'])
        actor = AndroidActor(llm, ActorConfig(max_steps=1))
        env = FakeEnv()
        adapter = FakeObservationAdapter()

        result = actor.run_subtask(
            env,
            ActorRequest(subtask="Tap create contact", observation=build_observation()),
            adapter,
        )

        self.assertEqual(result.status, "step_limit")
        self.assertEqual(result.steps[0].reason, "")
        self.assertEqual(result.steps[0].action.to_payload(), {"action_type": "click", "index": 3})

    def test_unstable_wait_step_marks_post_action_consistency_check(self) -> None:
        llm = FakeLLMClient(['Reason: UI is degraded.\nAction: {"action_type":"wait"}'])
        actor = AndroidActor(llm, ActorConfig(max_steps=1))
        env = FakeEnv()
        adapter = FakeObservationAdapter()
        observation = build_observation(
            foreground_package="com.google.android.contacts",
            app_name="com.android.systemui",
            warning="Only system UI elements were retained while foreground activity is contacts.",
            non_system_ui_count=0,
            observation_consistency="unstable",
        )

        result = actor.run_subtask(
            env,
            ActorRequest(subtask="Wait for the page to settle", observation=observation),
            adapter,
        )

        self.assertIn("requires_post_action_consistency_check=true", result.steps[0].summary)

    def test_execute_action_error_returns_execution_error(self) -> None:
        llm = FakeLLMClient(['Reason: tap create.\nAction: {"action_type":"click","index":3}'])
        actor = AndroidActor(llm)
        env = FakeEnv()
        env.fail_on_execute = True
        adapter = FakeObservationAdapter()

        result = actor.run_subtask(
            env,
            ActorRequest(subtask="Tap create contact", observation=build_observation()),
            adapter,
        )

        self.assertEqual(result.status, "execution_error")
        self.assertIn("boom", result.steps[0].execution_error or "")

    def test_explicit_open_app_goal_uses_direct_open_app_without_llm(self) -> None:
        llm = FakeLLMClient([])
        actor = AndroidActor(llm)
        env = FakeEnv()
        adapter = FakeObservationAdapter()
        observation = build_observation(
            foreground_package="com.google.android.apps.nexuslauncher",
            app_name="com.google.android.apps.nexuslauncher",
            activity="com.google.android.apps.nexuslauncher/.NexusLauncherActivity",
        )

        result = actor.run_subtask(
            env,
            ActorRequest(
                subtask="Precondition: The launcher is visible. Goal: Open the Settings app.",
                observation=observation,
            ),
            adapter,
        )

        self.assertEqual(result.status, "stopped")
        self.assertEqual(len(llm.messages), 0)
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].action.to_payload(), {"action_type": "open_app", "app_name": "Settings"})
        self.assertEqual(result.steps[0].prompt_text, "")
        self.assertEqual(result.steps[0].raw_response, "")
        self.assertIn("Direct open_app constraint applied", result.steps[0].reason)
        self.assertIn("Direct open_app executed", result.completion_message)
        self.assertEqual(getattr(env.actions[0], "action_type", None), "open_app")
        self.assertEqual(getattr(env.actions[0], "app_name", None), "Settings")

    def test_direct_open_app_not_triggered_when_already_in_target_app(self) -> None:
        llm = FakeLLMClient(['Reason: already open.\nAction: {"action_type":"status","goal_status":"complete","message":"done"}'])
        actor = AndroidActor(llm)
        env = FakeEnv()
        adapter = FakeObservationAdapter()
        observation = build_observation(
            foreground_package="com.android.settings",
            app_name="com.android.settings",
            activity="com.android.settings/.Settings",
        )

        result = actor.run_subtask(
            env,
            ActorRequest(
                subtask="Precondition: The launcher is visible. Goal: Open the Settings app.",
                observation=observation,
            ),
            adapter,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(llm.messages), 1)
        self.assertEqual(len(env.actions), 0)

    def test_direct_open_app_execution_error_returns_execution_error(self) -> None:
        llm = FakeLLMClient([])
        actor = AndroidActor(llm)
        env = FakeEnv()
        env.fail_on_execute = True
        adapter = FakeObservationAdapter()
        observation = build_observation(
            foreground_package="com.google.android.apps.nexuslauncher",
            app_name="com.google.android.apps.nexuslauncher",
            activity="com.google.android.apps.nexuslauncher/.NexusLauncherActivity",
        )

        result = actor.run_subtask(
            env,
            ActorRequest(
                subtask="Precondition: The launcher is visible. Goal: Open the Settings app.",
                observation=observation,
            ),
            adapter,
        )

        self.assertEqual(result.status, "execution_error")
        self.assertEqual(len(llm.messages), 0)
        self.assertEqual(len(result.steps), 1)
        self.assertIn("boom", result.steps[0].execution_error or "")

    def test_step_limit_returns_step_limit(self) -> None:
        llm = FakeLLMClient(
            [
                'Reason: wait for update.\nAction: {"action_type":"wait"}',
                'Reason: still waiting.\nAction: {"action_type":"wait"}',
            ]
        )
        actor = AndroidActor(llm, ActorConfig(max_steps=2))
        env = FakeEnv()
        adapter = FakeObservationAdapter()

        result = actor.run_subtask(
            env,
            ActorRequest(subtask="Wait until page settles", observation=build_observation()),
            adapter,
        )

        self.assertEqual(result.status, "step_limit")
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(len(adapter.calls), 2)

    def test_history_and_memory_are_carried_into_next_turn(self) -> None:
        llm = FakeLLMClient(
            [
                'Reason: tap create.\nAction: {"action_type":"click","index":3}',
                'Reason: now enter the name.\nAction: {"action_type":"status","goal_status":"complete","message":"done"}',
            ]
        )
        actor = AndroidActor(llm)
        env = FakeEnv()
        adapter = FakeObservationAdapter()

        result = actor.run_subtask(
            env,
            ActorRequest(
                subtask="Create a new contact",
                observation=build_observation(),
                memory_context="The create button is usually visible on the first screen.",
            ),
            adapter,
        )

        self.assertEqual(result.status, "completed")
        second_prompt = extract_user_text(llm.messages[1])
        self.assertIn('action={"action_type": "click", "index": 3}', second_prompt)
        self.assertIn("notes=Subtask=Create a new contact; action=click; status=progress;", second_prompt)
        self.assertIn("Retrieved memory context", second_prompt)
        self.assertIn("create button is usually visible", second_prompt)

    def test_normalized_action_is_recorded_in_step_result(self) -> None:
        llm = FakeLLMClient(['Reason: enter the first name.\nAction: {"action_type":"type","index":4,"text":"Sara"}'])
        actor = AndroidActor(llm, ActorConfig(max_steps=1))
        env = FakeEnv()
        adapter = FakeObservationAdapter()

        result = actor.run_subtask(
            env,
            ActorRequest(subtask="Enter the first name 'Sara'.", observation=build_observation()),
            adapter,
        )

        self.assertEqual(result.status, "step_limit")
        self.assertTrue(result.steps[0].action_normalization_applied)
        self.assertEqual(result.steps[0].normalized_action, {"action_type": "input_text", "index": 4, "text": "Sara"})
        self.assertIn("Recoverable actor schema mismatch handled locally", result.steps[0].summary)

    def test_parse_error_returns_parse_error(self) -> None:
        llm = FakeLLMClient(['Reason: broken output.\nAction: not-json'])
        actor = AndroidActor(llm)
        env = FakeEnv()
        adapter = FakeObservationAdapter()

        result = actor.run_subtask(
            env,
            ActorRequest(subtask="Tap create contact", observation=build_observation()),
            adapter,
        )

        self.assertEqual(result.status, "parse_error")
        self.assertIn("Failed to parse", result.steps[0].parse_error or "")


if __name__ == "__main__":
    unittest.main()
