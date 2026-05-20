from __future__ import annotations

import base64
import importlib.util
import unittest
from dataclasses import dataclass
from unittest.mock import patch

PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None

if PIL_AVAILABLE:
    from PIL import Image

    from dms_reproduction.envs.android_world_adapter import AndroidWorldObservationAdapter
    from dms_reproduction.envs.observation_utils import (
        bbox_to_tuple,
        build_ui_description,
        draw_labeled_screenshot,
        is_visible_candidate,
        standardize_ui_element,
    )


@dataclass
class FakeBBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass
class FakeUIElement:
    text: str | None = None
    content_description: str | None = None
    resource_name: str | None = None
    class_name: str | None = None
    bbox_pixels: FakeBBox | None = None
    is_clickable: bool | None = None
    is_editable: bool | None = None
    is_enabled: bool | None = None
    is_scrollable: bool | None = None
    is_visible: bool | None = None
    package_name: str | None = None
    is_selected: bool | None = None


@dataclass
class FakeState:
    pixels: object
    ui_elements: list[FakeUIElement]


class FakeEnv:
    def __init__(self) -> None:
        self.logical_screen_size = (200, 400)
        self.foreground_activity_name = "com.android.contacts/.activities.PeopleActivity"
        self.orientation = 0
        self._state = FakeState(
            pixels=object(),
            ui_elements=[
                FakeUIElement(
                    text="Create contact",
                    class_name="android.widget.Button",
                    bbox_pixels=FakeBBox(10, 20, 110, 60),
                    is_clickable=True,
                    is_enabled=True,
                    is_visible=True,
                    package_name="com.android.contacts",
                ),
                FakeUIElement(
                    text="tiny",
                    class_name="android.widget.TextView",
                    bbox_pixels=FakeBBox(1, 1, 4, 4),
                    is_visible=True,
                ),
                FakeUIElement(
                    text="Hidden",
                    class_name="android.widget.TextView",
                    bbox_pixels=FakeBBox(20, 90, 120, 120),
                    is_visible=False,
                ),
            ],
        )

    def get_state(self, wait_to_stabilize: bool = False) -> FakeState:
        self.wait_to_stabilize = wait_to_stabilize
        return self._state


class SequencedFakeEnv(FakeEnv):
    def __init__(self, states: list[FakeState], activity: str) -> None:
        super().__init__()
        self._states = list(states)
        self.foreground_activity_name = activity
        self._state = self._states[0]
        self._call_count = 0

    def get_state(self, wait_to_stabilize: bool = False) -> FakeState:
        self.wait_to_stabilize = wait_to_stabilize
        index = min(self._call_count, len(self._states) - 1)
        self._state = self._states[index]
        self._call_count += 1
        return self._state


@unittest.skipUnless(PIL_AVAILABLE, "Pillow is not installed in this environment.")
class ObservationUtilsTest(unittest.TestCase):
    def test_bbox_to_tuple(self) -> None:
        self.assertEqual(bbox_to_tuple(FakeBBox(1, 2, 3, 4)), (1, 2, 3, 4))

    def test_is_visible_candidate_filters_small_and_hidden_elements(self) -> None:
        visible = FakeUIElement(bbox_pixels=FakeBBox(0, 0, 20, 20), is_visible=True)
        tiny = FakeUIElement(bbox_pixels=FakeBBox(0, 0, 4, 4), is_visible=True)
        hidden = FakeUIElement(bbox_pixels=FakeBBox(0, 0, 20, 20), is_visible=False)

        self.assertTrue(is_visible_candidate(visible, (100, 100)))
        self.assertFalse(is_visible_candidate(tiny, (100, 100)))
        self.assertFalse(is_visible_candidate(hidden, (100, 100)))

    def test_build_ui_description_uses_original_indices(self) -> None:
        elements = [
            FakeUIElement(text="A", bbox_pixels=FakeBBox(0, 0, 20, 20), is_visible=True),
            FakeUIElement(text="B", bbox_pixels=FakeBBox(0, 0, 4, 4), is_visible=True),
            FakeUIElement(text="C", bbox_pixels=FakeBBox(10, 10, 30, 40), is_visible=True),
        ]
        ui_description, indices = build_ui_description(elements, (100, 100), max_elements=10)

        self.assertEqual(indices, [0, 2])
        self.assertIn("UI element 0", ui_description)
        self.assertIn("UI element 2", ui_description)
        self.assertNotIn("UI element 1", ui_description)

    def test_standardize_ui_element(self) -> None:
        element = FakeUIElement(
            text="Create contact",
            class_name="android.widget.Button",
            bbox_pixels=FakeBBox(1, 2, 11, 12),
            is_visible=True,
        )
        standardized = standardize_ui_element(7, element)
        self.assertEqual(standardized["index"], 7)
        self.assertEqual(standardized["bbox"], (1, 2, 11, 12))
        self.assertEqual(standardized["text"], "Create contact")

    @patch("dms_reproduction.envs.observation_utils.Image.fromarray")
    def test_draw_labeled_screenshot_can_hide_indices(self, utils_fromarray) -> None:
        utils_fromarray.return_value = Image.new("RGB", (20, 40), color="black")
        state = FakeState(
            pixels=object(),
            ui_elements=[
                FakeUIElement(
                    text="Create contact",
                    class_name="android.widget.Button",
                    bbox_pixels=FakeBBox(2, 3, 18, 20),
                    is_clickable=True,
                    is_enabled=True,
                    is_visible=True,
                    package_name="com.android.contacts",
                ),
            ],
        )

        image = draw_labeled_screenshot(state, [0], draw_indices=False)
        self.assertIsInstance(image, Image.Image)


@unittest.skipUnless(PIL_AVAILABLE, "Pillow is not installed in this environment.")
class AndroidWorldObservationAdapterTest(unittest.TestCase):
    @patch("dms_reproduction.envs.observation_utils.Image.fromarray")
    @patch("dms_reproduction.envs.android_world_adapter.Image.fromarray")
    def test_capture_observation_returns_expected_schema(
        self,
        adapter_fromarray,
        utils_fromarray,
    ) -> None:
        adapter_fromarray.return_value = Image.new("RGB", (20, 40), color="black")
        utils_fromarray.return_value = Image.new("RGB", (20, 40), color="black")
        adapter = AndroidWorldObservationAdapter(max_ui_elements=50)
        env = FakeEnv()

        observation = adapter.capture_observation(
            env,
            goal="Add a contact",
            step_id=3,
            include_screenshots=True,
        )

        self.assertEqual(observation["goal"], "Add a contact")
        self.assertEqual(
            observation["current_activity"],
            "com.android.contacts/.activities.PeopleActivity",
        )
        self.assertEqual(observation["foreground_package"], "com.android.contacts")
        self.assertEqual(observation["app_name"], "com.android.contacts")
        self.assertEqual(observation["screen_size"], {"width": 200, "height": 400})
        self.assertEqual(observation["valid_ui_indices"], [0])
        self.assertEqual(observation["visible_ui_count"], 1)
        self.assertEqual(observation["clickable_ui_count"], 1)
        self.assertEqual(observation["non_system_ui_count"], 1)
        self.assertIsNone(observation["observation_warning"])
        self.assertEqual(observation["observation_consistency"], "stable")
        self.assertEqual(observation["extra_state"]["step_id"], 3)
        self.assertEqual(observation["extra_state"]["orientation"], 0)
        self.assertEqual(observation["extra_state"]["observation_attempt"], 1)
        self.assertFalse(observation["extra_state"]["observation_resampled"])
        self.assertEqual(len(observation["ui_elements"]), 1)
        self.assertEqual(observation["ui_elements"][0]["index"], 0)
        self.assertIn("UI element 0", observation["ui_description"])
        self.assertTrue(env.wait_to_stabilize)

        raw = base64.b64decode(observation["screenshot_b64"])
        labeled = base64.b64decode(observation["labeled_screenshot_b64"])
        actor_labeled = base64.b64decode(observation["actor_labeled_screenshot_b64"])
        self.assertGreater(len(raw), 0)
        self.assertGreater(len(labeled), 0)
        self.assertGreater(len(actor_labeled), 0)

    @patch("dms_reproduction.envs.observation_utils.Image.fromarray")
    @patch("dms_reproduction.envs.android_world_adapter.Image.fromarray")
    def test_capture_observation_marks_unstable_and_resamples(
        self,
        adapter_fromarray,
        utils_fromarray,
    ) -> None:
        adapter_fromarray.return_value = Image.new("RGB", (20, 40), color="black")
        utils_fromarray.return_value = Image.new("RGB", (20, 40), color="black")
        env = FakeEnv()
        env.foreground_activity_name = "com.google.android.dialer/.DialtactsActivity"
        env._state = FakeState(
            pixels=object(),
            ui_elements=[
                FakeUIElement(
                    text="Phone",
                    class_name="android.widget.TextView",
                    bbox_pixels=FakeBBox(10, 20, 110, 60),
                    is_clickable=True,
                    is_enabled=True,
                    is_visible=True,
                    package_name="com.google.android.apps.nexuslauncher",
                ),
            ],
        )
        adapter = AndroidWorldObservationAdapter(max_ui_elements=50, max_resample_attempts=1, resample_delay_seconds=0.0)

        observation = adapter.capture_observation(
            env,
            goal="Add a contact",
            step_id=1,
            include_screenshots=True,
        )

        self.assertEqual(observation["observation_consistency"], "unstable")
        self.assertIn("launcher-dominated", observation["observation_warning"])
        self.assertEqual(observation["extra_state"]["observation_attempt"], 2)
        self.assertTrue(observation["extra_state"]["observation_resampled"])

    @patch("dms_reproduction.envs.observation_utils.Image.fromarray")
    @patch("dms_reproduction.envs.android_world_adapter.Image.fromarray")
    def test_capture_observation_marks_system_ui_only_as_unstable(
        self,
        adapter_fromarray,
        utils_fromarray,
    ) -> None:
        adapter_fromarray.return_value = Image.new("RGB", (20, 40), color="black")
        utils_fromarray.return_value = Image.new("RGB", (20, 40), color="black")
        env = FakeEnv()
        env.foreground_activity_name = "com.google.android.contacts/.activities.ContactEditorActivity"
        env._state = FakeState(
            pixels=object(),
            ui_elements=[
                FakeUIElement(
                    text=None,
                    content_description="Battery",
                    class_name="android.widget.TextView",
                    bbox_pixels=FakeBBox(10, 20, 110, 60),
                    is_clickable=False,
                    is_enabled=True,
                    is_visible=True,
                    package_name="com.android.systemui",
                ),
            ],
        )
        adapter = AndroidWorldObservationAdapter(max_ui_elements=50, max_resample_attempts=0, resample_delay_seconds=0.0)

        observation = adapter.capture_observation(
            env,
            goal="Add a contact",
            step_id=1,
            include_screenshots=True,
        )

        self.assertEqual(observation["observation_consistency"], "unstable")
        self.assertIn("system-ui-visible", " ".join(observation["unstable_reasons"]))

    @patch("dms_reproduction.envs.observation_utils.Image.fromarray")
    @patch("dms_reproduction.envs.android_world_adapter.Image.fromarray")
    def test_capture_observation_marks_keyboard_active_context_without_unstable_warning(
        self,
        adapter_fromarray,
        utils_fromarray,
    ) -> None:
        adapter_fromarray.return_value = Image.new("RGB", (20, 40), color="black")
        utils_fromarray.return_value = Image.new("RGB", (20, 40), color="black")
        env = FakeEnv()
        env.foreground_activity_name = "com.google.android.contacts/.activities.ContactEditorActivity"
        env._state = FakeState(
            pixels=object(),
            ui_elements=[
                FakeUIElement(
                    text="First name",
                    content_description="First name",
                    class_name="android.widget.EditText",
                    bbox_pixels=FakeBBox(10, 20, 110, 60),
                    is_clickable=True,
                    is_editable=True,
                    is_enabled=True,
                    is_visible=True,
                    package_name="com.google.android.contacts",
                ),
                FakeUIElement(
                    text="q",
                    class_name="android.widget.Button",
                    bbox_pixels=FakeBBox(10, 200, 40, 240),
                    is_clickable=True,
                    is_enabled=True,
                    is_visible=True,
                    package_name="com.google.android.inputmethod.latin",
                ),
                FakeUIElement(
                    text="w",
                    class_name="android.widget.Button",
                    bbox_pixels=FakeBBox(45, 200, 75, 240),
                    is_clickable=True,
                    is_enabled=True,
                    is_visible=True,
                    package_name="com.google.android.inputmethod.latin",
                ),
            ],
        )
        adapter = AndroidWorldObservationAdapter(max_ui_elements=50, max_resample_attempts=0, resample_delay_seconds=0.0)

        observation = adapter.capture_observation(
            env,
            goal="Add a contact",
            step_id=1,
            include_screenshots=True,
        )

        self.assertTrue(observation["keyboard_active_context"])
        self.assertEqual(observation["observation_consistency"], "stable")
        self.assertIn("Soft keyboard is active", observation["observation_warning"] or "")

    @patch("dms_reproduction.envs.observation_utils.Image.fromarray")
    @patch("dms_reproduction.envs.android_world_adapter.Image.fromarray")
    def test_capture_observation_resamples_when_selected_tab_snapshot_changes(
        self,
        adapter_fromarray,
        utils_fromarray,
    ) -> None:
        adapter_fromarray.return_value = Image.new("RGB", (20, 40), color="black")
        utils_fromarray.return_value = Image.new("RGB", (20, 40), color="black")
        stale_voicemail_state = FakeState(
            pixels=object(),
            ui_elements=[
                FakeUIElement(
                    content_description="Contacts",
                    resource_name="com.google.android.dialer:id/tab_contacts",
                    class_name="android.widget.FrameLayout",
                    bbox_pixels=FakeBBox(0, 300, 50, 340),
                    is_clickable=True,
                    is_enabled=True,
                    is_visible=True,
                    package_name="com.google.android.dialer",
                ),
                FakeUIElement(
                    content_description="Voicemail",
                    resource_name="com.google.android.dialer:id/tab_voicemail",
                    class_name="android.widget.FrameLayout",
                    bbox_pixels=FakeBBox(50, 300, 100, 340),
                    is_clickable=False,
                    is_enabled=True,
                    is_visible=True,
                    package_name="com.google.android.dialer",
                ),
                FakeUIElement(
                    text="You don't have any voicemail messages yet",
                    resource_name="com.google.android.dialer:id/empty_content_view_message",
                    class_name="android.widget.TextView",
                    bbox_pixels=FakeBBox(10, 120, 180, 170),
                    is_clickable=False,
                    is_enabled=True,
                    is_visible=True,
                    package_name="com.google.android.dialer",
                ),
            ],
        )
        stale_voicemail_state.ui_elements[1].is_visible = True
        stale_voicemail_state.ui_elements[1].is_selected = True
        correct_contacts_state = FakeState(
            pixels=object(),
            ui_elements=[
                FakeUIElement(
                    content_description="Contacts",
                    resource_name="com.google.android.dialer:id/tab_contacts",
                    class_name="android.widget.FrameLayout",
                    bbox_pixels=FakeBBox(0, 300, 50, 340),
                    is_clickable=True,
                    is_enabled=True,
                    is_visible=True,
                    package_name="com.google.android.dialer",
                ),
                FakeUIElement(
                    content_description="Voicemail",
                    resource_name="com.google.android.dialer:id/tab_voicemail",
                    class_name="android.widget.FrameLayout",
                    bbox_pixels=FakeBBox(50, 300, 100, 340),
                    is_clickable=False,
                    is_enabled=True,
                    is_visible=True,
                    package_name="com.google.android.dialer",
                ),
                FakeUIElement(
                    text="Create new contact",
                    resource_name="com.google.android.dialer:id/new_contact",
                    class_name="android.widget.TextView",
                    bbox_pixels=FakeBBox(10, 120, 180, 170),
                    is_clickable=True,
                    is_enabled=True,
                    is_visible=True,
                    package_name="com.google.android.dialer",
                ),
            ],
        )
        correct_contacts_state.ui_elements[0].is_selected = True
        env = SequencedFakeEnv(
            [stale_voicemail_state, correct_contacts_state, correct_contacts_state],
            "com.google.android.dialer/.DialtactsActivity",
        )
        adapter = AndroidWorldObservationAdapter(max_ui_elements=50, max_resample_attempts=2, resample_delay_seconds=0.0)

        observation = adapter.capture_observation(
            env,
            goal="Add a contact",
            step_id=1,
            include_screenshots=True,
        )

        self.assertEqual(observation["observation_consistency"], "stable")
        self.assertEqual(observation["extra_state"]["observation_attempt"], 3)
        self.assertTrue(observation["extra_state"]["observation_resampled"])
        self.assertIn("Create new contact", observation["ui_description"])
        self.assertNotIn("voicemail messages yet", observation["ui_description"])


if __name__ == "__main__":
    unittest.main()
