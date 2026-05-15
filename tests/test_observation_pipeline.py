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
        self.assertEqual(observation["app_name"], "com.android.contacts")
        self.assertEqual(observation["screen_size"], {"width": 200, "height": 400})
        self.assertEqual(observation["valid_ui_indices"], [0])
        self.assertEqual(observation["extra_state"]["step_id"], 3)
        self.assertEqual(observation["extra_state"]["orientation"], 0)
        self.assertEqual(len(observation["ui_elements"]), 1)
        self.assertEqual(observation["ui_elements"][0]["index"], 0)
        self.assertIn("UI element 0", observation["ui_description"])
        self.assertTrue(env.wait_to_stabilize)

        raw = base64.b64decode(observation["screenshot_b64"])
        labeled = base64.b64decode(observation["labeled_screenshot_b64"])
        self.assertGreater(len(raw), 0)
        self.assertGreater(len(labeled), 0)


if __name__ == "__main__":
    unittest.main()
