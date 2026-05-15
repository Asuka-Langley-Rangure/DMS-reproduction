from __future__ import annotations

from collections import Counter
from typing import Any

from PIL import Image

from dms_reproduction.envs.observation_utils import (
    build_ui_description,
    draw_labeled_screenshot,
    pil_to_base64_png,
    standardize_ui_element,
)


class AndroidWorldObservationAdapter:
    """Create the standard observation payload consumed by planner and actor."""

    def __init__(self, max_ui_elements: int = 50) -> None:
        self.max_ui_elements = max_ui_elements

    def capture_observation(
        self,
        env: Any,
        goal: str,
        *,
        step_id: int = 0,
        include_screenshots: bool = True,
    ) -> dict[str, Any]:
        state = env.get_state(wait_to_stabilize=True)
        screen_size_tuple = env.logical_screen_size
        ui_description, valid_ui_indices = build_ui_description(
            state.ui_elements,
            screen_size_tuple,
            self.max_ui_elements,
        )

        standardized_ui = [
            standardize_ui_element(index, state.ui_elements[index])
            for index in valid_ui_indices
        ]

        raw_image = Image.fromarray(state.pixels).convert("RGB")
        labeled_image = draw_labeled_screenshot(state, valid_ui_indices)
        app_name = self._infer_app_name(standardized_ui, env.foreground_activity_name)

        observation = {
            "goal": goal,
            "current_activity": env.foreground_activity_name,
            "app_name": app_name,
            "screen_size": {
                "width": int(screen_size_tuple[0]),
                "height": int(screen_size_tuple[1]),
            },
            "ui_elements": standardized_ui,
            "ui_description": ui_description,
            "valid_ui_indices": valid_ui_indices,
            "screenshot_b64": None,
            "labeled_screenshot_b64": None,
            "extra_state": {
                "step_id": step_id,
                "orientation": getattr(env, "orientation", None),
            },
        }

        if include_screenshots:
            observation["screenshot_b64"] = pil_to_base64_png(raw_image)
            observation["labeled_screenshot_b64"] = pil_to_base64_png(labeled_image)

        return observation

    @staticmethod
    def _infer_app_name(
        standardized_ui_elements: list[dict[str, Any]],
        current_activity: str,
    ) -> str | None:
        packages = [
            element.get("package_name")
            for element in standardized_ui_elements
            if element.get("package_name")
        ]
        if packages:
            return Counter(packages).most_common(1)[0][0]

        if current_activity and "/" in current_activity:
            return current_activity.split("/", 1)[0]
        if current_activity and "." in current_activity:
            parts = current_activity.split(".")
            if len(parts) > 1:
                return ".".join(parts[:-1])
        return None
