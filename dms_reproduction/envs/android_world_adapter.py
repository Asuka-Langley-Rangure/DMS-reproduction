from __future__ import annotations

from collections import Counter
import time
from typing import Any

from PIL import Image

from dms_reproduction.envs.observation_utils import (
    build_ui_description,
    draw_labeled_screenshot,
    pil_to_base64_png,
    standardize_ui_element,
)

IME_PACKAGES = {
    "com.google.android.inputmethod.latin",
    "com.android.inputmethod.latin",
}


class AndroidWorldObservationAdapter:
    """Create the standard observation payload consumed by planner and actor."""

    def __init__(
        self,
        max_ui_elements: int = 50,
        *,
        max_resample_attempts: int = 2,
        resample_delay_seconds: float = 0.2,
    ) -> None:
        self.max_ui_elements = max_ui_elements
        self.max_resample_attempts = max_resample_attempts
        self.resample_delay_seconds = resample_delay_seconds

    def capture_observation(
        self,
        env: Any,
        goal: str,
        *,
        step_id: int = 0,
        include_screenshots: bool = True,
    ) -> dict[str, Any]:
        attempts = 0
        last_observation: dict[str, Any] | None = None
        previous_stable_observation: dict[str, Any] | None = None
        while True:
            attempts += 1
            last_observation = self._capture_once(
                env,
                goal=goal,
                step_id=step_id,
                include_screenshots=include_screenshots,
                attempt=attempts,
            )
            if last_observation["observation_consistency"] == "stable":
                if (
                    previous_stable_observation is not None
                    and self._observations_match(previous_stable_observation, last_observation)
                ):
                    return last_observation
                if self._needs_stability_confirmation(last_observation):
                    previous_stable_observation = last_observation
                else:
                    return last_observation
            else:
                previous_stable_observation = None
            if attempts > self.max_resample_attempts:
                if (
                    previous_stable_observation is not None
                    and last_observation["observation_consistency"] == "stable"
                    and not self._observations_match(previous_stable_observation, last_observation)
                ):
                    notes = list(last_observation.get("unstable_reasons") or [])
                    notes.append(
                        "Consecutive stable captures disagreed; likely stale or mismatched UI tree."
                    )
                    last_observation["observation_consistency"] = "unstable"
                    last_observation["unstable_reasons"] = notes
                    last_observation["observation_warning"] = self._build_observation_warning(
                        foreground_package=last_observation.get("foreground_package"),
                        dominant_ui_package=last_observation.get("app_name"),
                        visible_ui_count=int(last_observation.get("visible_ui_count") or 0),
                        clickable_ui_count=int(last_observation.get("clickable_ui_count") or 0),
                        non_system_ui_count=int(last_observation.get("non_system_ui_count") or 0),
                        consistency_notes=notes,
                        observation_consistency="unstable",
                        keyboard_active_context=bool(last_observation.get("keyboard_active_context")),
                    )
                return last_observation
            time.sleep(self.resample_delay_seconds)

    def _capture_once(
        self,
        env: Any,
        *,
        goal: str,
        step_id: int,
        include_screenshots: bool,
        attempt: int,
    ) -> dict[str, Any]:
        state = env.get_state(wait_to_stabilize=True)
        current_activity = getattr(env, "foreground_activity_name", None)
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
        foreground_package = self._extract_foreground_package(current_activity)
        dominant_ui_package = self._infer_dominant_ui_package(standardized_ui)
        visible_ui_count = len(standardized_ui)
        clickable_ui_count = sum(1 for element in standardized_ui if element.get("is_clickable"))
        editable_ui_count = sum(1 for element in standardized_ui if element.get("is_editable"))
        non_system_ui_count = sum(
            1
            for element in standardized_ui
            if (element.get("package_name") or "") != "com.android.systemui"
        )
        keyboard_active_context = self._is_keyboard_active_context(
            foreground_package=foreground_package,
            dominant_ui_package=dominant_ui_package,
            editable_ui_count=editable_ui_count,
        )
        observation_consistency, consistency_notes = self._assess_observation_consistency(
            foreground_package=foreground_package,
            dominant_ui_package=dominant_ui_package,
            standardized_ui=standardized_ui,
            clickable_ui_count=clickable_ui_count,
            non_system_ui_count=non_system_ui_count,
            keyboard_active_context=keyboard_active_context,
        )
        observation_warning = self._build_observation_warning(
            foreground_package=foreground_package,
            dominant_ui_package=dominant_ui_package,
            visible_ui_count=visible_ui_count,
            clickable_ui_count=clickable_ui_count,
            non_system_ui_count=non_system_ui_count,
            consistency_notes=consistency_notes,
            observation_consistency=observation_consistency,
            keyboard_active_context=keyboard_active_context,
        )

        observation = {
            "goal": goal,
            "current_activity": current_activity,
            "foreground_package": foreground_package,
            "app_name": dominant_ui_package,
            "screen_size": {
                "width": int(screen_size_tuple[0]),
                "height": int(screen_size_tuple[1]),
            },
            "ui_elements": standardized_ui,
            "ui_description": ui_description,
            "valid_ui_indices": valid_ui_indices,
            "visible_ui_count": visible_ui_count,
            "clickable_ui_count": clickable_ui_count,
            "editable_ui_count": editable_ui_count,
            "non_system_ui_count": non_system_ui_count,
            "keyboard_active_context": keyboard_active_context,
            "observation_warning": observation_warning,
            "observation_consistency": observation_consistency,
            "unstable_reasons": consistency_notes,
            "screenshot_b64": None,
            "labeled_screenshot_b64": None,
            "extra_state": {
                "step_id": step_id,
                "orientation": getattr(env, "orientation", None),
                "observation_attempt": attempt,
                "observation_resampled": attempt > 1,
                "final_after_resample": attempt > 1,
            },
        }

        if include_screenshots:
            observation["screenshot_b64"] = pil_to_base64_png(raw_image)
            observation["labeled_screenshot_b64"] = pil_to_base64_png(labeled_image)

        return observation

    @staticmethod
    def _extract_foreground_package(current_activity: str | None) -> str | None:
        if current_activity and "/" in current_activity:
            return current_activity.split("/", 1)[0]
        if current_activity and "." in current_activity:
            parts = current_activity.split(".")
            if len(parts) > 1:
                return ".".join(parts[:-1])
        return None

    @staticmethod
    def _infer_dominant_ui_package(
        standardized_ui_elements: list[dict[str, Any]],
    ) -> str | None:
        packages = [
            element.get("package_name")
            for element in standardized_ui_elements
            if element.get("package_name")
        ]
        if packages:
            return Counter(packages).most_common(1)[0][0]
        return None

    @staticmethod
    def _build_observation_warning(
        *,
        foreground_package: str | None,
        dominant_ui_package: str | None,
        visible_ui_count: int,
        clickable_ui_count: int,
        non_system_ui_count: int,
        consistency_notes: list[str],
        observation_consistency: str,
        keyboard_active_context: bool,
    ) -> str | None:
        warnings: list[str] = list(consistency_notes)
        if visible_ui_count == 0:
            warnings.append("No visible UI elements were retained.")
        if clickable_ui_count == 0 and visible_ui_count > 0:
            warnings.append("No clickable UI elements were retained.")
        if foreground_package and foreground_package != "com.android.systemui" and non_system_ui_count == 0:
            warnings.append(
                f"Only system UI elements were retained while foreground activity is {foreground_package}."
            )
        if keyboard_active_context:
            warnings.append(
                f"Soft keyboard is active while editing fields in {foreground_package}."
            )
        elif foreground_package and dominant_ui_package and foreground_package != dominant_ui_package:
            warnings.append(
                f"Foreground package ({foreground_package}) differs from dominant visible UI package ({dominant_ui_package})."
            )
        if observation_consistency == "unstable" and not warnings:
            warnings.append("Observation consistency is unstable.")
        if not warnings:
            return None
        return " ".join(warnings)

    @staticmethod
    def _assess_observation_consistency(
        *,
        foreground_package: str | None,
        dominant_ui_package: str | None,
        standardized_ui: list[dict[str, Any]],
        clickable_ui_count: int,
        non_system_ui_count: int,
        keyboard_active_context: bool,
    ) -> tuple[str, list[str]]:
        if not foreground_package:
            return "stable", []

        notes: list[str] = []
        packages = {str(element.get("package_name") or "") for element in standardized_ui}
        packages.discard("")
        non_system_packages = {pkg for pkg in packages if pkg != "com.android.systemui"}
        launcher_packages = {"com.google.android.apps.nexuslauncher", "com.android.launcher3"}
        foreground_is_launcher = foreground_package in launcher_packages
        dominant_is_launcher = dominant_ui_package in launcher_packages if dominant_ui_package else False

        if keyboard_active_context:
            return "stable", []
        if not foreground_is_launcher and dominant_is_launcher:
            notes.append(
                f"Foreground package ({foreground_package}) differs from launcher-dominated visible UI package ({dominant_ui_package})."
            )
        if foreground_package != "com.android.systemui" and non_system_ui_count == 0:
            notes.append(
                f"Foreground package ({foreground_package}) has only system-ui-visible elements retained."
            )
        if foreground_package != "com.android.systemui" and non_system_ui_count == 0 and clickable_ui_count == 0:
            notes.append(
                f"Foreground package ({foreground_package}) has no non-system clickable UI elements retained."
            )
        if not foreground_is_launcher and non_system_packages and foreground_package not in non_system_packages:
            notes.append(
                f"Foreground package ({foreground_package}) is missing from retained non-system UI packages ({sorted(non_system_packages)})."
            )
        if notes:
            return "unstable", notes
        return "stable", []

    @staticmethod
    def _is_keyboard_active_context(
        *,
        foreground_package: str | None,
        dominant_ui_package: str | None,
        editable_ui_count: int,
    ) -> bool:
        return bool(
            foreground_package
            and foreground_package not in IME_PACKAGES
            and dominant_ui_package in IME_PACKAGES
            and editable_ui_count > 0
        )

    @staticmethod
    def _needs_stability_confirmation(observation: dict[str, Any]) -> bool:
        ui_elements = observation.get("ui_elements") or []
        for element in ui_elements:
            raw = element.get("raw") or {}
            resource_name = str(element.get("resource_name") or "")
            content_description = str(element.get("content_description") or "")
            if raw.get("is_selected"):
                return True
            if "tab_" in resource_name:
                return True
            if content_description.lower() in {"favorites", "recents", "contacts", "voicemail"}:
                return True
        return False

    @staticmethod
    def _observations_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return AndroidWorldObservationAdapter._observation_signature(
            left
        ) == AndroidWorldObservationAdapter._observation_signature(right)

    @staticmethod
    def _observation_signature(observation: dict[str, Any]) -> tuple[Any, ...]:
        ui_elements = observation.get("ui_elements") or []
        normalized_elements: list[tuple[Any, ...]] = []
        for element in ui_elements[:15]:
            raw = element.get("raw") or {}
            bbox = tuple(element.get("bbox") or ())
            normalized_elements.append(
                (
                    element.get("index"),
                    element.get("resource_name"),
                    element.get("text"),
                    element.get("content_description"),
                    element.get("class_name"),
                    bbox,
                    bool(raw.get("is_selected")),
                    bool(element.get("is_clickable")),
                    bool(element.get("is_editable")),
                    element.get("package_name"),
                )
            )
        return (
            observation.get("foreground_package"),
            observation.get("app_name"),
            observation.get("current_activity"),
            tuple(normalized_elements),
        )
