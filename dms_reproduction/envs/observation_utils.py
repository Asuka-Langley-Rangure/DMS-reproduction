from __future__ import annotations

import base64
import dataclasses
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def to_jsonable(obj: Any) -> Any:
    """Convert nested dataclasses and array-like objects into JSON-safe values."""
    if dataclasses.is_dataclass(obj):
        return {key: to_jsonable(value) for key, value in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def pil_to_base64_png(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_png_to_bytes(image_b64: str) -> bytes:
    """Decode a base64 PNG string into raw bytes."""
    return base64.b64decode(image_b64.encode("utf-8"))


def save_base64_png(image_b64: str, path: str | Path) -> None:
    """Persist a base64 PNG string to disk."""
    Path(path).write_bytes(base64_png_to_bytes(image_b64))


def bbox_to_tuple(bbox: Any) -> tuple[int, int, int, int] | None:
    """Convert AndroidWorld BoundingBox-like objects to integer tuples."""
    if bbox is None:
        return None
    return (
        int(getattr(bbox, "x_min", 0)),
        int(getattr(bbox, "y_min", 0)),
        int(getattr(bbox, "x_max", 0)),
        int(getattr(bbox, "y_max", 0)),
    )


def is_visible_candidate(element: Any, screen_size: tuple[int, int]) -> bool:
    """Filter elements down to visible, on-screen, useful prompt candidates."""
    bbox = getattr(element, "bbox_pixels", None)
    if bbox is None:
        return False

    x_min, y_min, x_max, y_max = bbox_to_tuple(bbox) or (0, 0, 0, 0)
    width = x_max - x_min
    height = y_max - y_min
    if width <= 5 or height <= 5:
        return False

    screen_width, screen_height = screen_size
    if x_max <= 0 or y_max <= 0 or x_min >= screen_width or y_min >= screen_height:
        return False

    if hasattr(element, "is_visible") and getattr(element, "is_visible") is False:
        return False

    return True


def element_brief(element: Any) -> str:
    """Render a compact one-line description for prompt consumption."""
    fields: list[str] = []

    for name in ("text", "content_description", "resource_name", "class_name"):
        value = getattr(element, name, None)
        if value:
            fields.append(f"{name}={value!r}")

    for name in ("is_clickable", "is_editable", "is_enabled", "is_scrollable"):
        if hasattr(element, name):
            fields.append(f"{name}={getattr(element, name)}")

    fields.append(f"bbox={bbox_to_tuple(getattr(element, 'bbox_pixels', None))}")
    return ", ".join(fields)


def build_ui_description(
    ui_elements: list[Any],
    screen_size: tuple[int, int],
    max_elements: int,
) -> tuple[str, list[int]]:
    """Build the visible UI text list and record retained original indices."""
    lines: list[str] = []
    valid_indices: list[int] = []

    for index, element in enumerate(ui_elements):
        if not is_visible_candidate(element, screen_size):
            continue

        valid_indices.append(index)
        lines.append(f"UI element {index}: {element_brief(element)}")
        if len(lines) >= max_elements:
            break

    return "\n".join(lines), valid_indices


def draw_labeled_screenshot(
    state: Any,
    valid_indices: list[int],
    *,
    draw_indices: bool = True,
) -> Image.Image:
    """Overlay element boxes and optional original UI indices onto the screenshot."""
    image = Image.fromarray(state.pixels).convert("RGB")
    draw = ImageDraw.Draw(image)

    for index in valid_indices:
        element = state.ui_elements[index]
        bbox = bbox_to_tuple(getattr(element, "bbox_pixels", None))
        if bbox is None:
            continue

        x_min, y_min, x_max, y_max = bbox
        x_min = max(0, x_min)
        y_min = max(0, y_min)

        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        if draw_indices:
            draw.rectangle([x_min, y_min, x_min + 48, y_min + 28], fill="red")
            draw.text((x_min + 3, y_min + 3), str(index), fill="white")

    return image


def standardize_ui_element(index: int, element: Any) -> dict[str, Any]:
    """Convert a UIElement-like object into the shared observation schema."""
    return {
        "index": index,
        "text": getattr(element, "text", None),
        "content_description": getattr(element, "content_description", None),
        "resource_name": getattr(element, "resource_name", None),
        "class_name": getattr(element, "class_name", None),
        "bbox": bbox_to_tuple(getattr(element, "bbox_pixels", None)),
        "is_clickable": getattr(element, "is_clickable", None),
        "is_editable": getattr(element, "is_editable", None),
        "is_enabled": getattr(element, "is_enabled", None),
        "is_scrollable": getattr(element, "is_scrollable", None),
        "is_visible": getattr(element, "is_visible", None),
        "package_name": getattr(element, "package_name", None),
        "raw": to_jsonable(element),
    }
