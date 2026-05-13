import argparse
import dataclasses
import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from android_world import registry
from android_world.env import env_launcher
from android_world.env import json_action


def find_adb_path() -> str:
    candidates = []

    which_adb = shutil.which("adb")
    if which_adb:
        candidates.append(which_adb)

    for env_name in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        sdk_root = os.environ.get(env_name)
        if sdk_root:
            candidates.append(str(Path(sdk_root) / "platform-tools" / "adb.exe"))
            candidates.append(str(Path(sdk_root) / "platform-tools" / "adb"))

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(
            str(Path(local_app_data) / "Android" / "Sdk" / "platform-tools" / "adb.exe")
        )

    for path in candidates:
        if path and Path(path).is_file():
            return path

    raise FileNotFoundError("adb not found. Please pass --adb_path explicitly.")


def to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def load_env(args):
    adb_path = args.adb_path or find_adb_path()
    kwargs = {
        "console_port": args.console_port,
        "emulator_setup": args.perform_emulator_setup,
        "adb_path": adb_path,
    }

    print(f"adb_path     = {adb_path}")
    print(f"console_port = {args.console_port}")
    print(f"grpc_port    = {args.grpc_port}")

    try:
        return env_launcher.load_and_setup_env(**kwargs, grpc_port=args.grpc_port)
    except TypeError:
        # Some AndroidWorld versions do not expose grpc_port in load_and_setup_env.
        return env_launcher.load_and_setup_env(**kwargs)


def save_state(env, label: str, output_dir: Path):
    state = env.get_state(wait_to_stabilize=True)

    screenshot_path = output_dir / f"{label}_screenshot.png"
    ui_path = output_dir / f"{label}_ui_elements.json"

    Image.fromarray(state.pixels).save(screenshot_path)

    with ui_path.open("w", encoding="utf-8") as f:
        json.dump(
            [to_jsonable(e) for e in state.ui_elements],
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print(f"\n========== {label} ==========")
    print(f"foreground_activity: {env.foreground_activity_name}")
    print(f"logical_screen_size: {env.logical_screen_size}")
    print(f"screenshot_shape   : {getattr(state.pixels, 'shape', None)}")
    print(f"ui_elements_count  : {len(state.ui_elements)}")
    print(f"screenshot_saved   : {screenshot_path}")
    print(f"ui_elements_saved  : {ui_path}")

    return state


def element_text(elem) -> str:
    parts = []
    if getattr(elem, "text", None):
        parts.append(str(elem.text))
    if getattr(elem, "content_description", None):
        parts.append(str(elem.content_description))
    if getattr(elem, "resource_name", None):
        parts.append(str(elem.resource_name))
    if getattr(elem, "class_name", None):
        parts.append(str(elem.class_name))
    return " | ".join(parts)


def is_clickable_candidate(elem) -> bool:
    bbox = getattr(elem, "bbox_pixels", None)
    if bbox is None:
        return False

    width = bbox.x_max - bbox.x_min
    height = bbox.y_max - bbox.y_min

    return bool(
        getattr(elem, "is_visible", True)
        and getattr(elem, "is_enabled", True)
        and getattr(elem, "is_clickable", False)
        and width > 5
        and height > 5
    )


def list_clickable_candidates(state, limit: int = 30):
    print("\n========== CLICKABLE CANDIDATES ==========")
    count = 0
    for i, elem in enumerate(state.ui_elements):
        if is_clickable_candidate(elem):
            bbox = getattr(elem, "bbox_pixels", None)
            print(f"[{i}] {element_text(elem)!r}, bbox={to_jsonable(bbox)}")
            count += 1
            if count >= limit:
                break
    if count == 0:
        print("No clickable candidate found.")


def find_click_index_by_text(state, target_text: str) -> Optional[int]:
    target = target_text.lower()
    for i, elem in enumerate(state.ui_elements):
        if not is_clickable_candidate(elem):
            continue
        label = element_text(elem).lower()
        if target in label:
            return i
    return None


def find_first_clickable_index(state) -> Optional[int]:
    for i, elem in enumerate(state.ui_elements):
        if is_clickable_candidate(elem):
            return i
    return None


def execute(env, action: json_action.JSONAction):
    print(f"\n>>> execute: {action}")
    env.execute_action(action)


def maybe_initialize_task(env, task_name: Optional[str]):
    if not task_name:
        return

    task_registry = registry.TaskRegistry()
    aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)

    if task_name not in aw_registry:
        raise ValueError(f"Task not found: {task_name}")

    task_type = aw_registry[task_name]
    params = task_type.generate_random_params()
    task = task_type(params)
    task.initialize_task(env)

    print("\n========== TASK ==========")
    print(f"task_name: {task_name}")
    print(f"params   : {params}")
    print(f"goal     : {task.goal}")


def main():
    parser = argparse.ArgumentParser(
        description="AndroidWorld manual action validation: no LLM, no default agent."
    )

    parser.add_argument("--task", default=None)
    parser.add_argument("--app_name", default="Settings")
    parser.add_argument("--console_port", type=int, default=5554)
    parser.add_argument("--grpc_port", type=int, default=8554)
    parser.add_argument("--adb_path", default=None)
    parser.add_argument("--perform_emulator_setup", action="store_true")
    parser.add_argument("--output_dir", default="manual_action_outputs")

    parser.add_argument(
        "--click_strategy",
        choices=["none", "first_clickable", "text", "index"],
        default="none",
    )
    parser.add_argument("--click_text", default=None)
    parser.add_argument("--click_index", type=int, default=None)

    parser.add_argument("--test_scroll", action="store_true")
    parser.add_argument("--test_back", action="store_true")

    args = parser.parse_args()

    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("GRPC_TRACE", "none")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = None

    try:
        print("[1/6] Loading env...")
        env = load_env(args)

        print("[2/6] Reset to home...")
        env.reset(go_home=True)
        save_state(env, "00_home", output_dir)

        print("[3/6] Optional task initialization...")
        maybe_initialize_task(env, args.task)
        save_state(env, "01_after_task_init", output_dir)

        print(f"[4/6] Open app: {args.app_name}")
        execute(env, json_action.JSONAction(action_type="open_app", app_name=args.app_name))
        state = save_state(env, "02_after_open_app", output_dir)

        list_clickable_candidates(state)

        print("[5/6] Optional click test...")
        click_index = None

        if args.click_strategy == "none":
            print("Skip click test. Use --click_strategy first_clickable/text/index to test click.")

        elif args.click_strategy == "first_clickable":
            click_index = find_first_clickable_index(state)

        elif args.click_strategy == "text":
            if not args.click_text:
                raise ValueError("--click_strategy text requires --click_text")
            click_index = find_click_index_by_text(state, args.click_text)

        elif args.click_strategy == "index":
            if args.click_index is None:
                raise ValueError("--click_strategy index requires --click_index")
            click_index = args.click_index

        if click_index is not None:
            if click_index < 0 or click_index >= len(state.ui_elements):
                raise ValueError(f"Invalid click index: {click_index}")

            print(f"Chosen click index: {click_index}")
            print(f"Chosen element    : {element_text(state.ui_elements[click_index])!r}")

            execute(env, json_action.JSONAction(action_type="click", index=click_index))
            state = save_state(env, "03_after_click", output_dir)
        elif args.click_strategy != "none":
            print("No valid clickable element found. Click test skipped.")

        if args.test_scroll:
            print("[extra] Scroll down...")
            execute(env, json_action.JSONAction(action_type="scroll", direction="down"))
            state = save_state(env, "04_after_scroll_down", output_dir)

        if args.test_back:
            print("[extra] Navigate back...")
            execute(env, json_action.JSONAction(action_type="navigate_back"))
            state = save_state(env, "05_after_back", output_dir)

        print("[6/6] Navigate home...")
        execute(env, json_action.JSONAction(action_type="navigate_home"))
        save_state(env, "06_after_home", output_dir)

        print("\n[OK] Manual action validation finished.")
        print("[OK] Observation -> Action -> Observation loop is available without LLM.")

    finally:
        if env is not None:
            print("\nClosing env...")
            env.close()


if __name__ == "__main__":
    main()