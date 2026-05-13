import argparse
import dataclasses
import json
import os
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from android_world import registry
from android_world.env import env_launcher


def find_adb_path() -> str:
    """Find adb executable on Windows / Linux / macOS."""
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

    candidates.extend(
        [
            str(Path.home() / "Android" / "Sdk" / "platform-tools" / "adb"),
            str(Path.home() / "Library" / "Android" / "sdk" / "platform-tools" / "adb"),
        ]
    )

    for path in candidates:
        if path and Path(path).is_file():
            return path

    raise FileNotFoundError(
        "adb not found. Please pass --adb_path explicitly, e.g. "
        r'--adb_path "%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe"'
    )


def to_jsonable(obj: Any) -> Any:
    """Convert dataclass / numpy-ish objects into JSON-serializable objects."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AndroidWorld env smoke test: no LLM, no default agent."
    )
    parser.add_argument("--task", default="ContactsAddContact")
    parser.add_argument("--console_port", type=int, default=5554)
    parser.add_argument("--grpc_port", type=int, default=8554)
    parser.add_argument("--adb_path", default=None)
    parser.add_argument("--perform_emulator_setup", action="store_true")
    parser.add_argument("--output_dir", default="smoke_outputs")
    parser.add_argument("--max_print_elements", type=int, default=20)
    parser.add_argument("--list_tasks", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

    task_registry = registry.TaskRegistry()
    aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)

    if args.list_tasks:
        print("Available AndroidWorld tasks:")
        for name in sorted(aw_registry.keys()):
            print(name)
        return

    if args.task not in aw_registry:
        print(f"[ERROR] Task not found: {args.task}")
        print("Use --list_tasks to inspect available task names.")
        raise SystemExit(1)

    adb_path = args.adb_path or find_adb_path()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = None
    try:
        print("[1/5] Loading AndroidWorld env...")
        print(f"      adb_path     = {adb_path}")
        print(f"      console_port = {args.console_port}")
        print(f"      grpc_port    = {args.grpc_port}")

        env = env_launcher.load_and_setup_env( # 启动虚拟机
            console_port=args.console_port,
            emulator_setup=args.perform_emulator_setup,
            adb_path=adb_path,
            grpc_port=args.grpc_port,
        )

        print("[2/5] Resetting env to home...")
        env.reset(go_home=True)

        print(f"[3/5] Initializing task: {args.task}")
        task_type = aw_registry[args.task]
        params = task_type.generate_random_params()
        task = task_type(params)
        task.initialize_task(env)

        print("\n========== TASK ==========")
        print(f"task_name: {args.task}")
        print(f"params   : {params}")
        print(f"goal     : {task.goal}")

        print("\n[4/5] Getting state / screenshot / UI elements...")
        state = env.get_state(wait_to_stabilize=True)

        screenshot_path = output_dir / f"{args.task}_screenshot.png"
        ui_json_path = output_dir / f"{args.task}_ui_elements.json"

        Image.fromarray(state.pixels).save(screenshot_path)

        ui_elements = state.ui_elements
        ui_json = [to_jsonable(e) for e in ui_elements]
        with ui_json_path.open("w", encoding="utf-8") as f:
            json.dump(ui_json, f, ensure_ascii=False, indent=2, default=str)

        print("\n========== STATE ==========")
        print(f"foreground_activity: {env.foreground_activity_name}")
        print(f"device_screen_size : {env.device_screen_size}")
        print(f"logical_screen_size: {env.logical_screen_size}")
        print(f"screenshot_shape   : {getattr(state.pixels, 'shape', None)}")
        print(f"ui_elements_count  : {len(ui_elements)}")
        print(f"screenshot_saved   : {screenshot_path}")
        print(f"ui_elements_saved  : {ui_json_path}")

        print("\n========== FIRST UI ELEMENTS ==========")
        for i, elem in enumerate(ui_elements[: args.max_print_elements]):
            data = to_jsonable(elem)
            text = data.get("text")
            desc = data.get("content_description")
            cls = data.get("class_name")
            clickable = data.get("is_clickable")
            editable = data.get("is_editable")
            bbox = data.get("bbox_pixels")
            print(
                f"[{i}] text={text!r}, desc={desc!r}, "
                f"class={cls!r}, clickable={clickable}, "
                f"editable={editable}, bbox={bbox}"
            )

        print("\n[5/5] Smoke test finished successfully.")
        print("[OK] AndroidWorld env can load task, observe screenshot and UI elements without LLM.")

    finally:
        if env is not None:
            print("\nClosing env...")
            env.close()


if __name__ == "__main__":
    main()