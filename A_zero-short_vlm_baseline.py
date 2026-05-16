import argparse
import base64
import dataclasses
import json
import os
import re
import shutil
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import requests
from PIL import Image, ImageDraw

from android_world import registry
from android_world.env import env_launcher
from android_world.env import json_action


VALID_ACTION_KEYS = {
    "action_type",
    "index",
    "x",
    "y",
    "text",
    "direction",
    "goal_status",
    "app_name",
    "keycode",
    "clear_text",
}


def find_adb_path() -> str: #自动寻找 adb.exe 的路径
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


def to_jsonable(obj: Any) -> Any: #把复杂 Python 对象转换成可以 json.dump() 保存的格式
    if dataclasses.is_dataclass(obj): # UIElement dataclass
        return dataclasses.asdict(obj)
    if isinstance(obj, (list, tuple)): # BBox 对象
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict): # numpy array
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "tolist"): # tuple
        return obj.tolist()
    return obj


def pil_to_base64_png(img: Image.Image) -> str: #把 PIL 图片转成 base64 字符串。
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def patch_a11y_forwarder_apk(local_apk: str | None) -> None:
    if not local_apk:
        return

    apk_path = Path(local_apk).resolve()
    if not apk_path.is_file():
        raise FileNotFoundError(f"A11y forwarder APK not found: {apk_path}")

    apk_bytes = apk_path.read_bytes()
    if len(apk_bytes) < 1024 * 1024:
        raise ValueError(
            f"A11y forwarder APK seems too small: {len(apk_bytes)} bytes. "
            f"Please check whether the file is fully downloaded."
        )

    from android_env.wrappers import a11y_grpc_wrapper

    def _local_get_accessibility_forwarder_apk():
        return apk_bytes

    a11y_grpc_wrapper._get_accessibility_forwarder_apk = _local_get_accessibility_forwarder_apk
    print(f"[PATCH] Using local accessibility forwarder APK: {apk_path}")
    print(f"[PATCH] APK size: {len(apk_bytes)} bytes")

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
        return env_launcher.load_and_setup_env(**kwargs)


def bbox_to_tuple(bbox): #把 AndroidWorld 的 bbox 对象转换成普通四元组
    if bbox is None:
        return None
    return (
        int(getattr(bbox, "x_min", 0)),
        int(getattr(bbox, "y_min", 0)),
        int(getattr(bbox, "x_max", 0)),
        int(getattr(bbox, "y_max", 0)),
    )


def is_visible_candidate(elem, screen_size) -> bool: #判断一个 UI element 是否值得放进 prompt
    bbox = getattr(elem, "bbox_pixels", None)
    if bbox is None:
        return False

    x_min, y_min, x_max, y_max = bbox_to_tuple(bbox)
    width = x_max - x_min
    height = y_max - y_min

    if width <= 5 or height <= 5:
        return False

    screen_w, screen_h = screen_size
    if x_max <= 0 or y_max <= 0 or x_min >= screen_w or y_min >= screen_h:
        return False

    if hasattr(elem, "is_visible") and not elem.is_visible:
        return False

    return True


def element_brief(elem) -> str: #把一个 UI element 压缩成一行文本描述
    fields = []

    for name in ["text", "content_description", "resource_name", "class_name"]:
        value = getattr(elem, name, None)
        if value:
            fields.append(f"{name}={value!r}")

    for name in ["is_clickable", "is_editable", "is_enabled", "is_scrollable"]:
        if hasattr(elem, name):
            fields.append(f"{name}={getattr(elem, name)}")

    bbox = bbox_to_tuple(getattr(elem, "bbox_pixels", None))
    fields.append(f"bbox={bbox}")

    return ", ".join(fields)


def build_ui_description(ui_elements, screen_size, max_elements: int) -> tuple[str, list[int]]: #构造给 VLM 的 UI element 文本列表
    lines = []
    valid_indices = []

    for i, elem in enumerate(ui_elements):
        if not is_visible_candidate(elem, screen_size):
            continue

        valid_indices.append(i)
        lines.append(f"UI element {i}: {element_brief(elem)}")

        if len(lines) >= max_elements:
            break

    return "\n".join(lines), valid_indices


def draw_labeled_screenshot(state, valid_indices: list[int]) -> Image.Image: #在截图上给 UI 元素画红框和数字编号
    img = Image.fromarray(state.pixels).convert("RGB")
    draw = ImageDraw.Draw(img)

    for idx in valid_indices:
        elem = state.ui_elements[idx]
        bbox = bbox_to_tuple(getattr(elem, "bbox_pixels", None))
        if bbox is None:
            continue

        x_min, y_min, x_max, y_max = bbox
        x_min = max(0, x_min)
        y_min = max(0, y_min)

        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        draw.rectangle([x_min, y_min, x_min + 48, y_min + 28], fill="red")
        draw.text((x_min + 3, y_min + 3), str(idx), fill="white")

    return img


def extract_json_object(text: str) -> Optional[dict]: #从模型中提取对应的Json action
    """
    Extract the first JSON-like object from model output.
    Supports pure JSON or text containing {...}.
    """
    text = text.strip()

    # Remove markdown fences if present.
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    candidates = []

    if text.startswith("{") and text.endswith("}"):
        candidates.append(text)

    matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    candidates.extend(matches)

    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            # Try a light fallback for single quotes.
            try:
                import ast

                value = ast.literal_eval(cand)
                if isinstance(value, dict):
                    return value
            except Exception:
                pass

    return None


def sanitize_action_dict(action_dict: dict) -> dict: #将得到的Json action中对应字段再次规范
    """
    action / type → action_type
    tap → click
    type → input_text
    press_enter → keyboard_enter
    """
    cleaned = {k: v for k, v in action_dict.items() if k in VALID_ACTION_KEYS}

    if "action" in action_dict and "action_type" not in cleaned:
        cleaned["action_type"] = action_dict["action"]

    if "type" in action_dict and "action_type" not in cleaned:
        cleaned["action_type"] = action_dict["type"]

    if cleaned.get("action_type") == "tap":
        cleaned["action_type"] = "click"

    if cleaned.get("action_type") == "type":
        cleaned["action_type"] = "input_text"

    if cleaned.get("action_type") == "press_enter":
        cleaned["action_type"] = "keyboard_enter"

    return cleaned


def make_prompt(goal: str, history: list[dict], ui_description: str) -> str:
    if history:
        history_text = "\n".join(
            [
                f"Step {h['step']}: action={h.get('action')}, "
                f"success_check={h.get('success_check')}, "
                f"note={h.get('note', '')}"
                for h in history[-4:]
            ]
        )
    else:
        history_text = "No previous action."

    return f"""
You are controlling an Android phone to complete the user's task.

User goal:
{goal}

Current screen:
You are given a screenshot with red bounding boxes and numeric labels.
The numeric labels correspond to UI element indexes below.

Visible UI elements:
{ui_description if ui_description else "No visible UI elements available."}

Recent action history:
{history_text}

You must output exactly ONE JSON object and nothing else.

Allowed actions:

1. Complete the task if the goal is already achieved:
{{"action_type": "status", "goal_status": "complete"}}

2. Stop if the task is impossible:
{{"action_type": "status", "goal_status": "infeasible"}}

3. Click a visible UI element:
{{"action_type": "click", "index": 12}}

4. Long press a visible UI element:
{{"action_type": "long_press", "index": 12}}

5. Input text into a visible text field:
{{"action_type": "input_text", "index": 12, "text": "hello", "clear_text": true}}

6. Press Enter:
{{"action_type": "keyboard_enter"}}

7. Navigate:
{{"action_type": "navigate_back"}}
{{"action_type": "navigate_home"}}

8. Scroll:
{{"action_type": "scroll", "direction": "down"}}
{{"action_type": "scroll", "direction": "up"}}
{{"action_type": "scroll", "direction": "left"}}
{{"action_type": "scroll", "direction": "right"}}

9. Open an app:
{{"action_type": "open_app", "app_name": "Contacts"}}

10. Wait:
{{"action_type": "wait"}}

Important rules:
- Use element indexes only from the visible UI element list.
- Do not invent indexes.
- Prefer open_app if you need to enter an app.
- Prefer input_text for typing; do not tap keyboard letters one by one.
- If an action did not change the screen before, try a different strategy.
- Output only JSON. No explanation. No markdown.
""".strip()


def call_openai_compatible_vlm(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    image: Image.Image,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    image_b64 = pil_to_base64_png(image)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def maybe_teardown(task, env):
    for name in ["tear_down", "teardown"]:
        fn = getattr(task, name, None)
        if callable(fn):
            try:
                fn(env)
            except Exception as e:
                print(f"[WARN] task teardown failed: {e}")
            return


def run_one_task(args) -> dict:
    env = None
    task = None

    run_name = f"{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history = []
    result = {
        "task": args.task,
        "goal": None,
        "success": False,
        "steps": 0,
        "run_dir": str(run_dir),
    }

    try:
        print("[1/5] Loading env...")
        env = load_env(args)

        print("[2/5] Reset env...")
        env.reset(go_home=True)

        print(f"[3/5] Initialize task: {args.task}")
        task_registry = registry.TaskRegistry()
        aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)

        if args.task not in aw_registry:
            raise ValueError(f"Task not found: {args.task}")

        task_type = aw_registry[args.task]
        params = task_type.generate_random_params()
        task = task_type(params)
        task.initialize_task(env)

        goal = str(task.goal)
        result["goal"] = goal

        print("\n========== TASK ==========")
        print(f"task   : {args.task}")
        print(f"params : {params}")
        print(f"goal   : {goal}")

        params_path = run_dir / "task_params.json"
        with params_path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(params), f, ensure_ascii=False, indent=2, default=str)

        print("[4/5] Start zero-shot VLM loop...")

        for step in range(1, args.max_steps + 1):
            print(f"\n---------- step {step}/{args.max_steps} ----------")

            state = env.get_state(wait_to_stabilize=True) # 获取截图和UI元素
            screen_size = env.logical_screen_size

            ui_description, valid_indices = build_ui_description(
                state.ui_elements,
                screen_size,
                args.max_ui_elements,
            )
            labeled_img = draw_labeled_screenshot(state, valid_indices)

            raw_img_path = run_dir / f"step_{step:02d}_raw.png"
            labeled_img_path = run_dir / f"step_{step:02d}_labeled.png"
            ui_path = run_dir / f"step_{step:02d}_ui.json"
            prompt_path = run_dir / f"step_{step:02d}_prompt.txt"
            response_path = run_dir / f"step_{step:02d}_response.txt"

            Image.fromarray(state.pixels).save(raw_img_path)
            labeled_img.save(labeled_img_path)

            with ui_path.open("w", encoding="utf-8") as f:
                json.dump(
                    [to_jsonable(e) for e in state.ui_elements],
                    f,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )

            prompt = make_prompt(goal, history, ui_description)
            prompt_path.write_text(prompt, encoding="utf-8")

            try:
                success_check = task.is_successful(env) == 1
            except Exception as e:
                print(f"[WARN] success check before action failed: {e}")
                success_check = False

            if success_check:
                print("[SUCCESS] Task already successful before model action.")
                result["success"] = True
                result["steps"] = step - 1
                break

            print("Calling VLM...")
            model_output = call_openai_compatible_vlm(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                prompt=prompt,
                image=labeled_img,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            response_path.write_text(model_output, encoding="utf-8")

            print("Model output:")
            print(model_output)

            action_dict = extract_json_object(model_output)
            if action_dict is None:
                note = "Failed to parse JSON action."
                print(f"[INVALID] {note}")
                history.append(
                    {
                        "step": step,
                        "action": None,
                        "raw_output": model_output,
                        "success_check": False,
                        "note": note,
                    }
                )
                continue

            action_dict = sanitize_action_dict(action_dict)

            try:
                action = json_action.JSONAction(**action_dict)
            except Exception as e:
                note = f"Invalid JSONAction: {e}"
                print(f"[INVALID] {note}")
                history.append(
                    {
                        "step": step,
                        "action": action_dict,
                        "raw_output": model_output,
                        "success_check": False,
                        "note": note,
                    }
                )
                continue

            print(f"Parsed action: {action}")

            if action.action_type == "status":
                final_success = task.is_successful(env) == 1
                print(f"Agent stopped with status={action.goal_status}; success_check={final_success}")
                result["success"] = bool(final_success)
                result["steps"] = step
                history.append(
                    {
                        "step": step,
                        "action": action.as_dict(),
                        "raw_output": model_output,
                        "success_check": bool(final_success),
                        "note": f"agent_status={action.goal_status}",
                    }
                )
                break

            try:
                env.execute_action(action)
                time.sleep(args.action_sleep)
            except Exception as e:
                note = f"Action execution failed: {e}"
                print(f"[EXEC ERROR] {note}")
                history.append(
                    {
                        "step": step,
                        "action": action.as_dict(),
                        "raw_output": model_output,
                        "success_check": False,
                        "note": note,
                    }
                )
                continue

            try:
                after_success = task.is_successful(env) == 1
            except Exception as e:
                print(f"[WARN] success check after action failed: {e}")
                after_success = False

            print(f"success_check_after_action = {after_success}")

            history.append(
                {
                    "step": step,
                    "action": action.as_dict(),
                    "raw_output": model_output,
                    "success_check": bool(after_success),
                    "note": "",
                }
            )

            if after_success:
                print("[SUCCESS] Task completed.")
                result["success"] = True
                result["steps"] = step
                break

            result["steps"] = step

        print("[5/5] Save logs...")
        history_path = run_dir / "history.json"
        result_path = run_dir / "result.json"

        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2, default=str)

        with result_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        print("\n========== RESULT ==========")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        return result

    finally:
        if task is not None and env is not None and args.teardown:
            maybe_teardown(task, env)

        if env is not None:
            print("Closing env...")
            env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot VLM baseline for AndroidWorld without default Agent or DMS memory."
    )

    parser.add_argument("--task", default="ContactsNewContactDraft")
    parser.add_argument("--max_steps", type=int, default=20)

    parser.add_argument("--console_port", type=int, default=5554)
    parser.add_argument("--grpc_port", type=int, default=8554)
    parser.add_argument("--adb_path", default="D:\\Android\\Sdk\\platform-tools\\adb.exe")
    parser.add_argument("--perform_emulator_setup", action="store_true")

    parser.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api_key", default="dms-qwen-secret")
    parser.add_argument("--model", default="qwen2.5-vl-7b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=120)

    parser.add_argument("--max_ui_elements", type=int, default=50)
    parser.add_argument("--action_sleep", type=float, default=1.0)
    parser.add_argument("--output_dir", default="zero_shot_runs")
    parser.add_argument("--teardown", action="store_true")
    parser.add_argument("--a11y_apk", default="F:\\baoyantest\\dms\\android_world\\third_party\\a11y_forwarder.apk")

    args = parser.parse_args()

    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("GRPC_TRACE", "none")

    patch_a11y_forwarder_apk(args.a11y_apk)
    run_one_task(args)


if __name__ == "__main__":
    main()