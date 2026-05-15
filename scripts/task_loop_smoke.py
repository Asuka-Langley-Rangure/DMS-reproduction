from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
VENDOR_DIR = ROOT_DIR / ".vendor"
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from dms_reproduction.agents.android_actor import ActorConfig, AndroidActor
from dms_reproduction.agents.planner import AndroidTaskPlanner
from dms_reproduction.agents.task_runner import AndroidTaskRunner, TaskRunConfig
from dms_reproduction.envs.android_world_adapter import AndroidWorldObservationAdapter
from dms_reproduction.envs.observation_utils import save_base64_png
from dms_reproduction.llm.base_client import OpenAICompatibleConfig
from dms_reproduction.llm.openai_compatible import OpenAICompatibleClient
from scripts.planner_smoke import (
    DEFAULT_A11Y_APK,
    ensure_emulator_running,
    ensure_ssh_tunnel,
    load_env,
    load_task,
    maybe_teardown,
    patch_a11y_forwarder_apk,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a planner + actor closed-loop smoke test on AndroidWorld."
    )
    parser.add_argument("--task", default="ContactsAddContact")
    parser.add_argument("--output_dir", default="task_loop_smoke_runs")
    parser.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api_key", default="dms-qwen-secret")
    parser.add_argument("--model", default="qwen2.5-vl-7b")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--ssh_host", default="114.212.165.149")
    parser.add_argument("--ssh_user", default="chencen")
    parser.add_argument("--ssh_password", default="")
    parser.add_argument("--local_port", type=int, default=8000)
    parser.add_argument("--remote_port", type=int, default=8007)
    parser.add_argument("--skip_ssh_tunnel", action="store_true")
    parser.add_argument("--healthcheck_timeout", type=int, default=5)
    parser.add_argument("--ssh_ready_timeout", type=int, default=20)

    parser.add_argument("--console_port", type=int, default=5554)
    parser.add_argument("--grpc_port", type=int, default=8554)
    parser.add_argument("--adb_path", default=r"D:\Android\Sdk\platform-tools\adb.exe")
    parser.add_argument("--perform_emulator_setup", action="store_true")
    parser.add_argument(
        "--emulator_start_script",
        default=str(ROOT_DIR / "start_androidworld_emulator.bat"),
    )
    parser.add_argument("--skip_emulator_launch", action="store_true")
    parser.add_argument("--emulator_ready_timeout", type=int, default=120)
    parser.add_argument("--a11y_apk", default=str(DEFAULT_A11Y_APK))
    parser.add_argument("--max_ui_elements", type=int, default=50)

    parser.add_argument("--max_planner_rounds", type=int, default=5)
    parser.add_argument("--max_actor_steps", type=int, default=8)
    parser.add_argument("--max_total_actor_steps", type=int, default=40)
    return parser.parse_args()


def make_run_dir(output_dir: str, task: str) -> Path:
    run_name = f"{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def maybe_save_observation_images(target_dir: Path, observation: dict[str, Any], prefix: str = "observation") -> dict[str, str]:
    artifact_paths: dict[str, str] = {}
    raw_b64 = observation.get("screenshot_b64")
    labeled_b64 = observation.get("labeled_screenshot_b64")
    if raw_b64:
        raw_path = target_dir / f"{prefix}_raw.png"
        save_base64_png(raw_b64, raw_path)
        artifact_paths["raw_image"] = str(raw_path)
    if labeled_b64:
        labeled_path = target_dir / f"{prefix}_labeled.png"
        save_base64_png(labeled_b64, labeled_path)
        artifact_paths["labeled_image"] = str(labeled_path)
    return artifact_paths


def maybe_save_actor_seen_image(target_dir: Path, step: dict[str, Any], step_id: int) -> str | None:
    for message in step.get("messages", []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") != "image_url":
                continue
            url = str(item.get("image_url", {}).get("url", ""))
            prefix = "data:image/png;base64,"
            if not url.startswith(prefix):
                continue
            image_path = target_dir / f"actor_seen_step_{step_id:02d}.png"
            save_base64_png(url[len(prefix) :], image_path)
            return str(image_path)
    return None


def observation_summary_lines(observation: dict[str, Any] | None) -> list[str]:
    if observation is None:
        return ["- Observation: None"]
    return [
        f"- Foreground package: {observation.get('foreground_package') or 'Unknown'}",
        f"- Dominant UI package: {observation.get('app_name') or 'Unknown'}",
        f"- Activity: {observation.get('current_activity') or 'Unknown'}",
        f"- Observation consistency: {observation.get('observation_consistency') or 'Unknown'}",
        f"- Visible UI count: {observation.get('visible_ui_count', 0)}",
        f"- Clickable UI count: {observation.get('clickable_ui_count', 0)}",
        f"- Non-system UI count: {observation.get('non_system_ui_count', 0)}",
        f"- Warning: {observation.get('observation_warning') or 'None'}",
    ]


def build_subtask_summary(subtask_run: dict[str, Any]) -> str:
    actor_result = subtask_run["actor_result"]
    success_check = subtask_run.get("subtask_success_check") or {}
    lines = [
        "# Subtask Summary",
        "",
        f"- Subtask: {subtask_run['subtask'].get('precondition', '')} -> {subtask_run['subtask'].get('goal', '')}",
        f"- Actor status: {actor_result.get('status')}",
        f"- Completion message: {actor_result.get('completion_message') or 'None'}",
        f"- Runner success override: {success_check.get('runner_overrode_to_completed', False)}",
        f"- Success rule: {success_check.get('success_rule') or 'None'}",
        f"- Final warning: {(subtask_run.get('post_observation') or {}).get('observation_warning') or 'None'}",
        f"- Text entry success detected: {str((success_check.get('success_rule') or '')).startswith('text_entry_')}",
        "",
        "| Step | Reason | Action | Result | Observation warning |",
        "| --- | --- | --- | --- | --- |",
    ]
    for step in actor_result.get("steps", []):
        action = step.get("action") or {}
        warning = ((step.get("after_observation") or step.get("before_observation") or {}).get("observation_warning") or "None")
        result = step.get("done_reason") or ("execution_error" if step.get("execution_error") else "progress")
        lines.append(
            f"| {int(step.get('step_id', 0)) + 1} | "
            f"{str(step.get('reason') or 'None').replace('|', '/')} | "
            f"{json.dumps(action, ensure_ascii=False)} | "
            f"{result} | "
            f"{str(warning).replace('|', '/')} |"
        )
    return "\n".join(lines) + "\n"


def build_ui_index_table(observation: dict[str, Any] | None) -> str:
    lines = [
        "# UI Index Table",
        "",
        "| Index | Label | Clickable | Editable | Package | Class |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for element in (observation or {}).get("ui_elements", []):
        label = element.get("text") or element.get("content_description") or element.get("resource_name") or "None"
        lines.append(
            f"| {element.get('index')} | {str(label).replace('|', '/')} | "
            f"{bool(element.get('is_clickable'))} | {bool(element.get('is_editable'))} | "
            f"{element.get('package_name') or 'Unknown'} | {element.get('class_name') or 'Unknown'} |"
        )
    return "\n".join(lines) + "\n"


def build_actor_decision_summary(subtask_text: str, step: dict[str, Any]) -> str:
    lines = [
        "# Actor Decision",
        "",
        f"- Subtask: {subtask_text}",
        f"- Observation consistency: {(step.get('before_observation') or {}).get('observation_consistency') or 'Unknown'}",
        f"- Raw response: {step.get('raw_response') or 'None'}",
        f"- Parsed action: {json.dumps(step.get('original_action') or {}, ensure_ascii=False)}",
        f"- Normalized action: {json.dumps(step.get('normalized_action') or {}, ensure_ascii=False) if step.get('normalized_action') else 'None'}",
        f"- Action normalization applied: {step.get('action_normalization_applied', False)}",
        f"- Normalization reason: {step.get('normalization_reason') or 'None'}",
        f"- Corrected action: {json.dumps(step.get('corrected_action') or {}, ensure_ascii=False) if step.get('corrected_action') else 'None'}",
        f"- Final executed action: {json.dumps(step.get('action') or {}, ensure_ascii=False)}",
        f"- Correction reason: {step.get('correction_reason') or 'None'}",
        f"- Failure reason: {step.get('parse_error') or step.get('execution_error') or step.get('done_reason') or 'None'}",
        "",
        "## Reason",
        step.get("reason") or "None",
        "",
    ]
    return "\n".join(lines)


def build_round_summary(round_record: dict[str, Any]) -> str:
    planner_result = round_record["planner_result"]
    lines = [
        "# Round Summary",
        "",
        "## Input Observation",
        *observation_summary_lines(round_record.get("input_observation")),
        "",
        "## Planner Output",
    ]
    normalized_subtasks = round_record.get("normalized_subtasks") or []
    if planner_result.get("is_goal_complete"):
        lines.append(f"- Planner declared goal complete: {planner_result.get('completion_message') or 'None'}")
    else:
        subtasks = planner_result.get("subtasks") or []
        if not subtasks:
            lines.append("- Planner returned no subtasks.")
        for index, subtask in enumerate(subtasks, start=1):
            lines.append(
                f"- Subtask {index}: Precondition={subtask.get('precondition')} | "
                f"Goal={subtask.get('goal')} | Reason={subtask.get('reason')}"
            )
        if normalized_subtasks:
            lines.extend(["", "## Normalized Subtasks"])
            for index, subtask in enumerate(normalized_subtasks, start=1):
                lines.append(
                    f"- Normalized {index}: Precondition={subtask.get('precondition')} | "
                    f"Goal={subtask.get('goal')} | Reason={subtask.get('reason')}"
                )
    lines.extend(
        [
            "",
            "## Round End",
            f"- Replan reason: {round_record.get('replan_reason') or 'None'}",
            f"- Subtask run count: {len(round_record.get('subtask_runs', []))}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_run_summary(task: str, goal: str, run_result: dict[str, Any]) -> str:
    failure_counts = {
        "invalid_index": 0,
        "unstable_observation": 0,
        "planner_subtask_invalid": 0,
        "actor_overshoot_after_goal": 0,
        "planner_near_json_repaired": 0,
        "actor_action_alias_normalized": 0,
        "recoverable_schema_error": 0,
        "text_entry_success_detected": 0,
    }
    for round_record in run_result.get("planner_rounds", []):
        reason = str(round_record.get("replan_reason") or "")
        if "invalid_index" in reason:
            failure_counts["invalid_index"] += 1
        if "unstable" in reason or "observation_degraded" in reason:
            failure_counts["unstable_observation"] += 1
        if reason == "planner_subtask_invalid":
            failure_counts["planner_subtask_invalid"] += 1
        if (round_record.get("planner_parse_repair") or {}).get("repaired_parse"):
            failure_counts["planner_near_json_repaired"] += 1
        for subtask_run in round_record.get("subtask_runs", []):
            if (subtask_run.get("subtask_success_check") or {}).get("runner_overrode_to_completed"):
                failure_counts["actor_overshoot_after_goal"] += 1
            if str((subtask_run.get("subtask_success_check") or {}).get("success_rule") or "").startswith("text_entry_"):
                failure_counts["text_entry_success_detected"] += 1
            for step in (subtask_run.get("actor_result") or {}).get("steps", []):
                if step.get("action_normalization_applied"):
                    failure_counts["actor_action_alias_normalized"] += 1
                if step.get("parse_error") and (
                    "failed to parse actor action json" in str(step.get("parse_error") or "").lower()
                    or "unsupported actor action type" in str(step.get("parse_error") or "").lower()
                ):
                    failure_counts["recoverable_schema_error"] += 1
    lines = [
        "# Run Summary",
        "",
        f"- Task: {task}",
        f"- Goal: {goal}",
        f"- Status: {run_result.get('status')}",
        f"- Completion message: {run_result.get('completion_message') or 'None'}",
        f"- Planner rounds: {len(run_result.get('planner_rounds', []))}",
        f"- Total actor steps: {run_result.get('total_actor_steps', 0)}",
        "",
        "## Round Highlights",
    ]
    for round_record in run_result.get("planner_rounds", []):
        lines.append(
            f"- Round {round_record['round_id']}: "
            f"replan_reason={round_record.get('replan_reason') or 'None'}; "
            f"subtasks={len(round_record.get('subtask_runs', []))}"
        )
    lines.extend(
        [
            "",
            "## Failure Pattern",
            f"- invalid_index: {failure_counts['invalid_index']}",
            f"- unstable_observation: {failure_counts['unstable_observation']}",
            f"- planner_subtask_invalid: {failure_counts['planner_subtask_invalid']}",
            f"- actor_overshoot_after_goal: {failure_counts['actor_overshoot_after_goal']}",
            f"- planner_near_json_repaired: {failure_counts['planner_near_json_repaired']}",
            f"- actor_action_alias_normalized: {failure_counts['actor_action_alias_normalized']}",
            f"- recoverable_schema_error: {failure_counts['recoverable_schema_error']}",
            f"- text_entry_success_detected: {failure_counts['text_entry_success_detected']}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_light_run_result(run_result: dict[str, Any], artifact_index: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": run_result.get("status"),
        "final_task_success": run_result.get("final_task_success"),
        "total_actor_steps": run_result.get("total_actor_steps"),
        "completion_message": run_result.get("completion_message"),
        "planner_round_count": len(run_result.get("planner_rounds", [])),
        "artifact_index": artifact_index,
    }


def build_meta(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "task": args.task,
        "model": args.model,
        "base_url": args.base_url,
        "console_port": args.console_port,
        "grpc_port": args.grpc_port,
        "output_dir": str(run_dir),
        "timestamp": datetime.now().isoformat(),
        "planner_call_seconds_total": None,
        "max_planner_rounds": args.max_planner_rounds,
        "max_actor_steps": args.max_actor_steps,
        "max_total_actor_steps": args.max_total_actor_steps,
        "error": None,
    }


def write_round_artifacts(run_dir: Path, round_record: dict[str, Any]) -> dict[str, Any]:
    round_dir = run_dir / f"round_{round_record['round_id']:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    save_json(round_dir / "observation_before.json", round_record["input_observation"])
    save_json(round_dir / "planner_messages.json", round_record["planner_messages"])
    save_text(round_dir / "planner_prompt.txt", round_record["planner_prompt"])
    save_text(round_dir / "planner_raw_response.txt", round_record["planner_raw_response"])
    save_json(round_dir / "planner_result.json", round_record["planner_result"])
    save_json(
        round_dir / "planner_parse_repair.json",
        round_record.get("planner_parse_repair") or {},
    )
    save_json(
        round_dir / "planner_subtasks_normalized.json",
        {
            "raw_subtasks": round_record["planner_result"].get("subtasks", []),
            "normalized_subtasks": round_record.get("normalized_subtasks", []),
        },
    )
    save_json(
        round_dir / "observation_consistency_report.json",
        {
            "consistency": round_record["input_observation"].get("observation_consistency"),
            "warning": round_record["input_observation"].get("observation_warning"),
            "unstable_reasons": round_record["input_observation"].get("unstable_reasons", []),
            "attempt": (round_record["input_observation"].get("extra_state") or {}).get("observation_attempt"),
            "resampled": (round_record["input_observation"].get("extra_state") or {}).get("observation_resampled"),
            "final_after_resample": (round_record["input_observation"].get("extra_state") or {}).get("final_after_resample"),
        },
    )
    save_json(
        round_dir / "planner_grounding_check.json",
        round_record.get("planner_grounding_check", []),
    )
    save_json(
        round_dir / "observation_transition_report.json",
        round_record.get("observation_transition_report", []),
    )
    observation_images = maybe_save_observation_images(round_dir, round_record["input_observation"])
    save_text(round_dir / "round_summary.md", build_round_summary(round_record))
    round_index: dict[str, Any] = {
        "round_dir": str(round_dir),
        "planner": {
            "observation_before": str(round_dir / "observation_before.json"),
            "planner_messages": str(round_dir / "planner_messages.json"),
            "planner_prompt": str(round_dir / "planner_prompt.txt"),
            "planner_raw_response": str(round_dir / "planner_raw_response.txt"),
            "planner_result": str(round_dir / "planner_result.json"),
            "planner_parse_repair": str(round_dir / "planner_parse_repair.json"),
            "planner_subtasks_normalized": str(round_dir / "planner_subtasks_normalized.json"),
            "observation_consistency_report": str(round_dir / "observation_consistency_report.json"),
            "planner_grounding_check": str(round_dir / "planner_grounding_check.json"),
            "observation_transition_report": str(round_dir / "observation_transition_report.json"),
            "round_summary": str(round_dir / "round_summary.md"),
            **observation_images,
        },
        "subtasks": [],
    }

    for subtask_index, subtask_run in enumerate(round_record["subtask_runs"], start=1):
        subtask_dir = round_dir / f"subtask_{subtask_index:02d}"
        subtask_dir.mkdir(parents=True, exist_ok=True)
        actor_result = subtask_run["actor_result"]
        save_json(subtask_dir / "actor_result.json", actor_result)
        save_json(subtask_dir / "subtask_success_check.json", subtask_run.get("subtask_success_check") or {})
        save_json(
            subtask_dir / "actor_action_normalization.json",
            [
                {
                    "step_id": step.get("step_id"),
                    "original_action": step.get("original_action"),
                    "normalized_action": step.get("normalized_action"),
                    "action_normalization_applied": step.get("action_normalization_applied", False),
                    "normalization_reason": step.get("normalization_reason"),
                }
                for step in actor_result.get("steps", [])
                if step.get("action_normalization_applied") or step.get("normalized_action")
            ],
        )
        if subtask_run["post_observation"] is not None:
            save_json(subtask_dir / "final_observation.json", subtask_run["post_observation"])
            maybe_save_observation_images(subtask_dir, subtask_run["post_observation"], prefix="final_observation")
        save_text(subtask_dir / "subtask_summary.md", build_subtask_summary(subtask_run))
        subtask_index_entry: dict[str, Any] = {
            "subtask_dir": str(subtask_dir),
            "actor_result": str(subtask_dir / "actor_result.json"),
            "subtask_success_check": str(subtask_dir / "subtask_success_check.json"),
            "actor_action_normalization": str(subtask_dir / "actor_action_normalization.json"),
            "subtask_summary": str(subtask_dir / "subtask_summary.md"),
            "steps": [],
        }
        if subtask_run["post_observation"] is not None:
            subtask_index_entry["final_observation"] = str(subtask_dir / "final_observation.json")
        for step in actor_result.get("steps", []):
            step_id = int(step["step_id"]) + 1
            save_json(subtask_dir / f"actor_messages_step_{step_id:02d}.json", step["messages"])
            save_text(subtask_dir / f"actor_prompt_step_{step_id:02d}.txt", step["prompt_text"])
            save_text(subtask_dir / f"actor_raw_response_step_{step_id:02d}.txt", step["raw_response"])
            save_text(
                subtask_dir / f"actor_step_{step_id:02d}_decision.md",
                build_actor_decision_summary(
                    f"Precondition: {subtask_run['subtask'].get('precondition', '')} Goal: {subtask_run['subtask'].get('goal', '')}",
                    step,
                ),
            )
            save_text(
                subtask_dir / f"ui_index_table_step_{step_id:02d}.md",
                build_ui_index_table(step.get("before_observation")),
            )
            step_entry = {
                "step_id": step_id,
                "messages": str(subtask_dir / f"actor_messages_step_{step_id:02d}.json"),
                "prompt": str(subtask_dir / f"actor_prompt_step_{step_id:02d}.txt"),
                "raw_response": str(subtask_dir / f"actor_raw_response_step_{step_id:02d}.txt"),
                "decision_summary": str(subtask_dir / f"actor_step_{step_id:02d}_decision.md"),
                "ui_index_table": str(subtask_dir / f"ui_index_table_step_{step_id:02d}.md"),
            }
            seen_image = maybe_save_actor_seen_image(subtask_dir, step, step_id)
            if seen_image:
                step_entry["actor_seen_image"] = seen_image
            subtask_index_entry["steps"].append(step_entry)
        round_index["subtasks"].append(subtask_index_entry)
    return round_index

def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = make_run_dir(args.output_dir, args.task)
    meta = build_meta(run_dir, args)
    tunnel_process = None
    env = None
    task = None

    try:
        tunnel_process = ensure_ssh_tunnel(args, meta)
        ensure_emulator_running(args, meta)
        patch_a11y_forwarder_apk(args.a11y_apk)
        env = load_env(args)
        env.reset(go_home=True)

        task, goal = load_task(args.task)
        save_text(run_dir / "task_goal.txt", goal)

        llm_client = OpenAICompatibleClient(
            OpenAICompatibleConfig(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
        )
        planner = AndroidTaskPlanner(llm_client)
        actor = AndroidActor(
            llm_client,
            ActorConfig(
                max_steps=args.max_actor_steps,
                temperature=args.temperature,
            ),
        )
        adapter = AndroidWorldObservationAdapter(max_ui_elements=args.max_ui_elements)
        runner = AndroidTaskRunner(
            planner=planner,
            actor=actor,
            observation_adapter=adapter,
            config=TaskRunConfig(
                max_planner_rounds=args.max_planner_rounds,
                max_total_actor_steps=args.max_total_actor_steps,
            ),
        )

        start_time = time.time()
        run_result = runner.run_task(env, task, goal)
        meta["planner_call_seconds_total"] = round(time.time() - start_time, 3)

        run_result_dict = run_result.to_dict()
        artifact_index = {"rounds": []}
        for round_record in run_result_dict["planner_rounds"]:
            artifact_index["rounds"].append(write_round_artifacts(run_dir, round_record))
        save_json(run_dir / "artifact_index.json", artifact_index)
        save_json(run_dir / "run_result.json", build_light_run_result(run_result_dict, artifact_index))
        save_text(run_dir / "run_summary.md", build_run_summary(args.task, goal, run_result_dict))
        save_json(run_dir / "meta.json", meta)

        print(f"Task: {args.task}")
        print(f"Goal: {goal}")
        print(f"Status: {run_result.status}")
        print(f"Planner rounds: {len(run_result.planner_rounds)}")
        print(f"Total actor steps: {run_result.total_actor_steps}")
        print(f"Run dir: {run_dir}")
        if run_result.final_task_success is not None:
            print(f"Final task success: {run_result.final_task_success}")

        return {"run_dir": str(run_dir), "run_result": run_result_dict, "meta": meta}
    except Exception as exc:
        meta["error"] = str(exc)
        save_json(run_dir / "meta.json", meta)
        raise
    finally:
        if task is not None and env is not None:
            try:
                maybe_teardown(task, env)
            except Exception:
                pass
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if tunnel_process is not None and meta.get("tunnel_started_by_script"):
            try:
                tunnel_process.terminate()
            except Exception:
                pass


def main() -> int:
    args = parse_args()
    try:
        run_smoke(args)
    except Exception as exc:
        print(f"[task_loop_smoke] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
