from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
TASK_LOOP_SMOKE = ROOT_DIR / "scripts" / "task_loop_smoke.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run task_loop_smoke.py repeatedly with a chosen task and memory backend."
    )
    parser.add_argument("--task", required=True, help="AndroidWorld task name, e.g. SystemWifiTurnOff")
    parser.add_argument("--runs", type=int, default=1, help="How many times to run the task loop smoke script.")
    parser.add_argument(
        "--memory_backend",
        default="none",
        choices=["none", "static", "dms"],
        help="Memory backend to use for every run.",
    )
    parser.add_argument(
        "--python_executable",
        default=sys.executable,
        help="Python interpreter used to launch task_loop_smoke.py.",
    )
    parser.add_argument(
        "task_loop_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed through to task_loop_smoke.py. Prefix with -- before extra args if needed.",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    return args


def _normalize_passthrough_args(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def build_command(args: argparse.Namespace) -> list[str]:
    passthrough = _normalize_passthrough_args(list(args.task_loop_args))
    return [
        args.python_executable,
        str(TASK_LOOP_SMOKE),
        "--task",
        args.task,
        "--memory_backend",
        args.memory_backend,
        *passthrough,
    ]


def extract_run_dir(stdout: str) -> str | None:
    for line in stdout.splitlines():
        if line.startswith("Run dir: "):
            return line.split("Run dir: ", 1)[1].strip()
    return None


def load_run_result(run_dir: str) -> dict | None:
    path = Path(run_dir) / "run_result.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_batch_outputs(batch_dir: Path, records: list[dict]) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fieldnames = [
        "task",
        "backend",
        "trial_index",
        "status",
        "final_task_success",
        "planner_round_count",
        "total_actor_steps",
        "planner_tokens_total",
        "actor_tokens_total",
        "verifier_tokens_total",
        "tokens_total",
        "completion_message",
        "run_dir",
    ]
    with (batch_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})
    success_count = sum(1 for record in records if record.get("final_task_success") is True)
    total_runs = len(records)
    average_tokens = (
        sum(int(record.get("tokens_total") or 0) for record in records) / total_runs
        if total_runs
        else 0.0
    )
    average_steps = (
        sum(int(record.get("total_actor_steps") or 0) for record in records) / total_runs
        if total_runs
        else 0.0
    )
    summary_lines = [
        "# Batch Summary",
        "",
        f"- Task: {records[0].get('task') if records else 'Unknown'}",
        f"- Backend: {records[0].get('backend') if records else 'Unknown'}",
        f"- Total runs: {total_runs}",
        f"- Success count: {success_count}",
        f"- Success rate: {success_count / total_runs:.3f}" if total_runs else "- Success rate: 0.000",
        f"- Average total tokens: {average_tokens:.2f}",
        f"- Average total actor steps: {average_steps:.2f}",
        "",
        "## Trial Results",
    ]
    for record in records:
        summary_lines.append(
            f"- Trial {record.get('trial_index')}: success={record.get('final_task_success')} "
            f"status={record.get('status')} tokens={record.get('tokens_total')} "
            f"steps={record.get('total_actor_steps')} run_dir={record.get('run_dir')}"
        )
    (batch_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    command = build_command(args)

    print(f"Task: {args.task}")
    print(f"Runs: {args.runs}")
    print(f"Memory backend: {args.memory_backend}")
    print(f"Python executable: {args.python_executable}")
    print(f"Delegated command: {' '.join(command)}")

    exit_codes: list[int] = []
    records: list[dict] = []
    batch_dir = ROOT_DIR / "batch_results" / f"{args.task}_{args.memory_backend}"
    for run_index in range(1, args.runs + 1):
        print(f"\n=== Batch Run {run_index}/{args.runs} ===")
        completed = subprocess.run(command, cwd=ROOT_DIR, capture_output=True, text=True)
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        exit_codes.append(int(completed.returncode))
        run_dir = extract_run_dir(completed.stdout)
        run_result = load_run_result(run_dir) if run_dir else None
        records.append(
            {
                "task": args.task,
                "backend": args.memory_backend,
                "trial_index": run_index,
                "status": (run_result or {}).get("status"),
                "final_task_success": (run_result or {}).get("final_task_success"),
                "planner_round_count": (run_result or {}).get("planner_round_count"),
                "total_actor_steps": (run_result or {}).get("total_actor_steps"),
                "planner_tokens_total": (run_result or {}).get("planner_tokens_total"),
                "actor_tokens_total": (run_result or {}).get("actor_tokens_total"),
                "verifier_tokens_total": (run_result or {}).get("verifier_tokens_total"),
                "tokens_total": (run_result or {}).get("tokens_total"),
                "completion_message": (run_result or {}).get("completion_message"),
                "run_dir": run_dir,
            }
        )
        if completed.returncode != 0:
            print(f"Run {run_index} failed with exit code {completed.returncode}.")
    write_batch_outputs(batch_dir, records)

    failed = [code for code in exit_codes if code != 0]
    print("\n=== Batch Summary ===")
    print(f"Total runs: {args.runs}")
    print(f"Successful process exits: {args.runs - len(failed)}")
    print(f"Failed process exits: {len(failed)}")
    print(f"Exit codes: {exit_codes}")
    print(f"Batch artifacts: {batch_dir}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
