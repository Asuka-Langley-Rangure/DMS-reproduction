from __future__ import annotations

import argparse
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


def main() -> int:
    args = parse_args()
    command = build_command(args)

    print(f"Task: {args.task}")
    print(f"Runs: {args.runs}")
    print(f"Memory backend: {args.memory_backend}")
    print(f"Python executable: {args.python_executable}")
    print(f"Delegated command: {' '.join(command)}")

    exit_codes: list[int] = []
    for run_index in range(1, args.runs + 1):
        print(f"\n=== Batch Run {run_index}/{args.runs} ===")
        completed = subprocess.run(command, cwd=ROOT_DIR)
        exit_codes.append(int(completed.returncode))
        if completed.returncode != 0:
            print(f"Run {run_index} failed with exit code {completed.returncode}.")

    failed = [code for code in exit_codes if code != 0]
    print("\n=== Batch Summary ===")
    print(f"Total runs: {args.runs}")
    print(f"Successful process exits: {args.runs - len(failed)}")
    print(f"Failed process exits: {len(failed)}")
    print(f"Exit codes: {exit_codes}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
