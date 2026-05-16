from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
TASK_LOOP_SMOKE = ROOT_DIR / "scripts" / "task_loop_smoke.py"


@dataclass
class EvalRunSummary:
    run_index: int
    status: str
    final_task_success: bool | None
    run_dir: str
    completion_message: str


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run task_loop_smoke multiple times and judge whether a prompt change is acceptable."
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max_failures", type=int, default=1)
    return parser.parse_known_args()


def extract_run_dir(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.startswith("Run dir: "):
            return line.split("Run dir: ", 1)[1].strip()
    raise ValueError("Could not find run directory in task_loop_smoke output.")


def load_run_summary(run_dir: str) -> EvalRunSummary:
    run_result_path = Path(run_dir) / "run_result.json"
    payload = json.loads(run_result_path.read_text(encoding="utf-8"))
    return EvalRunSummary(
        run_index=0,
        status=str(payload.get("status") or "unknown"),
        final_task_success=payload.get("final_task_success"),
        run_dir=run_dir,
        completion_message=str(payload.get("completion_message") or ""),
    )


def is_success(summary: EvalRunSummary) -> bool:
    if summary.final_task_success is True:
        return True
    return summary.status == "completed"


def aggregate_eval(task: str, run_summaries: list[EvalRunSummary], max_failures: int) -> dict[str, Any]:
    success_count = sum(1 for summary in run_summaries if is_success(summary))
    failure_count = len(run_summaries) - success_count
    accepted = failure_count <= max_failures
    return {
        "task": task,
        "runs": len(run_summaries),
        "success_count": success_count,
        "failure_count": failure_count,
        "max_failures": max_failures,
        "accepted": accepted,
        "judgement": "可接受" if accepted else "无效",
        "run_summaries": [
            {
                "run_index": summary.run_index,
                "status": summary.status,
                "final_task_success": summary.final_task_success,
                "run_dir": summary.run_dir,
                "completion_message": summary.completion_message,
            }
            for summary in run_summaries
        ],
    }


def run_eval(task: str, runs: int, max_failures: int, passthrough_args: list[str]) -> dict[str, Any]:
    run_summaries: list[EvalRunSummary] = []
    for run_index in range(1, runs + 1):
        command = [sys.executable, str(TASK_LOOP_SMOKE), "--task", task, *passthrough_args]
        completed = subprocess.run(
            command,
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        stdout = completed.stdout
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n")
        stderr = completed.stderr
        if stderr:
            print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")
        run_dir = extract_run_dir(stdout)
        summary = load_run_summary(run_dir)
        summary.run_index = run_index
        run_summaries.append(summary)

    return aggregate_eval(task, run_summaries, max_failures)


def main() -> int:
    args, passthrough_args = parse_args()
    try:
        result = run_eval(
            task=args.task,
            runs=args.runs,
            max_failures=args.max_failures,
            passthrough_args=passthrough_args,
        )
    except Exception as exc:
        print(f"[prompt_eval_smoke] ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
