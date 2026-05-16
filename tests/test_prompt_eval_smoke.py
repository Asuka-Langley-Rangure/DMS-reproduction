from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.prompt_eval_smoke import aggregate_eval, extract_run_dir, is_success, load_run_summary, EvalRunSummary


class PromptEvalSmokeTest(unittest.TestCase):
    def test_extract_run_dir(self) -> None:
        stdout = "Task: ContactsAddContact\nRun dir: task_loop_smoke_runs\\foo\n"
        self.assertEqual(extract_run_dir(stdout), "task_loop_smoke_runs\\foo")

    def test_load_run_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "run_result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "final_task_success": True,
                        "completion_message": "done",
                    }
                ),
                encoding="utf-8",
            )
            summary = load_run_summary(str(run_dir))
            self.assertEqual(summary.status, "completed")
            self.assertTrue(summary.final_task_success)
            self.assertEqual(summary.completion_message, "done")

    def test_is_success(self) -> None:
        self.assertTrue(is_success(EvalRunSummary(1, "completed", None, "a", "")))
        self.assertTrue(is_success(EvalRunSummary(1, "round_limit", True, "a", "")))
        self.assertFalse(is_success(EvalRunSummary(1, "round_limit", None, "a", "")))

    def test_aggregate_eval_rejects_when_failures_reach_two(self) -> None:
        result = aggregate_eval(
            "ContactsAddContact",
            [
                EvalRunSummary(1, "completed", True, "run1", "ok"),
                EvalRunSummary(2, "round_limit", None, "run2", "fail"),
                EvalRunSummary(3, "round_limit", None, "run3", "fail"),
            ],
            max_failures=1,
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["failure_count"], 2)
        self.assertEqual(result["judgement"], "无效")

    def test_aggregate_eval_accepts_when_successes_reach_two(self) -> None:
        result = aggregate_eval(
            "ContactsAddContact",
            [
                EvalRunSummary(1, "completed", True, "run1", "ok"),
                EvalRunSummary(2, "completed", True, "run2", "ok"),
                EvalRunSummary(3, "round_limit", None, "run3", "fail"),
            ],
            max_failures=1,
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["success_count"], 2)
        self.assertEqual(result["judgement"], "可接受")


if __name__ == "__main__":
    unittest.main()
