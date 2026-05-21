from __future__ import annotations

import unittest
from argparse import Namespace

from scripts.run_task_loop_batch import build_command


class RunTaskLoopBatchTest(unittest.TestCase):
    def test_build_command_includes_required_fields(self) -> None:
        args = Namespace(
            task="SystemWifiTurnOff",
            runs=3,
            memory_backend="dms",
            python_executable=r"F:\.conda\envs\android_world\python.exe",
            task_loop_args=["--skip_emulator_launch", "--max_planner_rounds", "7"],
        )

        command = build_command(args)

        self.assertEqual(command[0], r"F:\.conda\envs\android_world\python.exe")
        self.assertIn("--task", command)
        self.assertIn("SystemWifiTurnOff", command)
        self.assertIn("--memory_backend", command)
        self.assertIn("dms", command)
        self.assertIn("--skip_emulator_launch", command)
        self.assertIn("--max_planner_rounds", command)
        self.assertIn("7", command)

    def test_build_command_strips_remainder_separator(self) -> None:
        args = Namespace(
            task="ContactsAddContact",
            runs=2,
            memory_backend="static",
            python_executable="python",
            task_loop_args=["--", "--skip_emulator_launch"],
        )

        command = build_command(args)

        self.assertEqual(command[-1], "--skip_emulator_launch")
        self.assertNotIn("-- --skip_emulator_launch", " ".join(command))

    def test_build_command_preserves_custom_output_dir_passthrough(self) -> None:
        args = Namespace(
            task="ContactsAddContact",
            runs=1,
            memory_backend="static",
            python_executable="python",
            task_loop_args=["--output_dir", "custom_runs"],
        )

        command = build_command(args)

        self.assertIn("--output_dir", command)
        self.assertIn("custom_runs", command)
        self.assertIn("--memory_backend", command)
        self.assertIn("static", command)


if __name__ == "__main__":
    unittest.main()
