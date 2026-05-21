from __future__ import annotations

import sys
import unittest
from pathlib import Path
import shutil
from unittest.mock import patch

from scripts import task_loop_smoke


class TaskLoopSmokeConfigTest(unittest.TestCase):
    def test_parse_args_defaults_include_memory_backend_and_embedding_flags(self) -> None:
        with patch.object(sys, "argv", ["task_loop_smoke.py"]):
            args = task_loop_smoke.parse_args()

        self.assertEqual(args.memory_backend, "none")
        self.assertEqual(args.static_memory_retrieval_mode, "lexical_jaccard")
        self.assertEqual(args.dms_retrieval_mode, "lexical_jaccard")
        self.assertEqual(args.embedding_base_url, "http://127.0.0.1:19007/v1")
        self.assertEqual(args.embedding_api_key, "EMPTY")
        self.assertEqual(args.embedding_model, "bge-small-en-v1.5")
        self.assertEqual(args.embedding_timeout, 120)
        self.assertFalse(args.dms_enable_planner_risk_context)

    def test_validate_args_maps_legacy_use_static_memory_to_static_backend(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "task_loop_smoke.py",
                "--use_static_memory",
            ],
        ):
            args = task_loop_smoke.parse_args()

        task_loop_smoke.validate_args(args)
        self.assertEqual(args.memory_backend, "static")

    def test_validate_args_requires_embedding_model_for_static_embedding_modes(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "task_loop_smoke.py",
                "--memory_backend",
                "static",
                "--static_memory_retrieval_mode",
                "embedding_product",
                "--embedding_model",
                "",
            ],
        ):
            args = task_loop_smoke.parse_args()

        with self.assertRaises(ValueError):
            task_loop_smoke.validate_args(args)

    def test_validate_args_allows_static_embedding_mode_with_model(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "task_loop_smoke.py",
                "--memory_backend",
                "static",
                "--static_memory_retrieval_mode",
                "embedding_product",
            ],
        ):
            args = task_loop_smoke.parse_args()

        task_loop_smoke.validate_args(args)

    def test_validate_args_requires_embedding_model_for_dms_embedding_modes(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "task_loop_smoke.py",
                "--memory_backend",
                "dms",
                "--dms_retrieval_mode",
                "embedding_weighted_sum",
                "--embedding_model",
                "",
            ],
        ):
            args = task_loop_smoke.parse_args()

        with self.assertRaises(ValueError):
            task_loop_smoke.validate_args(args)

    def test_validate_args_allows_dms_embedding_mode_with_model(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "task_loop_smoke.py",
                "--memory_backend",
                "dms",
                "--dms_retrieval_mode",
                "embedding_product",
            ],
        ):
            args = task_loop_smoke.parse_args()

        task_loop_smoke.validate_args(args)

    def test_validate_args_allows_lexical_modes_without_embedding_requirements(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "task_loop_smoke.py",
                "--memory_backend",
                "dms",
                "--dms_retrieval_mode",
                "lexical_jaccard",
                "--embedding_model",
                "",
            ],
        ):
            args = task_loop_smoke.parse_args()

        task_loop_smoke.validate_args(args)

    def test_validate_args_rewrites_default_static_memory_path_by_backend(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "task_loop_smoke.py",
                "--memory_backend",
                "static",
            ],
        ):
            args = task_loop_smoke.parse_args()

        task_loop_smoke.validate_args(args)
        self.assertEqual(
            args.static_memory_path,
            str(Path("memory_bank") / "static" / "static" / "static_memory.jsonl"),
        )

    def test_validate_args_rewrites_default_dms_memory_root_by_backend(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "task_loop_smoke.py",
                "--memory_backend",
                "dms",
            ],
        ):
            args = task_loop_smoke.parse_args()

        task_loop_smoke.validate_args(args)
        self.assertEqual(
            args.dms_memory_root,
            str(Path("memory_bank") / "dms" / "dms"),
        )

    def test_validate_args_keeps_explicit_memory_paths(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "task_loop_smoke.py",
                "--memory_backend",
                "dms",
                "--static_memory_path",
                "custom/static.jsonl",
                "--dms_memory_root",
                "custom/dms_root",
            ],
        ):
            args = task_loop_smoke.parse_args()

        task_loop_smoke.validate_args(args)
        self.assertEqual(args.static_memory_path, "custom/static.jsonl")
        self.assertEqual(args.dms_memory_root, "custom/dms_root")

    def test_make_run_dir_nests_runs_under_backend(self) -> None:
        output_root = Path("task_loop_smoke_runs_test")
        try:
            with patch("scripts.task_loop_smoke.datetime") as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20260521_160000"
                run_dir = task_loop_smoke.make_run_dir(str(output_root), "SystemWifiTurnOff", "dms")

            self.assertEqual(
                run_dir,
                output_root / "dms" / "SystemWifiTurnOff_20260521_160000",
            )
            self.assertTrue(run_dir.exists())
        finally:
            shutil.rmtree(output_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
