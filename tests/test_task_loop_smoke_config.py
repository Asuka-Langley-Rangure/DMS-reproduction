from __future__ import annotations

import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
