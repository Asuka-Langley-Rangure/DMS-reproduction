from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from scripts import task_loop_smoke


class TaskLoopSmokeConfigTest(unittest.TestCase):
    def test_parse_args_defaults_include_embedding_flags(self) -> None:
        with patch.object(sys, "argv", ["task_loop_smoke.py"]):
            args = task_loop_smoke.parse_args()

        self.assertEqual(args.static_memory_retrieval_mode, "lexical_jaccard")
        self.assertEqual(args.embedding_base_url, "")
        self.assertEqual(args.embedding_api_key, "")
        self.assertEqual(args.embedding_model, "")
        self.assertEqual(args.embedding_timeout, 120)

    def test_validate_args_requires_embedding_model_for_embedding_modes(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "task_loop_smoke.py",
                "--use_static_memory",
                "--static_memory_retrieval_mode",
                "embedding_product",
            ],
        ):
            args = task_loop_smoke.parse_args()

        with self.assertRaises(ValueError):
            task_loop_smoke.validate_args(args)

    def test_validate_args_allows_embedding_mode_with_model_and_reused_base_url(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "task_loop_smoke.py",
                "--use_static_memory",
                "--static_memory_retrieval_mode",
                "embedding_product",
                "--embedding_model",
                "bge-small-en-v1.5",
            ],
        ):
            args = task_loop_smoke.parse_args()

        task_loop_smoke.validate_args(args)

    def test_validate_args_allows_lexical_mode_without_embedding_path(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "task_loop_smoke.py",
                "--use_static_memory",
                "--static_memory_retrieval_mode",
                "lexical_jaccard",
            ],
        ):
            args = task_loop_smoke.parse_args()

        task_loop_smoke.validate_args(args)


if __name__ == "__main__":
    unittest.main()
