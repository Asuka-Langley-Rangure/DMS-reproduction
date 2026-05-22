from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None

if PIL_AVAILABLE:
    from scripts.task_loop_smoke import (
        build_light_run_result,
        build_round_summary,
        build_run_summary,
        write_initial_stage_plan_artifacts,
        write_round_artifacts,
    )


PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0uoAAAAASUVORK5CYII="
)


def build_observation(tag: str) -> dict:
    return {
        "goal": "Create a contact",
        "current_activity": f"{tag}/activity",
        "foreground_package": "com.android.contacts",
        "app_name": "com.android.contacts",
        "screen_size": {"width": 1080, "height": 2400},
        "ui_elements": [{"index": 3, "text": "Create contact", "is_clickable": True}],
        "ui_description": "UI element 3: text='Create contact'",
        "valid_ui_indices": [3],
        "visible_ui_count": 1,
        "clickable_ui_count": 1,
        "non_system_ui_count": 1,
        "observation_warning": None,
        "screenshot_b64": PNG_B64,
        "labeled_screenshot_b64": PNG_B64,
        "extra_state": {
            "tag": tag,
            "observation_attempt": 1,
            "observation_resampled": False,
            "final_after_resample": False,
        },
    }


@unittest.skipUnless(PIL_AVAILABLE, "Pillow is not installed in this environment.")
class TaskLoopSmokeArtifactsTest(unittest.TestCase):
    def test_write_round_artifacts_outputs_images_and_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            initial_stage_plan = {
                "planner_messages": [{"role": "user", "content": "stage plan"}],
                "planner_prompt": "stage-plan prompt",
                "planner_raw_response": '{"stage_plan":[{"stage_id":1,"title":"Open app","success_signal":"App is open"},{"stage_id":2,"title":"Reach target area","success_signal":"Target area is visible"},{"stage_id":3,"title":"Delete target item","success_signal":"Item is deleted"}]}',
                "stage_plan_result": {
                    "stage_plan": [
                        {"stage_id": 1, "title": "Open app", "success_signal": "App is open"},
                        {"stage_id": 2, "title": "Reach target area", "success_signal": "Target area is visible"},
                        {"stage_id": 3, "title": "Delete target item", "success_signal": "Item is deleted"},
                    ]
                },
                "revision_reason": None,
            }
            round_record = {
                "round_id": 1,
                "input_observation": build_observation("initial"),
                "planner_messages": [{"role": "user", "content": "planner"}],
                "planner_prompt": "planner prompt",
                "planner_raw_response": '{"tool":"set_tasks"}',
                "planner_result": {
                    "is_goal_complete": False,
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    "completion_message": "",
                    "stage_plan": [
                        {
                            "stage_id": 1,
                            "title": "Open the Phone app",
                            "success_signal": "Phone app is foreground",
                        },
                        {
                            "stage_id": 2,
                            "title": "Reach the contact creation entry point",
                            "success_signal": "Create contact UI is visible",
                        },
                    ],
                    "current_stage_id": 2,
                    "subtasks": [
                        {
                            "precondition": "None",
                            "goal": "Open Contacts",
                            "reason": "Need app",
                            "agent": "android_actor",
                        }
                    ],
                },
                "subtask_runs": [
                    {
                        "subtask": {
                            "precondition": "None",
                            "goal": "Fill in Sara Khan and +15102899176 in the contact form.",
                            "reason": "Need app",
                            "agent": "android_actor",
                        },
                        "actor_result": {
                            "status": "completed",
                            "completion_message": "done",
                            "prompt_tokens_total": 8,
                            "completion_tokens_total": 4,
                            "total_tokens_total": 12,
                            "steps": [
                                {
                                    "step_id": 0,
                                    "reason": "tap create",
                                    "action": {"action_type": "click", "index": 3},
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": [
                                                {"type": "text", "text": "actor prompt"},
                                                {
                                                    "type": "image_url",
                                                    "image_url": {"url": f"data:image/png;base64,{PNG_B64}"},
                                                },
                                            ],
                                        }
                                    ],
                                    "prompt_text": "actor prompt",
                                    "raw_response": '{"action_type":"click","index":3}',
                                    "before_observation": build_observation("before"),
                                    "after_observation": build_observation("after"),
                                    "summary": "summary",
                                    "done": True,
                                    "done_reason": "completed",
                                    "parse_error": None,
                                    "execution_error": None,
                                    "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
                                }
                            ],
                        },
                        "subtask_success_check": {
                            "success_rule": "contact_form_fields_match_expected_values",
                            "runner_overrode_to_completed": True,
                            "progress_made": True,
                            "form_fill_progress": {
                                "expected_fields": {"first_name": "Sara", "last_name": "Khan", "phone": "+15102899176"},
                                "actual_values": {"first_name": "Sara", "last_name": "Khan", "phone": "+15102899176"},
                                "completed_fields": ["first_name", "last_name", "phone"],
                                "remaining_fields": [],
                            },
                        },
                        "subtask_verification": {
                            "status": "success",
                            "reason": "The evidence observation shows the contact form fields already match the goal.",
                            "memory_eligible": True,
                            "raw_response": '{"status":"success","reason":"ok","memory_eligible":true}',
                            "parse_error": None,
                            "prompt_text": "verifier prompt",
                            "messages": [{"role": "user", "content": "verifier prompt"}],
                            "usage": {"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8},
                        },
                        "post_observation": {
                            **build_observation("after"),
                            "extra_state": {
                                "tag": "after",
                                "observation_attempt": 2,
                                "observation_resampled": True,
                                "final_after_resample": True,
                            },
                        },
                    }
                ],
                "replan_reason": "subtasks_exhausted",
                "memory_metrics": {
                    "backend": "none",
                    "start_size_bytes": 0,
                    "end_size_bytes": 0,
                    "delta_size_bytes": 0,
                    "end_entry_count": 0,
                },
            }

            initial_index = write_initial_stage_plan_artifacts(run_dir, initial_stage_plan)
            artifact_index = write_round_artifacts(run_dir, round_record)
            round_dir = run_dir / "round_01"
            subtask_dir = round_dir / "subtask_01"

            self.assertTrue((run_dir / "initial_stage_plan.json").exists())
            self.assertTrue((run_dir / "initial_stage_plan_messages.json").exists())
            self.assertTrue((run_dir / "initial_stage_plan_prompt.txt").exists())
            self.assertTrue((run_dir / "initial_stage_plan_raw_response.txt").exists())
            self.assertTrue((round_dir / "observation_raw.png").exists())
            self.assertTrue((round_dir / "observation_labeled.png").exists())
            self.assertTrue((subtask_dir / "actor_seen_step_01.png").exists())
            self.assertTrue((round_dir / "round_summary.md").exists())
            self.assertTrue((round_dir / "planner_raw_quality.json").exists())
            self.assertTrue((round_dir / "planner_stage_plan.json").exists())
            self.assertTrue((round_dir / "planner_current_stage.json").exists())
            self.assertTrue((subtask_dir / "subtask_summary.md").exists())
            self.assertTrue((subtask_dir / "form_fill_progress.json").exists())
            self.assertTrue((subtask_dir / "verifier_result.json").exists())
            self.assertTrue((subtask_dir / "verifier_messages.json").exists())
            self.assertTrue((subtask_dir / "verifier_prompt.txt").exists())
            self.assertTrue((subtask_dir / "verifier_raw_response.txt").exists())
            subtask_summary = (subtask_dir / "subtask_summary.md").read_text(encoding="utf-8")
            self.assertIn("- Observation resampled: True", subtask_summary)
            self.assertEqual(artifact_index["planner"]["round_summary"], str(round_dir / "round_summary.md"))
            self.assertEqual(
                artifact_index["planner"]["planner_raw_quality"],
                str(round_dir / "planner_raw_quality.json"),
            )
            self.assertEqual(
                artifact_index["planner"]["planner_stage_plan"],
                str(round_dir / "planner_stage_plan.json"),
            )
            self.assertEqual(
                artifact_index["subtasks"][0]["steps"][0]["actor_seen_image"],
                str(subtask_dir / "actor_seen_step_01.png"),
            )
            self.assertEqual(
                artifact_index["subtasks"][0]["form_fill_progress"],
                str(subtask_dir / "form_fill_progress.json"),
            )
            self.assertEqual(
                artifact_index["subtasks"][0]["verifier_result"],
                str(subtask_dir / "verifier_result.json"),
            )
            self.assertEqual(
                initial_index["initial_stage_plan"],
                str(run_dir / "initial_stage_plan.json"),
            )

    def test_build_light_run_result_excludes_full_round_payloads(self) -> None:
        full_result = {
            "status": "round_limit",
            "planner_rounds": [{"round_id": 1, "input_observation": {"huge": "payload"}}],
            "final_task_success": None,
            "total_actor_steps": 3,
            "completion_message": "Round limit reached.",
            "memory_metrics": {"backend": "none", "start_size_bytes": 0, "end_size_bytes": 0, "delta_size_bytes": 0},
        }
        artifact_index = {"rounds": [{"round_dir": "x"}]}

        light_result = build_light_run_result(full_result, artifact_index)

        self.assertNotIn("planner_rounds", light_result)
        self.assertEqual(light_result["artifact_index"], artifact_index)
        self.assertEqual(light_result["planner_round_count"], 1)
        self.assertIn("tokens_total", light_result)
        self.assertIn("memory_metrics", light_result)

    def test_build_run_summary_is_readable(self) -> None:
        summary = build_run_summary(
            "ContactsAddContact",
            "Create a new contact",
            {
                "status": "round_limit",
                "completion_message": "Round limit reached.",
                "planner_rounds": [{"round_id": 1, "replan_reason": "x", "subtask_runs": [1], "planner_result": {"usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}}],
                "total_actor_steps": 2,
                "memory_metrics": {"backend": "none", "start_size_bytes": 0, "end_size_bytes": 0, "delta_size_bytes": 0, "end_entry_count": 0},
            },
            "none",
        )

        self.assertIn("# Run Summary", summary)
        self.assertIn("ContactsAddContact", summary)
        self.assertIn("replan_reason=x", summary)
        self.assertIn("group_form_subtask_used", summary)
        self.assertIn("Overall tokens total", summary)
        self.assertIn("## Memory Metrics", summary)

    def test_build_round_summary_includes_subtask_actor_and_verifier_outcomes(self) -> None:
        summary = build_round_summary(
            {
                "input_observation": build_observation("initial"),
                "planner_raw_response": '{"tool":"set_tasks"}',
                "planner_result": {
                    "is_goal_complete": False,
                    "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
                    "subtasks": [
                        {
                            "precondition": "Home screen is visible",
                            "goal": "Open Settings",
                            "reason": "Need app access",
                        }
                    ],
                },
                "subtask_runs": [
                    {
                        "subtask": {
                            "precondition": "Home screen is visible",
                            "goal": "Open Settings",
                        },
                        "actor_result": {
                            "status": "stopped",
                            "completion_message": "Stop for external verification.",
                            "prompt_tokens_total": 5,
                            "completion_tokens_total": 2,
                            "total_tokens_total": 7,
                            "steps": [
                                {
                                    "step_id": 0,
                                    "reason": "Settings is visible in the app drawer.",
                                    "action": {"action_type": "click", "index": 17},
                                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                                }
                            ],
                        },
                        "subtask_verification": {
                            "status": "uncertain",
                            "reason": "Settings is visible, but the Settings app is not yet open.",
                            "memory_eligible": False,
                            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                        },
                    }
                ],
                "replan_reason": "verifier_uncertain",
            }
        )

        self.assertIn("## Subtask Outcomes", summary)
        self.assertIn("### Subtask 1", summary)
        self.assertIn("Actor status: stopped", summary)
        self.assertIn('Actor final action: {"action_type": "click", "index": 17}', summary)
        self.assertIn("Actor final reason: Settings is visible in the app drawer.", summary)
        self.assertIn("Verifier status: uncertain", summary)
        self.assertIn("Verifier reason: Settings is visible, but the Settings app is not yet open.", summary)
        self.assertIn("Planner token usage: prompt=4, completion=2, total=6", summary)
        self.assertIn("Actor token usage: prompt=5, completion=2, total=7", summary)
        self.assertIn("Verifier token usage: prompt=2, completion=1, total=3", summary)

    def test_build_round_summary_handles_empty_steps_and_missing_verifier(self) -> None:
        summary = build_round_summary(
            {
                "input_observation": build_observation("initial"),
                "planner_raw_response": '{"tool":"set_tasks"}',
                "planner_result": {
                    "is_goal_complete": False,
                    "usage": None,
                    "subtasks": [
                        {
                            "precondition": "Launcher is visible",
                            "goal": "Open Settings",
                            "reason": "Need app access",
                        }
                    ],
                },
                "subtask_runs": [
                    {
                        "subtask": {
                            "precondition": "Launcher is visible",
                            "goal": "Open Settings",
                        },
                        "actor_result": {
                            "status": "parse_error",
                            "completion_message": "Action payload was invalid.",
                            "steps": [],
                        },
                        "subtask_verification": {},
                    }
                ],
                "replan_reason": "parse_error",
            }
        )

        self.assertIn("Actor final action: None", summary)
        self.assertIn("Actor final reason: Action payload was invalid.", summary)
        self.assertIn("Verifier status: None", summary)
        self.assertIn("Verifier reason: None", summary)


if __name__ == "__main__":
    unittest.main()
