from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None

if PIL_AVAILABLE:
    from scripts.task_loop_smoke import build_light_run_result, build_run_summary, write_round_artifacts


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
        "extra_state": {"tag": tag},
    }


@unittest.skipUnless(PIL_AVAILABLE, "Pillow is not installed in this environment.")
class TaskLoopSmokeArtifactsTest(unittest.TestCase):
    def test_write_round_artifacts_outputs_images_and_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            round_record = {
                "round_id": 1,
                "input_observation": build_observation("initial"),
                "planner_messages": [{"role": "user", "content": "planner"}],
                "planner_prompt": "planner prompt",
                "planner_raw_response": '{"tool":"set_tasks"}',
                "planner_result": {
                    "is_goal_complete": False,
                    "completion_message": "",
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
                            "goal": "Open Contacts",
                            "reason": "Need app",
                            "agent": "android_actor",
                        },
                        "actor_result": {
                            "status": "completed",
                            "completion_message": "done",
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
                                }
                            ],
                        },
                        "post_observation": build_observation("after"),
                    }
                ],
                "replan_reason": "subtasks_exhausted",
            }

            artifact_index = write_round_artifacts(run_dir, round_record)
            round_dir = run_dir / "round_01"
            subtask_dir = round_dir / "subtask_01"

            self.assertTrue((round_dir / "observation_raw.png").exists())
            self.assertTrue((round_dir / "observation_labeled.png").exists())
            self.assertTrue((subtask_dir / "actor_seen_step_01.png").exists())
            self.assertTrue((round_dir / "round_summary.md").exists())
            self.assertTrue((subtask_dir / "subtask_summary.md").exists())
            self.assertEqual(artifact_index["planner"]["round_summary"], str(round_dir / "round_summary.md"))
            self.assertEqual(
                artifact_index["subtasks"][0]["steps"][0]["actor_seen_image"],
                str(subtask_dir / "actor_seen_step_01.png"),
            )

    def test_build_light_run_result_excludes_full_round_payloads(self) -> None:
        full_result = {
            "status": "round_limit",
            "planner_rounds": [{"round_id": 1, "input_observation": {"huge": "payload"}}],
            "final_task_success": None,
            "total_actor_steps": 3,
            "completion_message": "Round limit reached.",
        }
        artifact_index = {"rounds": [{"round_dir": "x"}]}

        light_result = build_light_run_result(full_result, artifact_index)

        self.assertNotIn("planner_rounds", light_result)
        self.assertEqual(light_result["artifact_index"], artifact_index)
        self.assertEqual(light_result["planner_round_count"], 1)

    def test_build_run_summary_is_readable(self) -> None:
        summary = build_run_summary(
            "ContactsAddContact",
            "Create a new contact",
            {
                "status": "round_limit",
                "completion_message": "Round limit reached.",
                "planner_rounds": [{"round_id": 1, "replan_reason": "x", "subtask_runs": [1]}],
                "total_actor_steps": 2,
            },
        )

        self.assertIn("# Run Summary", summary)
        self.assertIn("ContactsAddContact", summary)
        self.assertIn("replan_reason=x", summary)


if __name__ == "__main__":
    unittest.main()
