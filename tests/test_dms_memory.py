import tempfile
import unittest
from pathlib import Path

from dms_reproduction.memory.dms import DMSMemoryConfig, DMSMemoryProvider
from dms_reproduction.memory.embedding import (
    OpenAICompatibleEmbeddingConfig,
    OpenAICompatibleEmbeddingProvider,
)
from dms_reproduction.memory.pruning import DMSPruner, PruningConfig
from dms_reproduction.memory.retrieval import RetrievalConfig
from dms_reproduction.memory.store import JsonDMSMemoryStore
from dms_reproduction.memory.survival_value import (
    SurvivalValueCalculator,
    SurvivalValueConfig,
)
from dms_reproduction.memory.types import (
    DMSMemoryMeta,
    DMSMemoryRecord,
    DMSMemoryStatus,
)


def _make_record(memory_id: str = "mem_test") -> DMSMemoryRecord:
    return DMSMemoryRecord(
        memory_id=memory_id,
        subtask_text="Precondition: App is open. Goal: Tap Save.",
        precondition="App is open.",
        goal="Tap Save.",
        user_goal="Save the form",
        app_name="Settings",
        current_activity="SettingsActivity",
        goal_embedding=[0.1, 0.2],
        precondition_embedding=[0.3, 0.4],
        meta=DMSMemoryMeta(memory_id=memory_id, success_count=1),
    )


class TestDMSMemoryModules(unittest.TestCase):
    def test_record_roundtrip(self) -> None:
        record = _make_record()
        payload = record.to_dict()
        restored = DMSMemoryRecord.from_dict(payload)

        self.assertEqual(restored.memory_id, record.memory_id)
        self.assertEqual(restored.goal, record.goal)
        self.assertEqual(restored.user_goal, record.user_goal)
        self.assertEqual(restored.app_name, record.app_name)
        self.assertEqual(restored.current_activity, record.current_activity)
        self.assertIsNotNone(restored.meta)
        self.assertEqual(restored.meta.memory_id, record.memory_id)

    def test_store_init_add_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonDMSMemoryStore(tmpdir)
            root = Path(tmpdir)

            self.assertTrue((root / "index.jsonl").exists())
            self.assertTrue((root / "event.jsonl").exists())
            self.assertTrue((root / "pruning_log.jsonl").exists())
            self.assertTrue((root / "trajectories").exists())

            record = _make_record()
            trajectory = [{"action": "tap_save", "result": "ok"}]
            store.add_memory(record, trajectory)

            loaded = store.get_memory(record.memory_id)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.memory_id, record.memory_id)

            loaded_trajectory = store.load_trajectory(record.memory_id)
            self.assertEqual(loaded_trajectory, trajectory)

            candidates = list(store.iter_active_candidates())
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].goal, record.goal)
            self.assertEqual(candidates[0].trajectory, [])

            store.append_event({"event_type": "unit_test"})
            store.append_pruning_log({"event_type": "unit_test_pruning"})

            stats = store.stats()
            self.assertEqual(stats["total_count"], 1)
            self.assertEqual(stats["active_count"], 1)

    def test_provider_and_helpers_instantiate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonDMSMemoryStore(tmpdir)
            provider = DMSMemoryProvider(
                store=store,
                config=DMSMemoryConfig(memory_root=Path(tmpdir)),
            )

            self.assertEqual(provider.build_context("goal", {}, []), "")
            self.assertEqual(provider.build_actor_context("goal", "bad", {}, []), "")

            calculator = SurvivalValueCalculator(SurvivalValueConfig())
            pruner = DMSPruner(PruningConfig(), calculator)

            self.assertIsInstance(calculator.compute(_make_record("mem_x"), 0), float)
            self.assertFalse(pruner.should_run(current_step=0, active_count=0))

    def test_provider_accepts_embedding_retrieval_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonDMSMemoryStore(tmpdir)

            class DummyEmbeddingProvider:
                def embed_text(self, text: str) -> list[float]:
                    return [0.1, 0.2]

            provider = DMSMemoryProvider(
                store=store,
                config=DMSMemoryConfig(memory_root=Path(tmpdir)),
                retriever_config=RetrievalConfig(retrieval_mode="embedding_product"),
                embedding_provider=DummyEmbeddingProvider(),
            )
            self.assertIsNotNone(provider)

            provider = DMSMemoryProvider(
                store=store,
                config=DMSMemoryConfig(memory_root=Path(tmpdir)),
                retriever_config=RetrievalConfig(retrieval_mode="embedding_weighted_sum"),
                embedding_provider=DummyEmbeddingProvider(),
            )
            self.assertIsNotNone(provider)

    def test_status_and_meta_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonDMSMemoryStore(tmpdir)
            record = _make_record("mem_status")
            store.add_memory(record, [{"action": "tap"}])

            updated = store.mark_status(
                "mem_status",
                DMSMemoryStatus.PRUNED,
                reason="unit_test",
            )
            self.assertEqual(updated.status, DMSMemoryStatus.PRUNED)

            refreshed = store.update_meta(
                "mem_status",
                {"reuse_count_delta": 2, "survival_value": 1.5},
            )
            self.assertEqual(refreshed.meta.reuse_count, 2)
            self.assertEqual(refreshed.meta.survival_value, 1.5)

    def test_record_consumes_selected_memory_feedback_and_task_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonDMSMemoryStore(tmpdir)
            record = _make_record("mem_feedback")
            store.add_memory(record, [{"action": "tap"}])
            provider = DMSMemoryProvider(
                store=store,
                config=DMSMemoryConfig(memory_root=Path(tmpdir)),
            )

            provider.record(
                {
                    "user_goal": "goal",
                    "round_id": 1,
                    "subtask": record.subtask_text,
                    "status": "completed",
                    "summary": "ok",
                    "verifier_status": "success",
                    "verifier_reason": "ok",
                    "memory_eligible": True,
                    "verifier_evidence_source": "actor_completed_frame",
                    "success_check": True,
                    "observation_digest": {},
                    "selected_memory_id": "mem_feedback",
                    "used_memory": True,
                    "should_replay": True,
                    "should_mutate": False,
                    "retrieval_score": 0.9,
                    "sim_goal": 0.8,
                    "sim_precondition": 0.7,
                    "survival_value": 1.2,
                    "risk_score": 0.1,
                    "memory_read_reason": "memory_hit",
                    "invalid_action_count": 1,
                    "no_state_change_count": 2,
                    "parse_error_count": 3,
                    "execution_error_count": 4,
                }
            )

            updated = store.get_memory("mem_feedback")
            self.assertEqual(updated.meta.reuse_count, 1)
            self.assertEqual(updated.meta.success_count, 2)
            self.assertEqual(updated.meta.invalid_action_count, 1)
            self.assertEqual(updated.meta.no_state_change_count, 2)
            self.assertEqual(updated.meta.parse_error_count, 3)
            self.assertEqual(updated.meta.execution_error_count, 4)

            provider.record(
                {
                    "event_type": "task_end",
                    "user_goal": "goal",
                    "round_id": 1,
                    "subtask": "",
                    "status": "completed",
                    "summary": "done",
                    "verifier_status": "",
                    "verifier_reason": "",
                    "memory_eligible": False,
                    "verifier_evidence_source": "",
                    "success_check": {},
                    "observation_digest": {},
                    "success": True,
                    "global_success": True,
                    "task_success": True,
                }
            )
            self.assertEqual(provider.active_memory_ids_for_task, set())


if __name__ == "__main__":
    unittest.main()
