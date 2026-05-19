from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dms_reproduction.memory.retrieval import (
    MemoryCandidate,
    MemoryQuery,
    RetrievalConfig,
    retrieve,
)
from dms_reproduction.memory.static import StaticJsonlMemoryStore, StaticMemoryConfig, StaticMemoryProvider


class FakeEmbeddingProvider:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[str] = []

    def embed_text(self, text: str) -> list[float]:
        self.calls.append(text)
        return list(self.vectors[text])


class FakeMemoryStore:
    def __init__(self, candidates: list[MemoryCandidate]) -> None:
        self._candidates = list(candidates)
        self.appended_records: list[dict] = []
        self.iter_calls = 0

    def iter_candidates(self):
        self.iter_calls += 1
        return iter(self._candidates)

    def append_record(self, record: dict) -> None:
        self.appended_records.append(record)


class FakeDMSMemoryProvider:
    def __init__(self, store: FakeMemoryStore, config: RetrievalConfig, embedding_provider=None) -> None:
        self.store = store
        self.config = config
        self.embedding_provider = embedding_provider

    def retrieve_for_actor(self, query: MemoryQuery):
        return retrieve(query, self.store.iter_candidates(), self.config, self.embedding_provider)


class StaticMemoryProviderTest(unittest.TestCase):
    def test_empty_memory_returns_empty_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = StaticMemoryProvider(
                StaticMemoryConfig(file_path=str(Path(temp_dir) / "static_memory.jsonl"), top_k=3)
            )

            context = provider.build_actor_context(
                user_goal="Create a contact",
                subtask="Precondition: Phone app is open Goal: Reach the contact creation entry point.",
                observation={},
                task_history=[],
            )

            self.assertEqual(context, "")

    def test_record_subtask_trajectory_appends_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "static_memory.jsonl"
            provider = StaticMemoryProvider(StaticMemoryConfig(file_path=str(memory_path), top_k=3))

            provider.record_subtask_trajectory(
                {
                    "subtask_text": "Precondition: None Goal: Open the Phone app.",
                    "precondition": "None",
                    "goal": "Open the Phone app.",
                    "user_goal": "Create a contact",
                    "round_id": 1,
                    "timestamp": "2026-05-18T00:00:01",
                    "app_name": "com.google.android.dialer",
                    "current_activity": "DialerActivity",
                    "verifier_status": "success",
                    "verifier_reason": "Phone app is visible.",
                    "completion_message": "done",
                    "trajectory": [{"reason": "open app", "action": {"action_type": "open_app"}, "summary": "opened"}],
                    "observation_digest": {"app_name": "com.google.android.dialer"},
                }
            )
            provider.record_subtask_trajectory(
                {
                    "subtask_text": "Precondition: Phone app is open Goal: Reach the contact creation entry point.",
                    "precondition": "Phone app is open",
                    "goal": "Reach the contact creation entry point.",
                    "user_goal": "Create a contact",
                    "round_id": 2,
                    "timestamp": "2026-05-18T00:00:02",
                    "app_name": "com.google.android.dialer",
                    "current_activity": "DialerContactsActivity",
                    "verifier_status": "success",
                    "verifier_reason": "Create contact entry point is visible.",
                    "completion_message": "done",
                    "trajectory": [{"reason": "open contacts", "action": {"action_type": "click"}, "summary": "navigated"}],
                    "observation_digest": {"app_name": "com.google.android.dialer"},
                }
            )

            lines = memory_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            payloads = [json.loads(line) for line in lines]
            self.assertEqual(payloads[0]["round_id"], 1)
            self.assertEqual(payloads[1]["round_id"], 2)

    def test_record_subtask_trajectory_writes_embeddings_when_provider_is_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "static_memory.jsonl"
            provider = StaticMemoryProvider(
                StaticMemoryConfig(file_path=str(memory_path), top_k=3),
                embedding_provider=FakeEmbeddingProvider(
                    {
                        "Open the Phone app.": [1.0, 0.0],
                        "None": [0.0, 1.0],
                    }
                ),
            )

            provider.record_subtask_trajectory(
                {
                    "subtask_text": "Precondition: None Goal: Open the Phone app.",
                    "precondition": "None",
                    "goal": "Open the Phone app.",
                    "user_goal": "Create a contact",
                    "round_id": 1,
                    "timestamp": "2026-05-18T00:00:01",
                    "app_name": "com.google.android.dialer",
                    "current_activity": "DialerActivity",
                    "verifier_status": "success",
                    "verifier_reason": "Phone app is visible.",
                    "completion_message": "done",
                    "trajectory": [{"reason": "open app", "action": {"action_type": "open_app"}, "summary": "opened"}],
                    "observation_digest": {"app_name": "com.google.android.dialer"},
                }
            )

            payload = json.loads(memory_path.read_text(encoding="utf-8").strip())
            self.assertEqual(payload["goal_embedding"], [1.0, 0.0])
            self.assertEqual(payload["precondition_embedding"], [0.0, 1.0])

    def test_similarity_retrieval_prefers_goal_match_and_orders_selected_by_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "static_memory.jsonl"
            provider = StaticMemoryProvider(StaticMemoryConfig(file_path=str(memory_path), top_k=2))
            records = [
                {
                    "subtask_text": "Precondition: Contacts tab is open Goal: Reach the contact creation entry point.",
                    "precondition": "Contacts tab is open",
                    "goal": "Reach the contact creation entry point.",
                    "user_goal": "Create a contact",
                    "round_id": 1,
                    "timestamp": "2026-05-18T00:00:03",
                    "app_name": "com.google.android.dialer",
                    "current_activity": "DialerActivity",
                    "verifier_status": "success",
                    "verifier_reason": "Create contact entry point was reached from contacts tab.",
                    "completion_message": "done",
                    "trajectory": [{"reason": "open create flow", "action": {"action_type": "click"}, "summary": "opened create flow"}],
                    "observation_digest": {"app_name": "com.google.android.dialer"},
                },
                {
                    "subtask_text": "Precondition: Phone app is open Goal: Reach the contact creation entry point.",
                    "precondition": "Phone app is open",
                    "goal": "Reach the contact creation entry point.",
                    "user_goal": "Create a contact",
                    "round_id": 2,
                    "timestamp": "2026-05-18T00:00:01",
                    "app_name": "com.google.android.dialer",
                    "current_activity": "DialerContactsActivity",
                    "verifier_status": "success",
                    "verifier_reason": "Create contact entry point is visible.",
                    "completion_message": "done",
                    "trajectory": [{"reason": "open contacts", "action": {"action_type": "click"}, "summary": "navigated"}],
                    "observation_digest": {"app_name": "com.google.android.dialer"},
                },
                {
                    "subtask_text": "Precondition: File list is visible Goal: Delete the selected audio file.",
                    "precondition": "File list is visible",
                    "goal": "Delete the selected audio file.",
                    "user_goal": "Delete a file",
                    "round_id": 3,
                    "timestamp": "2026-05-18T00:00:02",
                    "app_name": "com.google.android.documentsui",
                    "current_activity": "FilesActivity",
                    "verifier_status": "success",
                    "verifier_reason": "Audio file was deleted.",
                    "completion_message": "done",
                    "trajectory": [{"reason": "delete file", "action": {"action_type": "click"}, "summary": "deleted"}],
                    "observation_digest": {"app_name": "com.google.android.documentsui"},
                },
            ]
            for record in records:
                provider.record_subtask_trajectory(record)

            context = provider.build_actor_context(
                user_goal="Create a contact",
                subtask="Precondition: Phone app is open Goal: Reach the contact creation entry point.",
                observation={},
                task_history=[],
            )

            self.assertIn("Retrieved static memory experiences", context)
            self.assertIn("Reach the contact creation entry point.", context)
            self.assertIn("Contacts tab is open Goal: Reach the contact creation entry point.", context)
            self.assertNotIn("Delete the selected audio file.", context)
            self.assertLess(
                context.index("Timestamp: 2026-05-18T00:00:01"),
                context.index("Timestamp: 2026-05-18T00:00:03"),
            )

    def test_store_tolerates_bad_and_empty_jsonl_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "static_memory.jsonl"
            memory_path.write_text(
                '\n{"subtask_text":"Precondition: A Goal: B","precondition":"A","goal":"B","timestamp":"2026-05-18T00:00:01"}\n{bad json}\n',
                encoding="utf-8",
            )
            store = StaticJsonlMemoryStore(str(memory_path))

            candidates = list(store.iter_candidates())

            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].goal, "B")

    def test_provider_uses_store_and_retriever_not_inline_scoring(self) -> None:
        candidates = [
            MemoryCandidate(
                subtask_text="Precondition: Phone app is open Goal: Reach the contact creation entry point.",
                precondition="Phone app is open",
                goal="Reach the contact creation entry point.",
                timestamp="2026-05-18T00:00:01",
                app_name="com.google.android.dialer",
                current_activity="DialerContactsActivity",
                trajectory=[],
                verifier_reason="entry point visible",
            )
        ]
        store = FakeMemoryStore(candidates)
        provider = StaticMemoryProvider(
            StaticMemoryConfig(top_k=1),
            store=store,
        )

        context = provider.build_actor_context(
            user_goal="Create a contact",
            subtask="Precondition: Phone app is open Goal: Reach the contact creation entry point.",
            observation={"app_name": "com.google.android.dialer", "current_activity": "DialerContactsActivity"},
            task_history=[],
        )

        self.assertEqual(store.iter_calls, 1)
        self.assertIn("Retrieved static memory experiences", context)

    def test_reuse_contract_static_and_fake_dms_share_same_retriever_ranking(self) -> None:
        candidates = [
            MemoryCandidate(
                subtask_text="Precondition: Contacts tab is open Goal: Reach the contact creation entry point.",
                precondition="Contacts tab is open",
                goal="Reach the contact creation entry point.",
                timestamp="2026-05-18T00:00:02",
                app_name="com.google.android.dialer",
                current_activity="DialerActivity",
            ),
            MemoryCandidate(
                subtask_text="Precondition: Phone app is open Goal: Reach the contact creation entry point.",
                precondition="Phone app is open",
                goal="Reach the contact creation entry point.",
                timestamp="2026-05-18T00:00:01",
                app_name="com.google.android.dialer",
                current_activity="DialerContactsActivity",
            ),
        ]
        store = FakeMemoryStore(candidates)
        config = RetrievalConfig(top_k=2)
        query = MemoryQuery(
            subtask_text="Precondition: Phone app is open Goal: Reach the contact creation entry point.",
            precondition="Phone app is open",
            goal="Reach the contact creation entry point.",
        )
        static_results = retrieve(query, store.iter_candidates(), config)
        dms_results = FakeDMSMemoryProvider(store, config).retrieve_for_actor(query)

        self.assertEqual(
            [item.candidate.subtask_text for item in static_results],
            [item.candidate.subtask_text for item in dms_results],
        )

    def test_embedding_product_mode_uses_shared_retriever_formula(self) -> None:
        provider = FakeEmbeddingProvider(
            {
                "Reach the contact creation entry point.": [1.0, 0.0],
                "Phone app is open": [1.0, 0.0],
                "Reach the contact list.": [0.0, 1.0],
                "Contacts tab is open": [0.0, 1.0],
            }
        )
        query = MemoryQuery(
            subtask_text="Precondition: Phone app is open Goal: Reach the contact creation entry point.",
            precondition="Phone app is open",
            goal="Reach the contact creation entry point.",
        )
        results = retrieve(
            query,
            [
                MemoryCandidate(
                    subtask_text="Precondition: Phone app is open Goal: Reach the contact creation entry point.",
                    precondition="Phone app is open",
                    goal="Reach the contact creation entry point.",
                    timestamp="2026-05-18T00:00:01",
                    app_name="com.google.android.dialer",
                    current_activity="DialerContactsActivity",
                ),
                MemoryCandidate(
                    subtask_text="Precondition: Contacts tab is open Goal: Reach the contact list.",
                    precondition="Contacts tab is open",
                    goal="Reach the contact list.",
                    timestamp="2026-05-18T00:00:02",
                    app_name="com.google.android.dialer",
                    current_activity="DialerActivity",
                ),
            ],
            RetrievalConfig(top_k=2, retrieval_mode="embedding_product"),
            provider,
        )

        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0].sim_goal, 1.0)
        self.assertAlmostEqual(results[0].sim_precondition, 1.0)
        self.assertAlmostEqual(results[0].final_score, 1.0)
        self.assertGreater(results[0].final_score, results[1].final_score)

    def test_embedding_weighted_sum_mode_uses_config_weights(self) -> None:
        provider = FakeEmbeddingProvider(
            {
                "Goal query": [1.0, 0.0],
                "Precondition query": [1.0, 0.0],
                "Goal candidate": [1.0, 0.0],
                "Precondition candidate": [0.0, 1.0],
            }
        )
        query = MemoryQuery(
            subtask_text="Precondition: Precondition query Goal: Goal query",
            precondition="Precondition query",
            goal="Goal query",
        )
        results = retrieve(
            query,
            [
                MemoryCandidate(
                    subtask_text="Precondition: Precondition candidate Goal: Goal candidate",
                    precondition="Precondition candidate",
                    goal="Goal candidate",
                    timestamp="2026-05-18T00:00:01",
                    app_name="app",
                    current_activity="activity",
                )
            ],
            RetrievalConfig(
                top_k=1,
                retrieval_mode="embedding_weighted_sum",
                weighted_sum_goal_weight=0.7,
                weighted_sum_precondition_weight=0.3,
            ),
            provider,
        )

        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].sim_goal, 1.0)
        self.assertAlmostEqual(results[0].sim_precondition, 0.5)
        self.assertAlmostEqual(results[0].final_score, 0.85)

    def test_embedding_mode_without_provider_raises(self) -> None:
        query = MemoryQuery(
            subtask_text="Precondition: A Goal: B",
            precondition="A",
            goal="B",
        )
        with self.assertRaises(ValueError):
            retrieve(
                query,
                [],
                RetrievalConfig(top_k=1, retrieval_mode="embedding_product"),
            )


if __name__ == "__main__":
    unittest.main()
