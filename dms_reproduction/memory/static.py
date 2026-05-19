from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .retrieval import (
    EmbeddingProvider,
    MemoryCandidate,
    MemoryStore,
    RetrievalConfig,
    candidate_from_record,
    format_actor_memory_context,
    parse_subtask_text,
    retrieve,
)


@dataclass
class StaticMemoryConfig(RetrievalConfig):
    file_path: str = "memory_bank/static_memory.jsonl"
    max_trajectory_steps: int = 5


class StaticJsonlMemoryStore(MemoryStore):
    def __init__(self, file_path: str) -> None:
        self._file_path = Path(file_path)
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

    def iter_candidates(self) -> Iterable[MemoryCandidate]:
        for record in self.load_records():
            yield candidate_from_record(record)

    def append_record(self, record: dict[str, Any]) -> None:
        with self._file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_records(self) -> list[dict[str, Any]]:
        if not self._file_path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self._file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
        return records


class StaticMemoryProvider:
    """Persistent static memory store for actor-facing subtask retrieval."""

    def __init__(
        self,
        config: StaticMemoryConfig | None = None,
        *,
        store: MemoryStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.config = config or StaticMemoryConfig()
        self.embedding_provider = embedding_provider
        self.store = store or StaticJsonlMemoryStore(self.config.file_path)

    def build_context(
        self,
        user_goal: str,
        observation: Dict[str, Any],
        task_history: List[Dict[str, Any]],
    ) -> str:
        return ""

    def build_actor_context(
        self,
        user_goal: str,
        subtask: str,
        observation: Dict[str, Any],
        task_history: List[Dict[str, Any]],
    ) -> str:
        query = parse_subtask_text(subtask)
        if query is None:
            return ""
        query.user_goal = user_goal
        query.app_name = str(observation.get("app_name") or "")
        query.current_activity = str(observation.get("current_activity") or "")
        results = retrieve(
            query=query,
            candidates=self.store.iter_candidates(),
            config=self.config,
            embedding_provider=self.embedding_provider,
        )
        return format_actor_memory_context(results, self.config.max_trajectory_steps)

    def record(self, event: Dict[str, Any]) -> None:
        return None

    def record_subtask_trajectory(self, record: Dict[str, Any]) -> None:
        payload = dict(record)
        if self.embedding_provider is not None:
            goal = str(payload.get("goal") or "").strip()
            precondition = str(payload.get("precondition") or "").strip()
            if goal and not payload.get("goal_embedding"):
                payload["goal_embedding"] = self.embedding_provider.embed_text(goal)
            if precondition and not payload.get("precondition_embedding"):
                payload["precondition_embedding"] = self.embedding_provider.embed_text(precondition)
        self.store.append_record(payload)

    def reset(self) -> None:
        return None
