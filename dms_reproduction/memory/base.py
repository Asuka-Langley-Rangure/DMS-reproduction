from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Protocol


@dataclass
class MemoryEvent:
    user_goal: str
    round_id: int
    subtask: str
    status: str
    summary: str
    success_check: Dict[str, Any]
    observation_digest: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryProvider(Protocol):
    def build_context(
        self,
        user_goal: str,
        observation: Dict[str, Any],
        task_history: List[Dict[str, Any]],
    ) -> str:
        """Build planner/actor-facing memory context from structured runtime state."""

    def record(self, event: Dict[str, Any]) -> None:
        """Record a structured runtime event for future memory implementations."""

    def reset(self) -> None:
        """Reset provider state between tasks."""


class NoOpMemoryProvider:
    """Default provider that preserves current runtime behavior."""

    def build_context(
        self,
        user_goal: str,
        observation: Dict[str, Any],
        task_history: List[Dict[str, Any]],
    ) -> str:
        return ""

    def record(self, event: Dict[str, Any]) -> None:
        return None

    def reset(self) -> None:
        return None
