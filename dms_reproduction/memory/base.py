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
    verifier_status: str
    verifier_reason: str
    memory_eligible: bool
    verifier_evidence_source: str
    success_check: Dict[str, Any]
    observation_digest: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StaticMemoryRecord:
    subtask_text: str
    precondition: str
    goal: str
    user_goal: str
    round_id: int
    timestamp: str
    app_name: str
    current_activity: str
    verifier_status: str
    verifier_reason: str
    completion_message: str
    trajectory: List[Dict[str, Any]]
    observation_digest: Dict[str, Any]
    goal_embedding: List[float] | None = None
    precondition_embedding: List[float] | None = None

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

    def build_actor_context(
        self,
        user_goal: str,
        subtask: str,
        observation: Dict[str, Any],
        task_history: List[Dict[str, Any]],
    ) -> str:
        """Build actor-facing memory context for one planner subtask."""

    def record(self, event: Dict[str, Any]) -> None:
        """Record a structured runtime event for future memory implementations."""

    def record_subtask_trajectory(self, record: Dict[str, Any]) -> None:
        """Record one successful planner-subtask trajectory into persistent memory."""

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

    def build_actor_context(
        self,
        user_goal: str,
        subtask: str,
        observation: Dict[str, Any],
        task_history: List[Dict[str, Any]],
    ) -> str:
        return ""

    def record(self, event: Dict[str, Any]) -> None:
        return None

    def record_subtask_trajectory(self, record: Dict[str, Any]) -> None:
        return None

    def reset(self) -> None:
        return None
