from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Protocol, Sequence

RetrievalMode = Literal["lexical_jaccard", "embedding_product", "embedding_weighted_sum"]

STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "to",
    "of",
    "in",
    "on",
    "is",
    "are",
    "be",
    "for",
    "with",
    "that",
    "this",
}


@dataclass
class MemoryQuery:
    subtask_text: str
    precondition: str
    goal: str
    user_goal: str = ""
    app_name: str = ""
    current_activity: str = ""


@dataclass
class MemoryCandidate:
    subtask_text: str
    precondition: str
    goal: str
    timestamp: str
    app_name: str
    current_activity: str
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    verifier_reason: str = ""
    goal_embedding: list[float] | None = None
    precondition_embedding: list[float] | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    candidate: MemoryCandidate
    sim_goal: float
    sim_precondition: float
    final_score: float
    retrieval_mode: RetrievalMode
    rank: int = 0


@dataclass
class RetrievalConfig:
    top_k: int = 3
    retrieval_mode: RetrievalMode = "lexical_jaccard"
    weighted_sum_goal_weight: float = 0.7
    weighted_sum_precondition_weight: float = 0.3
    normalize_cosine_to_unit_interval: bool = True


class EmbeddingProvider(Protocol):
    def embed_text(self, text: str) -> list[float]:
        """Return one embedding vector for the given text."""


class MemoryStore(Protocol):
    def iter_candidates(self) -> Iterable[MemoryCandidate]:
        """Yield normalized retrieval candidates."""

    def append_record(self, record: dict[str, Any]) -> None:
        """Persist one memory record."""


def parse_subtask_text(subtask_text: str) -> MemoryQuery | None:
    match = re.match(
        r"^\s*Precondition\s*:\s*(?P<precondition>.*?)\s+Goal\s*:\s*(?P<goal>.+?)\s*$",
        subtask_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    precondition = match.group("precondition").strip()
    goal = match.group("goal").strip()
    if not precondition or not goal:
        return None
    return MemoryQuery(
        subtask_text=subtask_text.strip(),
        precondition=precondition,
        goal=goal,
    )


def candidate_from_record(record: dict[str, Any]) -> MemoryCandidate:
    return MemoryCandidate(
        subtask_text=str(record.get("subtask_text") or "Unknown"),
        precondition=str(record.get("precondition") or ""),
        goal=str(record.get("goal") or ""),
        timestamp=str(record.get("timestamp") or ""),
        app_name=str(record.get("app_name") or ""),
        current_activity=str(record.get("current_activity") or ""),
        trajectory=list(record.get("trajectory") or []),
        verifier_reason=str(record.get("verifier_reason") or ""),
        goal_embedding=_coerce_embedding(record.get("goal_embedding")),
        precondition_embedding=_coerce_embedding(record.get("precondition_embedding")),
        payload=record,
    )


def retrieve(
    query: MemoryQuery,
    candidates: Iterable[MemoryCandidate],
    config: RetrievalConfig,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[RetrievalResult]:
    results: list[RetrievalResult] = []
    query_goal_embedding: list[float] | None = None
    query_precondition_embedding: list[float] | None = None

    if config.retrieval_mode != "lexical_jaccard":
        if embedding_provider is None:
            raise ValueError("Embedding retrieval mode requires an embedding provider.")
        query_goal_embedding = embedding_provider.embed_text(query.goal)
        query_precondition_embedding = embedding_provider.embed_text(query.precondition)

    for candidate in candidates:
        if config.retrieval_mode == "lexical_jaccard":
            sim_goal, sim_precondition, final_score = score_lexical_jaccard(query, candidate)
        elif config.retrieval_mode == "embedding_product":
            sim_goal, sim_precondition, final_score = score_embedding_product(
                query=query,
                candidate=candidate,
                config=config,
                embedding_provider=embedding_provider,
                query_goal_embedding=query_goal_embedding,
                query_precondition_embedding=query_precondition_embedding,
            )
        elif config.retrieval_mode == "embedding_weighted_sum":
            sim_goal, sim_precondition, final_score = score_embedding_weighted_sum(
                query=query,
                candidate=candidate,
                config=config,
                embedding_provider=embedding_provider,
                query_goal_embedding=query_goal_embedding,
                query_precondition_embedding=query_precondition_embedding,
            )
        else:
            raise ValueError(f"Unsupported retrieval mode: {config.retrieval_mode}")
        if final_score <= 0.0:
            continue
        results.append(
            RetrievalResult(
                candidate=candidate,
                sim_goal=sim_goal,
                sim_precondition=sim_precondition,
                final_score=final_score,
                retrieval_mode=config.retrieval_mode,
            )
        )

    results.sort(
        key=lambda item: (
            -item.final_score,
            item.candidate.timestamp,
        )
    )
    selected = results[: config.top_k]
    selected.sort(key=lambda item: item.candidate.timestamp)
    for index, item in enumerate(selected, start=1):
        item.rank = index
    return selected


def format_actor_memory_context(results: Sequence[RetrievalResult], max_trajectory_steps: int) -> str:
    if not results:
        return ""
    lines = ["Retrieved static memory experiences:"]
    for result in results:
        candidate = result.candidate
        trajectory = list(candidate.trajectory)[:max_trajectory_steps]
        lines.extend(
            [
                "",
                f"Experience {result.rank}:",
                f"- Historical subtask: {candidate.subtask_text or 'Unknown'}",
                f"- Timestamp: {candidate.timestamp or 'Unknown'}",
                f"- App: {candidate.app_name or 'Unknown'}",
                f"- Activity: {candidate.current_activity or 'Unknown'}",
                f"- Verifier reason: {candidate.verifier_reason or 'None'}",
                (
                    f"- Retrieval score: final={result.final_score:.4f}; "
                    f"goal={result.sim_goal:.4f}; precondition={result.sim_precondition:.4f}; "
                    f"mode={result.retrieval_mode}"
                ),
                "- Core trajectory:",
            ]
        )
        if not trajectory:
            lines.append("  - None")
        for step_index, step in enumerate(trajectory, start=1):
            lines.append(
                f"  - Step {step_index}: reason={step.get('reason') or 'None'}; "
                f"action={json.dumps(step.get('action') or {}, ensure_ascii=False)}; "
                f"summary={step.get('summary') or 'None'}"
            )
    return "\n".join(lines)


def score_lexical_jaccard(query: MemoryQuery, candidate: MemoryCandidate) -> tuple[float, float, float]:
    query_goal_tokens = _tokenize(query.goal)
    query_precondition_tokens = _tokenize(query.precondition)
    candidate_goal_tokens = _tokenize(candidate.goal)
    candidate_precondition_tokens = _tokenize(candidate.precondition)
    sim_goal = _jaccard(query_goal_tokens, candidate_goal_tokens)
    sim_precondition = _jaccard(query_precondition_tokens, candidate_precondition_tokens)
    return sim_goal, sim_precondition, (2.0 * sim_goal) + sim_precondition


def score_embedding_product(
    *,
    query: MemoryQuery,
    candidate: MemoryCandidate,
    config: RetrievalConfig,
    embedding_provider: EmbeddingProvider | None,
    query_goal_embedding: list[float] | None,
    query_precondition_embedding: list[float] | None,
) -> tuple[float, float, float]:
    sim_goal = _similarity_from_embeddings(
        left=query_goal_embedding,
        right=_get_or_build_candidate_embedding(candidate.goal_embedding, candidate.goal, embedding_provider),
        normalize=config.normalize_cosine_to_unit_interval,
    )
    sim_precondition = _similarity_from_embeddings(
        left=query_precondition_embedding,
        right=_get_or_build_candidate_embedding(candidate.precondition_embedding, candidate.precondition, embedding_provider),
        normalize=config.normalize_cosine_to_unit_interval,
    )
    return sim_goal, sim_precondition, sim_goal * sim_precondition


def score_embedding_weighted_sum(
    *,
    query: MemoryQuery,
    candidate: MemoryCandidate,
    config: RetrievalConfig,
    embedding_provider: EmbeddingProvider | None,
    query_goal_embedding: list[float] | None,
    query_precondition_embedding: list[float] | None,
) -> tuple[float, float, float]:
    sim_goal = _similarity_from_embeddings(
        left=query_goal_embedding,
        right=_get_or_build_candidate_embedding(candidate.goal_embedding, candidate.goal, embedding_provider),
        normalize=config.normalize_cosine_to_unit_interval,
    )
    sim_precondition = _similarity_from_embeddings(
        left=query_precondition_embedding,
        right=_get_or_build_candidate_embedding(candidate.precondition_embedding, candidate.precondition, embedding_provider),
        normalize=config.normalize_cosine_to_unit_interval,
    )
    final_score = (
        config.weighted_sum_goal_weight * sim_goal
        + config.weighted_sum_precondition_weight * sim_precondition
    )
    return sim_goal, sim_precondition, final_score


def _tokenize(text: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9+]+", " ", text.lower())
    return {token for token in normalized.split() if token and token not in STOPWORDS}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _coerce_embedding(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    output: list[float] = []
    for item in value:
        if isinstance(item, (int, float)):
            output.append(float(item))
        else:
            return None
    return output


def _get_or_build_candidate_embedding(
    embedding: list[float] | None,
    text: str,
    embedding_provider: EmbeddingProvider | None,
) -> list[float]:
    if embedding is not None:
        return embedding
    if embedding_provider is None:
        raise ValueError("Embedding retrieval mode requires an embedding provider.")
    return embedding_provider.embed_text(text)


def _similarity_from_embeddings(
    *,
    left: list[float] | None,
    right: list[float] | None,
    normalize: bool,
) -> float:
    if left is None or right is None:
        return 0.0
    cosine = _cosine_similarity(left, right)
    if normalize:
        return (cosine + 1.0) / 2.0
    return cosine


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = sum(l * r for l, r in zip(left, right))
    return dot / (left_norm * right_norm)
