from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


LLMUsage = Dict[str, Optional[int]]


class BaseLLMClient(Protocol):
    """Minimal client contract shared by planner and actor integrations."""

    def generate(self, messages: List[Dict[str, Any]], temperature: float = 0.0) -> str:
        """Return the raw text content produced by the model."""


def normalize_usage(raw_usage: Any) -> LLMUsage | None:
    if not isinstance(raw_usage, dict):
        return None

    normalized: LLMUsage = {
        "prompt_tokens": _coerce_optional_int(raw_usage.get("prompt_tokens")),
        "completion_tokens": _coerce_optional_int(raw_usage.get("completion_tokens")),
        "total_tokens": _coerce_optional_int(raw_usage.get("total_tokens")),
    }
    if all(value is None for value in normalized.values()):
        return None
    return normalized


def get_client_usage(client: Any) -> LLMUsage | None:
    usage: Any = None
    getter = getattr(client, "get_last_usage", None)
    if callable(getter):
        usage = getter()
    elif hasattr(client, "last_usage"):
        usage = getattr(client, "last_usage")
    return normalize_usage(usage)


def sum_usages(usages: List[LLMUsage | None]) -> LLMUsage:
    return {
        "prompt_tokens": _sum_optional_ints([usage.get("prompt_tokens") for usage in usages if usage]),
        "completion_tokens": _sum_optional_ints([usage.get("completion_tokens") for usage in usages if usage]),
        "total_tokens": _sum_optional_ints([usage.get("total_tokens") for usage in usages if usage]),
    }


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sum_optional_ints(values: List[Optional[int]]) -> Optional[int]:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present)


@dataclass
class OpenAICompatibleConfig:
    base_url: str
    api_key: str
    model: str
    max_tokens: int = 512
    timeout: int = 120
