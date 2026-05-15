from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol


class BaseLLMClient(Protocol):
    """Minimal client contract shared by planner and actor integrations."""

    def generate(self, messages: List[Dict[str, Any]], temperature: float = 0.0) -> str:
        """Return the raw text content produced by the model."""


@dataclass
class OpenAICompatibleConfig:
    base_url: str
    api_key: str
    model: str
    max_tokens: int = 512
    timeout: int = 120
