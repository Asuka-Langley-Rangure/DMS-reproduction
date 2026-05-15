from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class ActorConfig:
    max_history_items: int = 8
    temperature: float = 0.0


@dataclass
class ActorRequest:
    task: str
    observation: Dict[str, Any]
    action_history: List[Dict[str, Any]] = field(default_factory=list)


class LLMClient(Protocol):
    def generate(self, messages: List[Dict[str, Any]], temperature: float = 0.0) -> str:
        """Generate a raw actor response."""


class AndroidActor:
    """Minimal Android actor scaffold shared by Baseline B and DMS."""

    def __init__(self, llm_client: LLMClient, config: Optional[ActorConfig] = None) -> None:
        self.llm_client = llm_client
        self.config = config or ActorConfig()

    def build_messages(self, request: ActorRequest) -> List[Dict[str, Any]]:
        history = request.action_history[-self.config.max_history_items :]
        history_text = "\n".join(str(item) for item in history) if history else "No previous action."
        prompt = (
            "You are controlling an Android phone to complete one subtask.\n\n"
            f"Subtask:\n{request.task}\n\n"
            "Current screen:\n"
            f"Activity: {request.observation.get('current_activity', '')}\n"
            f"Visible UI elements:\n{request.observation.get('ui_description', 'Not available')}\n\n"
            f"Recent action history:\n{history_text}\n"
        )
        return [{"role": "user", "content": prompt}]
