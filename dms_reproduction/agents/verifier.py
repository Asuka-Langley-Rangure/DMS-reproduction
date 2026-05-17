from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Literal, Optional, Protocol


VerifierStatus = Literal["success", "failure", "uncertain"]
EvidenceSource = Literal[
    "actor_completed_frame",
    "final_after_observation",
    "fallback_current_observation",
]


@dataclass
class VerifierConfig:
    max_history_items: int = 8
    max_ui_json_chars: int = 12000
    max_memory_context_chars: int = 6000
    temperature: float = 0.0


@dataclass
class VerifierRequest:
    subtask: str
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    before_observation: Dict[str, Any] = field(default_factory=dict)
    evidence_observation: Dict[str, Any] | None = None
    evidence_source: EvidenceSource = "fallback_current_observation"
    memory_context: str = ""


@dataclass
class VerifierResult:
    status: VerifierStatus
    reason: str
    memory_eligible: bool
    raw_response: str
    parse_error: str | None = None
    prompt_text: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "memory_eligible": self.memory_eligible,
            "raw_response": self.raw_response,
            "parse_error": self.parse_error,
            "prompt_text": self.prompt_text,
            "messages": self.messages,
        }


class LLMClient(Protocol):
    def generate(self, messages: List[Dict[str, Any]], temperature: float = 0.0) -> str:
        """Generate a raw verifier response."""


class AndroidVerifier:
    def __init__(self, llm_client: LLMClient, config: Optional[VerifierConfig] = None) -> None:
        self.llm_client = llm_client
        self.config = config or VerifierConfig()

    def build_messages(self, request: VerifierRequest) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_user_prompt(request)},
        ]

    @staticmethod
    def extract_user_text_prompt(messages: List[Dict[str, Any]]) -> str:
        for message in messages:
            if message.get("role") == "user":
                return str(message.get("content") or "")
        return ""

    def messages_to_jsonable(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return json.loads(json.dumps(messages, ensure_ascii=False))

    def run_verification(self, request: VerifierRequest) -> VerifierResult:
        messages = self.build_messages(request)
        prompt_text = self.extract_user_text_prompt(messages)
        messages_jsonable = self.messages_to_jsonable(messages)
        raw_response = self.llm_client.generate(
            messages=messages,
            temperature=self.config.temperature,
        )
        parsed = extract_json_object(raw_response)
        if parsed is None:
            return VerifierResult(
                status="uncertain",
                reason="Verifier response could not be parsed as JSON.",
                memory_eligible=False,
                raw_response=raw_response,
                parse_error="Failed to parse verifier response JSON.",
                prompt_text=prompt_text,
                messages=messages_jsonable,
            )
        try:
            return VerifierResult(
                status=_parse_status(parsed.get("status")),
                reason=str(parsed.get("reason", "")).strip() or "No verifier reason provided.",
                memory_eligible=bool(parsed.get("memory_eligible", False)),
                raw_response=raw_response,
                parse_error=None,
                prompt_text=prompt_text,
                messages=messages_jsonable,
            )
        except ValueError as exc:
            return VerifierResult(
                status="uncertain",
                reason="Verifier response schema was invalid.",
                memory_eligible=False,
                raw_response=raw_response,
                parse_error=str(exc),
                prompt_text=prompt_text,
                messages=messages_jsonable,
            )

    def _build_system_prompt(self) -> str:
        return (
            "You are an Android subtask verifier.\n\n"
            "Role:\n"
            "- You are not the planner.\n"
            "- You are not the actor.\n"
            "- You must judge whether the current subtask has already been completed.\n\n"
            "Evidence policy:\n"
            "- Treat evidence_observation as the main ground-truth evidence.\n"
            "- Treat action_history only as supporting evidence for or against what the screenshot shows.\n"
            "- Do not infer success from action history alone.\n"
            "- If action history and screenshot conflict, prioritize the screenshot.\n\n"
            "Output policy:\n"
            '- Return {"status":"success",...} only if the evidence_observation directly supports that the subtask goal is achieved.\n'
            '- Return {"status":"failure",...} if the evidence_observation directly contradicts the goal, or the action history clearly shows the subtask went off target.\n'
            '- Return {"status":"uncertain",...} if the evidence is insufficient to confirm success or failure.\n'
            '- memory_eligible=true means this trajectory may be worth saving for future memory; it does not decide task success.\n\n'
            "Return exactly one JSON object with keys:\n"
            '- "status": "success" | "failure" | "uncertain"\n'
            '- "reason": string\n'
            '- "memory_eligible": boolean\n'
        )

    def _build_user_prompt(self, request: VerifierRequest) -> str:
        evidence_observation = request.evidence_observation or {}
        return (
            f"Subtask:\n{request.subtask}\n\n"
            f"Evidence source: {request.evidence_source}\n\n"
            "Before observation:\n"
            f"{self._format_observation(request.before_observation)}\n\n"
            "Evidence observation:\n"
            f"{self._format_observation(evidence_observation)}\n\n"
            "Action history for this subtask:\n"
            f"{self._format_history(request.action_history)}\n\n"
            "Retrieved memory context:\n"
            f"{self._truncate(request.memory_context.strip(), self.config.max_memory_context_chars) or 'None'}\n\n"
            "Judge whether the subtask is already completed."
        )

    def _format_observation(self, observation: Dict[str, Any]) -> str:
        if not observation:
            return "None"
        ui_elements = observation.get("ui_elements") or []
        ui_json = self._truncate(
            json.dumps(ui_elements, ensure_ascii=False, indent=2),
            self.config.max_ui_json_chars,
        )
        return (
            f"- Foreground package: {observation.get('foreground_package') or 'Unknown'}\n"
            f"- Dominant visible UI package: {observation.get('app_name') or 'Unknown'}\n"
            f"- Current activity: {observation.get('current_activity') or 'Unknown'}\n"
            f"- Visible UI count: {observation.get('visible_ui_count', 0)}\n"
            f"- Clickable UI count: {observation.get('clickable_ui_count', 0)}\n"
            f"- Non-system UI count: {observation.get('non_system_ui_count', 0)}\n"
            f"- Observation consistency: {observation.get('observation_consistency') or 'Unknown'}\n"
            f"- Observation warning: {observation.get('observation_warning') or 'None'}\n"
            f"- UI description:\n{observation.get('ui_description') or 'None'}\n"
            f"- UI elements JSON:\n{ui_json}"
        )

    def _format_history(self, action_history: List[Dict[str, Any]]) -> str:
        if not action_history:
            return "None"
        lines: list[str] = []
        for index, item in enumerate(action_history[-self.config.max_history_items :], start=1):
            lines.append(
                f"{index}. status={item.get('status') or 'Unknown'}; "
                f"reason={item.get('reason') or 'None'}; "
                f"action={json.dumps(item.get('action') or {}, ensure_ascii=False)}; "
                f"summary={item.get('summary') or 'None'}; "
                f"error={item.get('error') or 'None'}"
            )
        return "\n".join(lines)

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 20] + "\n...[truncated]"


def _parse_status(value: Any) -> VerifierStatus:
    status = str(value or "").strip().lower()
    if status not in {"success", "failure", "uncertain"}:
        raise ValueError("verifier.status must be one of success/failure/uncertain.")
    return status  # type: ignore[return-value]


def extract_json_object(text: str) -> Optional[dict]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value
    balanced = _extract_balanced_json_object(cleaned)
    if balanced is None:
        return None
    try:
        value = json.loads(balanced)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _extract_balanced_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
