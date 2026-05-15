from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class PlannerSubtask:
    precondition: str
    goal: str
    reason: str
    agent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_prompt_text(self) -> str:
        return f"Precondition: {self.precondition}\nGoal: {self.goal}"

    def memory_key_text(self) -> str:
        return f"Precondition: {self.precondition}\nGoal: {self.goal}"

    @property
    def task(self) -> str:
        return f"Precondition: {self.precondition} Goal: {self.goal}"


@dataclass
class PlannerResult:
    is_goal_complete: bool
    completion_message: str = ""
    subtasks: List[PlannerSubtask] = field(default_factory=list)
    raw_response: str = ""
    parse_error: Optional[str] = None
    repaired_parse: bool = False
    repair_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_goal_complete": self.is_goal_complete,
            "completion_message": self.completion_message,
            "subtasks": [subtask.to_dict() for subtask in self.subtasks],
            "raw_response": self.raw_response,
            "parse_error": self.parse_error,
            "repaired_parse": self.repaired_parse,
            "repair_reason": self.repair_reason,
        }


@dataclass
class PlannerConfig:
    max_subtasks: int = 5
    max_ui_elements: int = 50
    max_history_items: int = 20
    max_memory_context_chars: int = 6000
    max_ui_json_chars: int = 12000
    temperature: float = 0.0
    default_actor_name: str = "android_actor"


class LLMClient(Protocol):
    def generate(self, messages: List[Dict[str, Any]], temperature: float = 0.0) -> str:
        """Generate a planner response from chat messages."""


class AndroidTaskPlanner:
    """Short-horizon planner shared by Baseline B and DMS."""

    def __init__(self, llm_client: LLMClient, config: Optional[PlannerConfig] = None) -> None:
        self.llm_client = llm_client
        self.config = config or PlannerConfig()

    def plan(
        self,
        user_goal: str,
        observation: Dict[str, Any],
        task_history: Optional[List[Dict[str, Any]]] = None,
        memory_context: str = "",
    ) -> PlannerResult:
        task_history = task_history or []
        messages = self.build_messages(
            user_goal=user_goal,
            observation=observation,
            task_history=task_history,
            memory_context=memory_context,
        )
        raw_response = self.llm_client.generate(
            messages=messages,
            temperature=self.config.temperature,
        )
        return self.parse_response(raw_response)

    def build_messages(
        self,
        user_goal: str,
        observation: Dict[str, Any],
        task_history: List[Dict[str, Any]],
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": self._build_system_prompt()},
            {
                "role": "user",
                "content": self._build_user_content(
                    user_goal=user_goal,
                    observation=observation,
                    task_history=task_history,
                    memory_context=memory_context,
                ),
            },
        ]

    def messages_to_jsonable(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return planner messages in a JSON-serializable structure."""
        return json.loads(json.dumps(messages, ensure_ascii=False))

    @staticmethod
    def extract_user_text_prompt(messages: List[Dict[str, Any]]) -> str:
        """Extract the planner user text prompt from a multimodal message list."""
        for message in messages:
            if message.get("role") != "user":
                continue

            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        return str(item.get("text", ""))
        return ""

    def parse_response(self, raw_response: str) -> PlannerResult:
        payload = extract_json_object(raw_response)
        repaired_parse = False
        repair_reason: str | None = None
        if payload is None:
            repaired_payload, repair_reason = repair_planner_payload(raw_response)
            if repaired_payload is None:
                return PlannerResult(
                    is_goal_complete=False,
                    raw_response=raw_response,
                    parse_error="Failed to parse planner JSON.",
                )
            payload = repaired_payload
            repaired_parse = True

        tool = payload.get("tool")
        if tool == "complete_goal":
            return PlannerResult(
                is_goal_complete=True,
                completion_message=str(payload.get("message", "")).strip(),
                raw_response=raw_response,
                repaired_parse=repaired_parse,
                repair_reason=repair_reason,
            )

        if tool not in {"set_tasks", "set_tasks_with_agents"}:
            return PlannerResult(
                is_goal_complete=False,
                raw_response=raw_response,
                parse_error=f"Unsupported planner tool: {tool!r}",
                repaired_parse=repaired_parse,
                repair_reason=repair_reason,
            )

        tasks = payload.get("tasks")
        if tasks is None:
            tasks = payload.get("task_assignments")
        if not isinstance(tasks, list) or not tasks:
            return PlannerResult(
                is_goal_complete=False,
                raw_response=raw_response,
                parse_error="Planner tasks must be a non-empty list.",
                repaired_parse=repaired_parse,
                repair_reason=repair_reason,
            )

        subtasks: list[PlannerSubtask] = []
        for index, item in enumerate(tasks):
            if not isinstance(item, dict):
                return PlannerResult(
                    is_goal_complete=False,
                    raw_response=raw_response,
                    parse_error=f"Planner task at index {index} is not an object.",
                    repaired_parse=repaired_parse,
                    repair_reason=repair_reason,
                )

            task_text = item.get("task")
            reason = item.get("reason")
            if not isinstance(task_text, str) or not task_text.strip():
                return PlannerResult(
                    is_goal_complete=False,
                    raw_response=raw_response,
                    parse_error=f"Planner task at index {index} is missing 'task'.",
                    repaired_parse=repaired_parse,
                    repair_reason=repair_reason,
                )
            if not isinstance(reason, str) or not reason.strip():
                return PlannerResult(
                    is_goal_complete=False,
                    raw_response=raw_response,
                    parse_error=f"Planner task at index {index} is missing 'reason'.",
                    repaired_parse=repaired_parse,
                    repair_reason=repair_reason,
                )

            parsed = parse_precondition_goal(task_text)
            if parsed is None:
                return PlannerResult(
                    is_goal_complete=False,
                    raw_response=raw_response,
                    parse_error=(
                        f"Planner task at index {index} must use "
                        "'Precondition: ... Goal: ...' format."
                    ),
                    repaired_parse=repaired_parse,
                    repair_reason=repair_reason,
                )

            precondition, goal = parsed
            subtasks.append(
                PlannerSubtask(
                    precondition=precondition,
                    goal=goal,
                    reason=reason.strip(),
                    agent=item.get("agent") or self.config.default_actor_name,
                )
            )

        return PlannerResult(
            is_goal_complete=False,
            subtasks=subtasks,
            raw_response=raw_response,
            repaired_parse=repaired_parse,
            repair_reason=repair_reason,
        )

    def _build_system_prompt(self) -> str:
        return (
            "You are an Android Task Planner. Your job is to create short, "
            f"functional plans (1-{self.config.max_subtasks} steps) to achieve "
            "a user's goal on an Android device.\n\n"
            "Inputs you receive:\n"
            "1. The user's overall goal.\n"
            "2. The current device state:\n"
            "   - A screenshot of the current screen.\n"
            "   - A labeled screenshot with indexed UI elements.\n"
            "   - JSON data of visible UI elements.\n"
            "   - The current visible Android activity.\n"
            "3. Complete task history for the current session.\n"
            "4. Optional retrieved memory context from previous trials.\n\n"
            "Your task:\n"
            f"- Devise the next 1-{self.config.max_subtasks} functional steps.\n"
            "- Focus on what to achieve, not how to click or type.\n"
            "- Planning fewer steps at a time improves accuracy because the state can change.\n\n"
            "Step format:\n"
            "- Each step must be a functional goal.\n"
            "- Each Goal must be one short natural-language description of a small objective.\n"
            "- Completing each Goal must produce a verifiable UI state change.\n"
            f"- Each Goal should usually take about 2-6 atomic actions, not one trivial click and not a long multi-stage workflow.\n"
            "- Do not describe the Goal as a low-level operation such as tap, click, input, type, swipe, scroll, or press unless that action itself is the user's goal.\n"
            "- Use 'Precondition: ... Goal: ...' for every step.\n"
            "- For the first step, use 'Precondition: None. Goal: ...' if needed.\n\n"
            "Examples:\n"
            "- Good: 'Open the Phone app.'\n"
            "- Good: 'Navigate to the contacts section.'\n"
            "- Good: 'Start creating a new contact.'\n"
            "- Bad: 'Tap the Phone app icon.'\n"
            "- Bad: 'Click the Contacts tab.'\n"
            "- Bad: 'Tap the Create new contact button.'\n\n"
            "Output:\n"
            "If the overall goal is achieved, return only:\n"
            '{"tool":"complete_goal","message":"..."}\n\n'
            "Otherwise return only:\n"
            '{"tool":"set_tasks","tasks":[{"task":"Precondition: ... Goal: ...","reason":"..."}]}\n\n'
            "Constraints:\n"
            "- Do not output low-level actions such as tap, swipe, scroll, press key, or input text.\n"
            "- Do not output Python code.\n"
            "- Return valid JSON only.\n"
            "- After the planned steps are executed, you will be called again with the new device state."
        )

    def _build_user_content(
        self,
        user_goal: str,
        observation: Dict[str, Any],
        task_history: List[Dict[str, Any]],
        memory_context: str,
    ) -> List[Dict[str, Any]]:
        prompt = self._build_user_prompt(
            user_goal=user_goal,
            observation=observation,
            task_history=task_history,
            memory_context=memory_context,
        )

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        screenshot_b64 = observation.get("screenshot_b64")
        labeled_screenshot_b64 = observation.get("labeled_screenshot_b64")
        image_b64 = labeled_screenshot_b64 or screenshot_b64
        if image_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                }
            )
        return content

    def _build_user_prompt(
        self,
        user_goal: str,
        observation: Dict[str, Any],
        task_history: List[Dict[str, Any]],
        memory_context: str,
    ) -> str:
        current_activity = observation.get("current_activity") or "Unknown"
        app_name = observation.get("app_name") or "Unknown"
        screen_size = observation.get("screen_size") or {}
        ui_elements = observation.get("ui_elements") or []
        ui_description = observation.get("ui_description") or "No visible UI elements available."
        ui_json = self._format_ui_json(ui_elements)
        history_text = self._format_task_history(task_history)
        memory_text = self._truncate(memory_context.strip(), self.config.max_memory_context_chars)

        return (
            f"User overall goal:\n{user_goal}\n\n"
            "Current device state:\n"
            f"- Current app: {app_name}\n"
            f"- Current activity: {current_activity}\n"
            f"- Screen size: {json.dumps(screen_size, ensure_ascii=False)}\n"
            "- You are given both the raw screenshot and the labeled screenshot in this message.\n\n"
            f"Visible UI elements summary:\n{ui_description}\n\n"
            f"Visible UI elements JSON:\n{ui_json}\n\n"
            f"Complete task history:\n{history_text}\n\n"
            "Retrieved memory context:\n"
            f"{memory_text if memory_text else 'None'}\n\n"
            "Planner instruction:\n"
            "- Assess whether the overall user goal is already complete.\n"
            f"- If not complete, return the next 1-{self.config.max_subtasks} functional steps.\n"
            "- Every step must use 'Precondition: ... Goal: ...'.\n"
            "- Do not output low-level actions or atomic UI operations.\n"
            "- Each Goal should be one short natural-language small objective.\n"
            "- Each Goal should produce a verifiable UI state change after completion.\n"
            "- Each Goal should usually take around 2-6 atomic actions.\n"
            "- If the UI already shows an entry point, describe the state to reach, not the tap itself.\n"
            "- For text-entry subtasks, describe the desired filled state, not actor protocol names like input_text or type.\n"
        )

    def _format_task_history(self, task_history: List[Dict[str, Any]]) -> str:
        if not task_history:
            return "No previous subtasks."

        stable_lines: list[str] = []
        warning_lines: list[str] = []
        recent_items = self._compress_history_items(task_history[-self.config.max_history_items :])
        for index, item in enumerate(recent_items, start=1):
            task = str(item.get("task") or item.get("subtask") or "").strip() or "Unknown task"
            status = str(item.get("status") or item.get("result") or "unknown").strip()
            reason = str(item.get("reason") or item.get("note") or "").strip()
            line = f"{index}. task={task}; status={status}"
            if reason:
                line += f"; reason={reason}"
            if item.get("observation_unreliable_context") or status == "warning":
                warning_lines.append(line)
            else:
                stable_lines.append(line)
        lines: list[str] = []
        if stable_lines:
            lines.append("Stable progress history:")
            lines.extend(stable_lines)
        if warning_lines:
            lines.append("Unstable warning history:")
            lines.extend(warning_lines)
        return "\n".join(lines)

    @staticmethod
    def _compress_history_items(task_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compressed: list[Dict[str, Any]] = []
        for item in task_history:
            if (
                compressed
                and str(item.get("subtask") or item.get("task") or "") == str(compressed[-1].get("subtask") or compressed[-1].get("task") or "")
                and str(item.get("status") or "") == str(compressed[-1].get("status") or "")
                and str(item.get("error") or "").lower() == str(compressed[-1].get("error") or "").lower()
                and "recoverable actor schema mismatch" in str(item.get("summary") or "").lower()
            ):
                continue
            compressed.append(item)
        return compressed

    def _format_ui_json(self, ui_elements: List[Dict[str, Any]]) -> str:
        serialized = json.dumps(ui_elements, ensure_ascii=False, indent=2)
        return self._truncate(serialized, self.config.max_ui_json_chars)

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 20] + "\n...[truncated]"


def extract_json_object(text: str) -> Optional[dict]:
    """Extract the first JSON object from free-form model output."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    cleaned = cleaned.strip()

    # Some models wrap the JSON object in an extra trailing quote or return a
    # JSON-encoded string containing the object. Normalize those forms first.
    direct_candidates = [cleaned]
    if cleaned.endswith('"') and cleaned.count("{") and cleaned.count("}"):
        direct_candidates.append(cleaned[:-1].rstrip())

    for candidate in direct_candidates:
        parsed = _decode_possible_json(candidate)
        if isinstance(parsed, dict):
            return parsed

    balanced = _extract_balanced_json_object(cleaned)
    if balanced is None:
        return None

    parsed = _decode_possible_json(balanced)
    if isinstance(parsed, dict):
        return parsed
    return None


def _decode_possible_json(candidate: str) -> Any:
    """Decode either a JSON object or a JSON string containing a JSON object."""
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            nested = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(nested, dict):
            return nested

    return None


def _extract_balanced_json_object(text: str) -> Optional[str]:
    """Return the first balanced JSON object substring from arbitrary text."""
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


def parse_precondition_goal(task_text: str) -> tuple[str, str] | None:
    """Parse 'Precondition: ... Goal: ...' task strings."""
    match = re.match(
        r"^\s*Precondition\s*:\s*(?P<precondition>.*?)\s+Goal\s*:\s*(?P<goal>.+?)\s*$",
        task_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None

    precondition = match.group("precondition").strip()
    goal = match.group("goal").strip()
    if not precondition or not goal:
        return None
    return precondition, goal


def repair_planner_payload(raw_response: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = raw_response.strip()
    tool_match = re.search(r'"tool"\s*:\s*"([^"]+)"', cleaned)
    if not tool_match:
        return None, None
    tool = tool_match.group(1)
    if tool not in {"set_tasks", "set_tasks_with_agents", "complete_goal"}:
        return None, None
    if tool == "complete_goal":
        message_match = re.search(r'"message"\s*:\s*"([^"]*)"', cleaned, flags=re.DOTALL)
        if not message_match:
            return None, None
        return {"tool": tool, "message": message_match.group(1)}, "Recovered complete_goal payload from near-JSON output."

    items: list[dict[str, Any]] = []
    pair_pattern = re.compile(
        r'"task"\s*:\s*"(?P<task>.+?)(?:"\s*,\s*"reason"\s*:\s*"|;\s*reason"\s*:\s*")(?P<reason>.+?)"',
        flags=re.DOTALL,
    )
    for match in pair_pattern.finditer(cleaned):
        items.append(
            {
                "task": match.group("task").strip(),
                "reason": match.group("reason").strip(),
            }
        )
    if not items:
        return None, None
    return {"tool": tool, "tasks": items}, "Recovered tasks payload from near-JSON output."
