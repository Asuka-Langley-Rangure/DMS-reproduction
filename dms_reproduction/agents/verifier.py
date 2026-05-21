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
    prompt_profile: Literal["self_written_json", "paper_history_first"] = "self_written_json"


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
        if self.config.prompt_profile == "paper_history_first":
            return [
                {
                    "role": "system",
                    "content": "You are an Android subtask verifier. Return only the final JSON verdict.",
                },
                {"role": "user", "content": self._build_example_prompt(request)},
            ]
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
            status = _parse_verifier_status(parsed)
            reason = str(parsed.get("reason", "")).strip() or "No verifier reason provided."
            status, reason = self._apply_local_verification_vetoes(
                request=request,
                status=status,
                reason=reason,
            )
            memory_eligible = _derive_memory_eligible(parsed, status)
            return VerifierResult(
                status=status,
                reason=reason,
                memory_eligible=memory_eligible,
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

    def _format_observation_compact(self, observation: Dict[str, Any]) -> str:
        if not observation:
            return "None"

        labels = []
        for element in observation.get("ui_elements") or []:
            label = (
                element.get("text")
                or element.get("content_description")
                or element.get("resource_name")
                or ""
            )
            label = str(label).strip()
            if label:
                labels.append(label)

        unique_labels = []
        seen = set()
        for label in labels:
            key = label.lower()
            if key not in seen:
                unique_labels.append(label)
                seen.add(key)

        return (
            f"- Foreground package: {observation.get('foreground_package') or 'Unknown'}\n"
            f"- App name: {observation.get('app_name') or 'Unknown'}\n"
            f"- Current activity: {observation.get('current_activity') or 'Unknown'}\n"
            f"- Observation consistency: {observation.get('observation_consistency') or 'Unknown'}\n"
            f"- Observation warning: {observation.get('observation_warning') or 'None'}\n"
            f"- Visible labels excerpt: {', '.join(unique_labels[:40]) or 'None'}"
        )

    def _build_user_prompt(self, request: VerifierRequest) -> str:
        evidence_observation = request.evidence_observation or {}
        return (
            f"Subtask:\n{request.subtask}\n\n""Critical verification rules:\n"
            "- Verify ONLY whether the Goal of the subtask is achieved.\n"
            "- Do NOT mark success merely because the Precondition is true.\n"
            "- Do NOT weaken or rewrite the Goal using the actor's reason.\n"
            "- If the Goal is to open an app, success requires the target app to be in foreground.\n"
            "- A visible launcher icon or actionable app shortcut is not enough.\n" 
            "- If before_observation and evidence_observation show the same app, activity, and UI, then a navigation/open/reach/toggle goal is NOT successful unless the Goal was already satisfied before.\n"
            "- Actor history is supporting evidence only. The final evidence_observation is the ground truth.\n"
            "- Return success only when the evidence directly supports the Goal state.\n\n"
            f"Evidence source: {request.evidence_source}\n\n"
            "Before observation summary:\n"
            f"{self._format_observation_compact(request.before_observation)}\n\n"
            "Evidence observation summary:\n"
            f"{self._format_observation_compact(evidence_observation)}\n\n"
            "Action history for this subtask:\n"
            f"{self._format_history(request.action_history)}\n\n"
            "Retrieved memory context:\n"
            f"{self._truncate(request.memory_context.strip(), self.config.max_memory_context_chars) or 'None'}\n\n"
            "Judge whether the subtask is already completed."
        )
    
    def _build_example_prompt(self, request: VerifierRequest) -> str:
        evidence_observation = request.evidence_observation or {}
        request_context = "\n\n".join(
            [
                f"Current Subtask:\n{request.subtask}",
                f"Evidence Source:\n{request.evidence_source}",
                f"Execution History:\n{self._format_history(request.action_history)}",
                f"Before Observation:\n{self._format_observation(request.before_observation)}",
                f"Evidence Observation:\n{self._format_observation(evidence_observation)}",
                "Retrieved Memory Context:\n"
                f"{self._truncate(request.memory_context.strip(), self.config.max_memory_context_chars) or 'None'}",
            ]
        )

        role = (
            "Role: You are an expert Android Task Verifier. "
            "Your job is to determine if the agent's execution history "
            "successfully achieved the user's goal."
        )

        input_info = """
    Input Information:

    1. Original Goal: The user's original objective.

    2. Execution History: The (Thought, json) steps the agent claims it just performed.
    This is your PRIMARY source of truth.

    3. Final Screenshot: The ground truth screenshot.
    This is your SECONDARY check for contradictions.
    """.strip()

        verification_logic = """
    YOUR VERIFICATION LOGIC (History-First):

    1. Analyze History (Trust):
    Read the Execution History. Did the agent perform the logical actions required to complete the Original Goal?
    For example, for "Save recording," did the agent tap("Save")?

    2. Assume Success:
    If the history looks correct, your default verdict is: {"verified_success": true}

    3. Visual Veto (Contradiction Check):
    Now, look at the Final Screenshot. Does this screenshot explicitly contradict the agent's claim of success?
    """.strip()

        contradiction_rules = """
    Contradiction examples that should lead to failure:
    - The screenshot shows an error message, such as "Password incorrect".
    - The screenshot shows the agent is in the wrong application.
    - The goal was "Dismiss the OK dialog", but the screenshot clearly shows the OK dialog is still visible.

    No-contradiction examples that should lead to success:
    - The goal was "Dismiss the OK dialog", and the screenshot shows the dialog is gone.
    - The goal was "Click the Save button", and the screenshot shows the app has moved to a different screen.
    """.strip()

        key_rule = """
    Key Rule:
    You must default to True, meaning success, if the history is sound AND the screenshot does not provide strong, undeniable proof of failure.
    """.strip()

        output_format = """
    Output Format:
    Respond ONLY with the JSON object:
    {"verified_success": <bool>, "reason": "<string>"}
    """.strip()

        return "\n\n".join([
            request_context,
            role,
            input_info,
            verification_logic,
            contradiction_rules,
            key_rule,
            output_format,
        ])

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

    def _apply_local_verification_vetoes(
        self,
        *,
        request: VerifierRequest,
        status: VerifierStatus,
        reason: str,
    ) -> tuple[VerifierStatus, str]:
        if status != "success":
            return status, reason

        goal_type = _classify_subtask_goal(request.subtask)
        evidence_observation = request.evidence_observation or {}
        if goal_type == "app_launch" and _should_veto_app_launch_success(request, evidence_observation):
            return (
                "failure",
                "Execution history does not show a real app launch and the final observation still does not show the target app in foreground.",
            )
        if goal_type == "navigation" and _should_veto_navigation_success(request, evidence_observation):
            return (
                "failure",
                "The target entry is visible, but there is no post-action evidence that the requested destination or section was actually reached.",
            )
        if goal_type == "toggle" and _should_veto_toggle_success(request, evidence_observation):
            return (
                "failure",
                "The control was acted on, but the final observation does not show reliable evidence that the requested control state actually changed.",
            )
        if status == "success" and _should_veto_no_progress_success(request, evidence_observation):
            return (
                "failure",
                "The evidence observation is unchanged from before, and the subtask Goal state was not achieved.",
            )
        return status, reason


def _parse_status(value: Any) -> VerifierStatus:
    status = str(value or "").strip().lower()
    if status not in {"success", "failure", "uncertain"}:
        raise ValueError("verifier.status must be one of success/failure/uncertain.")
    return status  # type: ignore[return-value]


def _parse_verifier_status(parsed: Dict[str, Any]) -> VerifierStatus:
    if "status" in parsed:
        return _parse_status(parsed.get("status"))
    if "verified_success" in parsed:
        verified_success = parsed.get("verified_success")
        if isinstance(verified_success, bool):
            return "success" if verified_success else "failure"
        raise ValueError("verifier.verified_success must be a boolean.")
    raise ValueError("verifier response must include either status or verified_success.")


def _derive_memory_eligible(parsed: Dict[str, Any], status: VerifierStatus) -> bool:
    if status != "success":
        return False
    if "memory_eligible" in parsed:
        return bool(parsed.get("memory_eligible"))
    return status == "success"


def _classify_subtask_goal(subtask: str) -> str:
    parsed = _parse_subtask_goal_text(subtask)
    lowered = parsed.lower()
    if re.search(r"\b(open|launch)\b.+\bapp\b", lowered):
        return "app_launch"
    if re.search(r"\b(turn off|turn on|toggle|switch off|switch on|enable|disable)\b", lowered):
        return "toggle"
    if re.search(r"\b(navigate|go to|open|access|reach|enter|switch to|click on)\b", lowered):
        return "navigation"
    if re.search(r"\b(type|enter|fill|input)\b", lowered):
        return "form_fill"
    if re.search(r"\b(save|submit|confirm)\b", lowered):
        return "submit"
    if re.search(r"\b(answer|report|tell|say)\b", lowered):
        return "qa"
    return "other"


def _parse_subtask_goal_text(subtask: str) -> str:
    match = re.search(r"Goal\s*:\s*(.+)$", subtask, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return str(subtask or "").strip()


def _extract_goal_app_name(subtask: str) -> str | None:
    goal = _parse_subtask_goal_text(subtask)
    match = re.search(r"\b(?:open|launch)\s+the\s+(.+?)\s+app\b", goal, flags=re.IGNORECASE)
    if not match:
        return None
    return re.sub(r"\s+", " ", match.group(1).strip().lower())


def _should_veto_app_launch_success(request: VerifierRequest, evidence_observation: Dict[str, Any]) -> bool:
    target_app = _extract_goal_app_name(request.subtask)
    if target_app is None:
        return False
    final_foreground = str(evidence_observation.get("foreground_package") or "").lower()
    final_app_name = str(evidence_observation.get("app_name") or "").lower()
    if target_app in final_foreground or target_app in final_app_name:
        return False
    if _history_contains_app_launch_evidence(request.action_history, target_app):
        return False
    return _history_is_non_progress_only(request.action_history)


def _history_contains_app_launch_evidence(action_history: List[Dict[str, Any]], target_app: str) -> bool:
    for item in action_history:
        action = item.get("action") or {}
        action_type = str(action.get("action_type") or "").strip().lower()
        app_name = str(action.get("app_name") or "").strip().lower()
        if action_type == "open_app" and target_app in app_name:
            return True
        summary = str(item.get("summary") or "").lower()
        reason = str(item.get("reason") or "").lower()
        if target_app in summary and re.search(r"\b(open|launch)\b", summary):
            return True
        if target_app in reason and re.search(r"\b(open|launch)\b", reason):
            return True
    return False


def _history_is_non_progress_only(action_history: List[Dict[str, Any]]) -> bool:
    if not action_history:
        return True
    for item in action_history:
        action = item.get("action") or {}
        action_type = str(action.get("action_type") or "").strip().lower()
        if action_type and action_type not in {"wait"}:
            return False
        if str(item.get("error") or "").strip():
            return False
    return True


def _should_veto_navigation_success(request: VerifierRequest, evidence_observation: Dict[str, Any]) -> bool:
    before = request.before_observation or {}
    before_activity = str(before.get("current_activity") or "")
    after_activity = str(evidence_observation.get("current_activity") or "")
    before_foreground = str(before.get("foreground_package") or "")
    after_foreground = str(evidence_observation.get("foreground_package") or "")
    if before_activity != after_activity or before_foreground != after_foreground:
        return False
    if any(str(item.get("error") or "").strip() for item in request.action_history):
        return True
    if _history_is_non_progress_only(request.action_history):
        return True
    return _goal_target_still_only_visible(request.subtask, before, evidence_observation)


def _goal_target_still_only_visible(subtask: str, before: Dict[str, Any], after: Dict[str, Any]) -> bool:
    goal = _parse_subtask_goal_text(subtask)
    quoted = re.search(r"'([^']+)'|\"([^\"]+)\"", goal)
    target = quoted.group(1) if quoted and quoted.group(1) else quoted.group(2) if quoted else None
    if not target:
        for marker in ("to ", "into ", "on ", "the "):
            if marker in goal.lower():
                target = goal.split(marker, 1)[-1].strip(" .")
                break
    if not target:
        return False
    before_desc = str(before.get("ui_description") or "").lower()
    after_desc = str(after.get("ui_description") or "").lower()
    target_lower = target.lower()
    return target_lower in before_desc and target_lower in after_desc


def _should_veto_toggle_success(request: VerifierRequest, evidence_observation: Dict[str, Any]) -> bool:
    before = request.before_observation or {}
    if _has_toggle_state_change(before, evidence_observation):
        return False
    return True

def _should_veto_no_progress_success(request: VerifierRequest, evidence_observation: dict[str, Any]) -> bool:
    before = request.before_observation or {}
    goal = _parse_subtask_goal_text(request.subtask).lower()

    same_screen = (
        before.get("foreground_package") == evidence_observation.get("foreground_package")
        and before.get("current_activity") == evidence_observation.get("current_activity")
        and before.get("ui_description") == evidence_observation.get("ui_description")
    )

    if not same_screen:
        return False

    progress_goals = [
        "open", "launch", "reach", "navigate", "turn on", "turn off",
        "enable", "disable", "set", "save", "delete", "create", "edit"
    ]
    return any(token in goal for token in progress_goals)


def _has_toggle_state_change(before: Dict[str, Any], after: Dict[str, Any]) -> bool:
    before_states = _collect_checkable_states(before)
    after_states = _collect_checkable_states(after)
    if not before_states or not after_states:
        return False
    for key, before_checked in before_states.items():
        after_checked = after_states.get(key)
        if after_checked is None:
            continue
        if before_checked != after_checked:
            return True
    return False


def _collect_checkable_states(observation: Dict[str, Any]) -> Dict[tuple[str, str, str, str], bool]:
    ui_elements = observation.get("ui_elements") or []
    states: Dict[tuple[str, str, str, str], bool] = {}
    for element in ui_elements:
        raw = element.get("raw") or {}
        if raw.get("is_checked") is None:
            continue
        key = (
            str(element.get("class_name") or ""),
            str(element.get("resource_name") or ""),
            str(element.get("content_description") or ""),
            str(element.get("text") or ""),
        )
        states[key] = bool(raw.get("is_checked"))
    return states


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
