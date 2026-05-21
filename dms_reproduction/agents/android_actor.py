from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol

try:
    from android_world.env import json_action
except ImportError:  # pragma: no cover - fallback for local package layout
    from android_world.android_world.env import json_action


ActorStatus = Literal[
    "completed",
    "stopped",
    "infeasible",
    "step_limit",
    "parse_error",
    "execution_error",
]

SYSTEM_UI_PACKAGE = "com.android.systemui"


@dataclass
class ActorConfig:
    max_history_items: int = 8
    max_steps: int = 8
    max_ui_json_chars: int = 12000
    max_memory_context_chars: int = 6000
    temperature: float = 0.0
    wait_after_action_seconds: float = 0.0
    prompt_profile: Literal[
        "generic_self_written",
        "generic_paper",
        "legacy_contact_tuned",
    ] = "generic_self_written"


@dataclass
class ActorRequest:
    subtask: str
    observation: Dict[str, Any]
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    memory_context: str = ""


@dataclass
class ActorAction:
    action_type: str

    def to_payload(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class StatusAction(ActorAction):
    goal_status: Literal["complete", "infeasible"]
    message: str = ""

    def __init__(self, goal_status: Literal["complete", "infeasible"], message: str = "") -> None:
        super().__init__("status")
        self.goal_status = goal_status
        self.message = message


@dataclass
class AnswerAction(ActorAction):
    text: str

    def __init__(self, text: str) -> None:
        super().__init__("answer")
        self.text = text


@dataclass
class ClickAction(ActorAction):
    index: int | None = None
    x: int | None = None
    y: int | None = None

    def __init__(self, index: int | None = None, x: int | None = None, y: int | None = None) -> None:
        super().__init__("click")
        self.index = index
        self.x = x
        self.y = y


@dataclass
class LongPressAction(ActorAction):
    index: int | None = None
    x: int | None = None
    y: int | None = None

    def __init__(self, index: int | None = None, x: int | None = None, y: int | None = None) -> None:
        super().__init__("long_press")
        self.index = index
        self.x = x
        self.y = y


@dataclass
class InputTextAction(ActorAction):
    index: int
    text: str
    clear_text: bool | None = None

    def __init__(self, index: int, text: str, clear_text: bool | None = None) -> None:
        super().__init__("input_text")
        self.index = index
        self.text = text
        self.clear_text = clear_text


@dataclass
class KeyboardEnterAction(ActorAction):
    def __init__(self) -> None:
        super().__init__("keyboard_enter")


@dataclass
class NavigateHomeAction(ActorAction):
    def __init__(self) -> None:
        super().__init__("navigate_home")


@dataclass
class NavigateBackAction(ActorAction):
    def __init__(self) -> None:
        super().__init__("navigate_back")


@dataclass
class ScrollAction(ActorAction):
    direction: Literal["up", "down", "left", "right"]
    index: int | None = None

    def __init__(self, direction: Literal["up", "down", "left", "right"], index: int | None = None) -> None:
        super().__init__("scroll")
        self.direction = direction
        self.index = index


@dataclass
class OpenAppAction(ActorAction):
    app_name: str

    def __init__(self, app_name: str) -> None:
        super().__init__("open_app")
        self.app_name = app_name


@dataclass
class WaitAction(ActorAction):
    def __init__(self) -> None:
        super().__init__("wait")


@dataclass
class ActorStepResult:
    step_id: int
    reason: str
    action: ActorAction | None
    original_action: Dict[str, Any] | None
    normalized_action: Dict[str, Any] | None
    action_normalization_applied: bool
    normalization_reason: str | None
    corrected_action: Dict[str, Any] | None
    correction_reason: str | None
    messages: List[Dict[str, Any]]
    prompt_text: str
    raw_response: str
    parse_error: str | None
    execution_error: str | None
    step_verification_triggered: bool = False
    step_verification: Dict[str, Any] | None = None
    step_verification_reason: str | None = None
    heuristic_stop_triggered: bool = False
    heuristic_stop_reason: str | None = None
    before_observation: Dict[str, Any] = field(default_factory=dict)
    after_observation: Dict[str, Any] | None = None
    summary: str = ""
    done: bool = False
    done_reason: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "reason": self.reason,
            "action": self.action.to_payload() if self.action else None,
            "original_action": self.original_action,
            "normalized_action": self.normalized_action,
            "action_normalization_applied": self.action_normalization_applied,
            "normalization_reason": self.normalization_reason,
            "corrected_action": self.corrected_action,
            "correction_reason": self.correction_reason,
            "messages": self.messages,
            "prompt_text": self.prompt_text,
            "raw_response": self.raw_response,
            "parse_error": self.parse_error,
            "execution_error": self.execution_error,
            "step_verification_triggered": self.step_verification_triggered,
            "step_verification": self.step_verification,
            "step_verification_reason": self.step_verification_reason,
            "heuristic_stop_triggered": self.heuristic_stop_triggered,
            "heuristic_stop_reason": self.heuristic_stop_reason,
            "before_observation": self.before_observation,
            "after_observation": self.after_observation,
            "summary": self.summary,
            "done": self.done,
            "done_reason": self.done_reason,
        }

    def to_history_item(self, subtask: str, status: str, error: str = "") -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "subtask": subtask,
            "reason": self.reason,
            "action": self.action.to_payload() if self.action else None,
            "summary": self.summary,
            "status": status,
            "error": error,
        }


@dataclass
class ActorRunResult:
    status: ActorStatus
    steps: List[ActorStepResult] = field(default_factory=list)
    final_observation: Dict[str, Any] | None = None
    completion_message: str = ""
    answer_text: str = ""
    last_action: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "steps": [step.to_dict() for step in self.steps],
            "final_observation": self.final_observation,
            "completion_message": self.completion_message,
            "answer_text": self.answer_text,
            "last_action": self.last_action,
        }


class LLMClient(Protocol):
    def generate(self, messages: List[Dict[str, Any]], temperature: float = 0.0) -> str:
        """Generate a raw actor response."""


class ObservationAdapter(Protocol):
    def capture_observation(
        self,
        env: Any,
        goal: str,
        *,
        step_id: int = 0,
        include_screenshots: bool = True,
    ) -> Dict[str, Any]:
        """Capture an observation from the environment."""


class AndroidActor:
    """ActorCode-aligned Android subtask executor backed by JSON actions."""

    def __init__(self, llm_client: LLMClient, config: Optional[ActorConfig] = None) -> None:
        self.llm_client = llm_client
        self.config = config or ActorConfig()

    def build_messages(self, request: ActorRequest) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_user_content(request)},
        ]

    @staticmethod
    def extract_user_text_prompt(messages: List[Dict[str, Any]]) -> str:
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

    def messages_to_jsonable(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return json.loads(json.dumps(messages, ensure_ascii=False))

    @staticmethod
    def _extract_open_app_target(goal: str) -> str | None:
        text = goal.strip()

        # Open the Settings app
        match = re.search(r"\b(?:open|launch)\s+(?:the\s+)?(.+?)\s+app\b", text, flags=re.I)
        if match:
            app = match.group(1).strip()
            if app.lower() in {"android settings", "settings"}:
                return "Settings"
            return app

        # Special but still generic system app wording
        if re.search(r"\bsettings\b", text, flags=re.I) and re.search(r"\b(open|launch)\b", text, flags=re.I):
            return "Settings"

        return None

    @staticmethod
    def _maybe_plan_direct_open_app_action(subtask: str, observation: dict[str, Any]) -> ActorAction | None:
        goal = _extract_subtask_goal(subtask)
        target_app = AndroidActor._extract_open_app_target(goal)
        if not target_app:
            return None

        current_text = " ".join(
            [
                str(observation.get("foreground_package") or ""),
                str(observation.get("app_name") or ""),
                str(observation.get("current_activity") or ""),
            ]
        ).lower()

        if target_app.lower() in current_text:
            return None

        return OpenAppAction(target_app)


    def run_subtask(
        self,
        env: Any,
        request: ActorRequest,
        observation_adapter: ObservationAdapter,
        step_verifier: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> ActorRunResult:
        current_observation = request.observation
        initial_observation = request.observation
        history = list(request.action_history)
        steps: list[ActorStepResult] = []
        answer_text = ""
        direct_open_action = self._maybe_plan_direct_open_app_action(request.subtask, current_observation)

        if direct_open_action is not None:
            after_observation: Dict[str, Any] | None = None
            execution_error: str | None = None
            try:
                env.execute_action(to_json_action(direct_open_action))
                after_observation = observation_adapter.capture_observation(
                    env,
                    request.subtask,
                    step_id=1,
                    include_screenshots=True,
                )
                current_observation = after_observation
            except Exception as exc:  # pylint: disable=broad-except
                execution_error = str(exc)

            heuristic_stop_reason: str | None = None
            if after_observation is not None:
                heuristic_stop_reason = _should_stop_for_external_verification(
                    subtask=request.subtask,
                    action=direct_open_action,
                    before_observation=initial_observation,
                    after_observation=after_observation,
                )
                if heuristic_stop_reason:
                    after_observation = _resample_observation_after_high_impact_stop(
                        observation_adapter=observation_adapter,
                        env=env,
                        goal=request.subtask,
                        step_id=1,
                        first_after_observation=after_observation,
                    )
                    current_observation = after_observation
            completion_message = "Direct open_app executed for explicit app-launch goal; stop for external verification."
            step_status = "execution_error" if execution_error else "stopped"
            done_reason = "execution_error" if execution_error else "heuristic_stop_for_external_verification"
            summary = _build_rule_summary(
                subtask=request.subtask,
                action=direct_open_action,
                reason="Direct open_app constraint applied for explicit app-launch goal.",
                status=step_status,
                error=execution_error or "",
            )
            if not execution_error:
                summary += f" Heuristic stop: {heuristic_stop_reason or completion_message}"
            step = ActorStepResult(
                step_id=0,
                reason="Direct open_app constraint applied for explicit app-launch goal.",
                action=direct_open_action,
                original_action=None,
                normalized_action=None,
                action_normalization_applied=False,
                normalization_reason=None,
                corrected_action=None,
                correction_reason=None,
                messages=[],
                prompt_text="",
                raw_response="",
                parse_error=None,
                execution_error=execution_error,
                heuristic_stop_triggered=not execution_error,
                heuristic_stop_reason=heuristic_stop_reason or completion_message if not execution_error else None,
                before_observation=initial_observation,
                after_observation=after_observation,
                summary=summary,
                done=True,
                done_reason=done_reason,
            )
            steps.append(step)
            if execution_error:
                return ActorRunResult(
                    status="execution_error",
                    steps=steps,
                    final_observation=current_observation,
                    answer_text=answer_text,
                    last_action=direct_open_action.to_payload(),
                )
            return ActorRunResult(
                status="stopped",
                steps=steps,
                final_observation=current_observation,
                completion_message=completion_message,
                answer_text=answer_text,
                last_action=direct_open_action.to_payload(),
            )

        for step_id in range(self.config.max_steps):
            step_request = ActorRequest(
                subtask=request.subtask,
                observation=current_observation,
                action_history=history,
                memory_context=request.memory_context,
            )
            messages = self.build_messages(step_request)
            prompt_text = self.extract_user_text_prompt(messages)
            messages_jsonable = self.messages_to_jsonable(messages)
            raw_response = self.llm_client.generate(
                messages=messages,
                temperature=self.config.temperature,
            )

            reason, action_payload = parse_reason_action_output(raw_response)
            original_action = dict(action_payload) if isinstance(action_payload, dict) else None
            if action_payload is None:
                step = ActorStepResult(
                    step_id=step_id,
                    reason=reason,
                    action=None,
                    original_action=None,
                    normalized_action=None,
                    action_normalization_applied=False,
                    normalization_reason=None,
                    corrected_action=None,
                    correction_reason=None,
                    messages=messages_jsonable,
                    prompt_text=prompt_text,
                    raw_response=raw_response,
                    parse_error="Failed to parse actor action JSON.",
                    execution_error=None,
                    heuristic_stop_triggered=False,
                    heuristic_stop_reason=None,
                    before_observation=current_observation,
                    after_observation=None,
                    summary=_build_rule_summary(
                        subtask=request.subtask,
                        action=None,
                        reason=reason,
                        status="parse_error",
                        error="Failed to parse actor action JSON.",
                    ),
                    done=True,
                    done_reason="parse_error",
                )
                steps.append(step)
                return ActorRunResult(
                    status="parse_error",
                    steps=steps,
                    final_observation=current_observation,
                    answer_text=answer_text,
                )

            try:
                action, normalized_action, normalization_applied, normalization_reason, corrected_action, correction_reason = parse_actor_action(
                    action_payload,
                    current_observation,
                    reason=reason,
                    subtask=request.subtask,
                )
            except ValueError as exc:
                step = ActorStepResult(
                    step_id=step_id,
                    reason=reason,
                    action=None,
                    original_action=original_action,
                    normalized_action=None,
                    action_normalization_applied=False,
                    normalization_reason=None,
                    corrected_action=None,
                    correction_reason=None,
                    messages=messages_jsonable,
                    prompt_text=prompt_text,
                    raw_response=raw_response,
                    parse_error=str(exc),
                    execution_error=None,
                    heuristic_stop_triggered=False,
                    heuristic_stop_reason=None,
                    before_observation=current_observation,
                    after_observation=None,
                    summary=_build_rule_summary(
                        subtask=request.subtask,
                        action=None,
                        reason=reason,
                        status="parse_error",
                        error=str(exc),
                    ),
                    done=True,
                    done_reason="parse_error",
                )
                steps.append(step)
                return ActorRunResult(
                    status="parse_error",
                    steps=steps,
                    final_observation=current_observation,
                    answer_text=answer_text,
                )

            done_reason: str | None = None
            after_observation: Dict[str, Any] | None = None
            execution_error: str | None = None
            step_status = "progress"
            step_completion_message = ""
            heuristic_stop_triggered = False
            heuristic_stop_reason: str | None = None

            if isinstance(action, StatusAction):
                done_reason = "completed" if action.goal_status == "complete" else "infeasible"
                step_status = done_reason
                step_completion_message = action.message
            else:
                if isinstance(action, AnswerAction):
                    answer_text = action.text
                try:
                    env.execute_action(to_json_action(action))
                    after_observation = observation_adapter.capture_observation(
                        env,
                        request.subtask,
                        step_id=step_id + 1,
                        include_screenshots=True,
                    )
                    current_observation = after_observation
                except Exception as exc:  # pylint: disable=broad-except
                    execution_error = str(exc)
                    done_reason = "execution_error"
                    step_status = "execution_error"

            if (
                done_reason is None
                and after_observation is not None
                and action is not None
            ):
                heuristic_stop_reason = _should_stop_for_external_verification(
                    subtask=request.subtask,
                    action=action,
                    before_observation=step_request.observation,
                    after_observation=after_observation,
                )
                if heuristic_stop_reason:
                    after_observation = _resample_observation_after_high_impact_stop(
                        observation_adapter=observation_adapter,
                        env=env,
                        goal=request.subtask,
                        step_id=step_id + 1,
                        first_after_observation=after_observation,
                    )
                    current_observation = after_observation
                    heuristic_stop_triggered = True
                    done_reason = "heuristic_stop_for_external_verification"
                    step_status = "stopped"

            summary = _build_rule_summary(
                subtask=request.subtask,
                action=action,
                reason=reason,
                status=step_status,
                error=execution_error or "",
            )
            if normalization_applied and normalization_reason:
                summary += f" Recoverable actor schema mismatch handled locally. {normalization_reason}"
            if heuristic_stop_triggered and heuristic_stop_reason:
                summary += f" Heuristic stop: {heuristic_stop_reason}"
            if (
                step_request.observation.get("observation_consistency") == "unstable"
                and action.action_type in {"wait", "navigate_back"}
            ):
                summary += " requires_post_action_consistency_check=true."
            step = ActorStepResult(
                step_id=step_id,
                reason=reason,
                action=action,
                original_action=original_action,
                normalized_action=normalized_action,
                action_normalization_applied=normalization_applied,
                normalization_reason=normalization_reason,
                corrected_action=corrected_action,
                correction_reason=correction_reason,
                messages=messages_jsonable,
                prompt_text=prompt_text,
                raw_response=raw_response,
                parse_error=None,
                execution_error=execution_error,
                heuristic_stop_triggered=heuristic_stop_triggered,
                heuristic_stop_reason=heuristic_stop_reason,
                before_observation=step_request.observation,
                after_observation=after_observation,
                summary=summary,
                done=done_reason is not None,
                done_reason=done_reason,
            )
            steps.append(step)
            history.append(
                step.to_history_item(
                    subtask=request.subtask,
                    status=step_status,
                    error=execution_error or "",
                )
            )

            if done_reason == "completed":
                return ActorRunResult(
                    status="completed",
                    steps=steps,
                    final_observation=current_observation,
                    completion_message=step_completion_message,
                    answer_text=answer_text,
                    last_action=action.to_payload(),
                )
            if done_reason == "heuristic_stop_for_external_verification":
                return ActorRunResult(
                    status="stopped",
                    steps=steps,
                    final_observation=current_observation,
                    completion_message=heuristic_stop_reason or "",
                    answer_text=answer_text,
                    last_action=action.to_payload(),
                )
            if done_reason == "infeasible":
                return ActorRunResult(
                    status="infeasible",
                    steps=steps,
                    final_observation=current_observation,
                    completion_message=action.message if isinstance(action, StatusAction) else "",
                    answer_text=answer_text,
                    last_action=action.to_payload(),
                )
            if done_reason == "execution_error":
                return ActorRunResult(
                    status="execution_error",
                    steps=steps,
                    final_observation=current_observation,
                    answer_text=answer_text,
                    last_action=action.to_payload(),
                )

        return ActorRunResult(
            status="step_limit",
            steps=steps,
            final_observation=current_observation,
            answer_text=answer_text,
            last_action=steps[-1].action.to_payload() if steps and steps[-1].action else None,
        )

    def _build_system_prompt(self) -> str:
        if self.config.prompt_profile == "legacy_contact_tuned":
            return self._build_legacy_system_prompt()
        return self._build_generic_system_prompt()

    def _build_generic_system_prompt(self) -> str:
        return (
            "You are an Android Actor executing one GUI subtask at a time.\n\n"
            "Role:\n"
            "- Execute only the current subtask on the Android device.\n"
            "- Do not re-plan the overall task.\n"
            "- Output exactly one action per turn.\n"
            "- The Runner will capture the next observation and verify progress after your action.\n\n"

            "Hard action rules:\n"
            "- If the Goal is to open or launch a named app, use open_app with that app name.\n"
            "- The app icon does not need to be visible for open_app.\n"
            "- For 'Open the Settings app', output {\"action_type\":\"open_app\",\"app_name\":\"Settings\"}.\n"
            "- From the launcher/home screen, never use wait or navigate_back to search for an app.\n"
            "- Do not return infeasible only because an app icon is not visible; open_app can launch installed apps directly.\n"
            "- Use wait only after an action was just executed and the screen may still be loading. Do not use wait as the first action on a stable launcher/home screen.\n"
            "- If the subtask precondition is clearly unmet and cannot be repaired by one valid action, return status infeasible.\n"
            "- Return status complete only when the current subtask Goal is already satisfied in the current observation.\n\n"

            "GUI grounding rules:\n"
            "- Prefer exact indexed UI actions when the target is visible and actionable.\n"
            "- For click and long_press, prefer a clickable index.\n"
            "- If the target label is visible but no matching clickable index is exposed, use x/y coordinates on the visible target region.\n"
            "- If the target is a visible switch, checkbox, radio button, or icon button and no clickable index is exposed, use a coordinate click on the control itself.\n"
            "- Do not click a known non-clickable label when an actionable parent row, container, or coordinate click is needed.\n"
            "- If a previous action failed or produced no progress, do not repeat it unchanged.\n"
            "- For toggle/enable/disable subtasks, act on the target control once, then stop; do not keep tapping nearby UI.\n"
            "- If an action changes the screen, do not predict follow-up actions; the Runner will provide the next observation.\n"
            "- If the observation is unstable, degraded, or only shows system UI, prefer a cautious recovery action over guessing.\n\n"

            "Available actions:\n"
            '- {"action_type":"status","goal_status":"complete","message":"..."}\n'
            '- {"action_type":"status","goal_status":"infeasible","message":"..."}\n'
            '- {"action_type":"answer","text":"..."}\n'
            '- {"action_type":"click","index":<target_index>}\n'
            '- {"action_type":"click","x":<screen_x>,"y":<screen_y>}\n'
            '- {"action_type":"long_press","index":<target_index>}\n'
            '- {"action_type":"long_press","x":<screen_x>,"y":<screen_y>}\n'
            '- {"action_type":"input_text","index":<target_index>,"text":"...","clear_text":true|false}\n'
            '- {"action_type":"keyboard_enter"}\n'
            '- {"action_type":"navigate_home"}\n'
            '- {"action_type":"navigate_back"}\n'
            '- {"action_type":"scroll","direction":"up|down|left|right","index":<optional_target_index>}\n'
            '- {"action_type":"open_app","app_name":"..."}\n'
            '- {"action_type":"wait"}\n\n'

            "Output format:\n"
            "Reason: <brief rationale naming the grounded target or tool>\n"
            'Action: {"action_type":"..."}\n'
        )

    def _build_legacy_system_prompt(self) -> str:
        return (
            "You are an Android Actor executing one GUI subtask at a time.\n\n"
            "Role:\n"
            "- You are responsible for executing a single subtask on the current Android device.\n"
            "- You are not responsible for re-planning the overall task.\n"
            "- Use the current GUI state, recent execution history, and retrieved memory context to decide the next action.\n\n"
            "Decision policy:\n"
            "- Focus on the smallest necessary action that moves the current subtask forward.\n"
            "- Return status complete only when the current subtask goal itself is achieved.\n"
            "- Do not mark a navigation or discovery subtask complete just because the app is open.\n"
            "- Example: opening the Phone app is not the same as reaching the Contacts tab.\n"
            "- Before choosing an index, verify the exact numbered UI element and its properties.\n"
            "- If you mention a specific target like Phone or Contacts, the action index must match that exact visible element.\n"
            "- For text entry, use action_type input_text, not type, enter_text, fill_text, or set_text.\n"
            "- If the subtask asks you to fill a coherent form section, fill only the required related fields for that subtask and stop once those fields are complete.\n"
            "- If the current path is not feasible, end explicitly with infeasible.\n"
            "- If the observation is unstable, degraded, or only shows system UI, prefer wait or navigate_back over guessing.\n"
            "- Treat history and memory as decision evidence, not as logs.\n"
            "- Do not output code snippets or natural-language-only actions.\n\n"
            "Available actions:\n"
            '- {"action_type":"status","goal_status":"complete","message":"..."}\n'
            '- {"action_type":"status","goal_status":"infeasible","message":"..."}\n'
            '- {"action_type":"answer","text":"..."}\n'
            '- {"action_type":"click","index":<target_index>}\n'
            '- {"action_type":"click","x":<screen_x>,"y":<screen_y>}\n'
            '- {"action_type":"long_press","index":<target_index>}\n'
            '- {"action_type":"long_press","x":<screen_x>,"y":<screen_y>}\n'
            '- {"action_type":"input_text","index":<target_index>,"text":"...","clear_text":true|false}\n'
            '- {"action_type":"keyboard_enter"}\n'
            '- {"action_type":"navigate_home"}\n'
            '- {"action_type":"navigate_back"}\n'
            '- {"action_type":"scroll","direction":"up|down|left|right","index":<optional_target_index>}\n'
            '- {"action_type":"open_app","app_name":"..."}\n'
            '- {"action_type":"wait"}\n\n'
            "Output format:\n"
            "Reason: <brief rationale>\n"
            'Action: {"action_type":"..."}\n'
            "Return exactly one action per turn."
        )

    def _build_user_content(self, request: ActorRequest) -> List[Dict[str, Any]]:
        prompt = self._build_user_prompt(request)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        labeled_screenshot_b64 = request.observation.get("actor_labeled_screenshot_b64") or request.observation.get("labeled_screenshot_b64")
        screenshot_b64 = request.observation.get("screenshot_b64")
        image_b64 = labeled_screenshot_b64 or screenshot_b64
        if image_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                }
            )
        return content

    def _build_user_prompt(self, request: ActorRequest) -> str:
        if self.config.prompt_profile == "legacy_contact_tuned":
            return self._build_legacy_user_prompt(request)
        if self.config.prompt_profile == "generic_paper":
            return self._build_generic_example_prompt(request)
        return self._build_generic_user_prompt(request)

    def _build_generic_user_prompt(self, request: ActorRequest) -> str:
        observation = request.observation
        foreground_package = observation.get("foreground_package") or "Unknown"
        dominant_ui_package = observation.get("app_name") or "Unknown"
        observation_warning = observation.get("observation_warning")
        observation_consistency = observation.get("observation_consistency") or "stable"

        return (
            f"Subtask:\n{request.subtask}\n\n"

            "Current screen state:\n"
            f"- Foreground package: {foreground_package}\n"
            f"- Dominant visible UI package: {dominant_ui_package}\n"
            f"- Current activity: {observation.get('current_activity') or 'Unknown'}\n"
            f"- Screen size: {json.dumps(observation.get('screen_size') or {}, ensure_ascii=False)}\n"
            f"- Visible UI count: {observation.get('visible_ui_count', 0)}\n"
            f"- Clickable UI count: {observation.get('clickable_ui_count', 0)}\n"
            f"- Non-system UI count: {observation.get('non_system_ui_count', 0)}\n"
            f"- Observation consistency: {observation_consistency}\n"
            f"- Observation warning: {observation_warning or 'None'}\n\n"

            "Visible UI index table:\n"
            f"{self._format_ui_index_table(observation.get('ui_elements') or [])}\n\n"

            "Visible UI elements JSON:\n"
            f"{self._format_ui_json(observation.get('ui_elements') or [])}\n\n"

            "Recent execution history:\n"
            f"{self._format_history(request.action_history)}\n\n"

            "Retrieved memory context:\n"
            f"{self._truncate(request.memory_context.strip(), self.config.max_memory_context_chars) or 'None'}\n\n"

            "Decision reminders:\n"
            "- Execute the current subtask with exactly one valid action.\n"
            "- Use the current screen, action history, and memory as evidence.\n"
            "- Use status complete only if the current subtask Goal is already satisfied now.\n"
            "- Use status infeasible only if the precondition or target cannot be repaired by one valid action.\n"
            "- If the Goal is to open or launch a named app, use open_app. The app icon does not need to be visible.\n"
            "- For 'Open the Settings app', use {\"action_type\":\"open_app\",\"app_name\":\"Settings\"}.\n"
            "- From a stable launcher/home screen, do not use wait or navigate_back to find an app.\n"
            "- If the exact target word is not visible, look for a visible parent category or semantically related entry before scrolling.\n"
            "- For Wi-Fi goals inside Android Settings, 'Internet', 'Network & internet', or 'Networks available' are relevant entries that may lead to Wi-Fi controls.\n"
            "- If a relevant entry is visible, click that entry instead of scrolling.\n"
            "- Use scroll only when no visible entry, category, row, or control is semantically related to the subtask goal.\n"
            "- Do not use wait as the first action unless the screen is unstable, loading, or recovering from a just-executed action.\n"
            "- For indexed actions, use only indices from the current Visible UI index table.\n"
            "- If a target is visible but has no legal clickable index, use x/y coordinates on the visible target region.\n"
            "- Do not invent indices from the screenshot.\n"
            "- Do not output action_type swipe; use scroll instead.\n"
            "- After a meaningful app launch, navigation, form edit, or control click, output only that action and let the Runner verify the next screen.\n\n"

            "Output format:\n"
            "Reason: <brief rationale naming the grounded target or tool>\n"
            'Action: {"action_type":"..."}\n'
        )

    def _build_generic_example_prompt(self, request: ActorRequest) -> str:
        observation = request.observation
        foreground_package = observation.get("foreground_package") or "Unknown"
        dominant_ui_package = observation.get("app_name") or "Unknown"
        observation_warning = observation.get("observation_warning")
        observation_consistency = observation.get("observation_consistency") or "stable"
        memory_context = self._truncate(request.memory_context.strip(), self.config.max_memory_context_chars) or "None"
        return (
            "Current subtask:\n"
            f"{request.subtask}\n\n"
            "Current device state:\n"
            "- You are given one screenshot in this message with red boxes around retained UI elements. Final action indices must come from the Visible UI index table below.\n"
            f"- Foreground package: {foreground_package}\n"
            f"- Dominant visible UI package: {dominant_ui_package}\n"
            f"- Current activity: {observation.get('current_activity') or 'Unknown'}\n"
            f"- Screen size: {json.dumps(observation.get('screen_size') or {}, ensure_ascii=False)}\n"
            f"- Visible UI count: {observation.get('visible_ui_count', 0)}\n"
            f"- Clickable UI count: {observation.get('clickable_ui_count', 0)}\n"
            f"- Non-system UI count: {observation.get('non_system_ui_count', 0)}\n"
            f"- Observation consistency: {observation_consistency}\n"
            "\n"
            f"Observation warning:\n{observation_warning or 'None'}\n\n"
            f"Visible UI index table:\n{self._format_ui_index_table(observation.get('ui_elements') or [])}\n\n"
            f"Visible UI elements summary:\n{observation.get('ui_description') or 'No visible UI elements available.'}\n\n"
            f"Recent execution history:\n{self._format_history(request.action_history)}\n\n"
            "Retrieved memory context:\n"
            f"{memory_context}\n\n"
            "Your task:\n"
            "- Execute the current subtask on this Android screen.\n"
            "- Output exactly one valid GUI action for the current screen.\n"
            "- If the subtask precondition is clearly unmet on the current screen, return status infeasible.\n"
            "- If the subtask goal is already satisfied, return status complete.\n\n"
            "Acting rules:\n"
            "- Execute exactly one GUI action at a time. Do not implicitly add follow-up actions.\n"
            "- Base the decision on the current screen state, recent execution history, and retrieved memory context.\n"
            "- Before choosing an indexed action, first identify the visible UI element that best matches the Goal of the current subtask.\n"
            "- Treat the Goal as the primary grounding target. Use the Reason to justify the match, not to invent a different target.\n"
            "- If the Goal is to open or launch an app and the current foreground package is not that app, prefer open_app instead of wait.\n"
            "- Do not use wait on launcher or other static screens when a concrete app-launch action is available.\n"
            "- Only use indices that appear in the current Visible UI index table.\n"
            "- If a number does not appear in the current Visible UI index table, it is not a legal action index.\n"
            "- Do not copy or invent numbers from the screenshot. Use the screenshot for layout and target appearance only; use the text index table for the final action index.\n"
            "- Do not output action_type swipe; use scroll instead.\n"
            "- Do not choose an index only because it is nearby, visually salient, or contains a partially related word.\n"
            "- Do not click a non-clickable label. If the label names the target but only the parent row or container is clickable, choose that actionable row or container.\n"
            "- If the target label is visible but no matching clickable index is available in the observation, use a coordinate click on the visible target region instead of selecting a known non-clickable label index.\n"
            "- If the target is a visible control widget such as a switch, checkbox, radio button, or icon button, and no clickable index is exposed, use a coordinate click on the control itself.\n"
            "- If you can see the target but cannot map it to a legal index from the Visible UI index table, use open_app, x/y coordinates, or status infeasible instead of inventing a new index.\n"
            "- If the current subtask is to toggle, enable, or disable a visible control, do not continue tapping nearby UI after you have acted on the target control once.\n"
            "- After a meaningful control click, prefer to stop and let the next observation verify the state change.\n"
            "- Do not click an unrelated tab or container just because its wording looks partially related.\n"
            "- In the Reason, explicitly name the grounded target element before giving the action.\n"
            "- If a previous action failed or produced no progress, do not repeat it unchanged. Change strategy or return infeasible.\n"
            "- If an action causes a visible UI change, stop rather than predicting the next screen.\n"
            "- For indexed actions, the chosen index must match the exact element named in your reasoning.\n"
            "- If the observation is degraded or unstable, prefer a cautious recovery action such as wait or navigate_back over guessing.\n\n"
            "Output format:\n"
            "Reason: <brief rationale that names the grounded target element>\n"
            'Action: {"action_type":"..."}\n'
            "Return exactly one action in the required JSON action format."
        )

    def _build_legacy_user_prompt(self, request: ActorRequest) -> str:
        observation = request.observation
        foreground_package = observation.get("foreground_package") or "Unknown"
        dominant_ui_package = observation.get("app_name") or "Unknown"
        observation_warning = observation.get("observation_warning")
        observation_consistency = observation.get("observation_consistency") or "stable"
        form_fill_context = observation.get("form_fill_context") or {}
        form_fill_block = ""
        if form_fill_context:
            form_fill_block = (
                "Grouped form-fill constraints:\n"
                f"- Target fields only: {json.dumps(form_fill_context.get('target_fields') or [], ensure_ascii=False)}\n"
                f"- Expected values: {json.dumps(form_fill_context.get('expected_fields') or {}, ensure_ascii=False)}\n"
                f"- Current values: {json.dumps(form_fill_context.get('current_values') or {}, ensure_ascii=False)}\n"
                f"- Remaining fields: {json.dumps(form_fill_context.get('remaining_fields') or [], ensure_ascii=False)}\n"
                f"- Required field indices: {json.dumps(form_fill_context.get('required_field_indices') or {}, ensure_ascii=False)}\n"
                "- Only edit those required fields.\n"
                "- Do not click Save, Done, Apply, or Submit until every required field matches the expected value.\n"
                "- If the keyboard is active while filling the form, keep working on the required fields instead of navigating away.\n\n"
            )
        return (
            f"Subtask:\n{request.subtask}\n\n"
            "Current screen state:\n"
            f"- Foreground package: {foreground_package}\n"
            f"- Dominant visible UI package: {dominant_ui_package}\n"
            f"- Current activity: {observation.get('current_activity') or 'Unknown'}\n"
            f"- Screen size: {json.dumps(observation.get('screen_size') or {}, ensure_ascii=False)}\n"
            f"- Visible UI count: {observation.get('visible_ui_count', 0)}\n"
            f"- Clickable UI count: {observation.get('clickable_ui_count', 0)}\n"
            f"- Non-system UI count: {observation.get('non_system_ui_count', 0)}\n"
            f"- Observation consistency: {observation_consistency}\n"
            "- You are given a screenshot in this message with red boxes around retained UI elements. Final action indices must come from the Visible UI index table below.\n\n"
            f"Observation warning:\n{observation_warning or 'None'}\n\n"
            f"Visible UI index table:\n{self._format_ui_index_table(observation.get('ui_elements') or [])}\n\n"
            f"Visible UI elements:\n{observation.get('ui_description') or 'No visible UI elements available.'}\n\n"
            f"Recent action history:\n{self._format_history(request.action_history)}\n\n"
            "Retrieved memory context:\n"
            f"{self._truncate(request.memory_context.strip(), self.config.max_memory_context_chars) or 'None'}\n\n"
            f"{form_fill_block}"
            "Decision policy:\n"
            "- Execute this subtask step by step using one valid GUI action.\n"
            "- Base the choice on the current GUI state, history, and memory.\n"
            "- Prefer the smallest necessary action that advances the subtask.\n"
            "- Use status complete only when the Goal in the current subtask is satisfied.\n"
            "- Do not use status complete for navigation subtasks solely because the app is already open.\n"
            "- Only use indices that appear in the current Visible UI index table.\n"
            "- If a number does not appear in the current Visible UI index table, it is not a legal action index.\n"
            "- Do not copy or invent numbers from the screenshot. Use the screenshot for layout and target appearance only; use the text index table for the final action index.\n"
            "- Do not output action_type swipe; use scroll instead.\n"
            "- If the subtask describes filling a form section, only edit the required related fields for that section and do not continue to save or navigate away.\n"
            "- In a grouped form-fill task, treat the allowed target fields as a hard constraint.\n"
            "- For indexed actions, the chosen index must match the exact element named in your reasoning.\n"
            "- If the observation warning indicates degraded UI, avoid assuming the goal is complete.\n"
            "- If observation consistency is unstable, prefer wait or navigate_back rather than blind taps.\n"
            "- If the current path is blocked or not feasible, use status infeasible.\n"
            "- Return exactly one action in the required JSON action format.\n"
        )

    def _format_history(self, action_history: List[Dict[str, Any]]) -> str:
        if not action_history:
            return "No previous action."
        stable_lines: list[str] = []
        warning_lines: list[str] = []
        lines: list[str] = []
        recent_items = action_history[-self.config.max_history_items :]
        invalid_click_counts: dict[int, int] = {}
        for item in recent_items:
            action = item.get("action") or {}
            error = str(item.get("error") or "")
            if action.get("action_type") in {"click", "long_press"} and "non-clickable" in error.lower():
                index = int(action.get("index", -1))
                invalid_click_counts[index] = invalid_click_counts.get(index, 0) + 1

        for index, item in enumerate(recent_items, start=1):
            action = item.get("action") or {}
            action_label = json.dumps(action, ensure_ascii=False) if action else "None"
            error = str(item.get("error") or "").strip() or "None"
            extras: list[str] = []
            if action.get("action_type") in {"click", "long_press"} and int(action.get("index", -1)) in invalid_click_counts:
                if invalid_click_counts[int(action.get("index", -1))] > 1:
                    extras.append("Repeated invalid click on non-clickable element.")
            if item.get("summary"):
                extras.append(str(item.get("summary")))
            line = (
                f"{index}. subtask={item.get('subtask', '')}\n"
                f"   action={action_label}\n"
                f"   status={item.get('status', '')}\n"
                f"   reason={item.get('reason', '') or 'None'}\n"
                f"   error={error}"
            )
            if extras:
                line += f"\n   notes={' | '.join(extras)}"
            if item.get("observation_unreliable_context") or item.get("status") == "warning":
                warning_lines.append(line)
            else:
                stable_lines.append(line)
        if stable_lines:
            lines.append("Stable progress history:")
            lines.extend(stable_lines)
        if warning_lines:
            lines.append("Unstable warning history:")
            lines.extend(warning_lines)
        return "\n".join(lines)

    def _format_ui_json(self, ui_elements: List[Dict[str, Any]]) -> str:
        serialized = json.dumps(ui_elements, ensure_ascii=False, indent=2)
        return self._truncate(serialized, self.config.max_ui_json_chars)

    @staticmethod
    def _format_ui_index_table(ui_elements: List[Dict[str, Any]]) -> str:
        if not ui_elements:
            return "No visible UI elements available."
        lines: list[str] = []
        for element in ui_elements:
            label = element.get("text") or element.get("content_description") or element.get("resource_name") or "None"
            lines.append(
                f"[#{element.get('index')}] label={label!r}; "
                f"clickable={bool(element.get('is_clickable'))}; "
                f"editable={bool(element.get('is_editable'))}; "
                f"package={element.get('package_name') or 'Unknown'}; "
                f"class={element.get('class_name') or 'Unknown'}"
            )
        return "\n".join(lines)

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 20] + "\n...[truncated]"


def parse_reason_action_output(raw_response: str) -> tuple[str, dict[str, Any] | None]:
    reason = ""
    reason_match = re.search(r"Reason\s*:\s*(.+?)(?:\nAction\s*:|$)", raw_response, flags=re.DOTALL | re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()

    action_match = re.search(r"Action\s*:\s*(.+)$", raw_response, flags=re.DOTALL | re.IGNORECASE)
    if action_match:
        payload = extract_json_object(action_match.group(1).strip())
        return reason, payload

    payload = extract_json_object(raw_response)
    return reason, payload


def parse_actor_action(
    payload: dict[str, Any],
    observation: Dict[str, Any],
    *,
    reason: str = "",
    subtask: str = "",
) -> tuple[ActorAction, Dict[str, Any] | None, bool, str | None, Dict[str, Any] | None, str | None]:
    normalized_payload, normalization_applied, normalization_reason = normalize_actor_action_payload(payload)
    action_type = str(normalized_payload.get("action_type", "")).strip()
    valid_indices = set(observation.get("valid_ui_indices") or [])
    ui_elements_by_index = {
        int(element.get("index")): element for element in (observation.get("ui_elements") or []) if element.get("index") is not None
    }
    observation_warning = str(observation.get("observation_warning") or "")

    if action_type == "status":
        goal_status = str(normalized_payload.get("goal_status", "")).strip()
        if goal_status not in {"complete", "infeasible"}:
            raise ValueError("status.goal_status must be 'complete' or 'infeasible'.")
        if goal_status == "complete" and _observation_is_degraded(observation_warning, observation):
            raise ValueError(
                "status.complete is not allowed when the observation is degraded or only system UI is visible."
            )
        return (
            StatusAction(goal_status=goal_status, message=str(normalized_payload.get("message", "")).strip()),
            normalized_payload if normalization_applied else None,
            normalization_applied,
            normalization_reason,
            None,
            None,
        )
    if action_type == "answer":
        text = str(normalized_payload.get("text", "")).strip()
        if not text:
            raise ValueError("answer.text must be non-empty.")
        return (
            AnswerAction(text=text),
            normalized_payload if normalization_applied else None,
            normalization_applied,
            normalization_reason,
            None,
            None,
        )
    if action_type == "click":
        corrected, correction_reason = _maybe_correct_pointing_payload(
            normalized_payload,
            observation=observation,
            reason=reason,
            subtask=subtask,
            require_clickable=True,
        )
        final_payload = corrected or normalized_payload
        index, x, y = _resolve_pointing_target(
            final_payload,
            valid_indices,
            ui_elements_by_index,
            "click",
            require_clickable=True,
        )
        return (
            ClickAction(index=index, x=x, y=y),
            normalized_payload if normalization_applied else None,
            normalization_applied,
            normalization_reason,
            corrected,
            correction_reason,
        )
    if action_type == "long_press":
        corrected, correction_reason = _maybe_correct_pointing_payload(
            normalized_payload,
            observation=observation,
            reason=reason,
            subtask=subtask,
            require_clickable=True,
        )
        final_payload = corrected or normalized_payload
        index, x, y = _resolve_pointing_target(
            final_payload,
            valid_indices,
            ui_elements_by_index,
            "long_press",
            require_clickable=True,
        )
        return (
            LongPressAction(index=index, x=x, y=y),
            normalized_payload if normalization_applied else None,
            normalization_applied,
            normalization_reason,
            corrected,
            correction_reason,
        )
    if action_type == "input_text":
        text = str(normalized_payload.get("text", "")).strip()
        if not text:
            raise ValueError("input_text.text must be non-empty.")
        clear_text = normalized_payload.get("clear_text")
        if clear_text is not None:
            clear_text = bool(clear_text)
        corrected, correction_reason = _maybe_correct_pointing_payload(
            normalized_payload,
            observation=observation,
            reason=reason,
            subtask=subtask,
            require_editable=True,
        )
        final_payload = corrected or normalized_payload
        return InputTextAction(
            index=_validate_index(
                final_payload,
                valid_indices,
                ui_elements_by_index,
                "input_text",
                require_editable=True,
            ),
            text=text,
            clear_text=clear_text,
        ), (
            normalized_payload if normalization_applied else None
        ), normalization_applied, normalization_reason, corrected, correction_reason
    if action_type == "keyboard_enter":
        return (
            KeyboardEnterAction(),
            normalized_payload if normalization_applied else None,
            normalization_applied,
            normalization_reason,
            None,
            None,
        )
    if action_type == "navigate_home":
        return (
            NavigateHomeAction(),
            normalized_payload if normalization_applied else None,
            normalization_applied,
            normalization_reason,
            None,
            None,
        )
    if action_type == "navigate_back":
        return (
            NavigateBackAction(),
            normalized_payload if normalization_applied else None,
            normalization_applied,
            normalization_reason,
            None,
            None,
        )
    if action_type == "scroll":
        direction = str(normalized_payload.get("direction", "")).strip()
        if direction not in {"up", "down", "left", "right"}:
            raise ValueError("scroll.direction must be one of up/down/left/right.")
        index = normalized_payload.get("index")
        if index is not None:
            index = _validate_index(normalized_payload, valid_indices, ui_elements_by_index, "scroll")
        return (
            ScrollAction(direction=direction, index=index),
            normalized_payload if normalization_applied else None,
            normalization_applied,
            normalization_reason,
            None,
            None,
        )
    if action_type == "open_app":
        app_name = str(normalized_payload.get("app_name", "")).strip()
        if not app_name:
            raise ValueError("open_app.app_name must be non-empty.")
        return (
            OpenAppAction(app_name=app_name),
            normalized_payload if normalization_applied else None,
            normalization_applied,
            normalization_reason,
            None,
            None,
        )
    if action_type == "wait":
        return (
            WaitAction(),
            normalized_payload if normalization_applied else None,
            normalization_applied,
            normalization_reason,
            None,
            None,
        )
    raise ValueError(f"Unsupported actor action type: {action_type!r}")


def normalize_actor_action_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], bool, str | None]:
    normalized = dict(payload)
    action_type = str(normalized.get("action_type", "")).strip()
    alias_map = {
        "type": "input_text",
        "enter_text": "input_text",
        "fill_text": "input_text",
        "set_text": "input_text",
        "swipe": "scroll",
    }
    canonical_action = alias_map.get(action_type)
    if canonical_action is None:
        return normalized, False, None
    normalized["action_type"] = canonical_action
    return normalized, True, f"Normalized action_type from {action_type!r} to {canonical_action!r}."


def _validate_index(
    payload: dict[str, Any],
    valid_indices: set[int],
    ui_elements_by_index: dict[int, dict[str, Any]],
    action_name: str,
    *,
    require_clickable: bool = False,
    require_editable: bool = False,
) -> int:
    if "index" not in payload:
        raise ValueError(f"{action_name}.index is required.")
    try:
        index = int(payload["index"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{action_name}.index must be an integer.") from exc
    if index not in valid_indices:
        raise ValueError(f"{action_name}.index must be one of the current valid_ui_indices.")
    element = ui_elements_by_index.get(index) or {}
    if require_clickable and not bool(element.get("is_clickable")):
        raise ValueError(f"{action_name}.index must point to a clickable UI element; non-clickable element selected.")
    if require_editable and not bool(element.get("is_editable")):
        raise ValueError(f"{action_name}.index must point to an editable UI element; non-editable element selected.")
    return index


def _resolve_pointing_target(
    payload: dict[str, Any],
    valid_indices: set[int],
    ui_elements_by_index: dict[int, dict[str, Any]],
    action_name: str,
    *,
    require_clickable: bool = False,
) -> tuple[int | None, int | None, int | None]:
    has_index = "index" in payload and payload.get("index") is not None
    has_x = "x" in payload and payload.get("x") is not None
    has_y = "y" in payload and payload.get("y") is not None

    if has_index and (has_x or has_y):
        raise ValueError(f"{action_name} must provide either index or x/y, not both.")
    if has_x != has_y:
        raise ValueError(f"{action_name} must provide both x and y together.")
    if has_index:
        index = _validate_index(
            payload,
            valid_indices,
            ui_elements_by_index,
            action_name,
            require_clickable=require_clickable,
        )
        return index, None, None
    if has_x and has_y:
        try:
            x = int(payload["x"])
            y = int(payload["y"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{action_name}.x and {action_name}.y must be integers.") from exc
        return None, x, y
    raise ValueError(f"{action_name} must provide either index or x/y.")


def _observation_is_degraded(observation_warning: str, observation: Dict[str, Any]) -> bool:
    if observation.get("observation_consistency") == "unstable":
        return True
    if observation.get("non_system_ui_count", 0) == 0 and (observation.get("foreground_package") or "") != SYSTEM_UI_PACKAGE:
        return True
    return bool(observation_warning)


def _maybe_correct_pointing_payload(
    payload: dict[str, Any],
    *,
    observation: Dict[str, Any],
    reason: str,
    subtask: str,
    require_clickable: bool = False,
    require_editable: bool = False,
) -> tuple[Dict[str, Any] | None, str | None]:
    if "index" not in payload:
        return None, None
    candidate = dict(payload)
    try:
        parsed_index = int(candidate["index"])
    except (TypeError, ValueError):
        parsed_index = None
    valid_indices = set(observation.get("valid_ui_indices") or [])
    ui_elements = observation.get("ui_elements") or []
    ui_by_index = {
        int(element.get("index")): element
        for element in ui_elements
        if element.get("index") is not None
    }
    current_target = ui_by_index.get(parsed_index) if parsed_index is not None else None
    target_token = _extract_target_token(subtask, reason, current_target)
    control_target_requested = require_clickable and current_target is not None and _is_control_like_target(current_target) and _goal_or_reason_requests_toggle(subtask, reason)
    if parsed_index is not None and parsed_index in valid_indices:
        target = ui_by_index.get(parsed_index) or {}
        target_is_usable = (not require_clickable or bool(target.get("is_clickable"))) and (
            not require_editable or bool(target.get("is_editable"))
        )
        if target_is_usable and (not target_token or _element_matches_token(target, target_token)):
            return None, None
    if not target_token and not control_target_requested:
        return None, None
    semantic_matches = [
        element
        for element in ui_elements
        if target_token and _element_matches_token(element, target_token)
    ]
    if len(semantic_matches) == 1:
        match = semantic_matches[0]
        if _element_satisfies_constraints(match, require_clickable=require_clickable, require_editable=require_editable):
            candidate["index"] = int(match["index"])
            return candidate, "Corrected click target from unique clickable semantic match."
        if control_target_requested and current_target is not None:
            fallback, fallback_kind = _build_coordinate_fallback_payload(
                candidate,
                current_target,
                target_token,
                subtask=subtask,
                reason=reason,
            )
            if fallback is not None and fallback_kind == "control":
                return fallback, (
                    "Original index pointed to a non-clickable control widget, no matching clickable index was available, "
                    "and the action was downgraded to a bbox-based coordinate click on the visible control widget because no clickable index was exposed."
                )
        fallback, fallback_kind = _build_coordinate_fallback_payload_from_match(
            candidate,
            match,
            subtask=subtask,
            reason=reason,
        )
        if fallback is not None and fallback_kind == "label":
            return fallback, "Corrected click target from unique non-clickable semantic match by using a bbox-based coordinate click."
        if fallback is not None and fallback_kind == "control":
            return fallback, (
                "Original index pointed to a non-clickable control widget, no matching clickable index was available, "
                "and the action was downgraded to a bbox-based coordinate click on the visible control widget because no clickable index was exposed."
            )
        return None, None

    if len(semantic_matches) > 1:
        constrained_clickable_matches = [
            element
            for element in semantic_matches
            if _element_satisfies_constraints(element, require_clickable=require_clickable, require_editable=require_editable)
        ]
        if len(constrained_clickable_matches) == 1:
            candidate["index"] = int(constrained_clickable_matches[0]["index"])
            return candidate, "Corrected click target from unique clickable match among multiple semantic matches."
        if not constrained_clickable_matches:
            if control_target_requested and current_target is not None:
                fallback, fallback_kind = _build_coordinate_fallback_payload(
                    candidate,
                    current_target,
                    target_token,
                    subtask=subtask,
                    reason=reason,
                )
                if fallback is not None and fallback_kind == "control":
                    return fallback, (
                        "Original index pointed to a non-clickable control widget, no matching clickable index was available, "
                        "and the action was downgraded to a bbox-based coordinate click on the visible control widget because no clickable index was exposed."
                    )
            exact_label_matches = [
                element
                for element in semantic_matches
                if _is_exact_visible_label_match(element, target_token)
                and (not require_editable or bool(element.get("is_editable")))
            ]
            if len(exact_label_matches) == 1:
                fallback, fallback_kind = _build_coordinate_fallback_payload_from_match(
                    candidate,
                    exact_label_matches[0],
                    subtask=subtask,
                    reason=reason,
                )
                if fallback is not None and fallback_kind == "label":
                    return fallback, (
                        "Corrected click target from unique exact visible label match among multiple non-clickable matches by using a bbox-based coordinate click."
                    )
        if require_clickable and current_target is not None:
            fallback, fallback_kind = _build_coordinate_fallback_payload(
                candidate,
                current_target,
                target_token,
                subtask=subtask,
                reason=reason,
            )
            if fallback is not None and fallback_kind == "control":
                return fallback, (
                    "Original index pointed to a non-clickable control widget, no matching clickable index was available, "
                    "and the action was downgraded to a bbox-based coordinate click on the visible control widget because no clickable index was exposed."
                )
        return None, None

    if require_clickable and current_target is not None:
        fallback, fallback_kind = _build_coordinate_fallback_payload(
            candidate,
            current_target,
            target_token,
            subtask=subtask,
            reason=reason,
        )
        if fallback is not None and fallback_kind == "control":
            return fallback, (
                "Original index pointed to a non-clickable control widget, no matching clickable index was available, "
                "and the action was downgraded to a bbox-based coordinate click on the visible control widget because no clickable index was exposed."
            )
    return None, None

def _extract_target_token(
    subtask: str,
    reason: str,
    current_target: dict[str, Any] | None,
) -> str | None:
    goal_text = _extract_subtask_goal(subtask)
    candidates = _quoted_candidates(goal_text) + _quoted_candidates(reason)
    for candidate in candidates:
        normalized = _normalize_match_text(candidate)
        if normalized:
            return normalized
    combined_text = " ".join(filter(None, [goal_text, reason])).lower()
    for token in (
        "network & internet",
        "wi-fi",
        "wifi",
        "bluetooth",
        "settings",
        "phone",
        "contacts",
    ):
        if token in combined_text:
            return _normalize_match_text(token)
    for source_text in (goal_text, reason):
        label = _extract_visible_label_candidate(source_text)
        normalized = _normalize_match_text(label)
        if normalized:
            return normalized
    fallback_fields = []
    if current_target:
        fallback_fields.extend(
            [
                str(current_target.get("text") or "").strip(),
                str(current_target.get("content_description") or "").strip(),
            ]
        )
    for candidate in fallback_fields:
        normalized = _normalize_match_text(candidate)
        if normalized:
            return normalized
    return None


def _extract_subtask_goal(subtask: str) -> str:
    match = re.search(r"Goal\s*:\s*(.+?)(?:$|\n)", subtask, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return subtask.strip()


def _quoted_candidates(text: str) -> list[str]:
    return [match.strip() for match in re.findall(r"['\"]([^'\"]+)['\"]", text) if match.strip()]


def _normalize_match_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return normalized


def _build_coordinate_fallback_payload(
    payload: dict[str, Any],
    element: dict[str, Any],
    target_token: str,
    *,
    subtask: str,
    reason: str,
) -> tuple[dict[str, Any] | None, str | None]:
    is_label_match = _element_matches_token(element, target_token)
    is_control_fallback = _is_control_like_target(element) and _goal_or_reason_requests_toggle(subtask, reason)
    if not is_label_match and not is_control_fallback:
        return None, None
    bbox = element.get("bbox") or ()
    if len(bbox) != 4:
        return None, None
    try:
        x_min, y_min, x_max, y_max = [int(value) for value in bbox]
    except (TypeError, ValueError):
        return None, None
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        return None, None
    if is_control_fallback and not is_label_match:
        x = int(round(x_min + (0.5 * width)))
        y = int(round(y_min + (0.5 * height)))
        fallback_kind = "control"
    else:
        x = int(round(x_min + (0.3 * width)))
        y = int(round(y_min + (0.5 * height)))
        fallback_kind = "label"
    return (
        {
            "action_type": payload.get("action_type"),
            "x": x,
            "y": y,
        },
        fallback_kind,
    )


def _build_coordinate_fallback_payload_from_match(
    payload: dict[str, Any],
    matched_element: dict[str, Any],
    *,
    subtask: str,
    reason: str,
) -> tuple[dict[str, Any] | None, str | None]:
    bbox = matched_element.get("bbox") or ()
    if len(bbox) != 4:
        return None, None
    try:
        x_min, y_min, x_max, y_max = [int(value) for value in bbox]
    except (TypeError, ValueError):
        return None, None
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        return None, None
    if _is_control_like_target(matched_element) and _goal_or_reason_requests_toggle(subtask, reason):
        x = int(round(x_min + (0.5 * width)))
        y = int(round(y_min + (0.5 * height)))
        fallback_kind = "control"
    else:
        x = int(round(x_min + (0.3 * width)))
        y = int(round(y_min + (0.5 * height)))
        fallback_kind = "label"
    return (
        {
            "action_type": payload.get("action_type"),
            "x": x,
            "y": y,
        },
        fallback_kind,
    )


def _goal_or_reason_requests_toggle(subtask: str, reason: str) -> bool:
    combined_text = " ".join(filter(None, [_extract_subtask_goal(subtask), reason])).lower()
    toggle_markers = (
        "toggle",
        "turn off",
        "turn on",
        "switch off",
        "switch on",
        "disable",
        "enable",
    )
    return any(marker in combined_text for marker in toggle_markers)


def _should_stop_for_external_verification(
    *,
    subtask: str,
    action: ActorAction | None,
    before_observation: Dict[str, Any],
    after_observation: Dict[str, Any],
) -> str | None:
    if action is None:
        return None
    if action.action_type in {"wait", "status", "answer"}:
        return None
    target = _resolve_action_target_element(action, before_observation)
    if target is not None and _is_control_like_target(target) and _goal_or_reason_requests_toggle(subtask, ""):
        return "Meaningful control action executed; stop and let the external verifier check the resulting state."
    if action.action_type == "open_app":
        return "App launch action executed; stop and let the external verifier check whether the target app is now open."
    if action.action_type in {"navigate_back", "navigate_home"}:
        return "Navigation action executed; stop and let the external verifier inspect the resulting screen."
    if action.action_type in {"input_text", "keyboard_enter"}:
        return "Form-editing action executed; stop and let the external verifier inspect the updated field state."
    if action.action_type in {"click", "long_press"}:
        return "High-impact pointing action executed; stop and let the external verifier inspect the resulting screen."
    return None


def _resolve_action_target_element(
    action: ActorAction,
    observation: Dict[str, Any],
) -> dict[str, Any] | None:
    ui_elements = observation.get("ui_elements") or []
    ui_by_index = {
        int(element.get("index")): element
        for element in ui_elements
        if element.get("index") is not None
    }
    index = getattr(action, "index", None)
    if index is not None:
        return ui_by_index.get(int(index))
    x = getattr(action, "x", None)
    y = getattr(action, "y", None)
    if x is None or y is None:
        return None
    for element in ui_elements:
        bbox = element.get("bbox") or ()
        if len(bbox) != 4:
            continue
        try:
            x_min, y_min, x_max, y_max = [int(value) for value in bbox]
        except (TypeError, ValueError):
            continue
        if x_min <= int(x) <= x_max and y_min <= int(y) <= y_max:
            return element
    return None


def _is_control_like_target(element: dict[str, Any]) -> bool:
    class_name = str(element.get("class_name") or "").lower()
    resource_name = str(element.get("resource_name") or "").lower()
    content_description = str(element.get("content_description") or "").lower()
    raw = element.get("raw") or {}
    raw_resource_name = str(raw.get("resource_name") or "").lower()
    raw_content_description = str(raw.get("content_description") or "").lower()

    control_class_markers = (
        "switch",
        "switchcompat",
        "checkbox",
        "radiobutton",
        "togglebutton",
        "imagebutton",
    )
    if any(marker in class_name for marker in control_class_markers):
        return True

    control_resource_markers = ("switch_widget", "checkbox", "radio", "toggle", "button")
    resource_fields = (resource_name, raw_resource_name, content_description, raw_content_description)
    if any(marker in value for value in resource_fields for marker in control_resource_markers if value):
        return True

    if bool(raw.get("is_checkable")) or raw.get("is_checked") is not None:
        return True
    return False


def _element_matches_token(element: dict[str, Any], token: str) -> bool:
    token = _normalize_match_text(token)
    if not token:
        return False
    fields = [
        _normalize_match_text(element.get("text") or ""),
        _normalize_match_text(element.get("content_description") or ""),
        _normalize_match_text(element.get("resource_name") or ""),
    ]
    aliases = {
        "wifi": {"wifi", "wi-fi"},
        "wi-fi": {"wifi", "wi-fi"},
    }
    alias_set = aliases.get(token, {token})
    for value in fields:
        if not value:
            continue
        if value in alias_set:
            return True
        if any(alias in value or value in alias for alias in alias_set):
            return True
    return False


def _element_satisfies_constraints(
    element: dict[str, Any],
    *,
    require_clickable: bool = False,
    require_editable: bool = False,
) -> bool:
    if require_clickable and not bool(element.get("is_clickable")):
        return False
    if require_editable and not bool(element.get("is_editable")):
        return False
    return True


def _is_exact_visible_label_match(element: dict[str, Any], target_token: str) -> bool:
    normalized_token = _normalize_match_text(target_token)
    if not normalized_token:
        return False
    for field in ("text", "content_description"):
        value = _normalize_match_text(element.get(field) or "")
        if value and value == normalized_token:
            return True
    return False


def _extract_visible_label_candidate(text: str) -> str | None:
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return None
    label_patterns = (
        r"\bsettings\b",
        r"\bnetwork\s*&\s*internet\b",
        r"\bwi-?fi\b",
        r"\bbluetooth\b",
        r"\bcontacts\b",
    )
    for pattern in label_patterns:
        match = re.search(pattern, normalized_text, flags=re.I)
        if match:
            return match.group(0)
    return None


def _resample_observation_after_high_impact_stop(
    *,
    observation_adapter: ObservationAdapter,
    env: Any,
    goal: str,
    step_id: int,
    first_after_observation: Dict[str, Any],
) -> Dict[str, Any]:
    resampled_observation = observation_adapter.capture_observation(
        env,
        goal,
        step_id=step_id,
        include_screenshots=True,
    )
    extra_state = dict(resampled_observation.get("extra_state") or {})
    first_extra_state = dict(first_after_observation.get("extra_state") or {})
    previous_attempt = int(first_extra_state.get("observation_attempt") or 1)
    extra_state["observation_attempt"] = max(previous_attempt + 1, 2)
    extra_state["observation_resampled"] = True
    extra_state["final_after_resample"] = True
    resampled_observation["extra_state"] = extra_state
    return resampled_observation


def to_json_action(action: ActorAction) -> json_action.JSONAction:
    return json_action.JSONAction(**action.to_payload())


def extract_json_object(text: str) -> Optional[dict]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

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


def _build_rule_summary(
    *,
    subtask: str,
    action: ActorAction | None,
    reason: str,
    status: str,
    error: str,
) -> str:
    action_name = action.action_type if action else "no_action"
    summary = (
        f"Subtask={subtask}; action={action_name}; status={status}; "
        f"reason={reason or 'not provided'}."
    )
    if error:
        summary += f" Error={error}."
    elif status in {"completed", "infeasible"}:
        summary += " This step ended the subtask."
    else:
        summary += " Use this to avoid repeating unproductive paths."
    return summary
