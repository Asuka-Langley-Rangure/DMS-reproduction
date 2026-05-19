from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol


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
class PlannerStage:
    stage_id: int
    title: str
    success_signal: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlannerResult:
    is_goal_complete: bool
    completion_message: str = ""
    subtasks: List[PlannerSubtask] = field(default_factory=list)
    stage_plan: List[PlannerStage] = field(default_factory=list)
    current_stage_id: int | None = None
    covered_stage_ids: List[int] = field(default_factory=list)
    raw_response: str = ""
    parse_error: Optional[str] = None
    parse_error_code: Optional[str] = None
    repaired_parse: bool = False
    repair_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_goal_complete": self.is_goal_complete,
            "completion_message": self.completion_message,
            "subtasks": [subtask.to_dict() for subtask in self.subtasks],
            "stage_plan": [stage.to_dict() for stage in self.stage_plan],
            "current_stage_id": self.current_stage_id,
            "covered_stage_ids": list(self.covered_stage_ids),
            "raw_response": self.raw_response,
            "parse_error": self.parse_error,
            "parse_error_code": self.parse_error_code,
            "repaired_parse": self.repaired_parse,
            "repair_reason": self.repair_reason,
        }


@dataclass
class StagePlanResult:
    stage_plan: List[PlannerStage] = field(default_factory=list)
    raw_response: str = ""
    parse_error: Optional[str] = None
    repaired_parse: bool = False
    repair_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_plan": [stage.to_dict() for stage in self.stage_plan],
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
    prompt_profile: Literal[
        "generic_self_written",
        "generic_paper",
        "legacy_contact_tuned",
    ] = "generic_self_written"


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
        return self.plan_current_subtasks(
            user_goal=user_goal,
            stage_plan=self._extract_stage_plan_from_history(task_history or []),
            observation=observation,
            task_history=task_history,
            memory_context=memory_context,
        )

    def plan_stage_milestones(self, user_goal: str) -> StagePlanResult:
        messages = self.build_stage_plan_messages(user_goal=user_goal)
        raw_response = self.llm_client.generate(
            messages=messages,
            temperature=self.config.temperature,
        )
        return self.parse_stage_plan_response(raw_response)

    def plan_current_subtasks(
        self,
        user_goal: str,
        stage_plan: List[PlannerStage],
        observation: Dict[str, Any],
        task_history: Optional[List[Dict[str, Any]]] = None,
        memory_context: str = "",
    ) -> PlannerResult:
        task_history = task_history or []
        messages = self.build_current_subtasks_messages(
            user_goal=user_goal,
            stage_plan=stage_plan,
            observation=observation,
            task_history=task_history,
            memory_context=memory_context,
        )
        raw_response = self.llm_client.generate(
            messages=messages,
            temperature=self.config.temperature,
        )
        return self.parse_response(raw_response)

    def build_stage_plan_messages(self, user_goal: str) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": self._build_stage_plan_system_prompt()},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self._build_stage_plan_user_prompt(user_goal=user_goal),
                    }
                ],
            },
        ]

    def build_messages(
        self,
        user_goal: str,
        observation: Dict[str, Any],
        task_history: List[Dict[str, Any]],
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        return self.build_current_subtasks_messages(
            user_goal=user_goal,
            stage_plan=self._extract_stage_plan_from_history(task_history),
            observation=observation,
            task_history=task_history,
            memory_context=memory_context,
        )

    def build_current_subtasks_messages(
        self,
        user_goal: str,
        stage_plan: List[PlannerStage],
        observation: Dict[str, Any],
        task_history: List[Dict[str, Any]],
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": self._build_current_subtasks_system_prompt()},
            {
                "role": "user",
                "content": self._build_current_subtasks_user_content(
                    user_goal=user_goal,
                    stage_plan=stage_plan,
                    observation=observation,
                    task_history=task_history,
                    memory_context=memory_context,
                ),
            },
        ]

    def messages_to_jsonable(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return json.loads(json.dumps(messages, ensure_ascii=False))

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
                    parse_error_code="planner_json_parse_failed",
                )
            payload = repaired_payload
            repaired_parse = True

        tool = payload.get("tool")
        if tool == "complete_goal":
            stage_plan, current_stage_id, covered_stage_ids, stage_error = parse_stage_plan_payload(payload)
            if stage_error is not None:
                return PlannerResult(
                    is_goal_complete=False,
                    raw_response=raw_response,
                    parse_error=stage_error,
                    parse_error_code="planner_stage_plan_invalid",
                    repaired_parse=repaired_parse,
                    repair_reason=repair_reason,
                )
            return PlannerResult(
                is_goal_complete=True,
                completion_message=str(payload.get("message", "")).strip(),
                stage_plan=stage_plan,
                current_stage_id=current_stage_id,
                covered_stage_ids=covered_stage_ids,
                raw_response=raw_response,
                repaired_parse=repaired_parse,
                repair_reason=repair_reason,
            )

        if tool not in {"set_tasks", "set_tasks_with_agents"}:
            return PlannerResult(
                is_goal_complete=False,
                raw_response=raw_response,
                parse_error=f"Unsupported planner tool: {tool!r}",
                parse_error_code="planner_tool_unsupported",
                repaired_parse=repaired_parse,
                repair_reason=repair_reason,
            )

        tasks = payload.get("tasks")
        if tasks is None:
            tasks = payload.get("task_assignments")
        stage_plan, current_stage_id, covered_stage_ids, stage_error = parse_stage_plan_payload(payload)
        if stage_error is not None:
            return PlannerResult(
                is_goal_complete=False,
                raw_response=raw_response,
                parse_error=stage_error,
                parse_error_code="planner_stage_plan_invalid",
                repaired_parse=repaired_parse,
                repair_reason=repair_reason,
            )
        if not isinstance(tasks, list) or not tasks:
            synthesized_subtask = synthesize_subtask_from_stage_plan(
                stage_plan=stage_plan,
                current_stage_id=current_stage_id,
                default_actor_name=self.config.default_actor_name,
            )
            if synthesized_subtask is None:
                return PlannerResult(
                    is_goal_complete=False,
                    raw_response=raw_response,
                    parse_error="Planner tasks must be a non-empty list.",
                    parse_error_code="planner_tasks_missing",
                    repaired_parse=repaired_parse,
                    repair_reason=repair_reason,
                )
            return PlannerResult(
                is_goal_complete=False,
                subtasks=[synthesized_subtask],
                stage_plan=stage_plan,
                current_stage_id=current_stage_id,
                covered_stage_ids=covered_stage_ids,
                raw_response=raw_response,
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
                    parse_error_code="planner_task_invalid",
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
                    parse_error_code="planner_task_invalid",
                    repaired_parse=repaired_parse,
                    repair_reason=repair_reason,
                )
            if not isinstance(reason, str) or not reason.strip():
                return PlannerResult(
                    is_goal_complete=False,
                    raw_response=raw_response,
                    parse_error=f"Planner task at index {index} is missing 'reason'.",
                    parse_error_code="planner_task_invalid",
                    repaired_parse=repaired_parse,
                    repair_reason=repair_reason,
                )

            parsed = parse_precondition_goal(task_text)
            if parsed is None:
                repaired_task_text = synthesize_precondition_goal_task(task_text)
                if repaired_task_text is None:
                    return PlannerResult(
                        is_goal_complete=False,
                        raw_response=raw_response,
                        parse_error=(
                            f"Planner task at index {index} must use "
                            "'Precondition: ... Goal: ...' format."
                        ),
                        parse_error_code="planner_task_format_invalid",
                        repaired_parse=repaired_parse,
                        repair_reason=repair_reason,
                    )
                parsed = parse_precondition_goal(repaired_task_text)
                if parsed is None:
                    return PlannerResult(
                        is_goal_complete=False,
                        raw_response=raw_response,
                        parse_error=(
                            f"Planner task at index {index} must use "
                            "'Precondition: ... Goal: ...' format."
                        ),
                        parse_error_code="planner_task_format_invalid",
                        repaired_parse=repaired_parse,
                        repair_reason=repair_reason,
                    )
                repaired_parse = True
                if repair_reason is None:
                    repair_reason = "synthesized_precondition_none_for_legacy_task_text"

            precondition, goal = parsed
            subtasks.append(
                PlannerSubtask(
                    precondition=precondition,
                    goal=goal,
                    reason=reason.strip(),
                    agent=item.get("agent") or self.config.default_actor_name,
                )
            )

        stage_alignment_error = _validate_stage_subtask_alignment(
            stage_plan=stage_plan,
            current_stage_id=current_stage_id,
            subtasks=subtasks,
        )
        if stage_alignment_error is not None:
            return PlannerResult(
                is_goal_complete=False,
                raw_response=raw_response,
                parse_error=stage_alignment_error,
                parse_error_code="planner_stage_subtask_misaligned",
                repaired_parse=repaired_parse,
                repair_reason=repair_reason,
            )

        return PlannerResult(
            is_goal_complete=False,
            subtasks=subtasks,
            stage_plan=stage_plan,
            current_stage_id=current_stage_id,
            covered_stage_ids=covered_stage_ids,
            raw_response=raw_response,
            repaired_parse=repaired_parse,
            repair_reason=repair_reason,
        )

    def parse_stage_plan_response(self, raw_response: str) -> StagePlanResult:
        payload: dict[str, Any] | None = None
        stripped = raw_response.strip()
        if stripped.startswith("[") or stripped.startswith("```json\n[") or stripped.startswith("```\n["):
            direct_stage_list = _extract_json_array(raw_response)
            if isinstance(direct_stage_list, list):
                payload = {"stage_plan": direct_stage_list}
        if payload is None:
            payload = extract_json_object(raw_response)
        if payload is None:
            direct_stage_list = _extract_json_array(raw_response)
            if isinstance(direct_stage_list, list):
                payload = {"stage_plan": direct_stage_list}
        repaired_parse = False
        repair_reason: str | None = None
        if payload is None:
            return StagePlanResult(
                raw_response=raw_response,
                parse_error="Failed to parse planner stage-plan JSON.",
            )
        stage_plan, current_stage_id, covered_stage_ids, stage_error = parse_stage_plan_payload(payload)
        if stage_error is not None:
            return StagePlanResult(
                raw_response=raw_response,
                parse_error=stage_error,
                repaired_parse=repaired_parse,
                repair_reason=repair_reason,
            )
        if current_stage_id is not None:
            return StagePlanResult(
                raw_response=raw_response,
                parse_error="Initial stage-plan generation must not include current_stage_id.",
                repaired_parse=repaired_parse,
                repair_reason=repair_reason,
            )
        if covered_stage_ids:
            return StagePlanResult(
                raw_response=raw_response,
                parse_error="Initial stage-plan generation must not include covered_stage_ids.",
                repaired_parse=repaired_parse,
                repair_reason=repair_reason,
            )
        if not 3 <= len(stage_plan) <= 5:
            return StagePlanResult(
                raw_response=raw_response,
                parse_error="Initial stage_plan must contain 3-5 milestones.",
                repaired_parse=repaired_parse,
                repair_reason=repair_reason,
            )
        return StagePlanResult(
            stage_plan=stage_plan,
            raw_response=raw_response,
            repaired_parse=repaired_parse,
            repair_reason=repair_reason,
        )

    def _build_system_prompt(self) -> str:
        if self.config.prompt_profile == "legacy_contact_tuned":
            return self._build_legacy_system_prompt()
        return self._build_generic_system_prompt()

    def _build_stage_plan_system_prompt(self) -> str:
        if self.config.prompt_profile == "legacy_contact_tuned":
            return self._build_legacy_stage_plan_system_prompt()
        return self._build_generic_stage_plan_system_prompt()

    def _build_current_subtasks_system_prompt(self) -> str:
        if self.config.prompt_profile == "legacy_contact_tuned":
            return self._build_legacy_system_prompt()
        return self._build_generic_current_subtasks_system_prompt()

    def _build_generic_stage_plan_system_prompt(self) -> str:
        return (
            "You are an Android Task Planner.\n\n"
            "Your only job in this call is to decompose the user's overall goal into a whole-task stage plan.\n"
            "You are not choosing the next tap, click, or current-round subtask.\n"
            "You do not see the current device state in this call.\n\n"
            "Stage-plan rules:\n"
            "- Build 3-5 high-level milestones for the whole task.\n"
            "- Prefer 3 or 4 milestones when the task is simple. Do not force 5 milestones.\n"
            "- The stage plan must cover the main path from start to verified completion.\n"
            "- Each milestone must describe a verifiable UI phase, task phase, or state transition.\n"
            "- A milestone must not be a single click, single button, single tab, single field, or single text-entry operation.\n"
            "- A milestone should usually correspond to a reusable procedure that may take multiple actor actions.\n"
            "- For form-filling tasks, combine related fields on the same form into one milestone, such as 'Fill all required form information'.\n"
            "- Do not create separate milestones for name, phone, amount, title, date, or other individual fields when they are filled on the same form.\n"
            "- Only split fields into separate milestones if they require different screens, different workflows, or independent verification.\n"
            "- Do not collapse the whole task into one stage such as 'Open app'.\n"
            "- Do not include current_stage_id.\n"
            "- Do not include tasks or subtasks.\n"
            "- Do not mention the current screen, because it is unknown in this call.\n"
            "- Do not use low-level action verbs such as tap, click, press, type, input, swipe, scroll, long press, or select.\n"
            "- Adjacent milestones must be distinct parts of the task, not repeated rephrasings.\n\n"
            "Output:\n"
            '{"stage_plan":[{"stage_id":1,"title":"...","success_signal":"..."}]}\n\n'
            "Constraints:\n"
            "- Return valid JSON only.\n"
            "- stage_plan must contain 3-5 items.\n"
            "- stage_id values must be unique integers.\n"
            "- title and success_signal must be non-empty strings."
        )

    def _build_generic_current_subtasks_system_prompt(self) -> str:
        return (
            "You are an Android Task Planner. In this call, a frozen whole-task stage plan already exists.\n\n"
            "Your job now is only to:\n"
            "1. Decide which existing milestone is the current stage.\n"
            "2. Output 1-2 executable functional subtasks that advance that stage.\n\n"
            "Rules:\n"
            "- Do not rewrite, replace, or expand the stage plan.\n"
            "- Use only the provided stage plan when choosing current_stage_id.\n"
            "- If adjacent frozen stages are over-fine field-level milestones on the same screen, you may execute them together in one subtask.\n"
            "- In that case, keep current_stage_id as the earliest merged stage id and also return covered_stage_ids with every merged frozen stage id.\n"
            "- covered_stage_ids must be a contiguous ascending list, must include current_stage_id, and must only contain adjacent frozen stages that this subtask really advances now.\n"
            "- This is not rewriting the frozen plan; it is grouping same-screen field-level milestones for execution.\n"
            "- Determine which frozen milestones are directly supported by the current observation before choosing current_stage_id.\n"
            "- Choose the earliest stage that is not yet directly satisfied by the current observation.\n"
            "- Do not jump to a later stage unless the current observation directly satisfies every earlier stage's success signal.\n"
            "- Default to one main subtask; use two only if there are truly distinct parallel objectives.\n"
            "- Do not repeat the same task wording multiple times in one response.\n"
            "- The subtasks must be milestone-shaped, not gesture-shaped.\n"
            "- Each subtask should usually take about 2-6 atomic actions, not one trivial click and not a full multi-stage workflow.\n"
            "- Use task history to avoid repeating failed or no-progress strategies.\n"
            "- Do not describe low-level actions such as tap, click, press, select, type, input, swipe, scroll, open icon, or long press unless the user's goal itself is that operation.\n"
            "- If you first think of a low-level interaction, rewrite it as the state change or functional result that interaction should achieve.\n"
            "- If the current observation already satisfies one stage's success signal, move to the next unmet stage instead of repeating the same one.\n"
            "- If the current screen conflicts with the frozen plan but there is no clear repair or state-loss evidence, interpret the screen using the existing milestones rather than inventing a new plan.\n"
            "- If the current screen is launcher, home, app entry, list, folder, navigation, or any otherwise ambiguous state, prefer the earliest directly supported stage rather than assuming later progress.\n"
            "- current_stage_id must match the returned subtasks semantically.\n"
            "- complete_goal is only a completion candidate, not final success.\n"
            "- Return complete_goal only if the current observation directly shows that the overall goal is already achieved.\n"
            "- Return complete_goal only if the current observation also directly satisfies the final milestone in the frozen stage plan.\n"
            "- Never return complete_goal based only on user_goal, common sense, inferred path completion, or the mere existence of the frozen stage plan.\n"
            "- If task history contains planner_complete_but_task_check_failed, treat that as a planner failure and return a repair, verification, or progress-making subtask unless the current screen now provides new direct evidence that the evaluator can pass.\n"
            "- A launcher, home, app entry, folder navigation, list, or detail screen without direct final-result evidence must not return complete_goal.\n\n"
            "Output:\n"
            '{"tool":"complete_goal","message":"..."}\n\n'
            "or\n\n"
            '{"tool":"set_tasks","current_stage_id":1,"tasks":[{"task":"Precondition: ... Goal: ...","reason":"..."}]}\n\n'
            "or, when one subtask intentionally advances multiple adjacent frozen stages:\n\n"
            '{"tool":"set_tasks","current_stage_id":1,"covered_stage_ids":[1,2],"tasks":[{"task":"Precondition: ... Goal: ...","reason":"..."}]}\n\n'
            "Constraints:\n"
            "- Return valid JSON only.\n"
            "- If tool is set_tasks, return at least one task.\n"
            "- Return covered_stage_ids only when one subtask intentionally advances multiple adjacent frozen stages.\n"
            "- Do not include stage_plan unless explicitly repairing it, which is not part of this call."
        )

    def _build_generic_system_prompt(self) -> str:
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
            "Your task has two layers:\n"
            "1. Build or refresh a high-level stage plan for the whole task.\n"
            "2. Choose the current stage and output only the next functional subtasks for that stage.\n\n"
            "Stage-plan rules:\n"
            f"- Build a stage plan with 1-{self.config.max_subtasks} stages.\n"
            "- Every stage must be a stable milestone, not a raw UI action.\n"
            "- Stages must cover the main path needed to finish the whole task.\n"
            "- Stage titles must not use low-level actions such as tap, click, type, input, swipe, scroll, or press.\n"
            "- When the task clearly spans navigation, editing, and verification, do not collapse the whole workflow into one generic stage.\n"
            "- Adjacent stages must describe distinct milestones, not the same milestone phrased three ways.\n"
            "- If a remembered stage plan is still consistent with the current screen, keep it stable instead of inventing a new decomposition.\n"
            "- Do not move the current stage backward unless the current screen or task history clearly shows state loss, validator failure, or a repair path.\n\n"
            "Current-step planning rules:\n"
            f"- Devise the next 1-{self.config.max_subtasks} functional steps.\n"
            "- If you return set_tasks, you must also return stage_plan, current_stage_id, and at least one task.\n"
            "- Focus on what state to reach next, not how to click or type.\n"
            "- The subtasks must advance the selected current stage.\n"
            "- If the current screen already satisfies a later stage, skip earlier stages and plan from the later stage.\n"
            "- Use task history to avoid repeating failed or no-progress strategies.\n"
            "- Default to one main subtask for the current stage, not multiple near-duplicate tasks.\n"
            "- Do not repeat the same task wording multiple times in one response.\n"
            "- Do not default to one field, one button, one tab, or one directory click per subtask.\n"
            "- Only isolate a single field, button, or entry-point retry when task history clearly shows that exact target failed and no higher-level milestone wording would be accurate.\n"
            "- The subtasks must semantically match current_stage_id. For example, a fill-form subtask cannot belong to a save-and-verify stage.\n"
            "- Do not declare complete_goal unless the current screen and history provide direct evidence that the requested result already exists.\n\n"
            "Step format:\n"
            "- Each step must be a functional goal.\n"
            "- Each step should produce a verifiable UI state change or a clearly checkable state.\n"
            f"- Each step should usually take about 2-6 atomic actions, not one trivial click and not a long multi-stage workflow.\n"
            "- Use 'Precondition: ... Goal: ...' for every step.\n"
            "- For the first step, use 'Precondition: None. Goal: ...' if needed.\n"
            "- Do not describe low-level operations such as tap, click, input, type, swipe, scroll, press, select, open icon, or long press unless the user's goal itself is the operation.\n"
            "- If you first think of a low-level interaction, rewrite it as the screen state or functional result that interaction should achieve, then output that higher-level goal instead.\n"
            "- If a step depends on information discovered earlier, include that information explicitly in the step.\n\n"
            "Output contract:\n"
            "- stage_plan is the whole-task milestone decomposition.\n"
            "- current_stage_id identifies the milestone being advanced now.\n"
            "- tasks contains the current actionable milestone wording for that stage.\n"
            "- The task goal should name the intended state change or functional result, not the literal UI gesture.\n"
            "- A valid task is milestone-shaped; an invalid task is gesture-shaped.\n\n"
            "Output:\n"
            "If the overall goal is achieved, return only:\n"
            '{"tool":"complete_goal","message":"..."}\n\n'
            "Otherwise return only:\n"
            '{"tool":"set_tasks","stage_plan":[{"stage_id":1,"title":"...","success_signal":"..."}],'
            '"current_stage_id":1,"tasks":[{"task":"Precondition: ... Goal: ...","reason":"..."}]}\n\n'
            "Constraints:\n"
            "- Return valid JSON only.\n"
            "- Do not output code.\n"
            "- Do not output low-level action scripts.\n"
            "- After the planned steps are executed, you will be called again with the new device state."
        )

    def _build_legacy_stage_plan_system_prompt(self) -> str:
        return (
            "You are an Android Task Planner.\n\n"
            "Your only job in this call is to decompose the user's overall goal into a whole-task stage plan.\n"
            "You do not see the current screen in this call.\n\n"
            "Rules:\n"
            "- Build exactly 3-5 high-level milestones for the full task.\n"
            "- For contact-creation tasks, use the canonical whole-task decomposition rather than a single generic stage.\n"
            "- Milestones must be verifiable outcomes, not taps or field-level actions.\n"
            "- Do not include current_stage_id.\n"
            "- Do not include tasks.\n"
            "- Return valid JSON only.\n\n"
            'Output:\n{"stage_plan":[{"stage_id":1,"title":"...","success_signal":"..."}]}'
        )

    def _build_legacy_system_prompt(self) -> str:
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
            "Your task has two layers:\n"
            "1. Build or refresh a high-level stage plan for the whole task.\n"
            "2. Choose the current stage and output only the next functional subtasks for that stage.\n\n"
            "Stage-plan rules:\n"
            f"- Build a stage plan with 1-{self.config.max_subtasks} stages.\n"
            "- Every stage must be a stable milestone, not a raw UI action.\n"
            "- Keep the stage plan stable across rounds unless the observed state clearly changed or a repair path is required.\n"
            "- When the task clearly spans navigation, editing, and verification, do not collapse it into one generic stage.\n"
            "- Do not move backward to an earlier stage unless history clearly shows state loss, validator failure, or a repair path.\n\n"
            "Current-step planning rules:\n"
            f"- Devise the next 1-{self.config.max_subtasks} functional steps.\n"
            "- Focus on what to achieve, not how to click or type.\n"
            "- Do not default to one field, one button, or one tab per subtask.\n"
            "- Use field-specific retries only when history shows a specific field failed.\n"
            "- Do not declare complete_goal unless the current screen and history directly support that the requested result already exists.\n\n"
            "Step format:\n"
            "- Each step must be a functional goal.\n"
            "- Each Goal must be one short natural-language description of a small objective.\n"
            "- Completing each Goal must produce a verifiable UI state change.\n"
            f"- Each Goal should usually take about 2-6 atomic actions, not one trivial click and not a long multi-stage workflow.\n"
            "- Do not describe the Goal as a low-level operation such as tap, click, input, type, swipe, scroll, or press unless that action itself is the user's goal.\n"
            "- Use 'Precondition: ... Goal: ...' for every step.\n"
            "- For the first step, use 'Precondition: None. Goal: ...' if needed.\n"
            "- When multiple related contact fields must be filled on a contact editor screen, prefer one coherent form-filling subtask.\n"
            "- Only split a contact form into individual field subtasks if a specific field previously failed and needs isolated retry.\n"
            "- If history shows saved_but_task_check_failed, saved_with_wrong_identity, or field_misgrounded, do not return complete_goal; return a repair-oriented functional subtask instead.\n"
            "- If history already shows grouped form partial progress on a contact editor, do not fall back to one field per step unless a specific field retry is explicitly required.\n"
            "- For an incomplete contact-creation task, do not return a one-stage plan. Use the canonical multi-stage decomposition and pick the current stage.\n\n"
            "Canonical contact-creation milestones:\n"
            "- Open the Phone app.\n"
            "- Reach the contact creation entry point.\n"
            "- Fill in <name> and <phone> in the contact form.\n"
            "- Save and verify the created contact.\n\n"
            "Examples:\n"
            "- Good: 'Open the Phone app.'\n"
            "- Good: 'Reach the contact creation entry point.'\n"
            "- Good: 'Fill in Mia Garcia and +18856139998 in the contact form.'\n"
            "- Good stage title: 'Save and verify the created contact.'\n"
            "- Bad: 'Tap the Phone app icon.'\n"
            "- Bad: 'Click the Contacts tab.'\n"
            "- Bad: 'Tap the Create new contact button.'\n"
            "- Bad: \"Enter the first name 'Mia' into the First name field.\"\n"
            "- Bad: \"Enter the last name 'Garcia' into the Last name field.\"\n"
            "- Bad: \"Enter the phone number '+18856139998' into the Phone field.\"\n\n"
            "Output:\n"
            "If the overall goal is achieved, return only:\n"
            '{"tool":"complete_goal","message":"..."}\n\n'
            "Otherwise return only:\n"
            '{"tool":"set_tasks","stage_plan":[{"stage_id":1,"title":"...","success_signal":"..."}],'
            '"current_stage_id":1,"tasks":[{"task":"Precondition: ... Goal: ...","reason":"..."}]}\n\n'
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
        prompt = self._build_current_subtasks_user_prompt(
            user_goal=user_goal,
            stage_plan=self._extract_stage_plan_from_history(task_history),
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
        return self._build_current_subtasks_user_prompt(
            user_goal=user_goal,
            stage_plan=self._extract_stage_plan_from_history(task_history),
            observation=observation,
            task_history=task_history,
            memory_context=memory_context,
        )

    def _build_stage_plan_user_prompt(self, user_goal: str) -> str:
        return (
            f"User overall goal:\n{user_goal}\n\n"
            "Planner instruction:\n"
            f"- Decompose this goal into 3-{self.config.max_subtasks} high-level whole-task milestones.\n"
            "- Prefer 3 or 4 milestones when the task is simple. Do not force 5 milestones.\n"
            "- Do not use current observation, task history, or current screen assumptions.\n"
            "- Cover the whole main path from task start to verified completion.\n"
            "- Each milestone must be a verifiable intermediate result.\n"
            "- A milestone must be a UI phase, task phase, or state transition, not a single field, button, tab, click, or text input.\n"
            "- For form-filling tasks, combine related fields on the same form into one milestone.\n"
            "- Do not split name and phone number, title and body, amount and note, or similar same-form fields into separate milestones unless they require different screens.\n"
            "- Do not return current_stage_id.\n"
            "- Do not return tasks or subtasks.\n"
            "- Do not collapse the plan into a single stage such as opening an app.\n"
        )

    def _build_current_subtasks_user_content(
        self,
        user_goal: str,
        stage_plan: List[PlannerStage],
        observation: Dict[str, Any],
        task_history: List[Dict[str, Any]],
        memory_context: str,
    ) -> List[Dict[str, Any]]:
        prompt = self._build_current_subtasks_user_prompt(
            user_goal=user_goal,
            stage_plan=stage_plan,
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

    def _build_current_subtasks_user_prompt(
        self,
        user_goal: str,
        stage_plan: List[PlannerStage],
        observation: Dict[str, Any],
        task_history: List[Dict[str, Any]],
        memory_context: str,
    ) -> str:
        if self.config.prompt_profile == "legacy_contact_tuned":
            return self._build_legacy_user_prompt(
                user_goal=user_goal,
                observation=observation,
                task_history=task_history,
                memory_context=memory_context,
            )
        if self.config.prompt_profile == "generic_paper":
            return self._build_generic_example_prompt(
                user_goal=user_goal,
                stage_plan=stage_plan,
                observation=observation,
                task_history=task_history,
                memory_context=memory_context,
            )
        return self._build_generic_user_prompt(
            user_goal=user_goal,
            stage_plan=stage_plan,
            observation=observation,
            task_history=task_history,
            memory_context=memory_context,
        )

    def _build_generic_example_prompt(
        self,
        user_goal: str,
        stage_plan: List[PlannerStage],
        observation: Dict[str, Any],
        task_history: List[Dict[str, Any]],
        memory_context: str,
    ) -> str:
        current_activity = observation.get("current_activity") or "Unknown"
        app_name = observation.get("app_name") or "Unknown"
        screen_size = observation.get("screen_size") or {}
        ui_description = observation.get("ui_description") or "No visible UI elements available."
        history_text = self._format_task_history(task_history)
        stage_plan_text = self._format_frozen_stage_plan(stage_plan)
        memory_text = self._truncate(memory_context.strip(), self.config.max_memory_context_chars)

        return (
            "You are an Android Task Planner. Your job is to create short, functional plans "
            f"(1-{self.config.max_subtasks} steps) to achieve a user's goal on an Android device.\n\n"
            "**Inputs You Receive:**\n"
            "1. **User's Overall Goal.**\n"
            "2. **Current Device State:**\n"
            "   * A **screenshot** of the current screen with red bounding boxes and numeric labels.\n"
            "   * The numeric labels correspond to the indexed UI elements described below.\n"
            "   * The current visible Android activity.\n"
            "   * A **Visible UI elements summary** describing the current screen.\n"
            "3. **Complete Task History:**\n"
            "   * A record of ALL tasks that have been completed or failed throughout the session.\n"
            "   * For completed tasks, the results and any discovered information.\n"
            "   * For failed tasks, the detailed reasons for failure.\n"
            "   * This history persists across all planning cycles and is never lost.\n"
            "4. **Retrieved memory context** from previous trials when available.\n\n"
            f"User's Overall Goal:\n{user_goal}\n\n"
            "Current Device State:\n"
            f"- Current app: {app_name}\n"
            f"- Current activity: {current_activity}\n"
            f"- Screen size: {json.dumps(screen_size, ensure_ascii=False)}\n"
            "Visible UI elements summary:\n"
            f"{ui_description}\n\n"
            "Complete Task History:\n"
            f"{history_text}\n\n"
            "Retrieved memory context:\n"
            f"{memory_text if memory_text else 'None'}\n\n"
            "Frozen stage plan:\n"
            f"{stage_plan_text}\n\n"
            "**Your Task:**\n"
            f"Given the goal, current state, and task history, devise the **next 1-{self.config.max_subtasks} functional steps**.\n"
            "Focus on what to achieve, not how. Planning fewer steps at a time improves accuracy, as the state can change.\n"
            "After your planned steps are executed, you will be invoked again with the new device state.\n"
            "At that time you must assess whether the overall user goal is complete, call `complete_goal` if it is complete, "
            "or return the next functional steps if it is not complete.\n\n"
            "**Step Format:**\n"
            "Return exactly one current subtask for the selected frozen stage.\n"
            "That subtask must be a functional goal.\n"
            "A **precondition** describing the expected starting screen/state for that subtask is required.\n"
            "Each task string must use \"Precondition: ... Goal: ...\".\n"
            "The Precondition must describe the current observable state BEFORE the Actor starts this subtask.\n"
            "It must be true in the current observation.\n"
            "Do not write a precondition that will only become true after completing an earlier stage.\n"
            "If no specific precondition is already true for the current screen, use "
            "\"Precondition: None. Goal: ...\".\n\n"
            "**Your Output:**\n"
            "If the overall user goal is complete, return only:\n"
            '{"tool":"complete_goal","message":"..."}\n\n'
            "Otherwise return only:\n"
            '{"tool":"set_tasks","current_stage_id":1,"tasks":[{"task":"Precondition: ... Goal: ...","reason":"..."}]}\n\n'
            "**Memory Persistence:**\n"
            "* You maintain a COMPLETE memory of ALL tasks across the entire session.\n"
            "* Every task that was completed or failed is preserved in your context.\n"
            "* Previously completed steps are never lost when returning new steps.\n"
            "* Use this accumulated knowledge to build progressively on successful steps.\n"
            "* When you see discovered information, use it explicitly in future tasks.\n\n"
            "**System Compatibility Constraints:**\n"
            "- The frozen stage plan already describes the whole task. Do not rewrite it in this call.\n"
            "- Use the frozen stage plan when choosing current_stage_id.\n"
            "- First determine which frozen milestones are directly supported by the current screen.\n"
            "- Then choose the earliest existing stage that is not yet directly satisfied.\n"
            "- Then return exactly one current subtask that advances that stage.\n"
            "- Use task history to avoid repeating failed or no-progress strategies.\n"
            "- Do not return duplicate or near-duplicate tasks.\n"
            "- The subtask must be milestone-shaped, not gesture-shaped.\n"
            "- Do not output low-level actions or atomic UI operations.\n"
            "- If current_stage_id refers to an app-opening stage, the returned subtask must still be about opening or launching that app. Do not write a precondition that assumes the app is already open while keeping the same current_stage_id.\n"
            "- Reason must be consistent with the Precondition. Do not say an app needs to be opened first while also writing a precondition that the app is already open.\n"
            "- If the current observation does not satisfy a precondition, do not write that precondition. Put the missing requirement into the Goal of the current or earlier stage instead.\n"
            "- Each goal should be a short, functional objective.\n"
            "- Each goal should produce a checkable state change after completion.\n"
            "- If the current screen already exposes a useful entry point, describe the next state to reach rather than the literal tap.\n"
            "- If the current screen is already on a form or detail page, do not fall back to navigation-stage subtasks.\n"
            "- If you think of a tap, click, press, select, long press, scroll, or type action, rewrite it as the milestone state that action is meant to achieve before you output the task.\n"
            "- Ensure current_stage_id and the returned tasks describe the same milestone.\n"
            "- Return complete_goal only if the current screen directly shows the final requested result and directly satisfies the final frozen milestone.\n"
            "- If task history includes planner_complete_but_task_check_failed, treat it as a planner failure and return a repair, verification, or progress-making subtask unless the current screen now provides new direct evidence that the evaluator can pass.\n"
            "- Return valid JSON only.\n"
        )

    def _build_generic_user_prompt(
        self,
        user_goal: str,
        stage_plan: List[PlannerStage],
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
        stage_plan_text = self._format_frozen_stage_plan(stage_plan)
        repeat_warning_text = self._format_stage_repeat_warnings(task_history)
        contextual_hint_text = self._build_generic_contextual_planning_hints(observation)
        memory_text = self._truncate(memory_context.strip(), self.config.max_memory_context_chars)

        return (
            f"User overall goal:\n{user_goal}\n\n"
            "Current device state:\n"
            # f"- Current app: {app_name}\n"
            # f"- Current activity: {current_activity}\n"
            # f"- Screen size: {json.dumps(screen_size, ensure_ascii=False)}\n"
            "- You are given a screenshot with red bounding boxes and numeric labels.\n"
            "- The numeric labels correspond to UI element indexes below.\n"
            f"Visible UI elements summary:\n{ui_description}\n\n"
            f"Complete task history:\n{history_text}\n\n"
            f"Frozen stage plan:\n{stage_plan_text}\n\n"
            f"Stage repetition warnings:\n{repeat_warning_text}\n\n"
            f"Contextual planning hints:\n{contextual_hint_text}\n\n"
            "Retrieved memory context:\n"
            f"{memory_text if memory_text else 'None'}\n\n"
            "Planner instruction:\n"
            "- The frozen stage plan already describes the whole task. Do not rewrite it in this call.\n"
            "- Do not create separate subtasks for same-form fields such as name and phone unless task history shows a specific field failed.\n"
            "- First determine which frozen milestones are directly supported by the current screen.\n"
            "- Then choose the earliest existing stage that is not yet directly satisfied.\n"
            "- Then return exactly one current subtask that advances that stage.\n"
            "- Consider complete_goal only as an exception after the previous three checks.\n"
            "- If you return set_tasks, you must return current_stage_id and at least one task.\n"
            "- Every step must use 'Precondition: ... Goal: ...'.\n"
            "- The Precondition must describe the current observable state BEFORE the Actor starts this subtask.\n"
            "- It must be true in the current observation.\n"
            "- Do not write a precondition that will only become true after completing an earlier stage.\n"
            "- Do not output low-level actions or atomic UI operations.\n"
            "- Each goal should be a short, functional objective.\n"
            "- Each goal should produce a checkable state change after completion.\n"
            "- Use the current screen as the source of truth.\n"
            "- Use task history to avoid repeating failed or no-progress strategies.\n"
            "- Return one current subtask for the selected stage, not multiple parallel subtasks.\n"
            "- Do not return duplicate or near-duplicate tasks in the same response.\n"
            "- Do not make a single field, single button, single tab, or single directory click the default subtask granularity.\n"
            "- If current_stage_id refers to an app-opening stage, the returned subtask must still be about opening or launching that app. Do not write a precondition that says the app is already open while keeping the same current_stage_id.\n"
            "- Reason must be consistent with the Precondition. Do not say an app needs to be opened first while also writing a precondition that the app is already open.\n"
            "- If the current observation does not satisfy a precondition, do not write that precondition. Put the missing requirement into the Goal of the current or earlier stage instead.\n"
            "- If the current screen already exposes a useful entry point, describe the next state to reach rather than the literal tap.\n"
            "- If the current screen is already on a form or detail page, do not fall back to navigation-stage subtasks.\n"
            "- If you think of a tap, click, press, select, long press, scroll, or type action, rewrite it as the milestone state that action is meant to achieve before you output the task.\n"
            "- Ensure current_stage_id and the returned tasks describe the same milestone.\n"
            "- If the current screen already satisfies one stage's success signal, choose the next unmet stage.\n"
            "- If the current screen seems surprising but there is no explicit repair or state-loss evidence in history, interpret it using the frozen milestones instead of inventing a new plan.\n"
            "- If the screen is unstable, degraded, or ambiguous, prefer a safer recovery or navigation objective instead of assuming success.\n"
            "- Do not skip to a later stage unless the current screen directly supports all earlier milestones as already satisfied.\n"
            "- Return complete_goal only if the current screen directly shows the final requested result and directly satisfies the final frozen milestone.\n"
            "- Do not return complete_goal from launcher, home, app entry, folder navigation, list, or detail views unless those screens themselves directly prove the final requested result.\n"
            "- If task history includes planner_complete_but_task_check_failed, treat it as a planner failure and return a repair, verification, or progress-making subtask unless the current screen now provides new direct evidence that the evaluator can pass.\n"
        )

    def _build_legacy_user_prompt(
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
        stage_plan_text = self._format_remembered_stage_plan(task_history)
        repeat_warning_text = self._format_stage_repeat_warnings(task_history)
        contextual_hint_text = self._build_legacy_contextual_planning_hints(user_goal, observation)
        memory_text = self._truncate(memory_context.strip(), self.config.max_memory_context_chars)

        return (
            f"User overall goal:\n{user_goal}\n\n"
            "Current device state:\n"
            f"- Current app: {app_name}\n"
            f"- Current activity: {current_activity}\n"
            f"- Screen size: {json.dumps(screen_size, ensure_ascii=False)}\n"
            "- You are given both the raw screenshot and the labeled screenshot in this message.\n\n"
            f"Visible UI elements summary:\n{ui_description}\n\n"
            f"Complete task history:\n{history_text}\n\n"
            f"Remembered stage plan:\n{stage_plan_text}\n\n"
            f"Stage repetition warnings:\n{repeat_warning_text}\n\n"
            f"Contextual planning hints:\n{contextual_hint_text}\n\n"
            "Retrieved memory context:\n"
            f"{memory_text if memory_text else 'None'}\n\n"
            "Planner instruction:\n"
            "- Assess whether the overall user goal is already complete.\n"
            f"- If not complete, first refresh a 1-{self.config.max_subtasks} stage high-level plan, then return the next functional steps for the current stage.\n"
            "- Return stage_plan as a list of high-level milestones plus current_stage_id.\n"
            "- Every step must use 'Precondition: ... Goal: ...'.\n"
            "- Do not output low-level actions or atomic UI operations.\n"
            "- Each Goal should be one short natural-language small objective.\n"
            "- Each Goal should produce a verifiable UI state change after completion.\n"
            "- Each Goal should usually take around 2-6 atomic actions.\n"
            "- If the UI already shows an entry point, describe the state to reach, not the tap itself.\n"
            "- For text-entry subtasks, describe the desired filled state, not actor protocol names like input_text or type.\n"
            "- On a contact editor screen, prefer one subtask that fills the visible contact form section instead of one subtask per individual field unless a field-specific retry is necessary.\n"
            "- For contact creation, prefer these milestone goals exactly when they fit: Open the Phone app; Reach the contact creation entry point; Fill in <name> and <phone> in the contact form; Save and verify the created contact.\n"
            "- For an incomplete contact-creation task, the stage plan should normally contain those 4 milestones or a close equivalent. Do not collapse them into one stage.\n"
            "- If the current screen is already on a contact editor or contact detail page, do not fall back to earlier navigation-stage subtasks.\n"
            "- Do not move backward to an earlier stage unless history clearly shows state loss, validator failure, or a repair path.\n"
            "- If task history says a referenced UI target was absent from the observation, replan to a safer functional milestone instead of repeating the same target claim.\n"
            "- If task history shows saved_but_task_check_failed, saved_with_wrong_identity, or field_misgrounded, output a repair subtask such as re-entering the contact editor or verifying and correcting the required contact fields.\n"
            "- If the current screen is a contact detail page but the validator has not yet succeeded, do not output complete_goal.\n"
            "- Ensure current_stage_id and the returned subtask describe the same milestone. For example, do not put a fill-form goal under a save stage.\n"
            "- If task history includes planner_complete_but_task_check_failed, do not return complete_goal again without new direct evidence on the current screen.\n"
        )

    @staticmethod
    def _format_frozen_stage_plan(stage_plan: List[PlannerStage]) -> str:
        if not stage_plan:
            return "None"
        lines = []
        for stage in stage_plan:
            lines.append(
                f"- Stage {stage.stage_id}: {stage.title} (success signal: {stage.success_signal})"
            )
        return "\n".join(lines)

    def _format_task_history(self, task_history: List[Dict[str, Any]]) -> str:
        if not task_history:
            return "No previous subtasks."

        stable_lines: list[str] = []
        warning_lines: list[str] = []
        recent_items = self._compress_history_items(task_history[-self.config.max_history_items :])
        summarized_subtasks = {
            str(item.get("task") or item.get("subtask") or "").strip()
            for item in recent_items
            if item.get("source") == "subtask_summary"
        }
        for index, item in enumerate(recent_items, start=1):
            if item.get("source") == "stage_plan":
                continue
            task = str(item.get("task") or item.get("subtask") or "").strip() or "Unknown task"
            if item.get("source") == "actor" and task in summarized_subtasks:
                continue
            status = str(item.get("status") or item.get("result") or "unknown").strip()
            reason = str(item.get("reason") or item.get("summary") or item.get("note") or "").strip()
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

    def _format_remembered_stage_plan(self, task_history: List[Dict[str, Any]]) -> str:
        recent_stage_entry = self._extract_recent_stage_plan_entry(task_history)
        if recent_stage_entry is None:
            return "None"
        stage_plan = recent_stage_entry.get("stage_plan") or []
        current_stage_id = recent_stage_entry.get("current_stage_id")
        covered_stage_ids = recent_stage_entry.get("covered_stage_ids") or []
        lines = []
        for stage in stage_plan:
            lines.append(
                f"- Stage {stage.get('stage_id')}: {stage.get('title')} "
                f"(success signal: {stage.get('success_signal')})"
            )
        lines.append(f"- Last known current stage: {current_stage_id if current_stage_id is not None else 'Unknown'}")
        if covered_stage_ids:
            lines.append(f"- Last covered stages: {covered_stage_ids}")
        return "\n".join(lines) if lines else "None"

    @staticmethod
    def _extract_stage_plan_from_history(task_history: List[Dict[str, Any]]) -> List[PlannerStage]:
        recent_stage_entry = AndroidTaskPlanner._extract_recent_stage_plan_entry(task_history)
        if recent_stage_entry is None:
            return []
        stage_plan: list[PlannerStage] = []
        for item in recent_stage_entry.get("stage_plan") or []:
            if not isinstance(item, dict):
                continue
            stage_id = item.get("stage_id")
            title = item.get("title")
            success_signal = item.get("success_signal")
            if isinstance(stage_id, int) and isinstance(title, str) and isinstance(success_signal, str):
                stage_plan.append(
                    PlannerStage(
                        stage_id=stage_id,
                        title=title,
                        success_signal=success_signal,
                    )
                )
        return stage_plan

    @staticmethod
    def _format_stage_repeat_warnings(task_history: List[Dict[str, Any]]) -> str:
        warnings: list[str] = []
        if any(str(item.get("status") or "") == "same_subtask_no_progress" for item in task_history):
            warnings.append("- Avoid repeating the same stage formulation that already produced no progress.")
        if any(str(item.get("status") or "") == "planner_stage_regressed" for item in task_history):
            warnings.append("- Do not regress to an earlier stage unless the current screen clearly shows state loss or a repair path.")
        if any(str(item.get("status") or "") == "planner_complete_but_task_check_failed" for item in task_history):
            warnings.append("- A previous complete_goal claim was rejected by the evaluator. Treat it as planner failure and return a repair, verification, or progress-making subtask unless the current screen now provides new direct evidence.")
        return "\n".join(warnings) if warnings else "None"

    @staticmethod
    def _build_generic_contextual_planning_hints(observation: Dict[str, Any]) -> str:
        hints: list[str] = []
        activity_lower = str(observation.get("current_activity") or "").lower()
        ui_summary_lower = str(observation.get("ui_description") or "").lower()
        current_app = str(observation.get("app_name") or "").lower()
        if any(token in activity_lower for token in ("editor", "compose", "detail", "viewer", "settings")):
            hints.append("- The current screen looks like an editing or detail stage. Do not fall back to early navigation-stage subtasks.")
        if any(token in ui_summary_lower for token in ("dialog", "confirm", "allow", "delete", "save")):
            hints.append("- The current screen may be asking for confirmation or finalization. Prefer a verify, confirm, delete, save, or submit milestone over restarting navigation.")
        if any(token in ui_summary_lower for token in ("search", "results", "list", "folder", "category")):
            hints.append("- The current screen looks like a navigation or selection stage. Describe the destination state to reach, not the literal tap or scroll.")
        if any(token in current_app for token in ("launcher", "home")):
            hints.append("- The current screen appears to be a launcher or home surface. The first milestone should usually be opening the target app, not interacting with unrelated UI.")
        if not hints and (activity_lower or ui_summary_lower):
            hints.append("- Use the current screen to infer whether the next milestone is opening, navigating, editing, verifying, deleting, submitting, or confirming.")
        if not hints:
            return "None"
        return "\n".join(hints)

    @staticmethod
    def _build_legacy_contextual_planning_hints(user_goal: str, observation: Dict[str, Any]) -> str:
        hints: list[str] = []
        goal_lower = user_goal.lower()
        activity_lower = str(observation.get("current_activity") or "").lower()
        ui_summary_lower = str(observation.get("ui_description") or "").lower()
        if "contact" in goal_lower:
            hints.append("- For contact tasks, prefer milestone wording over button-click wording.")
            hints.append("- If a create-contact entry button is visible, use the milestone 'Reach the contact creation entry point' rather than 'Tap the Create new contact button'.")
            hints.append("- If the contact editor is visible, use one grouped form-fill subtask with the target name and phone number, not one field at a time.")
            hints.append("- If the contact detail page is visible but success is not yet verified, use a save-or-verify milestone, not complete_goal.")
        if "contacteditoractivity" in activity_lower:
            hints.append("- The current screen is already a contact editor. Do not output navigation subtasks or single-field subtasks.")
        if "create new contact" in ui_summary_lower or "add contact" in ui_summary_lower:
            hints.append("- A create-contact entry point is visible now. The next milestone should describe reaching or using that entry point, not the literal tap action.")
        if not hints:
            return "None"
        return "\n".join(hints)

    @staticmethod
    def _extract_recent_stage_plan_entry(task_history: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        for item in reversed(task_history):
            if item.get("source") == "stage_plan" and item.get("stage_plan"):
                return item
        return None


def extract_json_object(text: str) -> Optional[dict]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    cleaned = cleaned.strip()

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


def _extract_json_array(text: str) -> Any:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def parse_precondition_goal(task_text: str) -> tuple[str, str] | None:
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


def synthesize_precondition_goal_task(task_text: str) -> str | None:
    stripped = str(task_text or "").strip()
    if not stripped:
        return None
    if re.search(r"\bprecondition\s*:", stripped, flags=re.IGNORECASE):
        return None
    if re.search(r"\bgoal\s*:", stripped, flags=re.IGNORECASE):
        return None
    return f"Precondition: None. Goal: {stripped}"


def _validate_stage_subtask_alignment(
    *,
    stage_plan: list[PlannerStage],
    current_stage_id: int | None,
    subtasks: list[PlannerSubtask],
) -> str | None:
    if not stage_plan or current_stage_id is None or not subtasks:
        return None
    current_stage = next((stage for stage in stage_plan if stage.stage_id == current_stage_id), None)
    if current_stage is None or not _looks_like_app_open_stage(current_stage.title):
        return None
    stage_target = _extract_named_app_target(current_stage.title)
    success_signal = str(current_stage.success_signal or "").strip().lower()
    if stage_target is None:
        return None
    for subtask in subtasks:
        if "app is open" in success_signal and "app is open" in subtask.precondition.lower() and not re.search(
            r"^(open|launch)\b",
            subtask.goal.lower(),
        ):
            return (
                "Planner returned a subtask whose precondition already assumes the current app-opening stage is satisfied. "
                "Keep the subtask aligned with current_stage_id until that stage is actually advanced."
            )
        if _mentions_open_app_completion(subtask.precondition, stage_target) and not _looks_like_open_app_goal(
            subtask.goal,
            stage_target,
        ):
            return (
                "Planner returned a subtask whose precondition already assumes the current app-opening stage is satisfied. "
                "Keep the subtask aligned with current_stage_id until that stage is actually advanced."
            )
    return None


def _looks_like_app_open_stage(text: str) -> bool:
    lowered = text.strip().lower()
    return bool(re.search(r"^(open|launch)\b", lowered))


def _extract_named_app_target(text: str) -> str | None:
    lowered = re.sub(r"\s+", " ", text.strip().lower())
    match = re.search(r"\b(?:open|launch)\s+the\s+(.+?)\s+app\b", lowered, flags=re.IGNORECASE)
    if match:
        return re.sub(r"\s+", " ", match.group(1).strip().lower())
    match = re.search(r"\b(?:open|launch)\s+(.+)$", lowered, flags=re.IGNORECASE)
    if not match:
        return None
    target = match.group(1).strip().strip(".")
    if not target:
        return None
    return target


def _mentions_open_app_completion(text: str, app_target: str) -> bool:
    lowered = re.sub(r"\s+", " ", text.strip().lower())
    if app_target not in lowered:
        return False
    return bool(
        re.search(r"\b(is|already)\s+open\b", lowered)
        or re.search(rf"\b{re.escape(app_target)}(?: app)? is open\b", lowered)
    )


def _looks_like_open_app_goal(text: str, app_target: str) -> bool:
    lowered = re.sub(r"\s+", " ", text.strip().lower())
    if app_target not in lowered:
        return False
    return bool(re.search(r"\b(open|launch|reach)\b", lowered))


def parse_stage_plan_payload(payload: dict[str, Any]) -> tuple[list[PlannerStage], int | None, list[int], str | None]:
    stage_plan_payload = payload.get("stage_plan")
    current_stage_id = payload.get("current_stage_id")
    covered_stage_ids_payload = payload.get("covered_stage_ids")
    if stage_plan_payload is None:
        if current_stage_id is not None and not isinstance(current_stage_id, int):
            return [], None, [], "Planner current_stage_id must be an integer when provided."
        covered_stage_ids, covered_error = _parse_covered_stage_ids(
            covered_stage_ids_payload,
            current_stage_id=current_stage_id,
            valid_stage_ids=None,
        )
        if covered_error is not None:
            return [], None, [], covered_error
        return [], current_stage_id, covered_stage_ids, None
    if not isinstance(stage_plan_payload, list) or not 1 <= len(stage_plan_payload) <= 5:
        return [], None, [], "Planner stage_plan must be a list with 1-5 stages."

    stage_plan: list[PlannerStage] = []
    seen_ids: set[int] = set()
    for index, item in enumerate(stage_plan_payload):
        if not isinstance(item, dict):
            return [], None, [], f"Planner stage_plan item at index {index} is not an object."
        stage_id = item.get("stage_id")
        title = item.get("title")
        success_signal = item.get("success_signal")
        if not isinstance(stage_id, int):
            return [], None, [], f"Planner stage_plan item at index {index} must include integer stage_id."
        if stage_id in seen_ids:
            return [], None, [], f"Planner stage_plan contains duplicate stage_id {stage_id}."
        if not isinstance(title, str) or not title.strip():
            return [], None, [], f"Planner stage_plan item at index {index} is missing title."
        if not isinstance(success_signal, str) or not success_signal.strip():
            return [], None, [], f"Planner stage_plan item at index {index} is missing success_signal."
        if _is_low_level_stage_title(title):
            return [], None, [], f"Planner stage title must be a high-level milestone, not a low-level action: {title!r}."
        stage_plan.append(
            PlannerStage(
                stage_id=stage_id,
                title=title.strip(),
                success_signal=success_signal.strip(),
            )
        )
        seen_ids.add(stage_id)

    if current_stage_id is not None:
        if not isinstance(current_stage_id, int):
            return [], None, [], "Planner current_stage_id must be an integer when provided."
        if current_stage_id not in seen_ids:
            return [], None, [], f"Planner current_stage_id {current_stage_id} does not exist in stage_plan."
    covered_stage_ids, covered_error = _parse_covered_stage_ids(
        covered_stage_ids_payload,
        current_stage_id=current_stage_id,
        valid_stage_ids=seen_ids,
    )
    if covered_error is not None:
        return [], None, [], covered_error
    return stage_plan, current_stage_id, covered_stage_ids, None


def _is_low_level_stage_title(title: str) -> bool:
    lowered = title.strip().lower()
    low_level_action_markers = (
        "click ",
        "tap ",
        "swipe ",
        "scroll ",
        "press ",
        "long press ",
    )

    return any(lowered.startswith(marker) for marker in low_level_action_markers)


def _parse_covered_stage_ids(
    covered_stage_ids_payload: Any,
    *,
    current_stage_id: int | None,
    valid_stage_ids: set[int] | None,
) -> tuple[list[int], str | None]:
    if covered_stage_ids_payload is None:
        return [], None
    if not isinstance(covered_stage_ids_payload, list) or not covered_stage_ids_payload:
        return [], "Planner covered_stage_ids must be a non-empty list of integers when provided."
    if not all(isinstance(stage_id, int) for stage_id in covered_stage_ids_payload):
        return [], "Planner covered_stage_ids must contain only integers."

    covered_stage_ids = list(covered_stage_ids_payload)
    if len(set(covered_stage_ids)) != len(covered_stage_ids):
        return [], "Planner covered_stage_ids must not contain duplicates."
    if covered_stage_ids != sorted(covered_stage_ids):
        return [], "Planner covered_stage_ids must be sorted in ascending order."
    if any(right - left != 1 for left, right in zip(covered_stage_ids, covered_stage_ids[1:])):
        return [], "Planner covered_stage_ids must contain only contiguous adjacent stage ids."
    if current_stage_id is not None and current_stage_id not in covered_stage_ids:
        return [], "Planner covered_stage_ids must include current_stage_id."
    if valid_stage_ids is not None and any(stage_id not in valid_stage_ids for stage_id in covered_stage_ids):
        return [], "Planner covered_stage_ids must only reference stage ids from stage_plan."
    return covered_stage_ids, None


def synthesize_subtask_from_stage_plan(
    *,
    stage_plan: list[PlannerStage],
    current_stage_id: int | None,
    default_actor_name: str,
) -> PlannerSubtask | None:
    if not stage_plan or current_stage_id is None:
        return None
    current_stage = next((stage for stage in stage_plan if stage.stage_id == current_stage_id), None)
    if current_stage is None:
        return None
    previous_stage = next((stage for stage in stage_plan if stage.stage_id == current_stage_id - 1), None)
    precondition = "None" if previous_stage is None else f"{previous_stage.title}."
    return PlannerSubtask(
        precondition=precondition,
        goal=current_stage.title,
        reason=f"Advance the current stage: {current_stage.success_signal}",
        agent=default_actor_name,
    )


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
