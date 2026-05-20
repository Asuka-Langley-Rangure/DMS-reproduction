# dms_reproduction/memory/formatting.py

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .types import DMSMemoryRecord


@dataclass
class DMSContextFormatterConfig:
    """Configuration for formatting DMS memory context."""

    max_trajectory_steps: int = 8
    include_scores: bool = True
    include_metadata: bool = True
    include_safety_note: bool = True
    max_planner_risk_items: int = 5


class DMSContextFormatter:
    """
    Convert DMS memory records into prompt text.

    This class does not retrieve, update, or prune memories.
    It only formats memory information for planner/actor consumption.
    """

    def __init__(
        self,
        config: Optional[DMSContextFormatterConfig] = None,
    ) -> None:
        self.config = config or DMSContextFormatterConfig()

    def format_actor_memory(
        self,
        record: DMSMemoryRecord,
        trajectory: List[Dict[str, Any]],
        *,
        sim_goal: Optional[float] = None,
        sim_precondition: Optional[float] = None,
        final_score: Optional[float] = None,
        survival_value: Optional[float] = None,
        risk_score: Optional[float] = None,
        should_replay: bool = False,
        should_mutate: bool = False,
        max_trajectory_steps: Optional[int] = None,
    ) -> str:
        """
        Format one retrieved DMS memory for the actor.

        The actor sees this as a historical successful pattern, but it is warned
        not to blindly replay the trajectory if the current UI state does not
        match the precondition.
        """
        max_steps = max_trajectory_steps or self.config.max_trajectory_steps

        lines: List[str] = []
        lines.append("[DMS Retrieved Memory]")
        lines.append(f"Memory ID: {record.memory_id}")

        if self.config.include_scores:
            score_parts = []
            if sim_precondition is not None:
                score_parts.append(f"Precondition Similarity={sim_precondition:.4f}")
            if sim_goal is not None:
                score_parts.append(f"Goal Similarity={sim_goal:.4f}")
            if final_score is not None:
                score_parts.append(f"Final Retrieval Score={final_score:.4f}")
            if survival_value is not None:
                score_parts.append(f"Survival Value={survival_value:.4f}")
            if risk_score is not None:
                score_parts.append(f"Risk Score={risk_score:.4f}")

            if score_parts:
                lines.append("Scores: " + "; ".join(score_parts))

        lines.append("")
        lines.append("Matched Historical Subtask:")
        lines.append(f"Precondition: {record.precondition or 'None'}")
        lines.append(f"Goal: {record.goal or record.subtask_text}")

        if self.config.include_metadata and record.meta is not None:
            meta = record.meta
            lines.append("")
            lines.append("Memory Metadata:")
            lines.append(f"- Reuse Count: {meta.reuse_count}")
            lines.append(f"- Success Count: {meta.success_count}")
            lines.append(f"- Failure Count: {meta.failure_count}")
            lines.append(f"- Strike Count: {meta.strike_count}")
            if meta.verifier_reason:
                lines.append(f"- Last Verifier Reason: {meta.verifier_reason}")
            if record.app_name or record.current_activity:
                lines.append(
                    f"- App/Activity: {record.app_name or 'Unknown'} / "
                    f"{record.current_activity or 'Unknown'}"
                )

        lines.append("")
        lines.append("Historical Actor Trajectory:")
        lines.extend(self.format_trajectory_steps(trajectory, max_steps=max_steps))

        lines.append("")
        if should_mutate:
            lines.append(
                "DMS Control: MUTATION mode is enabled. Use this memory as a "
                "reference, but try to solve the subtask from scratch if a better "
                "or shorter path is visible."
            )
        elif should_replay:
            lines.append(
                "DMS Control: REPLAY mode is suggested. You may follow this "
                "historical trajectory if the current UI state satisfies the "
                "precondition."
            )
        else:
            lines.append(
                "DMS Control: Use this memory as a reference only. Do not blindly "
                "execute actions if the screen differs."
            )

        if self.config.include_safety_note:
            lines.append("")
            lines.append("Important Safety Notes:")
            lines.append(
                "- First check whether the current screen satisfies the historical "
                "precondition."
            )
            lines.append(
                "- If the precondition is not met, do NOT replay the trajectory. "
                "Instead, fail the subtask or choose a new strategy."
            )
            lines.append(
                "- Stop immediately after the requested subtask goal is achieved."
            )

        return "\n".join(lines).strip()

    def format_planner_risk_context(
        self,
        risky_records: Iterable[DMSMemoryRecord],
        *,
        max_items: Optional[int] = None,
    ) -> str:
        """
        Format high-risk memories for planner-side feedback.

        This should not provide full successful trajectories. It only tells the
        planner which plan patterns were risky or repeatedly failed.
        """
        limit = max_items or self.config.max_planner_risk_items
        records = list(risky_records)[:limit]

        if not records:
            return ""

        lines: List[str] = []
        lines.append("[DMS Risk Feedback]")
        lines.append(
            "The following subtask patterns have recently failed or become risky. "
            "Avoid repeating them directly; rewrite the plan if possible."
        )

        for idx, record in enumerate(records, start=1):
            meta = record.meta
            risk_score = meta.risk_score if meta is not None else 0.0
            failure_count = meta.failure_count if meta is not None else 0
            strike_count = meta.strike_count if meta is not None else 0
            reason = meta.verifier_reason if meta is not None else None

            lines.append("")
            lines.append(f"{idx}. Memory ID: {record.memory_id}")
            lines.append(f"   Precondition: {record.precondition or 'None'}")
            lines.append(f"   Goal: {record.goal or record.subtask_text}")
            lines.append(f"   Risk Score: {risk_score:.4f}")
            lines.append(f"   Failure Count: {failure_count}")
            lines.append(f"   Strike Count: {strike_count}")
            if reason:
                lines.append(f"   Last Failure Reason: {reason}")

        return "\n".join(lines).strip()

    def format_trajectory_steps(
        self,
        trajectory: List[Dict[str, Any]],
        *,
        max_steps: Optional[int] = None,
    ) -> List[str]:
        """
        Format trajectory steps into readable prompt lines.

        The trajectory format may vary depending on your actor implementation,
        so this function is intentionally defensive.
        """
        if not trajectory:
            return ["- No trajectory steps are available."]

        max_steps = max_steps or self.config.max_trajectory_steps
        selected = trajectory[:max_steps]

        lines: List[str] = []

        for idx, step in enumerate(selected, start=1):
            action_text = self.extract_action_text(step)
            thought = self.extract_thought_text(step)
            result = self.extract_result_text(step)

            lines.append(f"{idx}. Action: {action_text}")

            if thought:
                lines.append(f"   Thought: {thought}")

            if result:
                lines.append(f"   Result: {result}")

        if len(trajectory) > len(selected):
            lines.append(
                f"... {len(trajectory) - len(selected)} more step(s) omitted."
            )

        return lines

    def extract_action_text(self, step: Dict[str, Any]) -> str:
        """
        Extract the most useful action representation from a trajectory step.

        Supported common keys:
        - action_code
        - code
        - action
        - tool_call
        - raw_response
        """
        if not isinstance(step, dict):
            return str(step)

        for key in ["action_code", "code", "action", "tool_call"]:
            value = step.get(key)
            if value:
                return self._stringify(value)

        # Some actor logs may store parsed code under nested fields.
        parsed = step.get("parsed_action") or step.get("parsed")
        if parsed:
            return self._stringify(parsed)

        raw = step.get("raw_response")
        if raw:
            return self._shorten(self._stringify(raw), max_chars=500)

        return self._shorten(self._stringify(step), max_chars=500)

    def extract_thought_text(self, step: Dict[str, Any]) -> str:
        """Extract thought/analysis text from one trajectory step."""
        if not isinstance(step, dict):
            return ""

        for key in ["thought", "analysis", "reasoning"]:
            value = step.get(key)
            if value:
                return self._shorten(self._stringify(value), max_chars=300)

        return ""

    def extract_result_text(self, step: Dict[str, Any]) -> str:
        """Extract execution result text from one trajectory step."""
        if not isinstance(step, dict):
            return ""

        for key in ["execution_result", "result", "observation_after", "error"]:
            value = step.get(key)
            if value:
                return self._shorten(self._stringify(value), max_chars=300)

        return ""

    def _stringify(self, value: Any) -> str:
        """Convert a value into a stable readable string."""
        if isinstance(value, str):
            return value.strip()

        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(value)

    def _shorten(self, text: str, max_chars: int) -> str:
        """Shorten long text to keep prompts compact."""
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."