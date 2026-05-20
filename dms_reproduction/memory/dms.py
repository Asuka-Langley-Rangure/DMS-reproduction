# dms_reproduction/memory/dms.py

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .base import MemoryProvider
from .formatting import DMSContextFormatter, DMSContextFormatterConfig
from .pruning import DMSPruner, PruningConfig
from .retrieval import MemoryCandidate, MemoryQuery, parse_subtask_text, retrieve
from .store import DMSMemoryStore, JsonDMSMemoryStore
from .survival_value import SurvivalValueCalculator, SurvivalValueConfig
from .types import (
    DMSMemoryMeta,
    DMSMemoryRecord,
    DMSMemoryStatus,
    MemoryReadResult,
    compute_trajectory_hash,
    make_memory_id,
    utc_timestamp,
)


@dataclass
class DMSMemoryConfig:
    """
    Main configuration for DMSMemoryProvider.

    This config intentionally keeps risk and mutation lightweight so the first
    implementation can run without extra modules.
    """

    memory_root: Optional[Path] = None

    # retrieval
    retrieval_threshold: float = 0.45
    retrieval_top_k: int = 3
    max_actor_memory_items: int = 1
    max_trajectory_steps: int = 8

    # storage
    min_trajectory_len_to_store: int = 2
    deduplicate_by_trajectory_hash: bool = True

    # mutation / evolutionary replacement
    enable_mutation: bool = False
    mutation_epsilon: float = 0.1
    replace_only_if_shorter: bool = True

    # lightweight risk control
    enable_risk_filter: bool = True
    risk_threshold: float = 0.70
    risky_memory_top_k_for_planner: int = 5

    # logical time
    increment_step_on_record: bool = True

    # sub-configs
    survival: SurvivalValueConfig = field(default_factory=SurvivalValueConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    formatter: DMSContextFormatterConfig = field(
        default_factory=DMSContextFormatterConfig
    )


class DMSMemoryProvider(MemoryProvider):
    """
    DMS implementation of MemoryProvider.

    Runtime responsibilities:
    1. Planner side:
       build_context() returns risk feedback.
    2. Actor side:
       build_actor_memory() retrieves active DMS memories and formats context.
    3. Feedback:
       record() updates reuse/success/failure/strike statistics.
    4. Storage:
       record_subtask_trajectory() stores successful subtask trajectories.
    5. Regulation:
       DMSPruner computes survival values and prunes low-value memories.
    """

    def __init__(
        self,
        store: Optional[DMSMemoryStore] = None,
        *,
        retriever_config: Optional[Any] = None,
        embedding_provider: Optional[Any] = None,
        config: Optional[DMSMemoryConfig] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.config = config or DMSMemoryConfig()

        if store is None:
            if self.config.memory_root is None:
                raise ValueError(
                    "Either store or config.memory_root must be provided."
                )
            store = JsonDMSMemoryStore(self.config.memory_root)

        self.store = store
        self.retriever_config = retriever_config
        self.embedding_provider = embedding_provider
        self.rng = rng or random.Random()

        self.survival_calculator = SurvivalValueCalculator(self.config.survival)
        self.pruner = DMSPruner(
            config=self.config.pruning,
            survival_calculator=self.survival_calculator,
        )
        self.formatter = DMSContextFormatter(self.config.formatter)

        self.logical_step: int = 0
        self.active_memory_ids_for_task: set[str] = set()
        self.last_read_result: Optional[MemoryReadResult] = None

    # ---------------------------------------------------------------------
    # Planner-side memory
    # ---------------------------------------------------------------------

    def build_context(
        self,
        user_goal: str,
        observation: Any,
        task_history: Any,
    ) -> str:
        """
        Build memory context for planner.

        We do not inject successful trajectories into planner.
        We only provide risk feedback to prevent repeated bad subtask plans.
        """
        risky_records = self._get_risky_records()
        return self.formatter.format_planner_risk_context(
            risky_records,
            max_items=self.config.risky_memory_top_k_for_planner,
        )

    # ---------------------------------------------------------------------
    # Actor-side memory
    # ---------------------------------------------------------------------

    def build_actor_context(
        self,
        user_goal: str,
        subtask: str,
        observation: Any,
        task_history: Any,
    ) -> str:
        """
        Compatibility interface.

        Existing runner can still call build_actor_context() and receive a
        string. New DMS-aware runner should call build_actor_memory().
        """
        return self.build_actor_memory(
            user_goal=user_goal,
            subtask=subtask,
            observation=observation,
            task_history=task_history,
        ).context

    def build_actor_memory(
        self,
        user_goal: str,
        subtask: str,
        observation: Any,
        task_history: Any,
    ) -> MemoryReadResult:
        """
        Retrieve DMS memories for the current actor subtask.

        This is the main read path:
        subtask -> MemoryQuery -> candidates -> retrieve -> selected memory
        -> load trajectory -> format actor context -> MemoryReadResult.
        """
        app_name = self._extract_app_name(observation)
        current_activity = self._extract_current_activity(observation)

        query = self._build_query(
            user_goal=user_goal,
            subtask=subtask,
            app_name=app_name,
            current_activity=current_activity,
        )
        if query is None:
            result = MemoryReadResult(
                context="",
                has_hit=False,
                reason="failed_to_parse_subtask",
            )
            self.last_read_result = result
            return result

        candidates = list(self.store.iter_active_candidates())
        if not candidates:
            result = MemoryReadResult(
                context="",
                has_hit=False,
                reason="no_active_memory",
            )
            self.last_read_result = result
            return result

        retrieval_results = self._retrieve(query, candidates)
        if not retrieval_results:
            result = MemoryReadResult(
                context="",
                has_hit=False,
                reason="retrieval_returned_empty",
            )
            self.last_read_result = result
            return result

        retrieval_results = sorted(
            retrieval_results,
            key=lambda item: item.final_score,
            reverse=True,
        )[: self.config.retrieval_top_k]

        selected = self._select_safe_retrieval_result(retrieval_results)

        if selected is None:
            result = MemoryReadResult(
                context="",
                has_hit=False,
                retrieval_results=retrieval_results,
                reason="no_result_above_threshold_or_all_risky",
            )
            self.last_read_result = result
            return result

        memory_id = self._get_candidate_memory_id(selected.candidate)
        if memory_id is None:
            result = MemoryReadResult(
                context="",
                has_hit=False,
                retrieval_results=retrieval_results,
                reason="selected_candidate_missing_memory_id",
            )
            self.last_read_result = result
            return result

        record = self.store.get_memory(memory_id)
        if record is None:
            result = MemoryReadResult(
                context="",
                has_hit=False,
                retrieval_results=retrieval_results,
                reason=f"selected_memory_not_found:{memory_id}",
            )
            self.last_read_result = result
            return result

        trajectory = self.store.load_trajectory(memory_id)
        risk_score = self._compute_light_risk_score(record)
        should_mutate = self._decide_mutation(selected, risk_score)
        should_replay = not should_mutate

        context = self.formatter.format_actor_memory(
            record=record,
            trajectory=trajectory,
            sim_goal=selected.sim_goal,
            sim_precondition=selected.sim_precondition,
            final_score=selected.final_score,
            survival_value=record.meta.survival_value if record.meta else None,
            risk_score=risk_score,
            should_replay=should_replay,
            should_mutate=should_mutate,
            max_trajectory_steps=self.config.max_trajectory_steps,
        )

        read_result = MemoryReadResult(
            context=context,
            has_hit=True,
            selected_memory_id=memory_id,
            retrieval_results=retrieval_results,
            should_replay=should_replay,
            should_mutate=should_mutate,
            replay_trajectory=trajectory,
            final_score=selected.final_score,
            sim_goal=selected.sim_goal,
            sim_precondition=selected.sim_precondition,
            survival_value=record.meta.survival_value if record.meta else None,
            risk_score=risk_score,
            reason="memory_hit",
        )

        self.active_memory_ids_for_task.add(memory_id)
        self.last_read_result = read_result
        return read_result

    # ---------------------------------------------------------------------
    # Runtime feedback
    # ---------------------------------------------------------------------

    def record(self, event: Dict[str, Any]) -> None:
        """
        Record one runtime event and update selected memory if available.

        This is usually called after each subtask.
        """
        event = dict(event)
        event.setdefault("timestamp", utc_timestamp())

        if event.get("event_type") == "task_end":
            self.finalize_task(event)
            return

        if self.config.increment_step_on_record:
            self.logical_step += 1

        self.store.append_event(event)

        selected_memory_id = event.get("selected_memory_id")
        if selected_memory_id:
            self._update_selected_memory_from_event(selected_memory_id, event)

        active_count = self.store.stats().get("active_count", 0)
        if self.pruner.should_run(self.logical_step, active_count):
            self.pruner.prune(self.store, self.logical_step)

    def record_subtask_trajectory(self, record: Dict[str, Any]) -> None:
        """
        Store a successful subtask trajectory as a new DMS memory.

        This is called by task_runner only when a subtask is verified successful
        and the trajectory is clean enough.
        """
        trajectory = self._extract_trajectory(record)
        if len(trajectory) < self.config.min_trajectory_len_to_store:
            return

        trajectory_hash = compute_trajectory_hash(trajectory)

        if self.config.deduplicate_by_trajectory_hash:
            if self._has_duplicate_trajectory(trajectory_hash):
                self.store.append_event(
                    {
                        "event_type": "memory_skipped_duplicate_trajectory",
                        "trajectory_hash": trajectory_hash,
                        "timestamp": utc_timestamp(),
                    }
                )
                return

        dms_record = self._build_record_from_success_payload(
            payload=record,
            trajectory=trajectory,
            trajectory_hash=trajectory_hash,
        )

        # Lightweight evolutionary replacement:
        # If this new trajectory comes from a mutation of an old memory and is
        # shorter, replace the old memory.
        if (
            self.config.enable_mutation
            and self.last_read_result is not None
            and self.last_read_result.should_mutate
            and self.last_read_result.selected_memory_id
        ):
            old_id = self.last_read_result.selected_memory_id
            old_record = self.store.get_memory(old_id)
            if old_record is not None and self._should_replace(
                old_record=old_record,
                new_record=dms_record,
            ):
                self.store.replace_memory(
                    old_memory_id=old_id,
                    new_record=dms_record,
                    new_trajectory=trajectory,
                )
                return

        self.store.add_memory(dms_record, trajectory)

    def finalize_task(self, event: Dict[str, Any]) -> None:
        """
        Apply global task-level feedback to active memories.

        If the whole task fails, all memories that participated in this task
        receive a soft failure penalty.
        """
        event = dict(event)
        event.setdefault("timestamp", utc_timestamp())
        event.setdefault("event_type", "task_end")
        self.store.append_event(event)

        global_success = self._extract_bool_success(event)

        if global_success is False:
            for memory_id in list(self.active_memory_ids_for_task):
                record = self.store.get_memory(memory_id)
                if record is None or record.meta is None:
                    continue

                self.store.update_meta(
                    memory_id,
                    {
                        "failure_count_delta": 1,
                        "strike_count_delta": 1,
                    },
                )
                self._refresh_one_survival_value(memory_id)

        self.active_memory_ids_for_task.clear()
        self.last_read_result = None

        active_count = self.store.stats().get("active_count", 0)
        if self.pruner.should_run(self.logical_step, active_count):
            self.pruner.prune(self.store, self.logical_step)

    def reset(self) -> None:
        """
        Reset session-scoped state.

        We keep logical_step because DMS temporal decay should span across
        multiple tasks/runs.
        """
        self.active_memory_ids_for_task.clear()
        self.last_read_result = None

    # ---------------------------------------------------------------------
    # Query / retrieval helpers
    # ---------------------------------------------------------------------

    def _build_query(
        self,
        *,
        user_goal: str,
        subtask: str,
        app_name: Optional[str],
        current_activity: Optional[str],
    ) -> Optional[MemoryQuery]:
        """Parse subtask text and attach runtime context."""
        query = parse_subtask_text(subtask)

        if query is None:
            try:
                query = MemoryQuery(
                    subtask_text=subtask,
                    precondition="",
                    goal=subtask,
                    user_goal=user_goal,
                    app_name=app_name,
                    current_activity=current_activity,
                )
            except TypeError:
                return None

        self._safe_setattr(query, "user_goal", user_goal)
        self._safe_setattr(query, "app_name", app_name)
        self._safe_setattr(query, "current_activity", current_activity)

        if not getattr(query, "subtask_text", None):
            self._safe_setattr(query, "subtask_text", subtask)

        return query

    def _retrieve(
        self,
        query: MemoryQuery,
        candidates: List[MemoryCandidate],
    ) -> List[Any]:
        """
        Call the project's retrieval engine.

        If retriever_config is not provided, use a small lexical fallback so
        the DMS store remains testable.
        """
        if self.retriever_config is not None:
            return retrieve(
                query=query,
                candidates=candidates,
                config=self.retriever_config,
                embedding_provider=self.embedding_provider,
            )

        return self._fallback_lexical_retrieve(query, candidates)

    def _fallback_lexical_retrieve(
        self,
        query: MemoryQuery,
        candidates: List[MemoryCandidate],
    ) -> List[Any]:
        """
        Minimal fallback retrieval when no retriever_config is provided.

        This creates small result-like objects with the same attributes used by
        DMSMemoryProvider.
        """

        class _Result:
            def __init__(
                self,
                candidate: MemoryCandidate,
                sim_goal: float,
                sim_precondition: float,
                final_score: float,
                rank: int,
            ) -> None:
                self.candidate = candidate
                self.sim_goal = sim_goal
                self.sim_precondition = sim_precondition
                self.final_score = final_score
                self.retrieval_mode = "fallback_lexical_product"
                self.rank = rank

        q_goal = getattr(query, "goal", "") or ""
        q_pre = getattr(query, "precondition", "") or ""

        scored = []
        for candidate in candidates:
            sim_goal = self._jaccard(q_goal, candidate.goal or "")
            sim_pre = self._jaccard(q_pre, candidate.precondition or "")
            final = sim_goal * sim_pre if q_pre else sim_goal
            scored.append((candidate, sim_goal, sim_pre, final))

        scored.sort(key=lambda item: item[3], reverse=True)

        return [
            _Result(
                candidate=item[0],
                sim_goal=item[1],
                sim_precondition=item[2],
                final_score=item[3],
                rank=rank,
            )
            for rank, item in enumerate(scored, start=1)
        ]

    def _select_safe_retrieval_result(
        self,
        retrieval_results: List[Any],
    ) -> Optional[Any]:
        """
        Select the best retrieval result above threshold and below risk limit.
        """
        for result in retrieval_results:
            if result.final_score < self.config.retrieval_threshold:
                continue

            memory_id = self._get_candidate_memory_id(result.candidate)
            if memory_id is None:
                continue

            record = self.store.get_memory(memory_id)
            if record is None:
                continue

            risk_score = self._compute_light_risk_score(record)
            if self.config.enable_risk_filter and risk_score >= self.config.risk_threshold:
                self.store.mark_status(
                    memory_id,
                    DMSMemoryStatus.RISKY,
                    reason=f"risk_score_above_threshold:{risk_score:.4f}",
                )
                continue

            return result

        return None

    def _get_candidate_memory_id(
        self,
        candidate: MemoryCandidate,
    ) -> Optional[str]:
        """Extract memory_id from candidate or candidate.payload."""
        memory_id = getattr(candidate, "memory_id", None)
        if memory_id:
            return memory_id

        payload = getattr(candidate, "payload", None)
        if isinstance(payload, dict):
            return payload.get("memory_id")

        return None

    # ---------------------------------------------------------------------
    # Feedback update helpers
    # ---------------------------------------------------------------------

    def _update_selected_memory_from_event(
        self,
        memory_id: str,
        event: Dict[str, Any],
    ) -> None:
        """Update metadata of the memory used by the just-finished subtask."""
        record = self.store.get_memory(memory_id)
        if record is None or record.meta is None:
            return

        success = self._extract_event_success(event)
        update: Dict[str, Any] = {
            "last_used_step": self.logical_step,
        }

        if success is True:
            update["reuse_count_delta"] = 1
            update["success_count_delta"] = 1
            update["strike_count"] = max(0, record.meta.strike_count - 1)
        elif success is False:
            update["failure_count_delta"] = 1
            update["strike_count_delta"] = 1

        update.update(self._extract_feedback_counter_deltas(event))

        verifier_reason = event.get("verifier_reason")
        if verifier_reason:
            update["verifier_reason"] = verifier_reason

        self.store.update_meta(memory_id, update)
        self._refresh_one_survival_value(memory_id)

    def _extract_feedback_counter_deltas(
        self,
        event: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract environment feedback counters from MemoryEvent."""
        update: Dict[str, Any] = {}

        for key in [
            "invalid_action_count",
            "no_state_change_count",
            "parse_error_count",
            "execution_error_count",
        ]:
            value = event.get(key)
            if value is None:
                continue

            try:
                int_value = int(value)
            except (TypeError, ValueError):
                continue

            if int_value > 0:
                update[f"{key}_delta"] = int_value

        return update

    def _refresh_one_survival_value(self, memory_id: str) -> None:
        """Recompute survival value of a single memory and write it back."""
        record = self.store.get_memory(memory_id)
        if record is None:
            return

        value = self.survival_calculator.compute(record, self.logical_step)
        self.store.update_meta(memory_id, {"survival_value": value})

    # ---------------------------------------------------------------------
    # Successful trajectory storage helpers
    # ---------------------------------------------------------------------

    def _build_record_from_success_payload(
        self,
        payload: Dict[str, Any],
        trajectory: List[Dict[str, Any]],
        trajectory_hash: str,
    ) -> DMSMemoryRecord:
        """Convert runner's successful trajectory payload to DMSMemoryRecord."""
        memory_id = make_memory_id()

        subtask_text = (
            payload.get("subtask_text")
            or payload.get("subtask")
            or payload.get("task")
            or ""
        )

        precondition = payload.get("precondition")
        goal = payload.get("goal")

        if not precondition or not goal:
            parsed = parse_subtask_text(subtask_text)
            if parsed is not None:
                precondition = precondition or getattr(parsed, "precondition", "")
                goal = goal or getattr(parsed, "goal", "")

        precondition = precondition or ""
        goal = goal or subtask_text

        goal_embedding = payload.get("goal_embedding")
        precondition_embedding = payload.get("precondition_embedding")

        if goal_embedding is None:
            goal_embedding = self._embed_text(goal)

        if precondition_embedding is None:
            precondition_embedding = self._embed_text(precondition)

        meta = DMSMemoryMeta(
            memory_id=memory_id,
            created_step=self.logical_step,
            last_used_step=self.logical_step,
            created_time=utc_timestamp(),
            updated_time=utc_timestamp(),
            success_count=1,
            failure_count=0,
            strike_count=0,
            verifier_reason=payload.get("verifier_reason"),
            completion_message=payload.get("completion_message"),
            status=DMSMemoryStatus.ACTIVE,
        )

        meta.invalid_action_count = int(payload.get("invalid_action_count") or 0)
        meta.no_state_change_count = int(payload.get("no_state_change_count") or 0)
        meta.parse_error_count = int(payload.get("parse_error_count") or 0)
        meta.execution_error_count = int(payload.get("execution_error_count") or 0)

        record = DMSMemoryRecord(
            memory_id=memory_id,
            subtask_text=subtask_text,
            precondition=precondition,
            goal=goal,
            user_goal=payload.get("user_goal") or "",
            app_name=payload.get("app_name"),
            current_activity=payload.get("current_activity"),
            goal_embedding=goal_embedding,
            precondition_embedding=precondition_embedding,
            trajectory_path=None,
            trajectory_len=len(trajectory),
            trajectory_hash=trajectory_hash,
            meta=meta,
            extra={
                "round_id": payload.get("round_id"),
                "observation_digest": payload.get("observation_digest"),
            },
        )

        survival_value = self.survival_calculator.compute(record, self.logical_step)
        record.meta.survival_value = survival_value

        return record

    def _extract_trajectory(
        self,
        record: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Extract trajectory list from runner payload."""
        trajectory = record.get("trajectory") or record.get("actor_trajectory") or []

        if not isinstance(trajectory, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for step in trajectory:
            if isinstance(step, dict):
                normalized.append(step)
            else:
                normalized.append({"raw": step})

        return normalized

    def _has_duplicate_trajectory(self, trajectory_hash: str) -> bool:
        """Check whether the same trajectory hash already exists."""
        for record in self.store.iter_records():
            if record.trajectory_hash == trajectory_hash:
                return True
        return False

    def _should_replace(
        self,
        old_record: DMSMemoryRecord,
        new_record: DMSMemoryRecord,
    ) -> bool:
        """Decide whether a mutated trajectory should replace the old memory."""
        if not self.config.replace_only_if_shorter:
            return True

        return new_record.trajectory_len < old_record.trajectory_len

    # ---------------------------------------------------------------------
    # Lightweight risk / mutation helpers
    # ---------------------------------------------------------------------

    def _compute_light_risk_score(self, record: DMSMemoryRecord) -> float:
        """
        Compute a lightweight risk score without a separate risk.py module.

        This is a simple smoothed failure ratio:
            risk = (failure + 1) / (success + failure + 2)
        """
        if record.meta is None:
            return 0.5

        success = max(0, record.meta.success_count)
        failure = max(0, record.meta.failure_count)
        return (failure + 1.0) / (success + failure + 2.0)

    def _decide_mutation(
        self,
        retrieval_result: Any,
        risk_score: float,
    ) -> bool:
        """Decide whether to mutate instead of replay."""
        if not self.config.enable_mutation:
            return False

        if risk_score >= self.config.risk_threshold:
            return True

        return self.rng.random() < self.config.mutation_epsilon

    def _get_risky_records(self) -> List[DMSMemoryRecord]:
        """Return top risky active records for planner feedback."""
        records = list(self.store.iter_active_records())

        scored = []
        for record in records:
            risk = self._compute_light_risk_score(record)
            if record.meta is not None:
                record.meta.risk_score = risk

            if risk > 0.5 or (
                record.meta is not None and record.meta.strike_count > 0
            ):
                scored.append((risk, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored]

    # ---------------------------------------------------------------------
    # Observation / event helpers
    # ---------------------------------------------------------------------

    def _extract_app_name(self, observation: Any) -> Optional[str]:
        """Extract app name from observation if possible."""
        return self._extract_from_obj(
            observation,
            keys=["app_name", "package_name", "current_app"],
        )

    def _extract_current_activity(self, observation: Any) -> Optional[str]:
        """Extract current Android activity from observation if possible."""
        return self._extract_from_obj(
            observation,
            keys=["current_activity", "activity", "activity_name"],
        )

    def _extract_from_obj(
        self,
        obj: Any,
        keys: List[str],
    ) -> Optional[str]:
        """Extract a string field from dict-like or object-like observation."""
        if obj is None:
            return None

        if isinstance(obj, dict):
            for key in keys:
                value = obj.get(key)
                if value:
                    return str(value)
            return None

        for key in keys:
            value = getattr(obj, key, None)
            if value:
                return str(value)

        return None

    def _extract_event_success(self, event: Dict[str, Any]) -> Optional[bool]:
        """
        Infer subtask success from event.

        Priority:
        1. explicit success_check
        2. verifier_status
        3. status
        """
        if "success_check" in event:
            return self._to_optional_bool(event.get("success_check"))

        if "verifier_status" in event:
            return self._status_to_success(event.get("verifier_status"))

        if "status" in event:
            return self._status_to_success(event.get("status"))

        return None

    def _extract_bool_success(self, event: Dict[str, Any]) -> Optional[bool]:
        """Infer global task success from event."""
        for key in ["success", "global_success", "task_success", "success_check"]:
            if key in event:
                return self._to_optional_bool(event.get(key))
        return self._extract_event_success(event)

    def _to_optional_bool(self, value: Any) -> Optional[bool]:
        """Convert value to Optional[bool]."""
        if isinstance(value, bool):
            return value

        if value is None:
            return None

        if isinstance(value, (int, float)):
            return bool(value)

        text = str(value).strip().lower()
        if text in {"true", "success", "succeeded", "pass", "passed", "1", "yes"}:
            return True
        if text in {"false", "fail", "failed", "0", "no"}:
            return False

        return None

    def _status_to_success(self, value: Any) -> Optional[bool]:
        """Convert status text to Optional[bool]."""
        if value is None:
            return None

        text = str(value).strip().lower()

        success_tokens = ["success", "succeeded", "verified_success", "complete"]
        failure_tokens = ["fail", "failed", "error", "timeout", "rejected"]

        if any(token in text for token in success_tokens):
            return True

        if any(token in text for token in failure_tokens):
            return False

        return None

    # ---------------------------------------------------------------------
    # Embedding / misc helpers
    # ---------------------------------------------------------------------

    def _embed_text(self, text: str) -> Optional[List[float]]:
        """
        Compute embedding using the injected embedding provider.

        This is defensive because different embedding providers may expose
        different method names.
        """
        if not self.embedding_provider or not text:
            return None

        provider = self.embedding_provider

        for method_name in ["embed", "embed_text", "get_embedding", "encode"]:
            method = getattr(provider, method_name, None)
            if method is None:
                continue

            result = method(text)

            if isinstance(result, list):
                if result and isinstance(result[0], list):
                    return result[0]
                return result

        return None

    def _safe_setattr(self, obj: Any, name: str, value: Any) -> None:
        """Set attribute if possible."""
        try:
            setattr(obj, name, value)
        except Exception:
            pass

    def _jaccard(self, a: str, b: str) -> float:
        """Small lexical similarity fallback."""
        set_a = self._tokenize(a)
        set_b = self._tokenize(b)

        if not set_a and not set_b:
            return 1.0

        if not set_a or not set_b:
            return 0.0

        return len(set_a & set_b) / len(set_a | set_b)

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize text for fallback lexical retrieval."""
        text = (text or "").lower()
        tokens = []
        current = []

        for ch in text:
            if ch.isalnum():
                current.append(ch)
            else:
                if current:
                    tokens.append("".join(current))
                    current = []

        if current:
            tokens.append("".join(current))

        return set(tokens)
