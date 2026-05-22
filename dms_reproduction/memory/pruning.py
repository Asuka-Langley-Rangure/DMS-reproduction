from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .store import DMSMemoryStore
from .survival_value import SurvivalValueCalculator
from .types import DMSMemoryRecord, DMSMemoryStatus, utc_timestamp

@dataclass
class PruningConfig:
    """
    目前策略：
    1. 硬性规则剪枝：某条 memory 被 verifier 多次判定失败后，直接剪掉
    2. 绝对生存值阈值剪枝
    3. 容量超出上限后触发后的长尾剪枝（论文中提及）
    4. 容量超出上限后先会尝试扩容，对肘部的survival_value和我们平均score做比较，如果大于就开始扩容
    """

    enabled: bool = True

    # When to run pruning
    prune_interval_steps: int = 10
    capacity_min: int = 50
    capacity_max: int = 150
    capacity_expand_step: int = 50

    # Hard rules
    strike_limit: int = 3
    min_survival_value: Optional[float] = None

    # Recent memory protection
    protect_recent_steps: int = 3

    # Capacity pruning
    use_elbow: bool = True
    elbow_min_records: int = 8
    low_value_prune_ratio: float = 0.2
    max_prune_per_run: int = 50

    # Capacity expansion safeguard
    allow_capacity_expansion: bool = True

    # Safety
    soft_delete: bool = True

@dataclass
class PruningDecision:
    """"""
    memory_id : str
    survival_value: float
    reason: str
    status_before: str = ""
    created_step: int = 0
    last_used_step: int = 0
    reuse_count: int = 0
    strike_count: int = 0

    def to_dict(self) -> Dict[str, object]:
        """for logging."""
        return {
            "memory_id": self.memory_id,
            "survival_value": self.survival_value,
            "reason": self.reason,
            "status_before": self.status_before,
            "created_step": self.created_step,
            "last_used_step": self.last_used_step,
            "reuse_count": self.reuse_count,
            "strike_count": self.strike_count,
        }

@dataclass
class PruningReport:
    """Result of one pruning run."""

    ran: bool
    current_step: int
    before_active_count: int
    after_active_count: int
    pruned_count: int
    decisions: List[PruningDecision] = field(default_factory=list)
    expanded_capacity: bool = False
    capacity_min_after: int = 0
    reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        """Convert report to dict for logging."""
        return {
            "event_type": "pruning_report",
            "ran": self.ran,
            "current_step": self.current_step,
            "before_active_count": self.before_active_count,
            "after_active_count": self.after_active_count,
            "pruned_count": self.pruned_count,
            "expanded_capacity": self.expanded_capacity,
            "capacity_min_after": self.capacity_min_after,
            "reason": self.reason,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "timestamp": utc_timestamp(),
        }
    

class DMSPruner:
    """
    The function of this class:
    1. refreshes survival values,
    2. decides which memories should be removed from prompt context,
    3. marks low-value memories as PRUNED in the store.
    """
    def __init__(
        self,
        config: PruningConfig | None = None,
        survival_calculator: SurvivalValueCalculator | None = None,
    ) -> None:
        self.config = config or PruningConfig()
        self.survival_calculator = survival_calculator or SurvivalValueCalculator()

    def should_run(self, current_step: int, active_count: int) -> bool:
        """
        Decide whether pruning should be triggered.

        Pruning is triggered when:
        1. pruning is enabled,
        2. active memory count exceeds capacity_min, or
        3. current_step reaches pruning interval.
        """
        if not self.config.enabled:
            return False

        if active_count > self.config.capacity_min:
            return True

        if self.config.prune_interval_steps <= 0:
            return False

        return current_step > 0 and current_step % self.config.prune_interval_steps == 0
    
    def refresh_survival_values(
        self,
        store: DMSMemoryStore,
        current_step: int,
    ) -> List[DMSMemoryRecord]:
        """
        Recompute survival values for all active memories and write them back.

        Returns the updated active records.
        """
        active_records = list(store.iter_active_records())
        updated_records: List[DMSMemoryRecord] = []

        for record in active_records:
            survival_value = self.survival_calculator.compute(
                record=record,
                current_step=current_step,
            )
            updated = store.update_meta(
                record.memory_id,
                {"survival_value": survival_value},
            )
            updated_records.append(updated)

        return updated_records


    def select_prunable_records(
        self,
        records: List[DMSMemoryRecord],
        current_step: int,
    ) -> List[PruningDecision]:
        """
        筛选需要被剪枝memory原则：
        1. 硬性规则剪枝：某条 memory 被 verifier 多次判定失败后，直接剪掉
        2. 绝对生存值阈值剪枝
        3. 容量超出上限后触发后的长尾剪枝（论文中提及）
        """
        if not self.config.enabled:
            return []
        
        protected_ids = self._get_recent_memory_ids(records, current_step)
        decisions: Dict[str, PruningDecision] = {}

        # 1. Hard pruning rules
        for record in records:
            if record.memory_id in protected_ids:
                continue

            decision = self._hard_rule_decision(record)
            if decision is not None:
                decisions[record.memory_id] = decision
        
        # 2. Optional absolute survival threshold.
        if self.config.min_survival_value is not None:
            for record in records:
                if record.memory_id in protected_ids:
                    continue
                if record.memory_id in decisions:
                    continue

                survival_value = self._get_survival_value(record)
                if survival_value < self.config.min_survival_value:
                    decisions[record.memory_id] = self._make_decision(
                        record=record,
                        reason=(
                            f"survival_value_below_threshold:"
                            f"{survival_value:.6f}<"
                            f"{self.config.min_survival_value:.6f}"
                        ),
                    )
        
        # 3. Capacity-based pruning.
        active_count_after_hard = len(records) - len(decisions)
        if active_count_after_hard > self.config.capacity_min:
            remaining = [
                record
                for record in records
                if record.memory_id not in decisions
                and record.memory_id not in protected_ids
            ]

            capacity_decisions = self._select_capacity_pruning_decisions(
                remaining_records=remaining,
                active_count_after_hard=active_count_after_hard,
            )

            for decision in capacity_decisions:
                decisions.setdefault(decision.memory_id, decision)
        
        all_decisions = list(decisions.values())
        all_decisions.sort(key=lambda item: item.survival_value)

        if self.config.max_prune_per_run > 0:
            all_decisions = all_decisions[: self.config.max_prune_per_run]

        return all_decisions
    
    def prune(
        self,
        store: DMSMemoryStore,
        current_step: int,
        force: bool = False,
    ) -> PruningReport:
        
        before_records = list(store.iter_active_records())
        before_active_count = len(before_records)
        # 1. 当剪枝未被触发时返回的日志
        if not force and not self.should_run(current_step, before_active_count):
            report = PruningReport(
                ran=False,
                current_step=current_step,
                before_active_count=before_active_count,
                after_active_count=before_active_count,
                pruned_count=0,
                capacity_min_after=self.config.capacity_min,
                reason="pruning_not_triggered",
            )
            return report
        
        refreshed_records = self.refresh_survival_values(
            store=store,
            current_step=current_step,
        )

        expanded = self._maybe_expand_capacity(refreshed_records) # 可能会扩容

        decisions = self.select_prunable_records(
            records=refreshed_records,
            current_step=current_step,
        )

        for decision in decisions:
            store.mark_status(
                memory_id=decision.memory_id,
                status=DMSMemoryStatus.PRUNED,
                reason=decision.reason,
            )
            store.append_pruning_log(
                {
                    "event_type": "memory_pruned",
                    "current_step": current_step,
                    **decision.to_dict(),
                }
            )

        after_active_count = len(list(store.iter_active_records()))

        report = PruningReport(
            ran=True,
            current_step=current_step,
            before_active_count=before_active_count,
            after_active_count=after_active_count,
            pruned_count=len(decisions),
            decisions=decisions,
            expanded_capacity=expanded,
            capacity_min_after=self.config.capacity_min,
            reason="pruning_finished",
        )

        store.append_pruning_log(report.to_dict())

        return report
    
    def _get_recent_memory_ids(
        self,
        records: List[DMSMemoryRecord],
        current_step: int,
    ) -> set[str]:
        """
        Protect very recent memories from pruning.

        This prevents newly created memories from being immediately removed
        before they have any chance to be reused.
        """
        protected: set[str] = set()

        if self.config.protect_recent_steps <= 0:
            return protected

        for record in records:
            if record.meta is None:
                continue

            age = current_step - int(record.meta.created_step)
            if age <= self.config.protect_recent_steps:
                protected.add(record.memory_id)

        return protected
    
    def _hard_rule_decision(
        self,
        record: DMSMemoryRecord,
    ) -> Optional[PruningDecision]:
        """
        硬性规则剪枝：strike_count >= strike_limit.
        """
        if record.meta is None:
            return None

        if self.config.strike_limit > 0:
            if record.meta.strike_count >= self.config.strike_limit:
                return self._make_decision(
                    record=record,
                    reason=(
                        f"strike_limit_reached:"
                        f"{record.meta.strike_count}>="
                        f"{self.config.strike_limit}"
                    ),
                )

        return None

    def _select_capacity_pruning_decisions(
        self,
        remaining_records: List[DMSMemoryRecord],
        active_count_after_hard: int,
    ) -> List[PruningDecision]:
        """
        Select capacity-based pruning candidates from the remaining records.

        This is the single entry point for the capacity pruning stage after
        hard-rule and threshold pruning have already been applied.
        """
        min_prune_count = active_count_after_hard - self.config.capacity_min
        if min_prune_count <= 0:
            return []

        if not remaining_records:
            return []

        if (
            self.config.use_elbow
            and len(remaining_records) >= self.config.elbow_min_records
        ):
            decisions = self._select_by_elbow(
                records=remaining_records,
                min_prune_count=min_prune_count,
            )
        else:
            decisions = self._select_by_bottom_ratio(
                records=remaining_records,
                min_prune_count=min_prune_count,
            )

        max_count = min(len(remaining_records), max(min_prune_count, 0))
        if max_count <= 0:
            return []

        return decisions[:max_count]
    

    # def _select_capacity_pruning_decisions(
    #     self,
    #     remaining_records: List[DMSMemoryRecord],
    #     active_count_after_hard: int,
    # ) -> List[PruningDecision]:
    #     """
    #     Select low-value memories when capacity is exceeded.

    #     If use_elbow is enabled and there are enough records, use an
    #     elbow-like cutoff. Otherwise prune the bottom ratio.
    #     """
    #     if not remaining_records:
    #         return []

    #     must_reduce = max(0, active_count_after_hard - self.config.capacity_min)
    #     if must_reduce <= 0:
    #         return []

    #     if (
    #         self.config.use_elbow
    #         and len(remaining_records) >= self.config.elbow_min_records
    #     ):
    #         return self._select_by_elbow(
    #             records=remaining_records,
    #             min_prune_count=must_reduce,
    #         )

    #     return self._select_by_bottom_ratio(
    #         records=remaining_records,
    #         min_prune_count=must_reduce,
    #     )
    def _select_by_bottom_ratio(
        self,
        records: List[DMSMemoryRecord],
        min_prune_count: int,
    ) -> List[PruningDecision]:
        """
        Simple fallback pruning strategy.

        Sort records by survival value and prune the lowest ratio.
        """
        sorted_records = sorted(records, key=self._get_survival_value)

        ratio_count = int(len(sorted_records) * self.config.low_value_prune_ratio)
        prune_count = max(min_prune_count, ratio_count, 1)
        prune_count = min(prune_count, len(sorted_records))

        selected = sorted_records[:prune_count]

        return [
            self._make_decision(
                record=record,
                reason="capacity_exceeded_bottom_ratio",
            )
            for record in selected
        ]

    def _select_by_elbow(
        self,
        records: List[DMSMemoryRecord],
        min_prune_count: int,
    ) -> List[PruningDecision]:
        """
        Select pruning candidates using an elbow-like cutoff.

        We sort survival values in descending order:
            high-value memories first, low-value memories last.

        Then we find the largest discrete second-order change. Memories after
        that cutoff are treated as the long tail.
        """
        sorted_records = sorted(
            records,
            key=self._get_survival_value,
            reverse=True,
        )
        scores = [self._get_survival_value(record) for record in sorted_records]

        if len(scores) < 3:
            return self._select_by_bottom_ratio(records, min_prune_count)

        elbow_index = self._find_elbow_index(scores)

        # keep records up to elbow_index, prune the long tail after it.
        keep_count = elbow_index + 1
        tail_records = sorted_records[keep_count:]

        if len(tail_records) < min_prune_count:
            tail_records = sorted_records[-min_prune_count:]

        if not tail_records:
            return []

        return [
            self._make_decision(
                record=record,
                reason=f"capacity_exceeded_elbow_cutoff:{elbow_index}",
            )
            for record in tail_records
        ]

    def _find_elbow_index(self, scores_desc: List[float]) -> int:
        """
        Find the elbow index using discrete second-order difference.

        The returned index is the last index to keep.
        """
        if len(scores_desc) < 3:
            return max(0, len(scores_desc) - 1)

        best_index = 1
        best_curvature = -1.0

        for i in range(1, len(scores_desc) - 1):
            curvature = abs(
                scores_desc[i - 1]
                - 2.0 * scores_desc[i]
                + scores_desc[i + 1]
            )
            if curvature > best_curvature:
                best_curvature = curvature
                best_index = i

        return best_index

    def _maybe_expand_capacity(
        self,
        records: List[DMSMemoryRecord],
    ) -> bool:
        """
        Optional capacity expansion safeguard.

        If the memory pool is full but the low-value tail is still above the
        population mean, we treat the overflow as valuable experience and
        expand capacity_min instead of aggressively pruning.
        """
        if not self.config.allow_capacity_expansion:
            return False

        if len(records) <= self.config.capacity_min:
            return False

        if self.config.capacity_min >= self.config.capacity_max:
            return False

        if len(records) < self.config.elbow_min_records:
            return False

        sorted_scores = sorted(
            [self._get_survival_value(record) for record in records],
            reverse=True,
        )

        if len(sorted_scores) < 3:
            return False

        elbow_index = self._find_elbow_index(sorted_scores)
        elbow_score = sorted_scores[elbow_index]
        mean_score = statistics.mean(sorted_scores)

        if elbow_score >= mean_score:
            self.config.capacity_min = min(
                self.config.capacity_min + self.config.capacity_expand_step,
                self.config.capacity_max,
            )
            return True

        return False

    def _make_decision(
        self,
        record: DMSMemoryRecord,
        reason: str,
    ) -> PruningDecision:
        """Create a pruning decision from a memory record."""
        if record.meta is None:
            raise ValueError(f"record.meta is None for memory {record.memory_id}")

        return PruningDecision(
            memory_id=record.memory_id,
            survival_value=self._get_survival_value(record),
            reason=reason,
            status_before=record.meta.status.value,
            created_step=record.meta.created_step,
            last_used_step=record.meta.last_used_step,
            reuse_count=record.meta.reuse_count,
            strike_count=record.meta.strike_count,
        )

    def _get_survival_value(self, record: DMSMemoryRecord) -> float:
        """Safely get survival value from a memory record."""
        if record.meta is None:
            return 0.0
        return float(record.meta.survival_value or 0.0)
