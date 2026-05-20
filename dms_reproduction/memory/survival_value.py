from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

from .types import DMSMemoryRecord

@dataclass
class SurvivalValueConfig:
    """
    存储计算一条memory的生存价值所需要的超参数
    在论文中：S = Utility * AdaptiveDecay * Reliability
    而在目前考核任务中还需要加入环境反馈：S = Utility * AdaptiveDecay * Reliability * CompletionFactor * EnvironmentFactor
    """

    # 复用价值 utility
    v_new: float = 1.0
    new_memory_grace_steps: int = 5 # 

    # 自适应时间衰减
    t_base:float = 30.0
    mu: float = 15.0
    beta: float = 0.5

    # 可靠性惩罚
    gamma: float = 1.0

    # 环境反馈
    lambda_invalid: float = 0.5
    lambda_no_change: float = 0.3
    lambda_parse_error: float = 0.7
    lambda_execution_error: float = 0.7

    completion_reward_min: float = 0.5

    # Numeric safety
    max_exp_input: float = 60.0
    min_value: float = 0.0

@dataclass
class SurvivalValueComponents: 
    """
    Detailed components of one survival value calculation for debugging
    """
    
    memory_id: str

    # raw statistics
    reuse_count: int
    success_count: int
    failure_count: int
    strike_count: int
    age: int
    delta_t: int

    # formula components
    novelty_bonus: float
    utility: float
    half_life: float
    decay: float
    reliability: float
    completion_factor: float
    environment_factor: float

    # final score
    final_value: float

    def to_dict(self) -> Dict[str, float | int | str]:
        """Convert components to a JSON-serializable dict."""
        return {
            "memory_id": self.memory_id,
            "reuse_count": self.reuse_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "strike_count": self.strike_count,
            "age": self.age,
            "delta_t": self.delta_t,
            "novelty_bonus": self.novelty_bonus,
            "utility": self.utility,
            "half_life": self.half_life,
            "decay": self.decay,
            "reliability": self.reliability,
            "completion_factor": self.completion_factor,
            "environment_factor": self.environment_factor,
            "final_value": self.final_value,
        }
    
class SurvivalValueCalculator:

    def __init__(self, config: SurvivalValueConfig | None = None) -> None:
        self.config = config or SurvivalValueConfig()

    def compute(self, record: DMSMemoryRecord, current_step: int) -> float:
        return self.compute_components(record, current_step).final_value
    
    def compute_components(
            self,
            record: DMSMemoryRecord,
            current_step: int,
    ) -> SurvivalValueComponents:
        if record.meta is None:
            raise ValueError(f"record.meta is None for memory {record.memory_id}")
        
        meta = record.meta

        created_step = max(0, int(meta.created_step))
        last_used_step = max(created_step, int(meta.last_used_step))
        current_step = max(current_step, last_used_step)

        age = current_step - created_step

        delta_t = current_step - last_used_step

        reuse_count = max(0, int(meta.reuse_count))
        success_count = max(0, int(meta.success_count))
        failure_count = max(0, int(meta.failure_count))
        strike_count = max(0, int(meta.strike_count))

        novelty_bonus = self._compute_novelty_bonus(age)
        utility = self._compute_utility(reuse_count, novelty_bonus)
        half_life = self._compute_half_life(reuse_count)
        decay = self._compute_decay(delta_t, half_life)
        reliability = self._compute_reliability(strike_count)
        completion_factor = self._compute_completion_factor(
            success_count=success_count,
            failure_count=failure_count,
        )
        environment_factor = self._compute_environment_factor(record)

        final_value = (
            utility
            * decay
            * reliability
            * completion_factor
            * environment_factor
        )

        final_value = max(self.config.min_value, float(final_value))

        return SurvivalValueComponents(
            memory_id=record.memory_id,
            reuse_count=reuse_count,
            success_count=success_count,
            failure_count=failure_count,
            strike_count=strike_count,
            age=age,
            delta_t=delta_t,
            novelty_bonus=novelty_bonus,
            utility=utility,
            half_life=half_life,
            decay=decay,
            reliability=reliability,
            completion_factor=completion_factor,
            environment_factor=environment_factor,
            final_value=final_value,
        )
    
    def _compute_novelty_bonus(self, age: int) -> float: #
        """
        给新记忆以冷启动保护
        """
        if age <= self.config.new_memory_grace_steps:
            return float(self.config.v_new)
        return 0.0
    
    def _compute_utility(self, reuse_count: int, novelity_bonus: float) -> float:
        
        return math.log1p(reuse_count) + novelity_bonus
    
    def _compute_half_life(self, reuse_count: int) -> float:

        return self.config.t_base + self.config.mu * math.log1p(reuse_count)

    def _compute_decay(self, delta_t, half_life) -> float:
        x = self.config.beta * (float(delta_t) - float(half_life))
        x = max(-self.config.max_exp_input, min(self.config.max_exp_input, x))
        return 1.0 / (1.0 + math.exp(x))
    
    def _compute_reliability(self, strike_count: int) -> float:
        return 1.0 / (1.0 + self.config.gamma * strike_count)
    
    def _compute_completion_factor(
        self, 
        success_count: int,
        failure_count: int,
    ) -> float:
        """
        给被证实成功/失败的策略一个soft reward，它的范围在completion_reward_min and 1.0
        但如果没有正反馈我们会返回1.0防止对新记忆干扰
        """
        total = success_count + failure_count
        if total <= 0:
            return 1.0

        success_rate = success_count / total
        low = self.config.completion_reward_min
        return low + (1.0 - low) * success_rate
    
    def _compute_environment_factor(self, record: DMSMemoryRecord) -> float:
        """
        我在这里对四种环境反馈的方式做了加权求和并仿照reliability转换成一个小于1的数作为权重
        """
        if record.meta is None:
            return 1.0

        meta = record.meta

        invalid = max(0, int(meta.invalid_action_count))
        no_change = max(0, int(meta.no_state_change_count))
        parse_error = max(0, int(meta.parse_error_count))
        execution_error = max(0, int(meta.execution_error_count))

        penalty = (
            self.config.lambda_invalid * invalid
            + self.config.lambda_no_change * no_change
            + self.config.lambda_parse_error * parse_error
            + self.config.lambda_execution_error * execution_error
        )

        return 1.0 / (1.0 + penalty)


# Backward-compatible alias for older imports.
SurvivalValueCaculator = SurvivalValueCalculator
