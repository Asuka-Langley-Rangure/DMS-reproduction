from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, List, Optional

class DMSMemoryStatus(str, Enum):

    ACTIVE = "active"
    PRUNED = "pruned"
    REPLACED = "replaced"
    RISKY = "risky"


def utc_timestamp() -> float:
    return time.time()

def make_memory_id(prefix: str = "mem") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"

def stable_json_dumps(obj: Any) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":")
    )

def compute_trajectory_hash(trajectory: List[Dict[str, Any]]) -> str:
    raw = stable_json_dumps(trajectory)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

@dataclass
class DMSMemoryMeta:
    """
    Metadata of one DMS memory.
    This corresponds to s_meta in the paper-level formulation:
        m = (p, tau, s_meta)
    It stores lifecycle, reuse, success/failure, feedback, and score signals.
    """

    memory_id:str

    #lifecycle
    created_step: int = 0
    last_used_step: int = 0
    created_time: float = field(default_factory=utc_timestamp)
    updated_time: float = field(default_factory=utc_timestamp)

    #reuse and verification statistics
    reuse_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    strike_count: int = 0

    # enviroment feedback signals
    invalid_action_count: int = 0
    no_state_change_count: int = 0
    parse_error_count: int = 0
    execution_error_count: int = 0

    # DMS scores
    survival_value: float = 0.0
    risk_score: float = 0.0

    # lifecycle status
    status: DMSMemoryStatus = DMSMemoryStatus.ACTIVE

    # debug / explanation
    verifier_reason: Optional[str] = None
    completion_message: Optional[str] = None
    replacement_of: Optional[str] = None
    replaced_by: Optional[str] = None
    prune_reason: Optional[str] = None

    # keep forward-compatible fields here
    extra: Dict[str, Any] = field(default_factory=dict) # Unknown fields will be stored in extra so old/new versions of the memory file remain compatible.

    @classmethod
    def from_dict(
        cls,
        data: Optional[Dict[str, Any]],
        default_memory_id: Optional[str] = None,
    ) -> "DMSMemoryMeta": # Build DMSMemoryMeta from a dict loaded from JSON.
        data = dict(data or {})

        if "memory_id" not in data:
            if default_memory_id is None:
                raise ValueError("memory_id is required for DMSMemoryMeta")
            data["memory_id"] = default_memory_id
        
        if "status" in data and not isinstance(data["status"], DMSMemoryStatus):
            data["status"] = DMSMemoryStatus(data["status"])
        
        valid_names = {f.name for f in fields(cls)}
        known_kwargs = {k: v for k, v in data.items() if k in valid_names}

        obj = cls(**known_kwargs)

        for key, value in data.items():
            if key not in valid_names:
                obj.extra[key] = value

        return obj
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a JSON-serializable dict."""
        data = {
            "memory_id": self.memory_id,
            "created_step": self.created_step,
            "last_used_step": self.last_used_step,
            "created_time": self.created_time,
            "updated_time": self.updated_time,
            "reuse_count": self.reuse_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "strike_count": self.strike_count,
            "invalid_action_count": self.invalid_action_count,
            "no_state_change_count": self.no_state_change_count,
            "parse_error_count": self.parse_error_count,
            "execution_error_count": self.execution_error_count,
            "survival_value": self.survival_value,
            "risk_score": self.risk_score,
            "status": self.status.value,
            "verifier_reason": self.verifier_reason,
            "completion_message": self.completion_message,
            "replacement_of": self.replacement_of,
            "replaced_by": self.replaced_by,
            "prune_reason": self.prune_reason,
            "extra": self.extra,
        }
        return data
    
    def apply_update(self, update: Dict[str, Any]) -> None:
        """
        Apply a partial metadata update.

        Supported styles:
        - {"reuse_count": 10} directly sets a value.
        - {"reuse_count_delta": 1} increments a value.
        """
        for key, value in update.items():
            if key.endswith("_delta"):
                base_key = key[: -len("_delta")]
                if not hasattr(self, base_key):
                    self.extra[key] = value
                    continue

                old_value = getattr(self, base_key)
                if old_value is None:
                    old_value = 0
                setattr(self, base_key, old_value + value)
                continue

            if key == "status":
                self.status = (
                    value
                    if isinstance(value, DMSMemoryStatus)
                    else DMSMemoryStatus(value)
                )
                continue

            if key == "extra" and isinstance(value, dict):
                self.extra.update(value)
                continue

            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.extra[key] = value

        self.updated_time = utc_timestamp()
    

@dataclass
class DMSMemoryRecord:
    memory_id: str

    subtask_text: str
    precondition: str
    goal: str
    user_goal: str

    app_name: Optional[str] = None
    current_activity: Optional[str] = None

    goal_embedding: Optional[List[float]] = None
    precondition_embedding: Optional[List[float]] = None

    trajectory_path: Optional[str] = None
    trajectory_len: int = 0
    trajectory_hash: Optional[str] = None

    meta: Optional[DMSMemoryMeta] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure meta exists and its memory_id is consistent."""
        if self.meta is None:
            self.meta = DMSMemoryMeta(memory_id=self.memory_id)
        else:
            self.meta.memory_id = self.memory_id
    
    @property
    def status(self) -> DMSMemoryStatus:
        """Return current lifecycle status."""
        assert self.meta is not None
        return self.meta.status

    def is_active(self) -> bool:
        """Whether this memory can be retrieved and injected into prompt."""
        return self.status == DMSMemoryStatus.ACTIVE
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DMSMemoryRecord":
        """Build a memory record from a JSON dict."""
        data = dict(data)

        if "memory_id" not in data:
            raise ValueError("memory_id is required for DMSMemoryRecord")

        meta = DMSMemoryMeta.from_dict(
            data.get("meta"),
            default_memory_id=data["memory_id"],
        )

        valid_names = {f.name for f in fields(cls)}
        known_kwargs = {
            k: v
            for k, v in data.items()
            if k in valid_names and k not in {"meta", "extra"}
        }

        extra = dict(data.get("extra") or {})
        for key, value in data.items():
            if key not in valid_names:
                extra[key] = value

        return cls(
            **known_kwargs,
            meta=meta,
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a JSON-serializable dict."""
        assert self.meta is not None

        return {
            "memory_id": self.memory_id,
            "subtask_text": self.subtask_text,
            "precondition": self.precondition,
            "goal": self.goal,
            "user_goal": self.user_goal,
            "app_name": self.app_name,
            "current_activity": self.current_activity,
            "goal_embedding": self.goal_embedding,
            "precondition_embedding": self.precondition_embedding,
            "trajectory_path": self.trajectory_path,
            "trajectory_len": self.trajectory_len,
            "trajectory_hash": self.trajectory_hash,
            "meta": self.meta.to_dict(),
            "extra": self.extra,
        }
    
@dataclass
class MemoryReadResult:
    """
    Structured result returned by DMSMemoryProvider before actor execution.

    Static memory only needs context string.
    DMS additionally needs selected_memory_id and feedback fields so that
    runner can later update reuse/failure/survival statistics.
    """

    context: str = ""

    # retrieval information
    has_hit: bool = False
    selected_memory_id: Optional[str] = None
    retrieval_results: List[Any] = field(default_factory=list)

    # DMS control
    should_replay: bool = False
    should_mutate: bool = False
    replay_trajectory: Optional[List[Dict[str, Any]]] = None

    # retrieval scores
    final_score: Optional[float] = None
    sim_goal: Optional[float] = None
    sim_precondition: Optional[float] = None

    # memory scores
    survival_value: Optional[float] = None
    risk_score: Optional[float] = None

    # debug
    reason: str = ""

    def to_event_fields(self) -> Dict[str, Any]:
        """
        Convert the read result to fields that can be merged into MemoryEvent.
        """
        return {
            "selected_memory_id": self.selected_memory_id,
            "used_memory": self.has_hit,
            "should_replay": self.should_replay,
            "should_mutate": self.should_mutate,
            "retrieval_score": self.final_score,
            "sim_goal": self.sim_goal,
            "sim_precondition": self.sim_precondition,
            "survival_value": self.survival_value,
            "risk_score": self.risk_score,
            "memory_read_reason": self.reason,
        }

@dataclass
class DMSStoreStats:
    """Summary statistics of the DMS memory store."""

    total_count: int
    active_count: int
    pruned_count: int
    replaced_count: int
    risky_count: int
    memory_size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dict for logging."""
        return {
            "total_count": self.total_count,
            "active_count": self.active_count,
            "pruned_count": self.pruned_count,
            "replaced_count": self.replaced_count,
            "risky_count": self.risky_count,
            "memory_size_bytes": self.memory_size_bytes,
        }

