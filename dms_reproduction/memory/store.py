from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Union

from .retrieval import MemoryCandidate
from .types import (
    DMSMemoryRecord,
    DMSMemoryStatus,
    DMSStoreStats,
    compute_trajectory_hash,
    utc_timestamp,
)

class DMSMemoryStore(Protocol):
    """
    Abstract interface for DMS memory storage.

    Later we can replace JsonDMSMemoryStore with SQLiteDMSMemoryStore
    without changing DMSMemoryProvider.
    """

    def add_memory(
        self,
        record: DMSMemoryRecord,
        trajectory: List[Dict[str, Any]],
    ) -> None:
        ...

    def get_memory(self, memory_id: str) -> Optional[DMSMemoryRecord]:
        ...

    def load_trajectory(self, memory_id: str) -> List[Dict[str, Any]]:
        ...

    def iter_records(self) -> Iterable[DMSMemoryRecord]:
        ...

    def iter_active_records(self) -> Iterable[DMSMemoryRecord]:
        ...

    def iter_active_candidates(self) -> Iterable[MemoryCandidate]:
        ...

    def update_meta(
        self,
        memory_id: str,
        meta_update: Dict[str, Any],
    ) -> DMSMemoryRecord:
        ...

    def mark_status(
        self,
        memory_id: str,
        status: DMSMemoryStatus,
        reason: str = "",
    ) -> DMSMemoryRecord:
        ...

    def replace_memory(
        self,
        old_memory_id: str,
        new_record: DMSMemoryRecord,
        new_trajectory: List[Dict[str, Any]],
    ) -> None:
        ...

    def append_event(self, event: Dict[str, Any]) -> None:
        ...

    def append_pruning_log(self, event: Dict[str, Any]) -> None:
        ...

    def stats(self) -> Dict[str, Any]:
        ...

class JsonDMSMemoryStore:
    """
    What will the dms_memory Directory look like?
    dms_memory/
        ├── index.jsonl
        ├── events.jsonl
        ├── pruning_log.jsonl
        └── trajectories/
            ├── mem_xxx.json
            └── mem_yyy.json
    index.jsonl stores index + metadata
    trajectories stores dense actor trajectories.
    """

    def __init__(
        self, 
        root_dir: Union[str, Path],
        index_filename: str = "index.jsonl",
        trajectories_dirname: str = "trajectories",
        event_filenames: str = "event.jsonl",
        pruning_log_filename: str = "pruning_log.jsonl"
    ) -> None:
        self.root_dir = Path(root_dir)
        self.index_path = self.root_dir / index_filename
        self.trajectories_dir = self.root_dir / trajectories_dirname
        self.events_path = self.root_dir / event_filenames
        self.pruning_log_path = self.root_dir / pruning_log_filename

        self._lock = threading.RLock()

        self._ensure_layout()

    def _ensure_layout(self) -> None:
        """Create store directory and required files."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.trajectories_dir.mkdir(parents=True, exist_ok=True)

        for path in [self.index_path, self.events_path, self.pruning_log_path]:
            if not path.exists():
                path.touch()

    def _read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Read a JSONL file into a list of dicts."""
        if not path.exists():
            return []

        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {path} at line {line_no}: {exc}"
                    ) from exc

        return rows
    
    def _write_jsonl_atomic(
        self,
        path: Path,
        rows: List[Dict[str, Any]],
    ) -> None:
        """
        Rewrite a JSONL file atomically.

        This is used for index updates, because metadata update requires
        modifying existing records.
        """
        tmp_path = path.with_name(path.name + ".tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        os.replace(tmp_path, path)

    def _append_jsonl(
        self,
        path: Path,
        row: Dict[str, Any],
    ) -> None:
        """Append one dict as one JSONL row."""
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _read_index_records(self) -> List[DMSMemoryRecord]:
        """Load all memory records from index.jsonl."""
        rows = self._read_jsonl(self.index_path)
        return [DMSMemoryRecord.from_dict(row) for row in rows]

    def _write_index_records_atomic(
        self,
        records: List[DMSMemoryRecord],
    ) -> None:
        """Rewrite index.jsonl with current records."""
        rows = [record.to_dict() for record in records]
        self._write_jsonl_atomic(self.index_path, rows)

    def _trajectory_rel_path(self, memory_id: str) -> str:
        """Return relative trajectory path for a memory id."""
        return f"{self.trajectories_dir.name}/{memory_id}.json"

    def _resolve_trajectory_path(self, record: DMSMemoryRecord) -> Path:
        """Resolve trajectory_path in a record to an absolute filesystem path."""
        if record.trajectory_path is None:
            return self.root_dir / self._trajectory_rel_path(record.memory_id)

        path = Path(record.trajectory_path)
        if path.is_absolute():
            return path

        return self.root_dir / path
    
    def _write_json_atomic(self, path: Path, obj: Any) -> None:
        """Write one JSON object atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

        os.replace(tmp_path, path)

    def _write_trajectory(
        self,
        record: DMSMemoryRecord,
        trajectory: List[Dict[str, Any]],
    ) -> None:
        """Persist one dense actor trajectory to trajectories/{memory_id}.json."""
        path = self._resolve_trajectory_path(record)

        payload = {
            "memory_id": record.memory_id,
            "trajectory": trajectory,
            "trajectory_len": len(trajectory),
            "trajectory_hash": compute_trajectory_hash(trajectory),
            "updated_time": utc_timestamp(),
        }

        self._write_json_atomic(path, payload)

    def _update_record(
        self,
        memory_id: str,
        updater: Callable[[DMSMemoryRecord], None],
    ) -> DMSMemoryRecord:
        """
        Find one record, apply updater(record), and rewrite index.

        This is the shared helper for update_meta, mark_status, and replacement.
        """
        with self._lock:
            records = self._read_index_records()

            for record in records:
                if record.memory_id == memory_id:
                    updater(record)
                    self._write_index_records_atomic(records)
                    return record

        raise KeyError(f"Memory not found: {memory_id}")

    def add_memory(
        self,
        record: DMSMemoryRecord,
        trajectory: List[Dict[str, Any]],
    ) -> None:
        """
        Add a new memory record and its trajectory.

        This is called after a subtask succeeds and the trajectory is eligible
        for memory storage.
        """
        with self._lock:
            records = self._read_index_records()
            if any(item.memory_id == record.memory_id for item in records):
                raise ValueError(f"Duplicate memory_id: {record.memory_id}")

            record.trajectory_path = record.trajectory_path or self._trajectory_rel_path(
                record.memory_id
            )
            record.trajectory_len = len(trajectory)
            record.trajectory_hash = compute_trajectory_hash(trajectory)

            if record.meta is not None:
                record.meta.updated_time = utc_timestamp()
                record.meta.status = DMSMemoryStatus.ACTIVE

            self._write_trajectory(record, trajectory)

            records.append(record)
            self._write_index_records_atomic(records)

    def get_memory(self, memory_id: str) -> Optional[DMSMemoryRecord]:
        """Return one memory record by id. Return None if not found."""
        with self._lock:
            for record in self._read_index_records():
                if record.memory_id == memory_id:
                    return record

        return None

    def load_trajectory(self, memory_id: str) -> List[Dict[str, Any]]:
        """
        Load dense actor trajectory by memory id.

        If the memory or trajectory file does not exist, return an empty list.
        """
        record = self.get_memory(memory_id)
        if record is None:
            return []

        path = self._resolve_trajectory_path(record)
        if not path.exists():
            return []

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "trajectory" in data:
            trajectory = data["trajectory"]
        else:
            trajectory = data

        if not isinstance(trajectory, list):
            raise ValueError(f"Invalid trajectory format for memory {memory_id}")

        return trajectory

    def iter_records(self) -> Iterable[DMSMemoryRecord]:
        """Iterate over all records, including pruned/replaced/risky ones."""
        with self._lock:
            records = self._read_index_records()

        return iter(records)

    def iter_active_records(self) -> Iterable[DMSMemoryRecord]:
        """Iterate over active memories only."""
        records = [record for record in self.iter_records() if record.is_active()]
        return iter(records)

    def _candidate_from_record(self, record: DMSMemoryRecord) -> MemoryCandidate:
        """
        Convert DMSMemoryRecord to retrieval.MemoryCandidate.

        The payload always carries memory_id and meta, even if the current
        MemoryCandidate dataclass does not yet define explicit DMS fields.
        """
        assert record.meta is not None

        payload = record.to_dict()
        payload["memory_id"] = record.memory_id

        extended_kwargs = {
            "memory_id": record.memory_id,
            "subtask_text": record.subtask_text,
            "precondition": record.precondition,
            "goal": record.goal,
            "timestamp": record.meta.created_time,
            "app_name": record.app_name,
            "current_activity": record.current_activity,
            "trajectory": [],
            "trajectory_len": record.trajectory_len,
            "verifier_reason": record.meta.verifier_reason,
            "goal_embedding": record.goal_embedding,
            "precondition_embedding": record.precondition_embedding,
            "payload": payload,
            "survival_value": record.meta.survival_value,
            "risk_score": record.meta.risk_score,
            "status": record.meta.status.value,
        }

        try:
            return MemoryCandidate(**extended_kwargs)
        except TypeError:
            # Compatibility with your current retrieval.py, whose MemoryCandidate
            # may not yet include memory_id / survival_value / risk_score / status.
            compatible_kwargs = {
                "subtask_text": record.subtask_text,
                "precondition": record.precondition,
                "goal": record.goal,
                "timestamp": record.meta.created_time,
                "app_name": record.app_name,
                "current_activity": record.current_activity,
                "trajectory": [],
                "verifier_reason": record.meta.verifier_reason,
                "goal_embedding": record.goal_embedding,
                "precondition_embedding": record.precondition_embedding,
                "payload": payload,
            }
            return MemoryCandidate(**compatible_kwargs)

    def iter_active_candidates(self) -> Iterable[MemoryCandidate]:
        """
        Iterate active records as retrieval candidates.

        Retrieval should use semantic fields and embeddings from candidates.
        The dense trajectory is not loaded here to keep retrieval lightweight.
        """
        candidates = [
            self._candidate_from_record(record)
            for record in self.iter_active_records()
        ]
        return iter(candidates)

    def update_meta(
        self,
        memory_id: str,
        meta_update: Dict[str, Any],
    ) -> DMSMemoryRecord:
        """
        Partially update metadata of one memory.

        Examples:
            update_meta(memory_id, {"reuse_count_delta": 1})
            update_meta(memory_id, {"strike_count_delta": 1})
            update_meta(memory_id, {"survival_value": 1.23})
        """

        def updater(record: DMSMemoryRecord) -> None:
            assert record.meta is not None
            record.meta.apply_update(meta_update)

        return self._update_record(memory_id, updater)

    def mark_status(
        self,
        memory_id: str,
        status: DMSMemoryStatus,
        reason: str = "",
    ) -> DMSMemoryRecord:
        """
        Soft-delete or block one memory by changing its status.

        We do not physically delete the memory so experiments can still
        analyze pruned/replaced/risky memories.
        """

        def updater(record: DMSMemoryRecord) -> None:
            assert record.meta is not None
            record.meta.status = status
            record.meta.updated_time = utc_timestamp()

            if status == DMSMemoryStatus.PRUNED:
                record.meta.prune_reason = reason
            else:
                record.meta.extra["status_reason"] = reason

        updated = self._update_record(memory_id, updater)

        self.append_event(
            {
                "event_type": "memory_status_changed",
                "memory_id": memory_id,
                "status": status.value,
                "reason": reason,
                "timestamp": utc_timestamp(),
            }
        )

        return updated

    def replace_memory(
        self,
        old_memory_id: str,
        new_record: DMSMemoryRecord,
        new_trajectory: List[Dict[str, Any]],
    ) -> None:
        """
        Replace an old memory with a new evolved memory.

        The old memory is marked as REPLACED.
        The new memory is inserted as ACTIVE.
        """
        with self._lock:
            records = self._read_index_records()

            old_record: Optional[DMSMemoryRecord] = None
            for record in records:
                if record.memory_id == old_memory_id:
                    old_record = record
                    break

            if old_record is None:
                raise KeyError(f"Old memory not found: {old_memory_id}")

            if any(record.memory_id == new_record.memory_id for record in records):
                raise ValueError(f"Duplicate new memory_id: {new_record.memory_id}")

            assert old_record.meta is not None
            assert new_record.meta is not None

            old_record.meta.status = DMSMemoryStatus.REPLACED
            old_record.meta.replaced_by = new_record.memory_id
            old_record.meta.updated_time = utc_timestamp()

            new_record.meta.status = DMSMemoryStatus.ACTIVE
            new_record.meta.replacement_of = old_memory_id
            new_record.meta.updated_time = utc_timestamp()

            new_record.trajectory_path = (
                new_record.trajectory_path
                or self._trajectory_rel_path(new_record.memory_id)
            )
            new_record.trajectory_len = len(new_trajectory)
            new_record.trajectory_hash = compute_trajectory_hash(new_trajectory)

            self._write_trajectory(new_record, new_trajectory)

            records.append(new_record)
            self._write_index_records_atomic(records)

        self.append_event(
            {
                "event_type": "memory_replaced",
                "old_memory_id": old_memory_id,
                "new_memory_id": new_record.memory_id,
                "timestamp": utc_timestamp(),
            }
        )

    def append_event(self, event: Dict[str, Any]) -> None:
        """
        Append one runtime memory event.

        This is useful for debugging, metric analysis, and later DMS feedback.
        """
        event = dict(event)
        event.setdefault("timestamp", utc_timestamp())

        with self._lock:
            self._append_jsonl(self.events_path, event)

    def append_pruning_log(self, event: Dict[str, Any]) -> None:
        """
        Append one pruning decision log.

        This should be called by DMSPruner later.
        """
        event = dict(event)
        event.setdefault("timestamp", utc_timestamp())

        with self._lock:
            self._append_jsonl(self.pruning_log_path, event)

    def _memory_size_bytes(self) -> int:
        """Return total disk usage of the whole DMS memory directory."""
        total = 0

        if not self.root_dir.exists():
            return 0

        for path in self.root_dir.rglob("*"):
            if path.is_file():
                total += path.stat().st_size

        return total

    def stats(self) -> Dict[str, Any]:
        """
        Return store-level statistics.

        These values are useful for drawing memory size and pruning curves.
        """
        records = list(self.iter_records())

        total_count = len(records)
        active_count = 0
        pruned_count = 0
        replaced_count = 0
        risky_count = 0

        for record in records:
            status = record.status
            if status == DMSMemoryStatus.ACTIVE:
                active_count += 1
            elif status == DMSMemoryStatus.PRUNED:
                pruned_count += 1
            elif status == DMSMemoryStatus.REPLACED:
                replaced_count += 1
            elif status == DMSMemoryStatus.RISKY:
                risky_count += 1

        return DMSStoreStats(
            total_count=total_count,
            active_count=active_count,
            pruned_count=pruned_count,
            replaced_count=replaced_count,
            risky_count=risky_count,
            memory_size_bytes=self._memory_size_bytes(),
        ).to_dict()


# Backward-compatible alias for older imports.
JsonlDMSMemoryStore = JsonDMSMemoryStore
