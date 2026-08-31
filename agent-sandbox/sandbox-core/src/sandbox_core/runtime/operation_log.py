import difflib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import TypeAdapter

from sandbox_core.schemas.operations import Operation, OperationRecord

RecordAdapter = TypeAdapter(OperationRecord)

DEFAULT_OPERATIONS_ROOT = Path("./operations")


def _unified_diff(before: dict, after: dict) -> str:
    before_lines = json.dumps(before, indent=2, sort_keys=True).splitlines(keepends=True)
    after_lines = json.dumps(after, indent=2, sort_keys=True).splitlines(keepends=True)
    return "".join(difflib.unified_diff(before_lines, after_lines, fromfile="before", tofile="after"))


class OperationLog:
    """Append-only JSONL log of every Operation applied to one spec, at
    <root>/<target_type>/<target_id>.jsonl. Mirrors EventLog's shape (seq
    assigned on append, continues from the existing file when reopened) but
    is keyed by a spec's (target_type, target_id) rather than a run_id,
    since these operations happen pre-run, against a draft spec that may
    outlive any particular run — or never be run at all."""

    def __init__(
        self,
        target_type: str,
        target_id: str,
        root: Path = DEFAULT_OPERATIONS_ROOT,
    ):
        self.target_type = target_type
        self.target_id = target_id
        self.root = Path(root)
        self.dir = self.root / target_type
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{target_id}.jsonl"
        self._seq = max((record.seq for record in read_operations(self.path)), default=0)

    def append(self, *, operation: Operation, actor: str, before: dict, after: dict) -> OperationRecord:
        """Assigns the next seq for this target and appends the record as
        one JSON line, including a unified diff of before/after for quick
        human review."""
        self._seq += 1
        record = OperationRecord(
            seq=self._seq,
            ts=datetime.now(timezone.utc),
            target_type=self.target_type,
            target_id=self.target_id,
            actor=actor,
            operation=operation,
            before=before,
            after=after,
            diff=_unified_diff(before, after),
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(RecordAdapter.dump_json(record).decode("utf-8") + "\n")
        return record


def read_operations(path: Path) -> list[OperationRecord]:
    """Reads a target's operations.jsonl back into typed OperationRecords, in file order."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(RecordAdapter.validate_json(line))
    return records
