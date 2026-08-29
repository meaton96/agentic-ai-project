from pathlib import Path

from pydantic import TypeAdapter

from sandbox_core.schemas.events import Event

EventAdapter = TypeAdapter(Event)

DEFAULT_OUTPUT_ROOT = Path("./runs")


class EventLog:
    """Append-only JSONL writer for one run's events, at <output_root>/<run_id>/events.jsonl."""

    def __init__(self, run_id: str, output_root: Path = DEFAULT_OUTPUT_ROOT):
        self.run_id = run_id
        self.output_root = Path(output_root)
        self.dir = self.output_root / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "events.jsonl"
        self._seq = max((event.seq for event in read_events(self.path)), default=0)

    def append(self, event: Event) -> Event:
        """Assigns the next seq for this run and appends the event as one JSON line."""
        self._seq += 1
        event = event.model_copy(update={"seq": self._seq})
        with self.path.open("a", encoding="utf-8") as f:
            f.write(EventAdapter.dump_json(event).decode("utf-8") + "\n")
        return event


def read_events(path: Path) -> list[Event]:
    """Reads a run's events.jsonl back into typed Event objects, in file order."""
    path = Path(path)
    if not path.exists():
        return []
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(EventAdapter.validate_json(line))
    return events
