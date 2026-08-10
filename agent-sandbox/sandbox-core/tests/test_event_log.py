from datetime import datetime, timezone

from sandbox_core.runtime.event_log import EventLog, read_events
from sandbox_core.schemas.events import AgentResultEvent, ErrorEvent, LlmRequestEvent

RUN_ID = "run-1"
AGENT_ID = "agent-1"
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_request_event(**overrides):
    fields = dict(run_id=RUN_ID, seq=0, ts=NOW, agent_id=AGENT_ID, messages=[{"role": "user", "content": "hi"}], model="m")
    fields.update(overrides)
    return LlmRequestEvent(**fields)


def test_append_assigns_monotonic_seq(tmp_path):
    log = EventLog(RUN_ID, output_root=tmp_path)

    first = log.append(make_request_event())
    second = log.append(make_request_event())
    third = log.append(make_request_event())

    assert [first.seq, second.seq, third.seq] == [1, 2, 3]


def test_append_writes_one_json_line_per_event(tmp_path):
    log = EventLog(RUN_ID, output_root=tmp_path)
    log.append(make_request_event())
    log.append(make_request_event())

    lines = log.path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_read_events_roundtrips_typed_events_in_order(tmp_path):
    log = EventLog(RUN_ID, output_root=tmp_path)
    log.append(make_request_event())
    log.append(AgentResultEvent(run_id=RUN_ID, seq=0, ts=NOW, agent_id=AGENT_ID, final_output="done", turns_used=1))

    events = read_events(log.path)

    assert len(events) == 2
    assert isinstance(events[0], LlmRequestEvent)
    assert isinstance(events[1], AgentResultEvent)
    assert [e.seq for e in events] == [1, 2]


def test_read_events_missing_file_returns_empty_list(tmp_path):
    assert read_events(tmp_path / "nope.jsonl") == []


def test_reopening_event_log_continues_seq_from_existing_file(tmp_path):
    first_log = EventLog(RUN_ID, output_root=tmp_path)
    first_log.append(make_request_event())
    first_log.append(make_request_event())

    reopened = EventLog(RUN_ID, output_root=tmp_path)
    third = reopened.append(ErrorEvent(run_id=RUN_ID, seq=0, ts=NOW, agent_id=AGENT_ID, message="boom"))

    assert third.seq == 3
    assert len(read_events(reopened.path)) == 3


def test_event_log_creates_output_directory(tmp_path):
    output_root = tmp_path / "does" / "not" / "exist"
    log = EventLog(RUN_ID, output_root=output_root)
    assert log.path.parent.is_dir()
    assert log.path == output_root / RUN_ID / "events.jsonl"
