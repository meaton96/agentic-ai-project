"""
GET /api/runs/{run_id} must behave identically for a run this server
launched and a run directory that only exists on disk from a prior CLI
invocation the server never saw start. This builds one by hand — same
events.jsonl line shape agentic_ml.events.make_event_logger writes, same
orchestrator_report.json shape run_orchestrator.py writes, same
leaderboard entry shape agentic_ml.harness.leaderboard.append_leaderboard_entry
writes — and confirms a fresh RunManager (that never tracked it) reads
it through the exact same assemble_run_summary() code path a live run
uses.
"""
from __future__ import annotations

import json
import time

from agentic_ml.harness.leaderboard import append_leaderboard_entry
from agentic_ml.paths import leaderboard_path as resolve_leaderboard_path
from agentic_ml.paths import run_dir as resolve_run_dir


def _write_historical_run(run_id: str) -> None:
    run_dir = resolve_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    events = [
        {"ts": time.time(), "run_id": run_id, "phase": "run", "type": "run_started",
         "payload": {"data": "/some/path.csv", "goal": "", "target": "churned"}},
        {"ts": time.time(), "run_id": run_id, "phase": "finalize", "type": "final_test_metrics",
         "payload": {"test_metrics": {"roc_auc": {"value": 0.9}}}},
        {"ts": time.time(), "run_id": run_id, "phase": "run", "type": "run_completed",
         "payload": {"status": "success"}},
    ]
    with open(run_dir / "events.jsonl", "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    report = {
        "run_id": run_id, "model": "fake-model", "status": "success",
        "final_test_metrics": {"roc_auc": {"value": 0.9}},
    }
    (run_dir / "orchestrator_report.json").write_text(json.dumps(report, indent=2))

    (run_dir / "transcripts").mkdir(exist_ok=True)
    (run_dir / "transcripts" / "profiler_01.json").write_text(json.dumps([{"role": "system", "content": "hi"}]))

    append_leaderboard_entry(resolve_leaderboard_path(), {
        "run_id": run_id, "candidate": "candidate_a", "template_id": "sklearn_mixed_pipeline",
        "source": "orchestrator", "model": "fake-model", "split": "validation",
        "metrics": {"roc_auc": {"value": 0.9}}, "verification_verdict": "approved",
    })


def test_historical_run_parity_with_live_run_shape(client):
    run_id = "run_historical_test"
    _write_historical_run(run_id)

    resp = client.get(f"/api/runs/{run_id}")
    assert resp.status_code == 200
    summary = resp.json()

    assert summary["run_id"] == run_id
    assert summary["orchestrator"] == "static"
    assert summary["status"] == "completed"
    assert summary["started_at"] is None  # never tracked by this RunManager
    assert summary["finished_at"] is None
    assert summary["error"] is None
    assert summary["n_events"] == 3
    assert summary["first_event"]["type"] == "run_started"
    assert summary["last_event"]["type"] == "run_completed"
    assert summary["report"]["status"] == "success"
    assert len(summary["leaderboard_entries"]) == 1
    assert summary["leaderboard_entries"][0]["candidate"] == "candidate_a"

    # same key set GET /api/runs/{id} returns for a server-launched run
    # (see test_run_lifecycle.test_launch_run_end_to_end)
    assert set(summary) == {
        "run_id", "orchestrator", "status", "started_at", "finished_at",
        "error", "n_events", "first_event", "last_event", "report", "leaderboard_entries",
    }

    # shows up in the list endpoint too, without ever being started via this API
    listing = client.get("/api/runs").json()
    assert any(r["run_id"] == run_id for r in listing)

    # transcripts + leaderboard endpoints work the same way for it
    names = client.get(f"/api/runs/{run_id}/transcripts").json()
    assert names == ["profiler_01.json"]
    assert client.get(f"/api/runs/{run_id}/transcripts/{names[0]}").status_code == 200

    lb = client.get(f"/api/leaderboard?run_id={run_id}").json()
    assert lb == summary["leaderboard_entries"]


def test_historical_run_sse_is_replay_only(client):
    run_id = "run_historical_sse"
    _write_historical_run(run_id)

    with client.stream("GET", f"/api/runs/{run_id}/events") as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line]

    assert len(lines) == 3
    assert json.loads(lines[0][len("data: "):])["type"] == "run_started"
    assert json.loads(lines[-1][len("data: "):])["type"] == "run_completed"
