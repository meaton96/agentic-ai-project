"""
SSE replay-then-tail: connecting while a run is in progress should get
whatever's already in events.jsonl, then keep receiving lines as the
run process appends more, then the connection should close once the run
finishes. Timing is made deterministic with a custom launch_fn (still a
real forked process, just not the real orchestrator) instead of relying
on the real pipeline's stub timing, which is too fast to reliably
observe a live tail over.
"""
from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from server.app import create_app
from server.run_manager import RunConfig

STEP_DELAY = 0.4
N_STEPS = 3


def _paced_launch_fn(run_id, config: RunConfig) -> None:
    from agentic_ml.events import emit_event, make_event_emitter, make_event_logger
    from agentic_ml.paths import run_dir as resolve_run_dir

    run_dir = resolve_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    emit = make_event_emitter(run_id, persist_fn=make_event_logger(run_dir))
    emit_event(emit, "run", "run_started", {"data": config.dataset})
    for i in range(N_STEPS):
        time.sleep(STEP_DELAY)
        emit_event(emit, "work", "step", {"i": i})
    time.sleep(STEP_DELAY)
    emit_event(emit, "run", "run_completed", {"status": "success"})


def _parse_sse_lines(raw_lines):
    events = []
    for line in raw_lines:
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


def test_sse_replay_then_tail(dataset_csv):
    app = create_app(launch_fn=_paced_launch_fn)
    with TestClient(app) as client:
        launch = client.post("/api/runs", json={
            "dataset": dataset_csv, "orchestrator": "static", "target": "x",
        })
        run_id = launch.json()["run_id"]

        # connect essentially immediately, well before run_completed
        started = time.time()
        with client.stream("GET", f"/api/runs/{run_id}/events") as resp:
            assert resp.status_code == 200
            raw_lines = [line for line in resp.iter_lines() if line]
        elapsed = time.time() - started

        events = _parse_sse_lines(raw_lines)
        types = [e["type"] for e in events]

        assert types[0] == "run_started"
        assert types[-1] == "run_completed"
        assert types.count("step") == N_STEPS
        # proves this wasn't just a replay of an already-finished file —
        # the connection stayed open across the run's own pacing.
        assert elapsed >= STEP_DELAY * N_STEPS


def test_sse_on_already_finished_run_replays_and_closes_immediately(dataset_csv):
    app = create_app(launch_fn=_paced_launch_fn)
    with TestClient(app) as client:
        launch = client.post("/api/runs", json={
            "dataset": dataset_csv, "orchestrator": "static", "target": "x",
        })
        run_id = launch.json()["run_id"]

        deadline = time.time() + 15
        while client.get(f"/api/runs/{run_id}").json()["status"] == "running":
            if time.time() > deadline:
                raise TimeoutError("paced run never finished")
            time.sleep(0.1)

        started = time.time()
        with client.stream("GET", f"/api/runs/{run_id}/events") as resp:
            raw_lines = [line for line in resp.iter_lines() if line]
        elapsed = time.time() - started

        events = _parse_sse_lines(raw_lines)
        assert events[0]["type"] == "run_started"
        assert events[-1]["type"] == "run_completed"
        # a finished run's stream is just a replay: fast, no tailing wait.
        assert elapsed < STEP_DELAY


def test_sse_unknown_run_is_404(client):
    resp = client.get("/api/runs/does_not_exist/events")
    assert resp.status_code == 404
