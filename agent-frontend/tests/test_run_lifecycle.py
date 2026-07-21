from __future__ import annotations

import time

from fastapi.testclient import TestClient

from server.app import create_app
from server.run_manager import RunConfig
from tests.conftest import wait_for_status


def test_launch_run_end_to_end(stubbed_client, dataset_csv):
    client = stubbed_client
    resp = client.post("/api/runs", json={
        "dataset": dataset_csv,
        "orchestrator": "static",
        "target": "churned",
        "skip_feature_engineering": True,
        "max_candidates": 1,
    })
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]
    assert resp.json()["status"] == "running"

    final = wait_for_status(client, run_id)
    assert final["status"] == "completed", final
    assert final["orchestrator"] == "static"
    assert final["error"] is None
    assert final["n_events"] > 0
    assert final["last_event"]["type"] == "run_completed"

    report = final["report"]
    assert report is not None
    assert report["status"] == "success"
    assert set(report["final_test_metrics"]) == {"roc_auc", "pr_auc", "f1", "accuracy"}

    assert len(final["leaderboard_entries"]) == 1
    assert final["leaderboard_entries"][0]["candidate"] == "candidate_a"
    assert final["leaderboard_entries"][0]["verification_verdict"] == "approved"

    # GET /api/runs includes it
    listing = client.get("/api/runs").json()
    assert any(r["run_id"] == run_id for r in listing)

    # GET /api/leaderboard, filtered by run_id, matches the summary's entries
    lb = client.get(f"/api/leaderboard?run_id={run_id}").json()
    assert lb == final["leaderboard_entries"]

    # transcripts: at least profiler/modeling/verification/analyst_summary
    names = client.get(f"/api/runs/{run_id}/transcripts").json()
    assert len(names) >= 3
    one = client.get(f"/api/runs/{run_id}/transcripts/{names[0]}")
    assert one.status_code == 200
    assert isinstance(one.json(), list)


def test_get_unknown_run_is_404(client):
    resp = client.get("/api/runs/does_not_exist")
    assert resp.status_code == 404


def _slow_launch_fn(run_id, config: RunConfig) -> None:
    from agentic_ml.events import emit_event, make_event_emitter, make_event_logger
    from agentic_ml.paths import run_dir as resolve_run_dir

    run_dir = resolve_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    emit = make_event_emitter(run_id, persist_fn=make_event_logger(run_dir))
    emit_event(emit, "run", "run_started", {"data": config.dataset})
    time.sleep(0.6)
    emit_event(emit, "run", "run_completed", {"status": "success"})


def test_second_run_rejected_while_one_active(dataset_csv):
    app = create_app(launch_fn=_slow_launch_fn)
    with TestClient(app) as client:
        body = {"dataset": dataset_csv, "orchestrator": "static", "target": "x"}
        first = client.post("/api/runs", json=body)
        assert first.status_code == 202
        run_id = first.json()["run_id"]

        second = client.post("/api/runs", json=body)
        assert second.status_code == 409

        final = wait_for_status(client, run_id)
        assert final["status"] == "completed"

        # now that the first run has finished, a new run is accepted
        third = client.post("/api/runs", json=body)
        assert third.status_code == 202
        wait_for_status(client, third.json()["run_id"])
