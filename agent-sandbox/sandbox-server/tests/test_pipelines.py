from fastapi.testclient import TestClient

from sandbox_server.main import create_app

PIPELINE_BODY = {
    "id": "pipe-1",
    "name": "Pipeline One",
    "steps": [
        {"step_id": "a", "agent_id": "agent-a", "task_template": "{{task}}"},
        {"step_id": "b", "agent_id": "agent-b", "task_template": "{{steps.a.output}}"},
    ],
}


def client(tmp_path) -> TestClient:
    app = create_app(
        specs_dir=tmp_path / "agents",
        output_root=tmp_path / "runs",
        pipelines_dir=tmp_path / "pipelines",
        pipeline_runs_root=tmp_path / "pipeline-runs",
    )
    return TestClient(app)


def test_list_pipelines_empty_when_specs_dir_absent(tmp_path):
    c = client(tmp_path)
    resp = c.get("/pipelines")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_then_get_pipeline(tmp_path):
    c = client(tmp_path)
    resp = c.post("/pipelines", json=PIPELINE_BODY)
    assert resp.status_code == 201
    assert resp.json()["id"] == "pipe-1"

    resp = c.get("/pipelines/pipe-1")
    assert resp.status_code == 200
    assert [s["step_id"] for s in resp.json()["steps"]] == ["a", "b"]

    resp = c.get("/pipelines")
    assert [p["id"] for p in resp.json()] == ["pipe-1"]


def test_create_duplicate_id_conflicts(tmp_path):
    c = client(tmp_path)
    c.post("/pipelines", json=PIPELINE_BODY)
    resp = c.post("/pipelines", json=PIPELINE_BODY)
    assert resp.status_code == 409


def test_get_missing_pipeline_is_404(tmp_path):
    c = client(tmp_path)
    resp = c.get("/pipelines/does-not-exist")
    assert resp.status_code == 404


def test_create_pipeline_with_zero_steps_returns_422_not_500(tmp_path):
    c = client(tmp_path)
    resp = c.post("/pipelines", json={"id": "empty", "name": "empty", "steps": []})
    assert resp.status_code == 422


def test_update_pipeline_round_trips_change(tmp_path):
    c = client(tmp_path)
    c.post("/pipelines", json=PIPELINE_BODY)
    updated = dict(PIPELINE_BODY, name="Renamed")
    resp = c.put("/pipelines/pipe-1", json=updated)
    assert resp.status_code == 200
    assert c.get("/pipelines/pipe-1").json()["name"] == "Renamed"


def test_update_pipeline_id_mismatch_is_400(tmp_path):
    c = client(tmp_path)
    c.post("/pipelines", json=PIPELINE_BODY)
    resp = c.put("/pipelines/pipe-1", json=dict(PIPELINE_BODY, id="someone-else"))
    assert resp.status_code == 400


def test_delete_pipeline(tmp_path):
    c = client(tmp_path)
    c.post("/pipelines", json=PIPELINE_BODY)
    resp = c.delete("/pipelines/pipe-1")
    assert resp.status_code == 204
    assert c.get("/pipelines/pipe-1").status_code == 404


def test_delete_missing_pipeline_is_404(tmp_path):
    c = client(tmp_path)
    resp = c.delete("/pipelines/nope")
    assert resp.status_code == 404
