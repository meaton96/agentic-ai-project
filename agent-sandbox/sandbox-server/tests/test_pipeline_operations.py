from fastapi.testclient import TestClient

from sandbox_server.main import create_app

AGENT_A = {
    "id": "agent-a",
    "name": "Agent A",
    "system_prompt": "be helpful",
    "model": {"base_url": "http://localhost:8000", "model_name": "test-model", "api_key_ref": "ref"},
}
AGENT_C = {**AGENT_A, "id": "agent-c", "name": "Agent C"}

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
        pipelines_dir=tmp_path / "pipelines",
        output_root=tmp_path / "runs",
        pipeline_runs_root=tmp_path / "pipeline-runs",
        operations_root=tmp_path / "operations",
    )
    return TestClient(app)


def test_swap_agent_operation_updates_the_named_step(tmp_path):
    c = client(tmp_path)
    c.post("/agents", json=AGENT_A)
    c.post("/agents", json=AGENT_C)
    c.post("/pipelines", json=PIPELINE_BODY)

    resp = c.post("/pipelines/pipe-1/operations", json={"type": "swap_agent", "step_id": "a", "new_agent_id": "agent-c"})

    assert resp.status_code == 201
    body = resp.json()
    steps = {s["step_id"]: s["agent_id"] for s in body["spec"]["steps"]}
    assert steps == {"a": "agent-c", "b": "agent-b"}
    assert body["record"]["operation"]["type"] == "swap_agent"

    # persisted
    steps = {s["step_id"]: s["agent_id"] for s in c.get("/pipelines/pipe-1").json()["steps"]}
    assert steps["a"] == "agent-c"


def test_swap_agent_to_unknown_agent_is_400(tmp_path):
    c = client(tmp_path)
    c.post("/agents", json=AGENT_A)
    c.post("/pipelines", json=PIPELINE_BODY)

    resp = c.post("/pipelines/pipe-1/operations", json={"type": "swap_agent", "step_id": "a", "new_agent_id": "ghost"})

    assert resp.status_code == 400
    # spec on disk is unchanged
    steps = {s["step_id"]: s["agent_id"] for s in c.get("/pipelines/pipe-1").json()["steps"]}
    assert steps["a"] == "agent-a"


def test_swap_agent_unknown_step_id_is_400(tmp_path):
    c = client(tmp_path)
    c.post("/agents", json=AGENT_A)
    c.post("/pipelines", json=PIPELINE_BODY)

    resp = c.post(
        "/pipelines/pipe-1/operations", json={"type": "swap_agent", "step_id": "nope", "new_agent_id": "agent-a"}
    )

    assert resp.status_code == 400


def test_alter_workflow_operation_replaces_steps(tmp_path):
    c = client(tmp_path)
    c.post("/agents", json=AGENT_A)
    c.post("/pipelines", json=PIPELINE_BODY)

    new_steps = [{"step_id": "only", "agent_id": "agent-a", "task_template": "{{task}}"}]
    resp = c.post("/pipelines/pipe-1/operations", json={"type": "alter_workflow", "steps": new_steps})

    assert resp.status_code == 201
    assert [s["step_id"] for s in resp.json()["spec"]["steps"]] == ["only"]


def test_alter_workflow_with_empty_steps_is_400(tmp_path):
    c = client(tmp_path)
    c.post("/pipelines", json=PIPELINE_BODY)

    resp = c.post("/pipelines/pipe-1/operations", json={"type": "alter_workflow", "steps": []})

    assert resp.status_code == 400
    assert len(c.get("/pipelines/pipe-1").json()["steps"]) == 2


def test_operation_on_missing_pipeline_is_404(tmp_path):
    c = client(tmp_path)
    resp = c.post(
        "/pipelines/does-not-exist/operations", json={"type": "swap_agent", "step_id": "a", "new_agent_id": "agent-a"}
    )
    assert resp.status_code == 404


def test_list_pipeline_operations_returns_history(tmp_path):
    c = client(tmp_path)
    c.post("/agents", json=AGENT_A)
    c.post("/agents", json=AGENT_C)
    c.post("/pipelines", json=PIPELINE_BODY)
    c.post("/pipelines/pipe-1/operations", json={"type": "swap_agent", "step_id": "a", "new_agent_id": "agent-c"})

    resp = c.get("/pipelines/pipe-1/operations")

    assert resp.status_code == 200
    records = resp.json()
    assert len(records) == 1
    assert records[0]["target_type"] == "pipeline"
    assert records[0]["target_id"] == "pipe-1"
    assert "agent-a" in records[0]["diff"]
    assert "agent-c" in records[0]["diff"]


def test_list_operations_on_missing_pipeline_is_404(tmp_path):
    c = client(tmp_path)
    resp = c.get("/pipelines/does-not-exist/operations")
    assert resp.status_code == 404
