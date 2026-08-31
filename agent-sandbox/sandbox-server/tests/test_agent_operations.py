from fastapi.testclient import TestClient

from sandbox_server.main import create_app

AGENT_BODY = {
    "id": "agent-1",
    "name": "Agent One",
    "system_prompt": "be helpful",
    "model": {"base_url": "http://localhost:8000", "model_name": "test-model", "api_key_ref": "ref"},
}


def client(tmp_path) -> TestClient:
    app = create_app(
        specs_dir=tmp_path / "agents",
        pipelines_dir=tmp_path / "pipelines",
        output_root=tmp_path / "runs",
        operations_root=tmp_path / "operations",
    )
    return TestClient(app)


def test_alter_model_operation_updates_spec(tmp_path):
    c = client(tmp_path)
    c.post("/agents", json=AGENT_BODY)

    resp = c.post(
        "/agents/agent-1/operations",
        json={"type": "alter_model", "model": {"base_url": "http://new", "model_name": "m2", "api_key_ref": "k2"}},
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["spec"]["model"]["base_url"] == "http://new"
    assert body["record"]["operation"]["type"] == "alter_model"
    assert body["record"]["target_type"] == "agent"
    assert body["record"]["target_id"] == "agent-1"
    assert body["record"]["seq"] == 1

    # persisted, not just returned
    assert c.get("/agents/agent-1").json()["model"]["base_url"] == "http://new"


def test_operation_on_missing_agent_is_404(tmp_path):
    c = client(tmp_path)
    resp = c.post(
        "/agents/does-not-exist/operations",
        json={"type": "alter_model", "model": {"base_url": "http://new", "model_name": "m2", "api_key_ref": "k2"}},
    )
    assert resp.status_code == 404


def test_swap_agent_operation_against_agent_target_is_400(tmp_path):
    """swap_agent only applies to pipelines — hitting the agent-scoped
    endpoint with one should be a clean 400, not a 500."""
    c = client(tmp_path)
    c.post("/agents", json=AGENT_BODY)

    resp = c.post("/agents/agent-1/operations", json={"type": "swap_agent", "step_id": "a", "new_agent_id": "other"})

    assert resp.status_code == 400


def test_list_operations_returns_history_in_order(tmp_path):
    c = client(tmp_path)
    c.post("/agents", json=AGENT_BODY)
    c.post(
        "/agents/agent-1/operations",
        json={"type": "alter_model", "model": {"base_url": "http://one", "model_name": "m", "api_key_ref": "k"}},
    )
    c.post(
        "/agents/agent-1/operations",
        json={"type": "alter_model", "model": {"base_url": "http://two", "model_name": "m", "api_key_ref": "k"}},
    )

    resp = c.get("/agents/agent-1/operations")

    assert resp.status_code == 200
    records = resp.json()
    assert [r["seq"] for r in records] == [1, 2]
    assert records[1]["operation"]["model"]["base_url"] == "http://two"


def test_list_operations_on_missing_agent_is_404(tmp_path):
    c = client(tmp_path)
    resp = c.get("/agents/does-not-exist/operations")
    assert resp.status_code == 404


def test_list_operations_empty_before_any_operation(tmp_path):
    c = client(tmp_path)
    c.post("/agents", json=AGENT_BODY)
    resp = c.get("/agents/agent-1/operations")
    assert resp.status_code == 200
    assert resp.json() == []
