from fastapi.testclient import TestClient

from sandbox_server.main import create_app

AGENT_BODY = {
    "id": "agent-1",
    "name": "Agent One",
    "system_prompt": "be helpful",
    "model": {"base_url": "http://localhost:8000", "model_name": "test-model", "api_key_ref": "ref"},
}


def client(tmp_path) -> TestClient:
    app = create_app(specs_dir=tmp_path / "agents", output_root=tmp_path / "runs")
    return TestClient(app)


def test_list_agents_empty_when_specs_dir_absent(tmp_path):
    c = client(tmp_path)
    resp = c.get("/agents")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_then_get_agent(tmp_path):
    c = client(tmp_path)
    resp = c.post("/agents", json=AGENT_BODY)
    assert resp.status_code == 201
    assert resp.json()["id"] == "agent-1"

    resp = c.get("/agents/agent-1")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Agent One"

    resp = c.get("/agents")
    assert [a["id"] for a in resp.json()] == ["agent-1"]


def test_create_duplicate_id_conflicts(tmp_path):
    c = client(tmp_path)
    c.post("/agents", json=AGENT_BODY)
    resp = c.post("/agents", json=AGENT_BODY)
    assert resp.status_code == 409


def test_get_missing_agent_is_404(tmp_path):
    c = client(tmp_path)
    resp = c.get("/agents/does-not-exist")
    assert resp.status_code == 404


def test_create_agent_with_invalid_spec_returns_422_not_500(tmp_path):
    c = client(tmp_path)
    # missing required fields (name, system_prompt, model) and a bad type for id
    resp = c.post("/agents", json={"id": 123})
    assert resp.status_code == 422
    body = resp.json()
    assert "detail" in body
    # pydantic's error shape: a list of {loc, msg, type} — not a generic message
    assert isinstance(body["detail"], list)
    locs = [tuple(err["loc"]) for err in body["detail"]]
    assert ("body", "name") in locs
    assert ("body", "system_prompt") in locs
    assert ("body", "model") in locs


def test_create_agent_with_bad_transport_literal_returns_422(tmp_path):
    c = client(tmp_path)
    body = dict(AGENT_BODY, id="agent-2", mcp_servers=[{"name": "fs", "transport": "carrier-pigeon", "connection": {}}])
    resp = c.post("/agents", json=body)
    assert resp.status_code == 422


def test_update_agent_round_trips_change(tmp_path):
    c = client(tmp_path)
    c.post("/agents", json=AGENT_BODY)
    updated = dict(AGENT_BODY, name="Renamed")
    resp = c.put("/agents/agent-1", json=updated)
    assert resp.status_code == 200
    assert c.get("/agents/agent-1").json()["name"] == "Renamed"


def test_update_agent_id_mismatch_is_400(tmp_path):
    c = client(tmp_path)
    c.post("/agents", json=AGENT_BODY)
    resp = c.put("/agents/agent-1", json=dict(AGENT_BODY, id="someone-else"))
    assert resp.status_code == 400


def test_delete_agent(tmp_path):
    c = client(tmp_path)
    c.post("/agents", json=AGENT_BODY)
    resp = c.delete("/agents/agent-1")
    assert resp.status_code == 204
    assert c.get("/agents/agent-1").status_code == 404


def test_delete_missing_agent_is_404(tmp_path):
    c = client(tmp_path)
    resp = c.delete("/agents/nope")
    assert resp.status_code == 404
