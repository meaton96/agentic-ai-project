from __future__ import annotations

from agentic_ml.orchestrator import agent_registry


def test_dynamic_catalog_node_ids_match_agent_registry(client):
    """Regression guard: if this ever regresses to a hand-duplicated agent
    list instead of deriving from agent_registry.AGENTS, adding/removing an
    agent there without updating the route would make this fail."""
    resp = client.get("/api/workflow-catalog", params={"type": "dynamic"})
    assert resp.status_code == 200
    body = resp.json()
    node_ids = {n["id"] for n in body["nodes"]}
    assert node_ids == set(agent_registry.AGENTS.keys())


def test_dynamic_catalog_carries_registry_fields(client):
    resp = client.get("/api/workflow-catalog", params={"type": "dynamic"})
    body = resp.json()
    by_id = {n["id"]: n for n in body["nodes"]}
    intake = by_id["intake"]
    spec = agent_registry.AGENTS["intake"]
    assert intake["description"] == spec.description
    assert intake["data"]["when_to_use"] == spec.when_to_use
    assert intake["data"]["required_state"] == spec.required_state


def test_static_topology_includes_agent_and_gate_kinds(client):
    resp = client.get("/api/workflow-catalog", params={"type": "static"})
    assert resp.status_code == 200
    body = resp.json()
    kinds = {n["kind"] for n in body["nodes"]}
    assert kinds == {"agent", "gate"}
    assert len(body["edges"]) > 0
