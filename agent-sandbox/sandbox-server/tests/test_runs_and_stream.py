"""Runs/stream tests against a fake runner — never a live model or MCP
server. `fake_runner_*` below stand in for sandbox_core.runtime.agent_loop's
`execute_run`, same shape (agent, run, *, resolver, event_log), the same
substitution point sandbox-core's own tests use for `run_agent` one level
down (FakeModelClient/FakeServer in test_agent_loop.py).

httpx's ASGITransport (not the sync FastAPI TestClient) is used for the
streaming tests specifically so the test coroutine and the server's
`asyncio.create_task`-launched run share one event loop — that's what makes
`await asyncio.sleep(...)` inside a fake runner an effective, deterministic
way to prove events arrive *while the stream is open*, not just after the
run finishes and get replayed.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from sandbox_core.schemas.agent_spec import AgentSpec, ModelConfig
from sandbox_core.schemas.events import AgentResultEvent, ErrorEvent, LlmRequestEvent, LlmResponseEvent
from sandbox_server.main import create_app
from sandbox_server.specs import write_agent_spec

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

AGENT = AgentSpec(
    id="agent-1",
    name="Agent One",
    system_prompt="be helpful",
    model=ModelConfig(base_url="http://localhost", model_name="test-model", api_key_ref="ref"),
)


async def fake_runner_success(agent, run, *, resolver, event_log):
    event_log.append(
        LlmRequestEvent(run_id=run.run_id, seq=0, ts=NOW, agent_id=agent.id, messages=[], model="test-model")
    )
    await asyncio.sleep(0.02)
    event_log.append(
        LlmResponseEvent(run_id=run.run_id, seq=0, ts=NOW, agent_id=agent.id, content="hi", duration_ms=1.0)
    )
    await asyncio.sleep(0.02)
    return event_log.append(
        AgentResultEvent(run_id=run.run_id, seq=0, ts=NOW, agent_id=agent.id, final_output="done", turns_used=1)
    )


async def fake_runner_max_turns_truncation(agent, run, *, resolver, event_log):
    return event_log.append(
        ErrorEvent(
            run_id=run.run_id, seq=0, ts=NOW, agent_id=agent.id,
            message="run truncated: hit max_turns (1)", context={"phase": "max_turns", "max_turns": 1},
        )
    )


async def fake_runner_raises_before_any_event(agent, run, *, resolver, event_log):
    raise RuntimeError("credential ref 'ref' not found")


def make_app(tmp_path: Path, runner):
    specs_dir = tmp_path / "agents"
    write_agent_spec(specs_dir, AGENT)
    return create_app(specs_dir=specs_dir, output_root=tmp_path / "runs", runner=runner)


async def _stream_lines(client: httpx.AsyncClient, url: str) -> list[dict]:
    events = []
    async with client.stream("GET", url) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
    return events


@pytest.mark.asyncio
async def test_stream_delivers_events_in_order_and_closes_on_completion(tmp_path):
    app = make_app(tmp_path, fake_runner_success)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/runs", json={"agent_id": "agent-1", "task": "do it"})
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]

        events = await _stream_lines(client, f"/runs/{run_id}/stream")

        assert [e["type"] for e in events] == ["llm_request", "llm_response", "agent_result"]
        assert [e["seq"] for e in events] == [1, 2, 3]

        # the stream closing on its own is the assertion above (aiter_lines
        # returned instead of hanging) — confirm the run is also reflected
        # as finished via the regular detail route
        detail = await client.get(f"/runs/{run_id}")
        assert detail.json()["status"] == "completed"


@pytest.mark.asyncio
async def test_two_simultaneous_clients_see_the_same_events(tmp_path):
    app = make_app(tmp_path, fake_runner_success)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/runs", json={"agent_id": "agent-1", "task": "do it"})
        run_id = resp.json()["run_id"]

        events_a, events_b = await asyncio.gather(
            _stream_lines(client, f"/runs/{run_id}/stream"),
            _stream_lines(client, f"/runs/{run_id}/stream"),
        )

        types_a = [e["type"] for e in events_a]
        types_b = [e["type"] for e in events_b]
        assert types_a == ["llm_request", "llm_response", "agent_result"]
        assert types_b == types_a


@pytest.mark.asyncio
async def test_connecting_after_run_already_finished_replays_and_closes(tmp_path):
    app = make_app(tmp_path, fake_runner_success)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/runs", json={"agent_id": "agent-1", "task": "do it"})
        run_id = resp.json()["run_id"]

        # drain once to let the (fast, sleep-based) fake runner actually finish
        await _stream_lines(client, f"/runs/{run_id}/stream")

        # second connection, well after completion — pure replay, no live wait
        events = await _stream_lines(client, f"/runs/{run_id}/stream")
        assert [e["type"] for e in events] == ["llm_request", "llm_response", "agent_result"]


@pytest.mark.asyncio
async def test_client_disconnect_mid_stream_does_not_kill_the_run(tmp_path):
    app = make_app(tmp_path, fake_runner_success)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/runs", json={"agent_id": "agent-1", "task": "do it"})
        run_id = resp.json()["run_id"]

        # connect and immediately walk away after the first line, well before
        # the run's own internal sleeps let it finish
        async with client.stream("GET", f"/runs/{run_id}/stream") as response:
            async for _ in response.aiter_lines():
                break

        # give the still-running background task time to finish on its own
        await asyncio.sleep(0.2)

        detail = await client.get(f"/runs/{run_id}")
        assert detail.json()["status"] == "completed"


@pytest.mark.asyncio
async def test_max_turns_truncation_reports_truncated_status(tmp_path):
    app = make_app(tmp_path, fake_runner_max_turns_truncation)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/runs", json={"agent_id": "agent-1", "task": "do it"})
        run_id = resp.json()["run_id"]

        events = await _stream_lines(client, f"/runs/{run_id}/stream")
        assert events[-1]["type"] == "error"

        detail = await client.get(f"/runs/{run_id}")
        assert detail.json()["status"] == "truncated"


@pytest.mark.asyncio
async def test_launch_failure_before_any_event_is_reported_as_errored(tmp_path):
    app = make_app(tmp_path, fake_runner_raises_before_any_event)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/runs", json={"agent_id": "agent-1", "task": "do it"})
        run_id = resp.json()["run_id"]

        events = await _stream_lines(client, f"/runs/{run_id}/stream")
        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert "credential ref" in events[0]["message"]

        detail = await client.get(f"/runs/{run_id}")
        assert detail.json()["status"] == "errored"


@pytest.mark.asyncio
async def test_launch_run_with_unknown_agent_id_is_404(tmp_path):
    app = make_app(tmp_path, fake_runner_success)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/runs", json={"agent_id": "no-such-agent", "task": "do it"})
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_runs_reports_turn_count_and_status(tmp_path):
    app = make_app(tmp_path, fake_runner_success)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/runs", json={"agent_id": "agent-1", "task": "do it"})
        run_id = resp.json()["run_id"]
        await _stream_lines(client, f"/runs/{run_id}/stream")

        runs = (await client.get("/runs")).json()
        assert len(runs) == 1
        assert runs[0]["run_id"] == run_id
        assert runs[0]["agent_id"] == "agent-1"
        assert runs[0]["status"] == "completed"
        assert runs[0]["turn_count"] == 1


@pytest.mark.asyncio
async def test_streaming_unknown_run_id_is_404(tmp_path):
    app = make_app(tmp_path, fake_runner_success)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/runs/no-such-run/stream")
        assert resp.status_code == 404
