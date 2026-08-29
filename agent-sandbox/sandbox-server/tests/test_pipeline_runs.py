"""Pipeline-run tests against a fake runner — never a live model or MCP
server. Mirrors test_runs_and_stream.py's fake_runner_* pattern one level up:
the fake stands in for sandbox_core.runtime.agent_loop's execute_run, the
same substitution point PipelineRunManager forwards into execute_pipeline.

Uses httpx's ASGITransport (not the sync FastAPI TestClient), same reason
test_runs_and_stream.py does: the test coroutine and the server's
asyncio.create_task-launched pipeline run need to share one event loop.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from sandbox_core.schemas.agent_spec import AgentSpec, ModelConfig
from sandbox_core.schemas.events import AgentResultEvent, ErrorEvent
from sandbox_core.schemas.pipeline_spec import PipelineSpec, PipelineStep
from sandbox_server.main import create_app
from sandbox_server.pipeline_specs import write_pipeline_spec
from sandbox_server.specs import write_agent_spec

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

AGENT_A = AgentSpec(
    id="agent-a", name="A", system_prompt="be helpful",
    model=ModelConfig(base_url="http://localhost", model_name="test-model", api_key_ref="ref"),
)
AGENT_B = AgentSpec(
    id="agent-b", name="B", system_prompt="be helpful",
    model=ModelConfig(base_url="http://localhost", model_name="test-model", api_key_ref="ref"),
)
PIPELINE = PipelineSpec(
    id="pipe-1",
    name="Pipeline One",
    steps=[
        PipelineStep(step_id="a", agent_id="agent-a", task_template="{{task}}"),
        PipelineStep(step_id="b", agent_id="agent-b", task_template="got: {{steps.a.output}}"),
    ],
)


async def fake_runner_success(agent, run, *, resolver, event_log):
    await asyncio.sleep(0.01)
    return event_log.append(
        AgentResultEvent(
            run_id=run.run_id, seq=1, ts=NOW, agent_id=agent.id, final_output=f"output of {agent.id}", turns_used=1
        )
    )


async def fake_runner_second_step_fails(agent, run, *, resolver, event_log):
    if agent.id == "agent-a":
        return event_log.append(
            AgentResultEvent(run_id=run.run_id, seq=1, ts=NOW, agent_id=agent.id, final_output="ok-a", turns_used=1)
        )
    return event_log.append(
        ErrorEvent(run_id=run.run_id, seq=1, ts=NOW, agent_id=agent.id, message="boom", context={"phase": "error"})
    )


async def fake_runner_raises_before_any_event(agent, run, *, resolver, event_log):
    raise RuntimeError("credential ref 'ref' not found")


def make_app(tmp_path: Path, runner):
    write_agent_spec(tmp_path / "agents", AGENT_A)
    write_agent_spec(tmp_path / "agents", AGENT_B)
    write_pipeline_spec(tmp_path / "pipelines", PIPELINE)
    return create_app(
        specs_dir=tmp_path / "agents",
        output_root=tmp_path / "runs",
        pipelines_dir=tmp_path / "pipelines",
        pipeline_runs_root=tmp_path / "pipeline-runs",
        runner=runner,
    )


async def _launch(client: httpx.AsyncClient, pipeline_id: str = "pipe-1", task: str = "seed") -> str:
    resp = await client.post("/pipeline-runs", json={"pipeline_id": pipeline_id, "task": task})
    assert resp.status_code == 202
    return resp.json()["pipeline_run_id"]


async def _await_terminal(client: httpx.AsyncClient, pipeline_run_id: str) -> dict:
    for _ in range(200):
        detail = (await client.get(f"/pipeline-runs/{pipeline_run_id}")).json()
        if detail["status"] != "running":
            return detail
        await asyncio.sleep(0.01)
    raise AssertionError("pipeline run never left 'running' status")


@pytest.mark.asyncio
async def test_launch_pipeline_run_returns_202_and_id(tmp_path):
    app = make_app(tmp_path, fake_runner_success)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        run_id = await _launch(client)
        assert run_id


@pytest.mark.asyncio
async def test_pipeline_run_detail_shows_both_steps_completed_with_substituted_task(tmp_path):
    app = make_app(tmp_path, fake_runner_success)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        run_id = await _launch(client)
        detail = await _await_terminal(client, run_id)

        assert detail["status"] == "completed"
        assert [s["step_id"] for s in detail["steps"]] == ["a", "b"]
        assert [s["status"] for s in detail["steps"]] == ["completed", "completed"]
        assert detail["steps"][0]["output"] == "output of agent-a"


@pytest.mark.asyncio
async def test_pipeline_run_halts_and_reports_errored_when_a_step_fails(tmp_path):
    app = make_app(tmp_path, fake_runner_second_step_fails)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        run_id = await _launch(client)
        detail = await _await_terminal(client, run_id)

        assert detail["status"] == "errored"
        assert len(detail["steps"]) == 2
        assert detail["steps"][1]["status"] == "errored"


@pytest.mark.asyncio
async def test_pipeline_run_failure_before_any_step_is_reported_as_errored(tmp_path):
    app = make_app(tmp_path, fake_runner_raises_before_any_event)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        run_id = await _launch(client)
        detail = await _await_terminal(client, run_id)

        assert detail["status"] == "errored"
        assert "credential" in detail["error"]


@pytest.mark.asyncio
async def test_launch_pipeline_run_for_missing_pipeline_is_404(tmp_path):
    app = make_app(tmp_path, fake_runner_success)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/pipeline-runs", json={"pipeline_id": "does-not-exist", "task": "seed"})
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_missing_pipeline_run_is_404(tmp_path):
    app = make_app(tmp_path, fake_runner_success)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/pipeline-runs/does-not-exist")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_pipeline_runs_includes_launched_run(tmp_path):
    app = make_app(tmp_path, fake_runner_success)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        run_id = await _launch(client)
        await _await_terminal(client, run_id)

        summaries = (await client.get("/pipeline-runs")).json()
        assert [s["pipeline_run_id"] for s in summaries] == [run_id]
