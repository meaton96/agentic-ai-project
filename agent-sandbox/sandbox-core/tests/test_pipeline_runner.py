from datetime import datetime, timezone

import pytest

from sandbox_core.runtime.pipeline_runner import (
    execute_pipeline,
    read_pipeline_run_record,
    save_pipeline_run_record,
)
from sandbox_core.schemas.agent_spec import AgentSpec, ModelConfig
from sandbox_core.schemas.events import AgentResultEvent, ErrorEvent
from sandbox_core.schemas.pipeline_run import PipelineRunSpec
from sandbox_core.schemas.pipeline_spec import PipelineSpec, PipelineStep

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_agent(agent_id: str) -> AgentSpec:
    return AgentSpec(
        id=agent_id,
        name=agent_id,
        system_prompt="be helpful",
        model=ModelConfig(base_url="http://localhost", model_name="test-model", api_key_ref="ref"),
    )


class FakeAgentLoader:
    def __init__(self, agents: dict[str, AgentSpec]):
        self._agents = agents

    def load(self, agent_id: str) -> AgentSpec:
        return self._agents[agent_id]


class FakeResolver:
    def resolve(self, ref: str) -> str:
        return "fake-key"


def make_pipeline(*steps: PipelineStep) -> PipelineSpec:
    return PipelineSpec(id="pipe-1", name="test pipeline", steps=list(steps))


def make_run(**overrides) -> PipelineRunSpec:
    fields = dict(pipeline_run_id="prun-1", pipeline_id="pipe-1", task="seed task")
    fields.update(overrides)
    return PipelineRunSpec(**fields)


def fake_runner_factory(results_by_agent: dict[str, list]):
    """Returns a runner(agent, run, *, resolver, event_log) fake that pops the
    next canned terminal Event for that agent_id and records the rendered task
    it was called with, so tests can assert on inter-step substitution."""
    calls: list[dict] = []

    async def runner(agent, run, *, resolver, event_log):
        calls.append({"agent_id": agent.id, "task": run.task})
        return results_by_agent[agent.id].pop(0)

    runner.calls = calls
    return runner


def result_event(run_id: str, agent_id: str, output: str) -> AgentResultEvent:
    return AgentResultEvent(
        run_id=run_id, seq=1, ts=NOW, agent_id=agent_id, final_output=output, turns_used=1
    )


def error_event(run_id: str, agent_id: str, *, phase: str = "error") -> ErrorEvent:
    return ErrorEvent(run_id=run_id, seq=1, ts=NOW, agent_id=agent_id, message="boom", context={"phase": phase})


@pytest.mark.asyncio
async def test_sequential_steps_substitute_prior_output_into_next_task(tmp_path):
    pipeline = make_pipeline(
        PipelineStep(step_id="summarize", agent_id="summarizer", task_template="Summarize: {{task}}"),
        PipelineStep(
            step_id="write",
            agent_id="writer",
            task_template="Write this to a file: {{steps.summarize.output}}",
        ),
    )
    runner = fake_runner_factory(
        {
            "summarizer": [result_event("r1", "summarizer", "a short summary")],
            "writer": [result_event("r2", "writer", "wrote it")],
        }
    )
    loader = FakeAgentLoader({"summarizer": make_agent("summarizer"), "writer": make_agent("writer")})

    record = await execute_pipeline(
        pipeline, make_run(), agent_loader=loader, resolver=FakeResolver(), output_root=tmp_path, runner=runner
    )

    assert record.status == "completed"
    assert [s.status for s in record.steps] == ["completed", "completed"]
    assert runner.calls[0] == {"agent_id": "summarizer", "task": "Summarize: seed task"}
    assert runner.calls[1] == {"agent_id": "writer", "task": "Write this to a file: a short summary"}


@pytest.mark.asyncio
async def test_step_failure_halts_the_pipeline_before_later_steps(tmp_path):
    pipeline = make_pipeline(
        PipelineStep(step_id="a", agent_id="agent-a", task_template="{{task}}"),
        PipelineStep(step_id="b", agent_id="agent-b", task_template="{{steps.a.output}}"),
    )
    runner = fake_runner_factory(
        {
            "agent-a": [error_event("r1", "agent-a")],
            "agent-b": [result_event("r2", "agent-b", "should never run")],
        }
    )
    loader = FakeAgentLoader({"agent-a": make_agent("agent-a"), "agent-b": make_agent("agent-b")})

    record = await execute_pipeline(
        pipeline, make_run(), agent_loader=loader, resolver=FakeResolver(), output_root=tmp_path, runner=runner
    )

    assert record.status == "errored"
    assert len(record.steps) == 1
    assert record.steps[0].status == "errored"
    assert len(runner.calls) == 1  # step b never ran


@pytest.mark.asyncio
async def test_max_turns_truncation_maps_to_truncated_step_status(tmp_path):
    pipeline = make_pipeline(PipelineStep(step_id="a", agent_id="agent-a", task_template="{{task}}"))
    runner = fake_runner_factory({"agent-a": [error_event("r1", "agent-a", phase="max_turns")]})
    loader = FakeAgentLoader({"agent-a": make_agent("agent-a")})

    record = await execute_pipeline(
        pipeline, make_run(), agent_loader=loader, resolver=FakeResolver(), output_root=tmp_path, runner=runner
    )

    assert record.steps[0].status == "truncated"
    assert record.status == "errored"


@pytest.mark.asyncio
async def test_on_step_callback_fires_after_each_step(tmp_path):
    pipeline = make_pipeline(
        PipelineStep(step_id="a", agent_id="agent-a", task_template="{{task}}"),
        PipelineStep(step_id="b", agent_id="agent-b", task_template="{{steps.a.output}}"),
    )
    runner = fake_runner_factory(
        {"agent-a": [result_event("r1", "agent-a", "out-a")], "agent-b": [result_event("r2", "agent-b", "out-b")]}
    )
    loader = FakeAgentLoader({"agent-a": make_agent("agent-a"), "agent-b": make_agent("agent-b")})
    seen = []

    await execute_pipeline(
        pipeline,
        make_run(),
        agent_loader=loader,
        resolver=FakeResolver(),
        output_root=tmp_path,
        runner=runner,
        on_step=lambda result: seen.append(result.step_id),
    )

    assert seen == ["a", "b"]


def test_save_and_read_pipeline_run_record_roundtrip(tmp_path):
    from sandbox_core.schemas.pipeline_run import PipelineRunRecord, PipelineStepResult

    record = PipelineRunRecord(
        pipeline_run_id="prun-1",
        pipeline_id="pipe-1",
        task="seed task",
        created_at=NOW,
        status="completed",
        steps=[PipelineStepResult(step_id="a", agent_id="agent-a", run_id="r1", status="completed", output="out-a")],
    )

    save_pipeline_run_record(record, tmp_path)
    loaded = read_pipeline_run_record(tmp_path, "prun-1")

    assert loaded == record


def test_read_pipeline_run_record_returns_none_when_missing(tmp_path):
    assert read_pipeline_run_record(tmp_path, "does-not-exist") is None
