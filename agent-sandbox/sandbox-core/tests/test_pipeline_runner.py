import json
from datetime import datetime, timezone

import fixture_gates
import pytest

from sandbox_core.runtime.pipeline_runner import (
    _resolve_gate,
    execute_pipeline,
    pipeline_run_path,
    read_pipeline_run_record,
    save_pipeline_run_record,
)
from sandbox_core.schemas.agent_spec import AgentSpec, ModelConfig
from sandbox_core.schemas.events import AgentResultEvent, ErrorEvent
from sandbox_core.schemas.pipeline_run import GateStepResult, PipelineRunSpec, PipelineStepResult
from sandbox_core.schemas.pipeline_spec import GateStep, PipelineSpec, PipelineStep

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


def make_pipeline(*steps: PipelineStep | GateStep, max_steps: int = 50) -> PipelineSpec:
    return PipelineSpec(id="pipe-1", name="test pipeline", steps=list(steps), max_steps=max_steps)


def make_gate(step_id: str, gate: str, on_result: dict[str, str]) -> GateStep:
    return GateStep(kind="gate", step_id=step_id, gate=gate, on_result=on_result)


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


def test_read_pipeline_run_record_returns_none_for_malformed_json(tmp_path):
    path = pipeline_run_path(tmp_path, "bad-json")
    path.parent.mkdir(parents=True)
    path.write_text("{not valid json")

    assert read_pipeline_run_record(tmp_path, "bad-json") is None


def test_read_pipeline_run_record_returns_none_for_pre_phase3_schema(tmp_path):
    """A record written before the `kind` discriminator existed on step
    results (no `kind` field on any step) must not raise — one such record
    must not be able to break listing every other pipeline run alongside
    it (PipelineRunManager.list_pipeline_runs calls this once per run_id)."""
    path = pipeline_run_path(tmp_path, "pre-phase3")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "pipeline_run_id": "pre-phase3",
                "pipeline_id": "pipe-1",
                "task": "seed task",
                "created_at": NOW.isoformat(),
                "status": "completed",
                "steps": [{"step_id": "a", "agent_id": "agent-a", "run_id": "r1", "status": "completed"}],
            }
        )
    )

    assert read_pipeline_run_record(tmp_path, "pre-phase3") is None


# -- _resolve_gate -----------------------------------------------------------


def test_resolve_gate_valid_path_returns_the_callable():
    assert _resolve_gate("fixture_gates:always_approve") is fixture_gates.always_approve


def test_resolve_gate_rejects_malformed_path():
    with pytest.raises(ValueError, match="module.path:function_name"):
        _resolve_gate("no-colon-in-this-string")


def test_resolve_gate_missing_module_raises_clear_value_error():
    with pytest.raises(ValueError, match="could not import module"):
        _resolve_gate("no_such_module_xyz:fn")


def test_resolve_gate_missing_function_raises_clear_value_error():
    with pytest.raises(ValueError, match="is not defined"):
        _resolve_gate("fixture_gates:no_such_function")


def test_resolve_gate_non_callable_attribute_raises_clear_value_error():
    with pytest.raises(ValueError, match="is not callable"):
        _resolve_gate("fixture_gates:not_callable")


# -- gate routing in execute_pipeline ----------------------------------------


@pytest.mark.asyncio
async def test_gate_routes_to_one_branch_and_the_other_never_runs(tmp_path):
    # "untaken" is declared before "taken" so a stray linear fallthrough
    # after "taken" (the last declared step) can't accidentally run it —
    # only the gate's jump could reach either, and it always picks "taken".
    pipeline = make_pipeline(
        PipelineStep(step_id="a", agent_id="agent-a", task_template="{{task}}"),
        make_gate("check", "fixture_gates:always_approve", {"approved": "taken", "rejected": "untaken"}),
        PipelineStep(step_id="untaken", agent_id="agent-untaken", task_template="never"),
        PipelineStep(step_id="taken", agent_id="agent-taken", task_template="{{steps.a.output}}"),
    )
    runner = fake_runner_factory(
        {
            "agent-a": [result_event("r1", "agent-a", "out-a")],
            "agent-taken": [result_event("r2", "agent-taken", "out-taken")],
            "agent-untaken": [result_event("r3", "agent-untaken", "should never run")],
        }
    )
    loader = FakeAgentLoader(
        {
            "agent-a": make_agent("agent-a"),
            "agent-taken": make_agent("agent-taken"),
            "agent-untaken": make_agent("agent-untaken"),
        }
    )

    record = await execute_pipeline(
        pipeline, make_run(), agent_loader=loader, resolver=FakeResolver(), output_root=tmp_path, runner=runner
    )

    assert record.status == "completed"
    assert [s.step_id for s in record.steps] == ["a", "check", "taken"]
    assert {c["agent_id"] for c in runner.calls} == {"agent-a", "agent-taken"}
    gate_result = record.steps[1]
    assert isinstance(gate_result, GateStepResult)
    assert gate_result.decision == "approved"
    assert gate_result.routed_to == "taken"


@pytest.mark.asyncio
async def test_gate_end_sentinel_stops_before_later_declared_steps(tmp_path):
    pipeline = make_pipeline(
        PipelineStep(step_id="a", agent_id="agent-a", task_template="{{task}}"),
        make_gate("check", "fixture_gates:always_approve", {"approved": "__end__"}),
        PipelineStep(step_id="b", agent_id="agent-b", task_template="never"),
    )
    runner = fake_runner_factory(
        {
            "agent-a": [result_event("r1", "agent-a", "out-a")],
            "agent-b": [result_event("r2", "agent-b", "should never run")],
        }
    )
    loader = FakeAgentLoader({"agent-a": make_agent("agent-a"), "agent-b": make_agent("agent-b")})

    record = await execute_pipeline(
        pipeline, make_run(), agent_loader=loader, resolver=FakeResolver(), output_root=tmp_path, runner=runner
    )

    assert record.status == "completed"
    assert [s.step_id for s in record.steps] == ["a", "check"]
    assert record.steps[-1].routed_to is None
    assert all(c["agent_id"] != "agent-b" for c in runner.calls)


@pytest.mark.asyncio
async def test_gate_reject_retry_loop_reruns_the_looped_back_step_with_a_new_run_id(tmp_path):
    fixture_gates.reject_then_approve_calls = 0
    pipeline = make_pipeline(
        PipelineStep(step_id="intake", agent_id="agent-intake", task_template="{{task}}"),
        PipelineStep(step_id="modeling", agent_id="agent-modeling", task_template="{{steps.intake.output}}"),
        make_gate("verify", "fixture_gates:reject_then_approve", {"rejected": "modeling", "approved": "finalize"}),
        PipelineStep(step_id="finalize", agent_id="agent-finalize", task_template="{{steps.modeling.output}}"),
    )
    runner = fake_runner_factory(
        {
            "agent-intake": [result_event("r1", "agent-intake", "out-intake")],
            "agent-modeling": [
                result_event("r2", "agent-modeling", "attempt-1"),
                result_event("r3", "agent-modeling", "attempt-2"),
            ],
            "agent-finalize": [result_event("r4", "agent-finalize", "done")],
        }
    )
    loader = FakeAgentLoader(
        {
            "agent-intake": make_agent("agent-intake"),
            "agent-modeling": make_agent("agent-modeling"),
            "agent-finalize": make_agent("agent-finalize"),
        }
    )

    record = await execute_pipeline(
        pipeline, make_run(), agent_loader=loader, resolver=FakeResolver(), output_root=tmp_path, runner=runner
    )

    assert record.status == "completed"
    modeling_results = [s for s in record.steps if isinstance(s, PipelineStepResult) and s.step_id == "modeling"]
    assert len(modeling_results) == 2
    assert len({s.run_id for s in modeling_results}) == 2
    assert [s.step_id for s in record.steps] == ["intake", "modeling", "verify", "modeling", "verify", "finalize"]


@pytest.mark.asyncio
async def test_gate_decision_not_in_on_result_halts_with_a_clear_error(tmp_path):
    pipeline = make_pipeline(
        PipelineStep(step_id="a", agent_id="agent-a", task_template="{{task}}"),
        make_gate("check", "fixture_gates:always_approve", {"only_this_key": "somewhere"}),
    )
    runner = fake_runner_factory({"agent-a": [result_event("r1", "agent-a", "out-a")]})
    loader = FakeAgentLoader({"agent-a": make_agent("agent-a")})

    record = await execute_pipeline(
        pipeline, make_run(), agent_loader=loader, resolver=FakeResolver(), output_root=tmp_path, runner=runner
    )

    assert record.status == "errored"
    assert "approved" in record.error
    assert "check" in record.error
    assert "only_this_key" in record.error


@pytest.mark.asyncio
async def test_gate_routing_to_unknown_step_id_halts_with_a_clear_error(tmp_path):
    pipeline = make_pipeline(
        PipelineStep(step_id="a", agent_id="agent-a", task_template="{{task}}"),
        make_gate("check", "fixture_gates:always_approve", {"approved": "does-not-exist"}),
    )
    runner = fake_runner_factory({"agent-a": [result_event("r1", "agent-a", "out-a")]})
    loader = FakeAgentLoader({"agent-a": make_agent("agent-a")})

    record = await execute_pipeline(
        pipeline, make_run(), agent_loader=loader, resolver=FakeResolver(), output_root=tmp_path, runner=runner
    )

    assert record.status == "errored"
    assert "does-not-exist" in record.error


@pytest.mark.asyncio
async def test_unterminated_gate_loop_halts_at_max_steps_instead_of_hanging(tmp_path):
    pipeline = make_pipeline(
        PipelineStep(step_id="a", agent_id="agent-a", task_template="{{task}}"),
        make_gate("check", "fixture_gates:always_loop", {"loop": "a"}),
        max_steps=4,
    )
    runner = fake_runner_factory({"agent-a": [result_event(f"r{i}", "agent-a", "out-a") for i in range(10)]})
    loader = FakeAgentLoader({"agent-a": make_agent("agent-a")})

    record = await execute_pipeline(
        pipeline, make_run(), agent_loader=loader, resolver=FakeResolver(), output_root=tmp_path, runner=runner
    )

    assert record.status == "errored"
    assert "max_steps" in record.error


@pytest.mark.asyncio
async def test_async_gate_function_is_awaited_not_ignored(tmp_path):
    pipeline = make_pipeline(
        PipelineStep(step_id="a", agent_id="agent-a", task_template="{{task}}"),
        make_gate("check", "fixture_gates:async_always_approve", {"approved": "__end__"}),
    )
    runner = fake_runner_factory({"agent-a": [result_event("r1", "agent-a", "out-a")]})
    loader = FakeAgentLoader({"agent-a": make_agent("agent-a")})

    record = await execute_pipeline(
        pipeline, make_run(), agent_loader=loader, resolver=FakeResolver(), output_root=tmp_path, runner=runner
    )

    assert record.status == "completed"
    gate_result = record.steps[-1]
    assert isinstance(gate_result, GateStepResult)
    assert gate_result.decision == "approved"
