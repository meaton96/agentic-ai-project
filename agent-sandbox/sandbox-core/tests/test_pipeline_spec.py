import pytest
from pydantic import ValidationError

from sandbox_core.schemas.pipeline_spec import GateStep, PipelineSpec, PipelineStep


def test_pipeline_spec_roundtrip():
    pipeline = PipelineSpec(
        id="pipe-1",
        name="test pipeline",
        steps=[
            PipelineStep(step_id="a", agent_id="agent-a", task_template="{{task}}"),
            PipelineStep(step_id="b", agent_id="agent-b", task_template="{{steps.a.output}}"),
        ],
    )
    assert PipelineSpec.model_validate(pipeline.model_dump()) == pipeline


def test_pipeline_spec_rejects_zero_steps():
    with pytest.raises(ValidationError, match="at least one step"):
        PipelineSpec(id="pipe-1", name="empty", steps=[])


def test_pipeline_spec_rejects_duplicate_step_ids():
    with pytest.raises(ValidationError, match="unique"):
        PipelineSpec(
            id="pipe-1",
            name="dup",
            steps=[
                PipelineStep(step_id="a", agent_id="agent-a", task_template="x"),
                PipelineStep(step_id="a", agent_id="agent-b", task_template="y"),
            ],
        )


def test_pipeline_step_kind_defaults_to_agent_so_existing_yaml_stays_valid():
    step = PipelineStep(step_id="a", agent_id="agent-a", task_template="x")
    assert step.kind == "agent"


def test_pipeline_spec_max_steps_defaults_to_50():
    pipeline = PipelineSpec(id="pipe-1", name="x", steps=[PipelineStep(step_id="a", agent_id="a", task_template="x")])
    assert pipeline.max_steps == 50


def test_pipeline_spec_roundtrip_with_a_gate_step():
    pipeline = PipelineSpec(
        id="pipe-1",
        name="with gate",
        steps=[
            PipelineStep(step_id="a", agent_id="agent-a", task_template="{{task}}"),
            GateStep(
                kind="gate",
                step_id="check",
                gate="mypkg.gates:decide",
                on_result={"approved": "__end__", "rejected": "a"},
            ),
        ],
    )
    reloaded = PipelineSpec.model_validate(pipeline.model_dump())
    assert reloaded == pipeline
    assert isinstance(reloaded.steps[1], GateStep)


def test_gate_step_requires_kind_gate_explicitly():
    with pytest.raises(ValidationError):
        GateStep(step_id="check", gate="mypkg.gates:decide", on_result={})
