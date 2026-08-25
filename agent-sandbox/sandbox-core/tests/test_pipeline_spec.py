import pytest
from pydantic import ValidationError

from sandbox_core.schemas.pipeline_spec import PipelineSpec, PipelineStep


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
