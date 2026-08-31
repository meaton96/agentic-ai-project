import pytest

from sandbox_core.schemas.agent_spec import AgentSpec, ModelConfig
from sandbox_core.schemas.operations import (
    AlterModelOperation,
    AlterWorkflowOperation,
    OperationError,
    SwapAgentOperation,
    apply_operation,
)
from sandbox_core.schemas.pipeline_spec import GateStep, PipelineSpec, PipelineStep


def make_pipeline(**overrides) -> PipelineSpec:
    fields = dict(
        id="pipe-1",
        name="test pipeline",
        steps=[
            PipelineStep(step_id="a", agent_id="agent-a", task_template="{{task}}"),
            PipelineStep(step_id="b", agent_id="agent-b", task_template="{{steps.a.output}}"),
        ],
    )
    fields.update(overrides)
    return PipelineSpec(**fields)


def make_agent(**overrides) -> AgentSpec:
    fields = dict(
        id="agent-a",
        name="Agent A",
        system_prompt="be helpful",
        model=ModelConfig(base_url="http://x", model_name="m", api_key_ref="k"),
    )
    fields.update(overrides)
    return AgentSpec(**fields)


class FakeAgentLoader:
    def __init__(self, known_ids: set[str]):
        self._known_ids = known_ids

    def load(self, agent_id: str) -> AgentSpec:
        if agent_id not in self._known_ids:
            raise LookupError(f"agent {agent_id!r} not found")
        return make_agent(id=agent_id)


def test_swap_agent_repoints_the_named_step_only():
    pipeline = make_pipeline()

    result = apply_operation(pipeline, SwapAgentOperation(step_id="a", new_agent_id="agent-c"))

    assert isinstance(result, PipelineSpec)
    assert result.steps[0].agent_id == "agent-c"
    assert result.steps[1].agent_id == "agent-b"
    # task_template and step_id are untouched
    assert result.steps[0].task_template == "{{task}}"
    assert result.steps[0].step_id == "a"


def test_swap_agent_validates_new_agent_id_against_loader():
    pipeline = make_pipeline()
    loader = FakeAgentLoader(known_ids={"agent-a", "agent-b"})

    with pytest.raises(OperationError, match="does not resolve"):
        apply_operation(pipeline, SwapAgentOperation(step_id="a", new_agent_id="ghost"), agent_loader=loader)


def test_swap_agent_skips_loader_check_when_none_given():
    pipeline = make_pipeline()
    result = apply_operation(pipeline, SwapAgentOperation(step_id="a", new_agent_id="unregistered"))
    assert result.steps[0].agent_id == "unregistered"


def test_swap_agent_unknown_step_id_raises():
    pipeline = make_pipeline()
    with pytest.raises(OperationError, match="no step"):
        apply_operation(pipeline, SwapAgentOperation(step_id="does-not-exist", new_agent_id="agent-c"))


def test_swap_agent_on_gate_step_raises():
    pipeline = make_pipeline(
        steps=[
            PipelineStep(step_id="a", agent_id="agent-a", task_template="{{task}}"),
            GateStep(kind="gate", step_id="g", gate="mod:fn", on_result={"pass": "__end__"}),
        ]
    )
    with pytest.raises(OperationError, match="not an agent step"):
        apply_operation(pipeline, SwapAgentOperation(step_id="g", new_agent_id="agent-c"))


def test_swap_agent_against_agent_spec_raises():
    agent = make_agent()
    with pytest.raises(OperationError, match="PipelineSpec"):
        apply_operation(agent, SwapAgentOperation(step_id="a", new_agent_id="agent-c"))


def test_alter_model_replaces_model_config():
    agent = make_agent()
    new_model = ModelConfig(base_url="http://y", model_name="m2", api_key_ref="k2", temperature=0.7)

    result = apply_operation(agent, AlterModelOperation(model=new_model))

    assert isinstance(result, AgentSpec)
    assert result.model == new_model
    assert result.id == agent.id  # rest of the spec untouched


def test_alter_model_against_pipeline_spec_raises():
    pipeline = make_pipeline()
    new_model = ModelConfig(base_url="http://y", model_name="m2", api_key_ref="k2")
    with pytest.raises(OperationError, match="AgentSpec"):
        apply_operation(pipeline, AlterModelOperation(model=new_model))


def test_alter_workflow_replaces_step_list():
    pipeline = make_pipeline()
    new_steps = [PipelineStep(step_id="only", agent_id="agent-z", task_template="{{task}}")]

    result = apply_operation(pipeline, AlterWorkflowOperation(steps=new_steps))

    assert [s.step_id for s in result.steps] == ["only"]


def test_alter_workflow_can_update_max_steps():
    pipeline = make_pipeline()
    new_steps = [PipelineStep(step_id="only", agent_id="agent-z", task_template="{{task}}")]

    result = apply_operation(pipeline, AlterWorkflowOperation(steps=new_steps, max_steps=5))

    assert result.max_steps == 5


def test_alter_workflow_still_enforces_pipeline_spec_validators():
    """_revalidate() must run PipelineSpec's own validators, not just splice
    fields in — an empty or duplicate-id step list should fail the same way
    constructing PipelineSpec directly would."""
    pipeline = make_pipeline()
    with pytest.raises(OperationError):
        apply_operation(pipeline, AlterWorkflowOperation(steps=[]))
