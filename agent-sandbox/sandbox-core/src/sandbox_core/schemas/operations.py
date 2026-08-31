"""Operations: explicit, logged mutations of a draft AgentSpec/PipelineSpec,
replacing the raw write_agent_spec()/write_pipeline_spec() overwrites that
sandbox-server's routes used to call directly. See
agent-sandbox/docs/future-work-roadmap.md Phase 1.

Pre-run only for now: every Operation here targets a spec sitting in a specs
directory, never a live run's in-memory state (that's Phase 3's
SPAWN_SUBAGENT/mid-run-mutation territory, and is deliberately not modeled
here yet).

apply_operation() is pure — it takes a spec and an Operation and returns a
new, fully re-validated spec. It does no file I/O and knows nothing about
directories or YAML; callers (sandbox-server's operations.py, sandbox_core's
own CLI if it grows one) are responsible for loading the current spec,
calling this, logging the result via an OperationLog, and persisting the
returned spec.
"""

from datetime import datetime
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, ValidationError, field_validator

from .agent_spec import AgentSpec, AgentSpecLoader, ModelConfig
from .pipeline_spec import PipelineSpec, PipelineStep, Step, default_step_kind_to_agent


class SwapAgentOperation(BaseModel):
    """Repoints one pipeline step at a different agent, leaving everything
    else about the step (its task_template, its position) unchanged."""

    type: Literal["swap_agent"] = "swap_agent"
    step_id: str
    new_agent_id: str


class AlterModelOperation(BaseModel):
    """Replaces an AgentSpec's model configuration wholesale."""

    type: Literal["alter_model"] = "alter_model"
    model: ModelConfig


class AlterWorkflowOperation(BaseModel):
    """Replaces a PipelineSpec's step list (and optionally its max_steps)
    wholesale — the general-purpose edit behind add/remove/rewire-node UIs
    (Phase 6), of which SwapAgentOperation is a common, narrower special
    case."""

    type: Literal["alter_workflow"] = "alter_workflow"
    steps: list[Step]
    max_steps: int | None = None

    @field_validator("steps", mode="before")
    @classmethod
    def _default_step_kind_to_agent(cls, steps):
        return default_step_kind_to_agent(steps)


Operation = Annotated[
    Union[SwapAgentOperation, AlterModelOperation, AlterWorkflowOperation],
    Field(discriminator="type"),
]


class OperationRecord(BaseModel):
    """One applied operation's forensic record: what was requested, against
    which target, and the before/after spec state it produced. Written by an
    OperationLog and never mutated after append — the append-only trail the
    roadmap's Phase 1 wants "almost for free" from the existing EventLog
    pattern."""

    seq: int = 0
    ts: datetime
    target_type: Literal["agent", "pipeline"]
    target_id: str
    actor: str = "user"
    operation: Operation
    before: dict
    after: dict
    diff: str


class OperationError(ValueError):
    """An Operation that doesn't apply to the given spec — wrong target
    type, a step_id that doesn't exist, a new_agent_id that doesn't resolve.
    Callers (e.g. sandbox-server's routes) should surface this as a 400, not
    a 500."""


def _revalidate(spec: AgentSpec | PipelineSpec, updates: dict) -> AgentSpec | PipelineSpec:
    """Merges `updates` into spec's dumped state and re-validates the whole
    thing through the model's own validators (field validators included) —
    unlike model_copy(update=...), which skips them. This is what makes an
    Operation "validate against the current spec" rather than just splice
    new data in. A resulting ValidationError (e.g. an AlterWorkflowOperation
    that empties out a pipeline's steps) is surfaced as an OperationError,
    the one exception type callers need to treat as "bad request"."""
    data = spec.model_dump(mode="json")
    data.update(updates)
    try:
        return type(spec).model_validate(data)
    except ValidationError as exc:
        raise OperationError(f"operation produces an invalid {type(spec).__name__}: {exc}") from exc


def apply_operation(
    spec: AgentSpec | PipelineSpec,
    operation: Operation,
    *,
    agent_loader: AgentSpecLoader | None = None,
) -> AgentSpec | PipelineSpec:
    """Returns a new, validated spec with `operation` applied. Raises
    OperationError if the operation doesn't apply to `spec` (wrong spec
    type, missing step_id, an unresolvable new_agent_id when `agent_loader`
    is supplied). `agent_loader` is optional so pure schema-level callers
    (e.g. tests) don't need a full spec store just to apply an operation."""
    if isinstance(operation, SwapAgentOperation):
        if not isinstance(spec, PipelineSpec):
            raise OperationError(f"swap_agent targets a PipelineSpec, got {type(spec).__name__}")
        if agent_loader is not None:
            try:
                agent_loader.load(operation.new_agent_id)
            except Exception as exc:
                raise OperationError(
                    f"new_agent_id {operation.new_agent_id!r} does not resolve: {exc}"
                ) from exc
        steps = list(spec.steps)
        for index, step in enumerate(steps):
            if step.step_id != operation.step_id:
                continue
            if not isinstance(step, PipelineStep):
                raise OperationError(
                    f"step {operation.step_id!r} is a {step.kind!r} step, not an agent step"
                )
            steps[index] = step.model_copy(update={"agent_id": operation.new_agent_id})
            break
        else:
            raise OperationError(f"pipeline {spec.id!r} has no step {operation.step_id!r}")
        return _revalidate(spec, {"steps": [s.model_dump(mode="json") for s in steps]})

    if isinstance(operation, AlterModelOperation):
        if not isinstance(spec, AgentSpec):
            raise OperationError(f"alter_model targets an AgentSpec, got {type(spec).__name__}")
        return _revalidate(spec, {"model": operation.model.model_dump(mode="json")})

    if isinstance(operation, AlterWorkflowOperation):
        if not isinstance(spec, PipelineSpec):
            raise OperationError(f"alter_workflow targets a PipelineSpec, got {type(spec).__name__}")
        updates = {"steps": [s.model_dump(mode="json") for s in operation.steps]}
        if operation.max_steps is not None:
            updates["max_steps"] = operation.max_steps
        return _revalidate(spec, updates)

    raise OperationError(f"unsupported operation: {operation!r}")
