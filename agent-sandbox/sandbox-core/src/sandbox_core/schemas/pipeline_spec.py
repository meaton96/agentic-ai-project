from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator


class PipelineStep(BaseModel):
    """One step in a PipelineSpec: run `agent_id` with a task rendered from
    `task_template`. `task_template` supports two placeholders — `{{task}}`
    (the pipeline run's seed task) and `{{steps.<step_id>.output}}` (a prior
    step's AgentResultEvent.final_output) — deliberately plain string
    substitution, not a templating engine; typed step output is future work."""

    kind: Literal["agent"] = "agent"
    step_id: str
    agent_id: str
    task_template: str


class GateStep(BaseModel):
    """A deterministic, non-LLM step: `gate` (an importable `module:function`
    path) is called with every completed step's output so far and returns a
    decision string, which `on_result` maps to the step_id to run next. The
    sentinel `"__end__"` in `on_result`'s values means "pipeline completes
    successfully here". This is agentic_ml's "agents propose, harness
    decides" pattern made generic — the gate function is operator-supplied,
    resolved by import path, and runs trusted, in-process Python only (no
    sandboxed/subprocess isolation; see docs/architecture.md's Known gaps)."""

    kind: Literal["gate"]
    step_id: str
    gate: str
    on_result: dict[str, str]


Step = Annotated[Union[PipelineStep, GateStep], Field(discriminator="kind")]


def default_step_kind_to_agent(steps):
    """Backfills "kind": "agent" onto step dicts that omit it. A
    discriminated union's tag must be present in the raw input — pydantic
    does not fall back to PipelineStep.kind's default during tag lookup,
    only afterwards — so any step dict omitting "kind" (every pre-existing
    pipeline YAML, and any API caller relying on the documented default)
    would otherwise fail validation before it ever reaches PipelineStep.
    Shared by PipelineSpec and AlterWorkflowOperation, the two places a bare
    list[Step] is validated from raw input. Already-constructed Step
    instances pass through untouched."""
    if not isinstance(steps, list):
        return steps
    return [
        {**step, "kind": step.get("kind", "agent")} if isinstance(step, dict) else step
        for step in steps
    ]


class PipelineSpec(BaseModel):
    """A deterministic, harness-driven sequence of agent/gate steps. Each
    agent step is an ordinary agent run (its own run_id/events.jsonl,
    unchanged); each gate step is a plain Python call. This spec only
    decides what order to run them in, how one step's output feeds the
    next, and — via gates — which step runs next. Not LLM-orchestrated
    (contrast AgentSpec.sub_agents/as_tool()): the sandbox itself walks
    `steps`, matching agentic_ml's "agents propose, harness decides" split."""

    id: str
    name: str
    steps: list[Step] = Field(default_factory=list)
    # Safety cap on total step executions per run — a gate that always routes
    # backward would otherwise loop forever. The pipeline-level analog of
    # AgentSpec.max_turns.
    max_steps: int = 50

    @field_validator("steps", mode="before")
    @classmethod
    def _default_step_kind_to_agent(cls, steps):
        return default_step_kind_to_agent(steps)

    @field_validator("steps")
    @classmethod
    def _at_least_one_step_with_unique_ids(cls, steps: list[Step]) -> list[Step]:
        if not steps:
            raise ValueError("a pipeline must have at least one step")
        step_ids = [step.step_id for step in steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError(f"step_id values must be unique within a pipeline, got: {step_ids}")
        return steps
