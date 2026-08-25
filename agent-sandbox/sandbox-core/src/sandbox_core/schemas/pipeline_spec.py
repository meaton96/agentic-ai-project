from pydantic import BaseModel, Field, field_validator


class PipelineStep(BaseModel):
    """One step in a PipelineSpec: run `agent_id` with a task rendered from
    `task_template`. `task_template` supports two placeholders — `{{task}}`
    (the pipeline run's seed task) and `{{steps.<step_id>.output}}` (a prior
    step's AgentResultEvent.final_output) — deliberately plain string
    substitution, not a templating engine; typed step output is future work."""

    step_id: str
    agent_id: str
    task_template: str


class PipelineSpec(BaseModel):
    """A deterministic, harness-driven sequence of agent steps. Each step is
    an ordinary agent run (its own run_id/events.jsonl, unchanged) — this
    spec only decides what order to run them in and how one step's output
    feeds the next step's task. Not LLM-orchestrated (contrast
    AgentSpec.sub_agents/as_tool()): the sandbox itself walks `steps` in
    order, matching agentic_ml's "agents propose, harness decides" split."""

    id: str
    name: str
    steps: list[PipelineStep] = Field(default_factory=list)

    @field_validator("steps")
    @classmethod
    def _at_least_one_step_with_unique_ids(cls, steps: list[PipelineStep]) -> list[PipelineStep]:
        if not steps:
            raise ValueError("a pipeline must have at least one step")
        step_ids = [step.step_id for step in steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError(f"step_id values must be unique within a pipeline, got: {step_ids}")
        return steps
