from datetime import datetime, timezone
from typing import Annotated, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, Field

PipelineStepStatus = Literal["completed", "errored", "truncated"]
PipelineRunStatus = Literal["running", "completed", "errored"]


class PipelineRunSpec(BaseModel):
    """Mirrors RunSpec, one level up: the seed request for one pipeline
    execution, before any of its steps have run."""

    pipeline_run_id: str = Field(default_factory=lambda: str(uuid4()))
    pipeline_id: str
    task: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PipelineStepResult(BaseModel):
    """The outcome of one already-finished agent pipeline step. `run_id` is
    the underlying agent run's own run_id — its full events.jsonl/trace is
    reachable through the exact same endpoints a standalone agent run is."""

    kind: Literal["agent"] = "agent"
    step_id: str
    agent_id: str
    run_id: str
    status: PipelineStepStatus
    output: str | None = None


class GateStepResult(BaseModel):
    """The outcome of one already-finished gate step: no agent run backs it,
    just a decision and where it routed to. `routed_to` is None when the
    gate's decision resolved to the "__end__" sentinel.

    `output`, like PipelineStepResult.output, feeds `{{steps.<step_id>.output}}`
    for later steps — a gate is deterministic Python, so this is deliberately
    *not* restricted to prose: a gate wrapping a real data-pipeline stage
    (e.g. a DataFrame in, DataFrame out transform) has nothing sensible to
    put in a string except a reference to where it wrote the real artifact
    (a parquet/joblib file path) — the sandbox never holds that payload
    itself, same as it never holds an agent's full conversation, only its
    final_output text."""

    kind: Literal["gate"] = "gate"
    step_id: str
    decision: str
    routed_to: str | None
    output: str | None = None


StepResult = Annotated[Union[PipelineStepResult, GateStepResult], Field(discriminator="kind")]


class PipelineRunRecord(BaseModel):
    """The pipeline-level trace: an ordered list of step outcomes plus
    overall status. Persisted as a single JSON file (not a JSONL event log —
    there's no live per-event fan-out at this level; each step's own run
    already has one) so a pipeline run survives a server restart the same
    way an agent run's events.jsonl does."""

    pipeline_run_id: str
    pipeline_id: str
    task: str
    created_at: datetime
    status: PipelineRunStatus
    steps: list[StepResult] = Field(default_factory=list)
    error: str | None = None
