"""Tracks in-flight and historical pipeline runs. Mirrors run_manager.py's
shape (in-memory tracked runs + disk fallback, launched as an
asyncio.create_task on the server's own event loop) but delegates every step
to sandbox_core's own execute_run — this manager owns pipeline-level
bookkeeping only, no execution logic of its own, same division of
responsibility sandbox-server keeps everywhere else.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sandbox_core.runtime.agent_loop import execute_run
from sandbox_core.runtime.credential_store import YamlCredentialStore
from sandbox_core.runtime.pipeline_runner import Runner, execute_pipeline, read_pipeline_run_record, save_pipeline_run_record
from sandbox_core.schemas.agent_spec import AgentSpec
from sandbox_core.schemas.pipeline_run import PipelineRunRecord, PipelineRunSpec, StepResult
from sandbox_core.schemas.pipeline_spec import PipelineSpec

from .specs import read_agent_spec


class AgentNotFoundError(LookupError):
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        super().__init__(f"pipeline references agent {agent_id!r}, which does not exist")


class ServerAgentSpecLoader:
    """AgentSpecLoader over sandbox-server's own agent specs directory —
    raises (rather than returning None, like read_agent_spec does) since a
    pipeline step referencing a missing agent is a hard failure, not
    something the caller can shrug off."""

    def __init__(self, specs_dir: Path):
        self._specs_dir = specs_dir

    def load(self, agent_id: str) -> AgentSpec:
        agent = read_agent_spec(self._specs_dir, agent_id)
        if agent is None:
            raise AgentNotFoundError(agent_id)
        return agent


@dataclass
class TrackedPipelineRun:
    pipeline_run_id: str
    pipeline_id: str
    task: str
    created_at: datetime
    status: str = "running"
    steps: list[StepResult] = field(default_factory=list)
    error: str | None = None

    def to_record(self) -> PipelineRunRecord:
        return PipelineRunRecord(
            pipeline_run_id=self.pipeline_run_id,
            pipeline_id=self.pipeline_id,
            task=self.task,
            created_at=self.created_at,
            status=self.status,
            steps=self.steps,
            error=self.error,
        )


class PipelineRunManager:
    def __init__(self, *, specs_dir: Path, output_root: Path, pipeline_runs_root: Path, runner: Runner = execute_run):
        self.specs_dir = Path(specs_dir)
        self.output_root = Path(output_root)
        # Deliberately a directory RunManager never scans — see config.py's
        # DEFAULT_PIPELINE_RUNS_ROOT comment for why a pipeline manifest can't
        # live under output_root without corrupting the plain agent-runs list.
        self.pipeline_runs_root = Path(pipeline_runs_root)
        self._runner = runner
        self._tracked: dict[str, TrackedPipelineRun] = {}

    def launch_run(self, pipeline: PipelineSpec, task: str) -> str:
        run = PipelineRunSpec(pipeline_id=pipeline.id, task=task)
        tracked = TrackedPipelineRun(
            pipeline_run_id=run.pipeline_run_id,
            pipeline_id=pipeline.id,
            task=task,
            created_at=run.created_at,
        )
        self._tracked[run.pipeline_run_id] = tracked
        asyncio.create_task(self._execute(pipeline, run, tracked))
        return run.pipeline_run_id

    async def _execute(self, pipeline: PipelineSpec, run: PipelineRunSpec, tracked: TrackedPipelineRun) -> None:
        try:
            record = await execute_pipeline(
                pipeline,
                run,
                agent_loader=ServerAgentSpecLoader(self.specs_dir),
                resolver=YamlCredentialStore(),
                output_root=self.output_root,
                runner=self._runner,
                on_step=lambda result: self._on_step(tracked, result),
            )
        except Exception as exc:
            # A failure before any step ever ran (missing agent, bad
            # credential) — still needs to reach the UI as a terminal state.
            tracked.status = "errored"
            tracked.error = str(exc)
            save_pipeline_run_record(tracked.to_record(), self.pipeline_runs_root)
            return

        tracked.status = record.status
        tracked.error = record.error
        save_pipeline_run_record(record, self.pipeline_runs_root)

    def _on_step(self, tracked: TrackedPipelineRun, result: StepResult) -> None:
        tracked.steps.append(result)
        save_pipeline_run_record(tracked.to_record(), self.pipeline_runs_root)

    def list_pipeline_runs(self) -> list[dict]:
        run_ids = set(self._tracked)
        if self.pipeline_runs_root.exists():
            run_ids.update(p.name for p in self.pipeline_runs_root.iterdir() if p.is_dir())

        summaries = []
        for run_id in run_ids:
            summary = self._summarize(run_id)
            if summary is not None:
                summaries.append(summary)
        summaries.sort(key=lambda s: s["created_at"])
        return summaries

    def get_pipeline_run_detail(self, pipeline_run_id: str) -> dict | None:
        tracked = self._tracked.get(pipeline_run_id)
        if tracked is not None:
            return tracked.to_record().model_dump(mode="json")
        record = read_pipeline_run_record(self.pipeline_runs_root, pipeline_run_id)
        return record.model_dump(mode="json") if record is not None else None

    def _summarize(self, pipeline_run_id: str) -> dict | None:
        tracked = self._tracked.get(pipeline_run_id)
        if tracked is not None:
            return {
                "pipeline_run_id": tracked.pipeline_run_id,
                "pipeline_id": tracked.pipeline_id,
                "created_at": tracked.created_at,
                "status": tracked.status,
                "step_count": len(tracked.steps),
            }
        record = read_pipeline_run_record(self.pipeline_runs_root, pipeline_run_id)
        if record is None:
            return None
        return {
            "pipeline_run_id": record.pipeline_run_id,
            "pipeline_id": record.pipeline_id,
            "created_at": record.created_at,
            "status": record.status,
            "step_count": len(record.steps),
        }
