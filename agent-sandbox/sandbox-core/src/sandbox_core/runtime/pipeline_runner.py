import json
from pathlib import Path
from typing import Awaitable, Callable

from sandbox_core.schemas.agent_spec import AgentSpecLoader
from sandbox_core.schemas.credentials import CredentialResolver
from sandbox_core.schemas.events import AgentResultEvent, ErrorEvent, Event
from sandbox_core.schemas.pipeline_run import PipelineRunRecord, PipelineRunSpec, PipelineStepResult
from sandbox_core.schemas.pipeline_spec import PipelineSpec
from sandbox_core.schemas.run_spec import RunSpec

from .agent_loop import execute_run
from .event_log import EventLog

Runner = Callable[..., Awaitable[Event]]


def _render_task(template: str, *, seed_task: str, outputs: dict[str, str]) -> str:
    """Substitutes `{{task}}` (the pipeline's seed task) and
    `{{steps.<step_id>.output}}` (a prior step's final_output) into a step's
    task_template. Plain string replacement, not a templating engine —
    typed step output (Phase 4) is what eventually replaces this."""
    rendered = template.replace("{{task}}", seed_task)
    for step_id, output in outputs.items():
        rendered = rendered.replace(f"{{{{steps.{step_id}.output}}}}", output)
    return rendered


def _status_for(result: Event) -> tuple[str, str | None]:
    """(status, output) for one step's terminal Event."""
    if isinstance(result, AgentResultEvent):
        return "completed", result.final_output
    if isinstance(result, ErrorEvent):
        phase = (result.context or {}).get("phase")
        return ("truncated" if phase == "max_turns" else "errored"), None
    return "errored", None


async def execute_pipeline(
    pipeline: PipelineSpec,
    run: PipelineRunSpec,
    *,
    agent_loader: AgentSpecLoader,
    resolver: CredentialResolver,
    output_root: Path,
    runner: Runner = execute_run,
    on_step: Callable[[PipelineStepResult], None] | None = None,
) -> PipelineRunRecord:
    """Walks a PipelineSpec's steps in order, calling `runner` (normally
    agent_loop.execute_run) once per step and substituting each completed
    step's output into the next step's task. Each step is a completely
    ordinary agent run with its own run_id/events.jsonl — this function is a
    thin coordinator over existing runs, not a new execution engine. Halts at
    the first step that doesn't finish with AgentResultEvent, leaving later
    steps un-run."""
    outputs: dict[str, str] = {}
    step_results: list[PipelineStepResult] = []

    for step in pipeline.steps:
        agent = agent_loader.load(step.agent_id)
        task = _render_task(step.task_template, seed_task=run.task, outputs=outputs)
        step_run = RunSpec(agent_id=agent.id, task=task)
        event_log = EventLog(step_run.run_id, output_root=output_root)

        result = await runner(agent, step_run, resolver=resolver, event_log=event_log)
        status, output = _status_for(result)
        if output is not None:
            outputs[step.step_id] = output

        step_result = PipelineStepResult(
            step_id=step.step_id, agent_id=agent.id, run_id=step_run.run_id, status=status, output=output
        )
        step_results.append(step_result)
        if on_step is not None:
            on_step(step_result)

        if status != "completed":
            return PipelineRunRecord(
                pipeline_run_id=run.pipeline_run_id,
                pipeline_id=pipeline.id,
                task=run.task,
                created_at=run.created_at,
                status="errored",
                steps=step_results,
                error=f"step {step.step_id!r} did not complete (status={status})",
            )

    return PipelineRunRecord(
        pipeline_run_id=run.pipeline_run_id,
        pipeline_id=pipeline.id,
        task=run.task,
        created_at=run.created_at,
        status="completed",
        steps=step_results,
    )


def pipeline_run_path(output_root: Path, pipeline_run_id: str) -> Path:
    return Path(output_root) / pipeline_run_id / "pipeline.json"


def save_pipeline_run_record(record: PipelineRunRecord, output_root: Path) -> None:
    path = pipeline_run_path(output_root, record.pipeline_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2))


def read_pipeline_run_record(output_root: Path, pipeline_run_id: str) -> PipelineRunRecord | None:
    path = pipeline_run_path(output_root, pipeline_run_id)
    if not path.exists():
        return None
    return PipelineRunRecord.model_validate(json.loads(path.read_text()))
