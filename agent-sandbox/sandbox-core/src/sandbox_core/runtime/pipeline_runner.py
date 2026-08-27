import asyncio
import importlib
import inspect
import json
from pathlib import Path
from typing import Awaitable, Callable

from pydantic import ValidationError

from sandbox_core.schemas.agent_spec import AgentSpecLoader
from sandbox_core.schemas.credentials import CredentialResolver
from sandbox_core.schemas.events import AgentResultEvent, ErrorEvent, Event
from sandbox_core.schemas.pipeline_run import GateStepResult, PipelineRunRecord, PipelineRunSpec, PipelineStepResult, StepResult
from sandbox_core.schemas.pipeline_spec import GateStep, PipelineSpec, PipelineStep, Step
from sandbox_core.schemas.run_spec import RunSpec

from .agent_loop import execute_run
from .event_log import EventLog

Runner = Callable[..., Awaitable[Event]]
GateFn = Callable[[dict[str, str]], str]


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


def _resolve_gate(path: str) -> GateFn:
    """Resolves a "module.path:function_name" gate reference to a callable.
    Not a plugin directory — a gate is an ordinary importable Python
    function, so whoever authors a pipeline that uses one must have that
    package installed in the sandbox's venv. Every failure mode here raises
    a ValueError naming the gate path and what's wrong, mirroring
    strands_adapter._mcp_client_for()'s _require() — a bare ImportError/
    AttributeError/KeyError gives no indication of which gate is broken."""
    module_path, _, func_name = path.rpartition(":")
    if not module_path or not func_name:
        raise ValueError(f"gate {path!r} must be of the form 'module.path:function_name'")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ValueError(f"gate {path!r}: could not import module {module_path!r}: {exc}") from exc
    fn = getattr(module, func_name, None)
    if fn is None:
        raise ValueError(f"gate {path!r}: {func_name!r} is not defined in module {module_path!r}")
    if not callable(fn):
        raise ValueError(f"gate {path!r}: {func_name!r} in module {module_path!r} is not callable")
    return fn


async def _run_gate(fn: GateFn, outputs: dict[str, str]) -> str:
    """Runs a resolved gate function, sync or async. Most real gates
    (agentic_ml.harness.*) are plain sync functions, so async is supported
    without forcing every gate to be one — mirrors the asyncio.to_thread
    pattern agent_loop.execute_run() already uses for building the Strands
    agent."""
    if inspect.iscoroutinefunction(fn):
        return await fn(outputs)
    return await asyncio.to_thread(fn, outputs)


def _next_step_id(steps: list[Step], current_id: str) -> str | None:
    """The step_id declared immediately after `current_id` in `steps`, or
    None if `current_id` is last. This is how a plain agent step advances —
    only a gate can jump elsewhere."""
    ids = [step.step_id for step in steps]
    idx = ids.index(current_id)
    return ids[idx + 1] if idx + 1 < len(ids) else None


async def execute_pipeline(
    pipeline: PipelineSpec,
    run: PipelineRunSpec,
    *,
    agent_loader: AgentSpecLoader,
    resolver: CredentialResolver,
    output_root: Path,
    runner: Runner = execute_run,
    on_step: Callable[[StepResult], None] | None = None,
) -> PipelineRunRecord:
    """Walks a PipelineSpec's steps starting from the first declared step_id,
    calling `runner` (normally agent_loop.execute_run) once per agent step
    and substituting each completed step's output into the next step's task.
    A gate step instead calls a resolved Python function with every
    completed step's output and jumps to whatever step_id its decision maps
    to via `on_result` (or ends the pipeline on the "__end__" sentinel) —
    this is what lets a pipeline express a reject/retry loop, which a
    strictly linear step list can't. `pipeline.max_steps` caps total step
    executions so a gate that always routes backward can't loop forever.
    Halts (returning a PipelineRunRecord with status="errored") at the
    first agent step that doesn't finish with AgentResultEvent, the first
    gate that fails to resolve/run or returns an unconfigured decision, or
    once max_steps is exceeded — in every case leaving later steps un-run."""
    outputs: dict[str, str] = {}
    step_results: list[StepResult] = []
    steps_by_id = {step.step_id: step for step in pipeline.steps}
    current_id: str | None = pipeline.steps[0].step_id
    executions = 0

    def halt(error: str) -> PipelineRunRecord:
        return PipelineRunRecord(
            pipeline_run_id=run.pipeline_run_id,
            pipeline_id=pipeline.id,
            task=run.task,
            created_at=run.created_at,
            status="errored",
            steps=step_results,
            error=error,
        )

    while current_id is not None:
        executions += 1
        if executions > pipeline.max_steps:
            return halt(f"pipeline exceeded max_steps ({pipeline.max_steps}) — likely an unterminated gate loop")

        step = steps_by_id[current_id]

        if isinstance(step, PipelineStep):
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
                return halt(f"step {step.step_id!r} did not complete (status={status})")

            current_id = _next_step_id(pipeline.steps, current_id)

        else:  # GateStep
            assert isinstance(step, GateStep)
            try:
                gate_fn = _resolve_gate(step.gate)
                decision = await _run_gate(gate_fn, outputs)
            except Exception as exc:
                return halt(f"gate {step.step_id!r} ({step.gate!r}) failed: {exc}")

            if decision not in step.on_result:
                return halt(
                    f"gate {step.step_id!r} returned decision {decision!r}, which is not configured in "
                    f"on_result (configured decisions: {sorted(step.on_result)})"
                )

            next_id = step.on_result[decision]
            routed_to = None if next_id == "__end__" else next_id
            if routed_to is not None and routed_to not in steps_by_id:
                return halt(
                    f"gate {step.step_id!r} routed decision {decision!r} to step_id {routed_to!r}, which "
                    f"does not exist in this pipeline (known step ids: {sorted(steps_by_id)})"
                )

            step_result = GateStepResult(step_id=step.step_id, decision=decision, routed_to=routed_to)
            step_results.append(step_result)
            if on_step is not None:
                on_step(step_result)

            current_id = routed_to

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
    """None means "nothing usable at this id" — covers a missing file, but
    also a file that's present but unreadable (corrupted JSON, or written
    under a schema this version no longer accepts — e.g. a pre-Phase-3
    record with no `kind` discriminator on its steps). Callers already
    treat None as "not found"; one such record must not be able to break
    listing every *other* pipeline run alongside it."""
    path = pipeline_run_path(output_root, pipeline_run_id)
    if not path.exists():
        return None
    try:
        return PipelineRunRecord.model_validate(json.loads(path.read_text()))
    except (json.JSONDecodeError, ValidationError):
        return None
