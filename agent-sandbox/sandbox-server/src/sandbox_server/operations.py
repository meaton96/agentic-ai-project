"""Applies an Operation to a stored AgentSpec/PipelineSpec, logging the
before/after/diff to that target's OperationLog before persisting the new
spec — the formalized replacement for routes/agents.py and
routes/pipelines.py calling write_agent_spec()/write_pipeline_spec()
directly. Pre-run only: an Operation here targets a draft spec on disk,
never a live run (see agent-sandbox/docs/future-work-roadmap.md Phase 1).

The log is written before the spec file, not after: if the process dies
between the two writes, the operation log stays the authoritative record of
intent (and can be replayed), whereas the reverse order would let a spec
change land on disk with no audit trail at all if the log write then failed.
"""

from pathlib import Path

from sandbox_core.runtime.operation_log import OperationLog
from sandbox_core.schemas.agent_spec import AgentSpec
from sandbox_core.schemas.operations import Operation, OperationRecord, apply_operation
from sandbox_core.schemas.pipeline_spec import PipelineSpec

from .pipeline_run_manager import ServerAgentSpecLoader
from .pipeline_specs import read_pipeline_spec, write_pipeline_spec
from .specs import read_agent_spec, write_agent_spec


def apply_agent_operation(
    *,
    specs_dir: Path,
    operations_root: Path,
    agent_id: str,
    operation: Operation,
    actor: str = "user",
) -> tuple[AgentSpec, OperationRecord]:
    """Loads agent `agent_id`, applies `operation`, logs it, writes the
    result. Raises LookupError if the agent doesn't exist and
    sandbox_core.schemas.operations.OperationError if the operation doesn't
    apply (e.g. targets a pipeline-only operation type)."""
    spec = read_agent_spec(specs_dir, agent_id)
    if spec is None:
        raise LookupError(f"agent {agent_id!r} not found")

    new_spec = apply_operation(spec, operation)

    log = OperationLog("agent", agent_id, root=operations_root)
    record = log.append(
        operation=operation,
        actor=actor,
        before=spec.model_dump(mode="json"),
        after=new_spec.model_dump(mode="json"),
    )
    write_agent_spec(specs_dir, new_spec)
    return new_spec, record


def apply_pipeline_operation(
    *,
    pipelines_dir: Path,
    specs_dir: Path,
    operations_root: Path,
    pipeline_id: str,
    operation: Operation,
    actor: str = "user",
) -> tuple[PipelineSpec, OperationRecord]:
    """Loads pipeline `pipeline_id`, applies `operation`, logs it, writes the
    result. `specs_dir` is only used to validate a swap_agent's
    new_agent_id against the real agent store; see ServerAgentSpecLoader."""
    spec = read_pipeline_spec(pipelines_dir, pipeline_id)
    if spec is None:
        raise LookupError(f"pipeline {pipeline_id!r} not found")

    new_spec = apply_operation(spec, operation, agent_loader=ServerAgentSpecLoader(specs_dir))

    log = OperationLog("pipeline", pipeline_id, root=operations_root)
    record = log.append(
        operation=operation,
        actor=actor,
        before=spec.model_dump(mode="json"),
        after=new_spec.model_dump(mode="json"),
    )
    write_pipeline_spec(pipelines_dir, new_spec)
    return new_spec, record
