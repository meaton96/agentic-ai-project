"""agent-sandbox GateStep functions for resource_scheduler's six agents,
each resolvable as a GateStep.gate path
("resource_scheduler.gate_adapters:prepare_load_monitor" etc.). Every
function here is purely deterministic: no ModelClient, no
ToolCallingAgent, no LLM call of any kind. Direct port of the pattern
proven out in the sibling agentic-ml-classification project's
src/agentic_ml/gate_adapters.py -- read that module's docstring first;
this one only calls out where resource_scheduler's shape genuinely
differs (the mailbox, below).

Shape: each stage that needs a judgment call is split into a pair of
gates with an agent step, owned entirely outside resource_scheduler, in
between:

    prepare_<stage>  ->  (external agent proposes)  ->  <stage>_decide

prepare_<stage> computes the stage's harness facts with the same calls
tools/*_tool.py's build_*_fact functions already make, persists them via
mcp_facts.fact_store.write_fact so mcp_facts/server.py can serve them,
and outputs only the bare run_id. It never waits for or produces a
proposal -- except where a stage's whole point is conditional on a
mailbox message existing (Resource Allocation, Failure Recovery, both
Human Oversight paths), in which case prepare_* itself may short-circuit
with a terminal decision (no LLM call to skip *to* if the precondition
isn't met) rather than always returning "ready" the way every agentic_ml
prepare_* gate does.

<stage>_decide re-reads that stage's prepare_* manifest by run_id, takes
the propose_* agent step's raw output (held to exactly the JSON contract
the stage's own steps/*_step.py function already expects) and runs the
stage's post-proposal logic -- calling straight into environment/*.py's
existing pure validate_*/check_constraints/parse_oversight_verdict
functions, so a proposal is judged identically however it was produced.
Unlike agentic_ml's pre-refactor steps/*_step.py, resource_scheduler's
post-proposal logic was never tangled up with its LLM call in the first
place -- environment/*.py's validate_* functions are already pure and
already the single source of truth steps/*_step.py itself calls, so
there was no extraction work to do here (see spec/sandbox-port-spec.md
§0). steps/*_step.py, tools/*.py, agent_runtime.py, and every
scripts/run_*_agent.py stay exactly as they are, unmodified, for
standalone CLI use outside the sandbox.

Data flow: every gate writes a JSON "manifest" (paths + small JSON-safe
fields) under run_dir, named "<step_id>_manifest.json" (the step's own
step_id, whether prepare_* or *_decide). A prepare_* gate's output is the
bare run_id (an agent step substitutes a prior output into its prompt
verbatim, can't read this filesystem, and must never see raw paths); a
*_decide gate's output is its own manifest's path (only ever consumed by
another gate, which can read a path fine). prepare_* manifests are the
upstream manifest plus one "prepare" key (stage name + which MCP tools
the proposer should read, audit-only -- the agent's actual tool choice
comes from its own AgentSpec/prompt); decide gates strip it before
writing their own.

`run_dir` is created once, by prepare_load_monitor (or, for a pipeline
whose seed task is itself another run's run_id -- prepare_failure_recovery,
and prepare_optimization when run as its own dedicated pipeline --
freshly by that prepare_* gate), via cli_common.make_run_dir(). Later
stages read it from the manifest chain. Set RESOURCE_SCHEDULER_DATA_ROOT
in the repo-root .env (same mechanism as agentic_ml's own
AGENTIC_ML_DATA_ROOT) so run_dir lands under resource-scheduler/, not
agent-sandbox/runs/.

Every gate builds one on_event (via make_event_emitter/make_event_logger)
appending to that run's own events.jsonl, so tool_called-shaped events
(here: facts_ready, a2a_sent, and each stage's own *_report event) are
recorded exactly as the step functions' equivalents already emit them.

THE ONE REAL STRUCTURAL DIFFERENCE FROM agentic_ml: resource_scheduler's
steps/*_step.py additionally depend on a live, in-process Mailbox shared
across an entire orchestrator run (a2a/mailbox.py) -- agent-to-agent
messages that don't fit agentic_ml's "everything flows through
outputs/manifests" shape. In the sandbox, every gate call is a separate
process, so an in-memory Mailbox can't cross that boundary. Fix:
a2a.mailbox.PersistentMailbox, a disk-backed mailbox with the identical
send/inbox_for/peek API, bound to a Tier-1 EnvironmentSpec session
container via each mailbox-touching GateStep's `environment_id` (see
agent-sandbox/docs/persistent-environment-spec.md and this repo's own
spec/sandbox-port-spec.md §1) -- a session container gives that
container's own filesystem real per-pipeline-run continuity across
otherwise-fresh-process gate calls, which PersistentMailbox's on-disk
queues rely on. Every *_decide gate that sends a message, and every
prepare_* gate that reads one, constructs `PersistentMailbox(run_id,
on_event=on_event)`; every other gate here never touches it. Pipelines
wiring these gates must set the matching GateStep.environment_id on each
one -- see pipelines/resource-scheduler-*.yaml in agent-sandbox.

v1 scope: the continuous optimization loop (scripts/run_optimization_loop.py)
stays outside the sandbox -- a cross-run hill-climbing search that itself
re-invokes two other agents many times per iteration doesn't fit
PipelineSpec's single-run shape. Failure Recovery is its own separate
pipeline (seed task: an existing resource-scheduler-main run's run_id),
not chained automatically off that run, since it's meaningfully a
different triggering event (an incident, not "the next step of
scheduling").
"""

import json
from pathlib import Path
from typing import Optional

from resource_scheduler.a2a.mailbox import PersistentMailbox
from resource_scheduler.cli_common import make_run_dir
from resource_scheduler.environment.allocation import (
    DEFAULT_SLICE_CAPACITY,
    build_allocation_events,
    check_constraints,
    compute_slice_load,
    identify_risky_assignments,
    validate_allocation_structure,
)
from resource_scheduler.environment.incidents import reroutes_avoid_source_machine, validate_recovery_proposal
from resource_scheduler.environment.oversight import (
    build_decision_review_bundle,
    build_oversight_review_bundle,
    parse_oversight_verdict,
)
from resource_scheduler.environment.policy_evidence import validate_policy_proposal
from resource_scheduler.environment.queue import is_ranking_score_consistent, validate_ranking_proposal
from resource_scheduler.environment.state import compute_snapshot, load_task_table
from resource_scheduler.events import emit_event, make_event_emitter, make_event_logger
from resource_scheduler.mcp_facts.fact_store import read_fact, write_fact
from resource_scheduler.mcp_facts.server import FACT_TOOL_NAMES
from resource_scheduler.paths import run_dir as resolve_run_dir
from resource_scheduler.tools.failure_recovery_tool import build_incident_fact
from resource_scheduler.tools.load_monitor_tool import build_resource_snapshot_fact
from resource_scheduler.tools.optimization_tool import build_policy_evidence_fact
from resource_scheduler.tools.resource_allocation_tool import build_allocation_context_fact
from resource_scheduler.tools.task_prioritization_tool import build_task_queue_fact

STEP_PREPARE_LOAD_MONITOR = "prepare_load_monitor"
STEP_PROPOSE_LOAD_MONITOR = "propose_load_monitor"
STEP_LOAD_MONITOR_DECIDE = "load_monitor_decide"

STEP_PREPARE_TASK_PRIORITIZATION = "prepare_task_prioritization"
STEP_PROPOSE_TASK_PRIORITIZATION = "propose_task_prioritization"
STEP_TASK_PRIORITIZATION_DECIDE = "task_prioritization_decide"

STEP_PREPARE_RESOURCE_ALLOCATION = "prepare_resource_allocation"
STEP_PROPOSE_RESOURCE_ALLOCATION = "propose_resource_allocation"
STEP_RESOURCE_ALLOCATION_DECIDE = "resource_allocation_decide"

STEP_REROUTE_VALIDATION_DECIDE = "reroute_validation_decide"

STEP_PREPARE_FAILURE_RECOVERY = "prepare_failure_recovery"
STEP_PROPOSE_FAILURE_RECOVERY = "propose_failure_recovery"
STEP_FAILURE_RECOVERY_DECIDE = "failure_recovery_decide"

STEP_PREPARE_OPTIMIZATION = "prepare_optimization"
STEP_PROPOSE_OPTIMIZATION = "propose_optimization"
STEP_OPTIMIZATION_DECIDE = "optimization_decide"

STEP_PREPARE_OVERSIGHT = "prepare_oversight"
STEP_PROPOSE_OVERSIGHT = "propose_oversight"
STEP_OVERSIGHT_DECIDE = "oversight_decide"

STEP_PREPARE_DECISION_OVERSIGHT = "prepare_decision_oversight"
STEP_PROPOSE_DECISION_OVERSIGHT = "propose_decision_oversight"
STEP_DECISION_OVERSIGHT_DECIDE = "decision_oversight_decide"

_DEFAULT_QUEUE_SIZE = 8
_DEFAULT_SNAPSHOT_WINDOW = 200
_DEFAULT_SLICE_CAPACITY = DEFAULT_SLICE_CAPACITY
_DEFAULT_N_RUNS = 10
_DEFAULT_BEFORE_OFFSET = 50  # mirrors run_orchestrator.py's before_row default (len(df) - 50)

_PREPARE_KEY = "prepare"


def _write_manifest(path: Path, data: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    return str(path)


def _read_manifest(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _without(manifest: dict, key: str) -> dict:
    return {k: v for k, v in manifest.items() if k != key}


def _manifest_path(run_dir: Path, step_id: str) -> Path:
    return run_dir / f"{step_id}_manifest.json"


def _parse_proposal(raw: object) -> Optional[dict]:
    if not isinstance(raw, str):
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _publish_fact(run_id: str, on_event, stage: str, fact_name: str, payload: dict) -> dict:
    """Writes one fact for mcp_facts/server.py to serve and returns the
    manifest's "prepare" block -- audit-only (which stage now awaits a
    proposal, and the one MCP tool its proposer should read); the
    proposer's actual tool choice comes from its own AgentSpec/prompt,
    not from this block."""
    write_fact(run_id, fact_name, payload)
    mcp_tools = [FACT_TOOL_NAMES[fact_name]]
    emit_event(on_event, stage, "facts_ready", {"mcp_tools": mcp_tools})
    return {"stage": stage, "mcp_tools": mcp_tools}


def _write_legacy_evidence_report(run_dir: Path, name: str, payload: dict) -> None:
    """environment/policy_evidence.py::collect_policy_evidence reads
    task_prioritization_report.json / resource_allocation_report.json /
    failure_recovery_report.json by exact filename -- the shape
    scripts/run_orchestrator.py's CLI path already wrote, unrelated to
    and unaware of gate_adapters.py's own "<step_id>_manifest.json"
    convention. Writing this small subset alongside the real manifest
    (never in place of it -- this is purely additive) is what keeps
    Optimization's cross-run evidence aggregation working for
    sandbox-driven runs too, without changing environment/policy_evidence.py's
    file-reading contract or the CLI path's own report shape at all."""
    (run_dir / name).write_text(json.dumps(payload, default=str))


def _flags_match(reported: object, authoritative: list[dict]) -> bool:
    """Order-independent comparison on (scope, id, severity) triples.
    Mirrors steps/load_monitor_step.py's private _flags_match exactly
    (kept in sync by hand, same as agent-sandbox's own entrypoint.py
    mirrors sandbox_core helpers by hand per its own docstring) --
    gate_adapters.py deliberately never imports from steps/*.py, since
    those modules own the LLM-calling standalone-CLI path this port
    leaves untouched."""
    if not isinstance(reported, list):
        return False
    try:
        reported_keys = {(f["scope"], f["id"], f["severity"]) for f in reported}
    except (TypeError, KeyError):
        return False
    authoritative_keys = {(f["scope"], f["id"], f["severity"]) for f in authoritative}
    return reported_keys == authoritative_keys


# -- 1. Load Monitor -------------------------------------------------


def prepare_load_monitor(outputs: dict[str, str]) -> tuple[str, str]:
    csv_path = outputs["__task__"]
    run_id, run_dir = make_run_dir(None)
    on_event = make_event_emitter(run_id, persist_fn=make_event_logger(run_dir))

    df, variance_injected = load_task_table(csv_path)
    fact = build_resource_snapshot_fact(df, variance_injected, window=_DEFAULT_SNAPSHOT_WINDOW)
    prepare = _publish_fact(run_id, on_event, "load_monitor", "resource_snapshot", fact)

    manifest = {"run_id": run_id, "run_dir": str(run_dir), "csv_path": csv_path, _PREPARE_KEY: prepare}
    _write_manifest(_manifest_path(run_dir, STEP_PREPARE_LOAD_MONITOR), manifest)
    return "reported", run_id


def load_monitor_decide(outputs: dict[str, str]) -> tuple[str, str]:
    prep = _without(_read_manifest(
        str(resolve_run_dir(outputs[STEP_PREPARE_LOAD_MONITOR]) / f"{STEP_PREPARE_LOAD_MONITOR}_manifest.json")
    ), _PREPARE_KEY)
    run_id, run_dir = prep["run_id"], Path(prep["run_dir"])
    on_event = make_event_emitter(run_id, persist_fn=make_event_logger(run_dir))

    reported = _parse_proposal(outputs.get(STEP_PROPOSE_LOAD_MONITOR))
    authoritative_flags = read_fact(run_id, "resource_snapshot")["flags"]
    flag_mismatch = not _flags_match((reported or {}).get("flags"), authoritative_flags)

    emit_event(on_event, "load_monitor", "load_monitor_report", {"flag_mismatch": flag_mismatch})
    manifest = {**prep, "flag_mismatch": flag_mismatch}
    return "reported", _write_manifest(_manifest_path(run_dir, STEP_LOAD_MONITOR_DECIDE), manifest)


# -- 2. Task Prioritization -------------------------------------------


def prepare_task_prioritization(outputs: dict[str, str]) -> tuple[str, str]:
    prior = _read_manifest(outputs[STEP_LOAD_MONITOR_DECIDE])
    run_id, run_dir, csv_path = prior["run_id"], Path(prior["run_dir"]), prior["csv_path"]
    on_event = make_event_emitter(run_id, persist_fn=make_event_logger(run_dir))

    df, variance_injected = load_task_table(csv_path)
    fact = build_task_queue_fact(df, variance_injected, queue_size=_DEFAULT_QUEUE_SIZE, snapshot_window=_DEFAULT_SNAPSHOT_WINDOW)
    prepare = _publish_fact(run_id, on_event, "task_prioritization", "task_queue_profile", fact)

    manifest = {"run_id": run_id, "run_dir": str(run_dir), "csv_path": csv_path, _PREPARE_KEY: prepare}
    _write_manifest(_manifest_path(run_dir, STEP_PREPARE_TASK_PRIORITIZATION), manifest)
    return "reported", run_id


def task_prioritization_decide(outputs: dict[str, str]) -> tuple[str, str]:
    prep = _without(_read_manifest(
        str(resolve_run_dir(outputs[STEP_PREPARE_TASK_PRIORITIZATION]) / f"{STEP_PREPARE_TASK_PRIORITIZATION}_manifest.json")
    ), _PREPARE_KEY)
    run_id, run_dir = prep["run_id"], Path(prep["run_dir"])
    on_event = make_event_emitter(run_id, persist_fn=make_event_logger(run_dir))

    proposal = _parse_proposal(outputs.get(STEP_PROPOSE_TASK_PRIORITIZATION))
    fact = read_fact(run_id, "task_queue_profile")
    pending_task_ids = [t["task_id"] for t in fact["pending_tasks"]]

    if proposal is None:
        errors = ["propose_task_prioritization output was not valid JSON"]
    else:
        errors = validate_ranking_proposal(pending_task_ids, proposal)
    valid = not errors
    score_inconsistent = valid and not is_ranking_score_consistent(proposal)

    if valid:
        mailbox = PersistentMailbox(run_id, on_event=on_event)
        mailbox.send(
            sender="task_prioritization", recipient="resource_allocation", message_type="task_ranking",
            payload={
                "ranked_task_ids": proposal["ranked_task_ids"],
                "score_breakdown": proposal["score_breakdown"],
                "reasoning": proposal.get("reasoning", ""),
            },
        )
        emit_event(on_event, "task_prioritization", "a2a_sent", {"recipient": "resource_allocation"})

    emit_event(on_event, "task_prioritization", "task_prioritization_report", {
        "valid": valid, "validation_errors": errors, "score_inconsistent": score_inconsistent,
    })
    _write_legacy_evidence_report(run_dir, "task_prioritization_report.json", {
        "valid": valid, "score_inconsistent": score_inconsistent,
    })
    manifest = {**prep, "valid": valid, "validation_errors": errors, "score_inconsistent": score_inconsistent}
    decision = "ranked" if valid else "invalid"
    return decision, _write_manifest(_manifest_path(run_dir, STEP_TASK_PRIORITIZATION_DECIDE), manifest)


# -- 3. Resource Allocation --------------------------------------------


def prepare_resource_allocation(outputs: dict[str, str]) -> tuple[str, str]:
    prior = _read_manifest(outputs[STEP_TASK_PRIORITIZATION_DECIDE])
    run_id, run_dir, csv_path = prior["run_id"], Path(prior["run_dir"]), prior["csv_path"]
    on_event = make_event_emitter(run_id, persist_fn=make_event_logger(run_dir))

    mailbox = PersistentMailbox(run_id, on_event=on_event)
    inbox = mailbox.inbox_for("resource_allocation", message_type="task_ranking")
    if not inbox:
        emit_event(on_event, "resource_allocation", "no_ranking_available", {})
        manifest = {"run_id": run_id, "run_dir": str(run_dir), "csv_path": csv_path, "errors": ["no task_ranking message available"]}
        return "no_ranking", _write_manifest(_manifest_path(run_dir, STEP_PREPARE_RESOURCE_ALLOCATION), manifest)

    ranking_message = inbox[-1]  # MVP: only the most recent ranking is actionable
    ranked_task_ids = ranking_message.payload["ranked_task_ids"]
    score_breakdown = ranking_message.payload.get("score_breakdown", {})
    emit_event(on_event, "resource_allocation", "ranking_received", {
        "sender": ranking_message.sender, "n_tasks": len(ranked_task_ids),
    })

    df, variance_injected = load_task_table(csv_path)
    fact = build_allocation_context_fact(
        df, variance_injected, ranked_task_ids, score_breakdown,
        snapshot_window=_DEFAULT_SNAPSHOT_WINDOW, slice_capacity=_DEFAULT_SLICE_CAPACITY,
    )
    prepare = _publish_fact(run_id, on_event, "resource_allocation", "allocation_context", fact)

    manifest = {
        "run_id": run_id, "run_dir": str(run_dir), "csv_path": csv_path,
        "ranked_task_ids": ranked_task_ids, "score_breakdown": score_breakdown, _PREPARE_KEY: prepare,
    }
    _write_manifest(_manifest_path(run_dir, STEP_PREPARE_RESOURCE_ALLOCATION), manifest)
    return "ready", run_id


def resource_allocation_decide(outputs: dict[str, str]) -> tuple[str, str]:
    prep = _without(_read_manifest(
        str(resolve_run_dir(outputs[STEP_PREPARE_RESOURCE_ALLOCATION]) / f"{STEP_PREPARE_RESOURCE_ALLOCATION}_manifest.json")
    ), _PREPARE_KEY)
    run_id, run_dir, csv_path = prep["run_id"], Path(prep["run_dir"]), prep["csv_path"]
    ranked_task_ids = prep["ranked_task_ids"]
    score_breakdown = prep["score_breakdown"]
    on_event = make_event_emitter(run_id, persist_fn=make_event_logger(run_dir))

    proposal = _parse_proposal(outputs.get(STEP_PROPOSE_RESOURCE_ALLOCATION))
    errors = ["propose_resource_allocation output was not valid JSON"] if proposal is None \
        else validate_allocation_structure(ranked_task_ids, proposal)
    valid = not errors

    accepted: list[dict] = []
    environment_rejected: list[dict] = []
    agent_rejected: list[dict] = []
    events: list[dict] = []
    risky: list[dict] = []

    if valid:
        assignments = proposal.get("assignments", [])
        agent_rejected = proposal.get("rejected", [])

        fact = read_fact(run_id, "allocation_context")
        machine_status = {m["machine_id"]: m["status"] for m in fact["available_machines"]}
        slice_current_load = {s["slice_id"]: s["current_load"] for s in fact["available_slices"]}
        violations = check_constraints(assignments, machine_status, slice_current_load, slice_capacity=_DEFAULT_SLICE_CAPACITY)
        violations_by_task = {v["task_id"]: v for v in violations}

        for a in assignments:
            if a.get("task_id") in violations_by_task:
                environment_rejected.append({"task_id": a.get("task_id"), "proposed": a, "violation": violations_by_task[a["task_id"]]})
            else:
                accepted.append(a)

        trigger_signal_by_task = {
            a.get("task_id"): f"final_score={(score_breakdown.get(a.get('task_id')) or {}).get('final_score')}"
            for a in assignments
        }
        events = build_allocation_events(assignments, violations_by_task, trigger_signal_by_task)

        # Published so a LATER resource-scheduler-failure-recovery pipeline's
        # prepare_failure_recovery can read this run's committed assignments
        # by run_id alone, without this pipeline still being alive.
        write_fact(run_id, "accepted_assignments", {"accepted_assignments": accepted, "csv_path": csv_path})

        if accepted:
            risky = identify_risky_assignments(accepted, machine_status)
            if risky:
                mailbox = PersistentMailbox(run_id, on_event=on_event)
                mailbox.send(
                    sender="resource_allocation", recipient="human_oversight", message_type="risky_decision",
                    payload={"source": "resource_allocation", "risky_decisions": risky},
                )
                emit_event(on_event, "resource_allocation", "a2a_sent", {"recipient": "human_oversight", "n_risky": len(risky)})

    emit_event(on_event, "resource_allocation", "resource_allocation_report", {
        "valid": valid, "n_accepted": len(accepted), "n_environment_rejected": len(environment_rejected),
        "n_agent_rejected": len(agent_rejected), "n_risky": len(risky),
    })
    _write_legacy_evidence_report(run_dir, "resource_allocation_report.json", {
        "accepted_assignments": accepted, "environment_rejected": environment_rejected, "agent_rejected": agent_rejected,
    })

    manifest = {
        **prep, "validation_errors": errors, "accepted_assignments": accepted,
        "environment_rejected": environment_rejected, "agent_rejected": agent_rejected, "events": events,
    }
    if not valid:
        decision = "invalid"
    elif risky:
        decision = "accepted_with_risk"
    else:
        decision = "accepted_clean"
    return decision, _write_manifest(_manifest_path(run_dir, STEP_RESOURCE_ALLOCATION_DECIDE), manifest)


def reroute_validation_decide(outputs: dict[str, str]) -> tuple[str, str]:
    """Purely deterministic -- no propose step, no LLM call, same as
    steps/resource_allocation_step.py::run_reroute_validation_step. Reuses
    check_constraints/build_allocation_events unchanged; Failure Recovery
    already did the reasoning about where to reroute, this gate's only
    job is to be the deterministic gate-keeper."""
    prep = _read_manifest(outputs[STEP_FAILURE_RECOVERY_DECIDE])
    run_id, run_dir, csv_path = prep["run_id"], Path(prep["run_dir"]), prep["csv_path"]
    on_event = make_event_emitter(run_id, persist_fn=make_event_logger(run_dir))

    mailbox = PersistentMailbox(run_id, on_event=on_event)
    inbox = mailbox.inbox_for("resource_allocation", message_type="reroute_request")
    if not inbox:
        emit_event(on_event, "resource_allocation", "no_reroute_request_available", {})
        manifest = {**prep, "errors": ["no reroute_request message available"]}
        return "no_reroute", _write_manifest(_manifest_path(run_dir, STEP_REROUTE_VALIDATION_DECIDE), manifest)

    message = inbox[-1]  # MVP: only the most recent reroute request is actionable
    reroute_proposals = message.payload["reroute_proposals"]
    emit_event(on_event, "resource_allocation", "reroute_request_received", {
        "sender": message.sender, "n_reroutes": len(reroute_proposals),
    })

    df, variance_injected = load_task_table(csv_path)
    snapshot = compute_snapshot(df, variance_injected, window=_DEFAULT_SNAPSHOT_WINDOW)
    machine_status = {m["machine_id"]: m["status"] for m in snapshot["machines"]}
    slice_current_load = compute_slice_load(df, window=max(len(reroute_proposals), 1))

    as_assignments = [
        {"task_id": r["task_id"], "machine_id": r.get("new_machine_id"),
         "network_slice_id": r.get("new_network_slice_id"), "rationale": r.get("reasoning")}
        for r in reroute_proposals
    ]
    violations = check_constraints(as_assignments, machine_status, slice_current_load, slice_capacity=_DEFAULT_SLICE_CAPACITY)
    violations_by_task = {v["task_id"]: v for v in violations}

    accepted: list[dict] = []
    environment_rejected: list[dict] = []
    for a in as_assignments:
        if a["task_id"] in violations_by_task:
            environment_rejected.append({"task_id": a["task_id"], "proposed": a, "violation": violations_by_task[a["task_id"]]})
        else:
            accepted.append(a)

    trigger_signal_by_task = {a["task_id"]: "reroute_request (Failure Recovery, incident-triggered)" for a in as_assignments}
    events = build_allocation_events(as_assignments, violations_by_task, trigger_signal_by_task)

    risky: list[dict] = []
    if accepted:
        risky = identify_risky_assignments(accepted, machine_status)
        if risky:
            # sender is the reroute's true origin (Failure Recovery), not this
            # gate -- this code re-validates the reroute, but the decision to
            # place it here is Failure Recovery's, same attribution
            # build_allocation_events already gives it via trigger_signal_by_task.
            mailbox.send(
                sender=message.sender, recipient="human_oversight", message_type="risky_decision",
                payload={"source": message.sender, "risky_decisions": risky},
            )
            emit_event(on_event, "resource_allocation", "a2a_sent", {"recipient": "human_oversight", "n_risky": len(risky)})

    emit_event(on_event, "resource_allocation", "reroute_validation_report", {
        "n_accepted": len(accepted), "n_environment_rejected": len(environment_rejected), "n_risky": len(risky),
    })
    _write_legacy_evidence_report(run_dir, "failure_recovery_report.json", {
        "reroute_validation": {"accepted_reroutes": accepted, "environment_rejected": environment_rejected},
    })

    manifest = {**prep, "accepted_reroutes": accepted, "environment_rejected": environment_rejected, "events": events}
    if risky:
        decision = "accepted_with_risk"
    elif accepted:
        decision = "accepted_clean"
    else:
        decision = "no_reroute"
    return decision, _write_manifest(_manifest_path(run_dir, STEP_REROUTE_VALIDATION_DECIDE), manifest)


# -- 4. Failure Recovery -------------------------------------------------


def prepare_failure_recovery(outputs: dict[str, str]) -> tuple[str, str]:
    """Seed task is an EXISTING resource-scheduler-main run's run_id, not
    a CSV path -- this pipeline starts its own fresh run_id/run_dir (via
    make_run_dir), and reads the source run's committed assignments (the
    accepted_assignments fact resource_allocation_decide publishes) plus
    its csv_path as the only things it needs from that other run."""
    source_run_id = outputs["__task__"]
    source = read_fact(source_run_id, "accepted_assignments")
    committed_assignments, csv_path = source["accepted_assignments"], source["csv_path"]

    run_id, run_dir = make_run_dir(None)
    on_event = make_event_emitter(run_id, persist_fn=make_event_logger(run_dir))

    df, variance_injected = load_task_table(csv_path)
    before_row = max(len(df) - _DEFAULT_BEFORE_OFFSET, 1)
    after_row = len(df)
    fact = build_incident_fact(df, variance_injected, committed_assignments, before_row, after_row, window=_DEFAULT_SNAPSHOT_WINDOW)
    emit_event(on_event, "failure_recovery", "incident_scan", {
        "n_incidents": len(fact["incidents"]), "n_affected": len(fact["affected_tasks"]),
    })

    base_manifest = {
        "run_id": run_id, "run_dir": str(run_dir), "csv_path": csv_path,
        "source_run_id": source_run_id, "before_row": before_row, "after_row": after_row,
    }
    if not fact["incidents"]:
        emit_event(on_event, "failure_recovery", "no_incidents_detected", {})
        return "no_incidents", _write_manifest(_manifest_path(run_dir, STEP_PREPARE_FAILURE_RECOVERY), base_manifest)
    if not fact["affected_tasks"]:
        emit_event(on_event, "failure_recovery", "no_affected_tasks", {})
        manifest = {**base_manifest, "incidents": fact["incidents"]}
        return "no_affected_tasks", _write_manifest(_manifest_path(run_dir, STEP_PREPARE_FAILURE_RECOVERY), manifest)

    prepare = _publish_fact(run_id, on_event, "failure_recovery", "incident_report", fact)
    manifest = {**base_manifest, _PREPARE_KEY: prepare}
    _write_manifest(_manifest_path(run_dir, STEP_PREPARE_FAILURE_RECOVERY), manifest)
    return "ready", run_id


def failure_recovery_decide(outputs: dict[str, str]) -> tuple[str, str]:
    prep = _without(_read_manifest(
        str(resolve_run_dir(outputs[STEP_PREPARE_FAILURE_RECOVERY]) / f"{STEP_PREPARE_FAILURE_RECOVERY}_manifest.json")
    ), _PREPARE_KEY)
    run_id, run_dir = prep["run_id"], Path(prep["run_dir"])
    on_event = make_event_emitter(run_id, persist_fn=make_event_logger(run_dir))

    proposal = _parse_proposal(outputs.get(STEP_PROPOSE_FAILURE_RECOVERY))
    fact = read_fact(run_id, "incident_report")
    affected_task_ids = [t["task_id"] for t in fact["affected_tasks"]]

    errors = ["propose_failure_recovery output was not valid JSON"] if proposal is None \
        else validate_recovery_proposal(affected_task_ids, proposal)
    valid = not errors
    reroute_avoids_source = True

    if valid:
        reroutes = proposal.get("reroute_proposals", [])
        reroute_avoids_source = reroutes_avoid_source_machine(fact["affected_tasks"], proposal)
        mailbox = PersistentMailbox(run_id, on_event=on_event)
        mailbox.send(
            sender="failure_recovery", recipient="resource_allocation", message_type="reroute_request",
            payload={"reroute_proposals": reroutes, "incidents": fact["incidents"]},
        )
        emit_event(on_event, "failure_recovery", "a2a_sent", {"recipient": "resource_allocation"})

    emit_event(on_event, "failure_recovery", "failure_recovery_report", {
        "valid": valid, "reroute_avoids_source": reroute_avoids_source,
    })
    manifest = {**prep, "validation_errors": errors, "reroute_avoids_source": reroute_avoids_source}
    decision = "rerouted" if valid else "invalid"
    return decision, _write_manifest(_manifest_path(run_dir, STEP_FAILURE_RECOVERY_DECIDE), manifest)


# -- 5. Optimization (one-shot) -------------------------------------------


def prepare_optimization(outputs: dict[str, str]) -> tuple[str, str]:
    """Reachable two ways: as a fan-in step inside resource-scheduler-main
    (after resource_allocation_decide or decision_oversight_decide -- both
    keys are checked since `outputs` carries every prior step's output,
    not just the immediately preceding one), or as the entry gate of the
    standalone resource-scheduler-optimization pipeline (neither key
    present -- starts its own fresh run_id, since Optimization reads
    cross-run history, not this run's own facts, and needs no upstream
    manifest to inherit from)."""
    prior_manifest_path = outputs.get(STEP_RESOURCE_ALLOCATION_DECIDE) or outputs.get(STEP_DECISION_OVERSIGHT_DECIDE)
    if prior_manifest_path:
        prior = _read_manifest(prior_manifest_path)
        run_id, run_dir = prior["run_id"], Path(prior["run_dir"])
    else:
        run_id, run_dir = make_run_dir(None)
    on_event = make_event_emitter(run_id, persist_fn=make_event_logger(run_dir))

    evidence = build_policy_evidence_fact(_DEFAULT_N_RUNS)
    emit_event(on_event, "optimization", "evidence_collected", {
        "n_runs_scanned": evidence["n_runs_scanned"], "n_run_dirs_considered": evidence["n_run_dirs_considered"],
    })

    if not evidence["n_runs_scanned"]:
        emit_event(on_event, "optimization", "no_run_history_available", {})
        manifest = {"run_id": run_id, "run_dir": str(run_dir), "errors": ["no run history available"]}
        return "no_history", _write_manifest(_manifest_path(run_dir, STEP_PREPARE_OPTIMIZATION), manifest)

    prepare = _publish_fact(run_id, on_event, "optimization", "policy_evidence", evidence)
    manifest = {"run_id": run_id, "run_dir": str(run_dir), _PREPARE_KEY: prepare}
    _write_manifest(_manifest_path(run_dir, STEP_PREPARE_OPTIMIZATION), manifest)
    return "ready", run_id


def optimization_decide(outputs: dict[str, str]) -> tuple[str, str]:
    prep = _without(_read_manifest(
        str(resolve_run_dir(outputs[STEP_PREPARE_OPTIMIZATION]) / f"{STEP_PREPARE_OPTIMIZATION}_manifest.json")
    ), _PREPARE_KEY)
    run_id, run_dir = prep["run_id"], Path(prep["run_dir"])
    on_event = make_event_emitter(run_id, persist_fn=make_event_logger(run_dir))

    proposal = _parse_proposal(outputs.get(STEP_PROPOSE_OPTIMIZATION))
    errors = ["propose_optimization output was not valid JSON"] if proposal is None else validate_policy_proposal(proposal)
    valid = not errors

    if valid:
        evidence = read_fact(run_id, "policy_evidence")
        mailbox = PersistentMailbox(run_id, on_event=on_event)
        mailbox.send(
            sender="optimization", recipient="human_oversight", message_type="policy_update_proposal",
            payload={
                "policy_updates": proposal["policy_updates"], "evidence": proposal["evidence"],
                "recommend_apply": proposal["recommend_apply"], "underlying_evidence": evidence,
            },
        )
        emit_event(on_event, "optimization", "a2a_sent", {"recipient": "human_oversight"})

    emit_event(on_event, "optimization", "optimization_report", {"valid": valid})
    manifest = {**prep, "validation_errors": errors}
    decision = "proposed" if valid else "invalid"
    return decision, _write_manifest(_manifest_path(run_dir, STEP_OPTIMIZATION_DECIDE), manifest)


# -- 6. Human Oversight: policy review -------------------------------------


def prepare_oversight(outputs: dict[str, str]) -> tuple[str, str]:
    prior = _read_manifest(outputs[STEP_OPTIMIZATION_DECIDE])
    run_id, run_dir = prior["run_id"], Path(prior["run_dir"])
    on_event = make_event_emitter(run_id, persist_fn=make_event_logger(run_dir))

    mailbox = PersistentMailbox(run_id, on_event=on_event)
    inbox = mailbox.inbox_for("human_oversight", message_type="policy_update_proposal")
    if not inbox:
        emit_event(on_event, "human_oversight", "no_proposal_available", {})
        manifest = {"run_id": run_id, "run_dir": str(run_dir), "errors": ["no policy_update_proposal message available"]}
        return "no_proposal", _write_manifest(_manifest_path(run_dir, STEP_PREPARE_OVERSIGHT), manifest)

    message = inbox[-1]  # MVP: only the most recent proposal is actionable
    proposal = message.payload
    emit_event(on_event, "human_oversight", "proposal_received", {"sender": message.sender})

    bundle = build_oversight_review_bundle(proposal, proposal.get("underlying_evidence") or {})
    prepare = _publish_fact(run_id, on_event, "human_oversight", "policy_review_bundle", bundle)

    manifest = {"run_id": run_id, "run_dir": str(run_dir), _PREPARE_KEY: prepare}
    _write_manifest(_manifest_path(run_dir, STEP_PREPARE_OVERSIGHT), manifest)
    return "ready", run_id


def oversight_decide(outputs: dict[str, str]) -> tuple[str, str]:
    prep = _without(_read_manifest(
        str(resolve_run_dir(outputs[STEP_PREPARE_OVERSIGHT]) / f"{STEP_PREPARE_OVERSIGHT}_manifest.json")
    ), _PREPARE_KEY)
    run_id, run_dir = prep["run_id"], Path(prep["run_dir"])
    on_event = make_event_emitter(run_id, persist_fn=make_event_logger(run_dir))

    raw = _parse_proposal(outputs.get(STEP_PROPOSE_OVERSIGHT))
    verdict = parse_oversight_verdict(raw)

    emit_event(on_event, "human_oversight", "oversight_report", {"verdict": verdict["verdict"], "n_concerns": len(verdict["concerns"])})
    manifest = {**prep, **verdict}
    return verdict["verdict"], _write_manifest(_manifest_path(run_dir, STEP_OVERSIGHT_DECIDE), manifest)


# -- 7. Human Oversight: risky-decision review ------------------------------


def prepare_decision_oversight(outputs: dict[str, str]) -> tuple[str, str]:
    """Reachable from resource_allocation_decide (resource-scheduler-main)
    or reroute_validation_decide (resource-scheduler-failure-recovery) --
    both keys are checked, same reasoning as prepare_optimization above."""
    prior_manifest_path = outputs.get(STEP_RESOURCE_ALLOCATION_DECIDE) or outputs.get(STEP_REROUTE_VALIDATION_DECIDE)
    prior = _read_manifest(prior_manifest_path)
    run_id, run_dir = prior["run_id"], Path(prior["run_dir"])
    on_event = make_event_emitter(run_id, persist_fn=make_event_logger(run_dir))

    mailbox = PersistentMailbox(run_id, on_event=on_event)
    inbox = mailbox.inbox_for("human_oversight", message_type="risky_decision")
    if not inbox:
        emit_event(on_event, "human_oversight", "no_risky_decision_available", {})
        manifest = {"run_id": run_id, "run_dir": str(run_dir), "errors": ["no risky_decision message available"]}
        return "no_decision", _write_manifest(_manifest_path(run_dir, STEP_PREPARE_DECISION_OVERSIGHT), manifest)

    message = inbox[-1]  # MVP: only the most recent risky-decision batch is actionable
    emit_event(on_event, "human_oversight", "risky_decision_received", {
        "sender": message.sender, "n_decisions": len(message.payload.get("risky_decisions", [])),
    })

    bundle = build_decision_review_bundle(message.payload)
    prepare = _publish_fact(run_id, on_event, "human_oversight", "decision_review_bundle", bundle)

    manifest = {"run_id": run_id, "run_dir": str(run_dir), _PREPARE_KEY: prepare}
    _write_manifest(_manifest_path(run_dir, STEP_PREPARE_DECISION_OVERSIGHT), manifest)
    return "ready", run_id


def decision_oversight_decide(outputs: dict[str, str]) -> tuple[str, str]:
    prep = _without(_read_manifest(
        str(resolve_run_dir(outputs[STEP_PREPARE_DECISION_OVERSIGHT]) / f"{STEP_PREPARE_DECISION_OVERSIGHT}_manifest.json")
    ), _PREPARE_KEY)
    run_id, run_dir = prep["run_id"], Path(prep["run_dir"])
    on_event = make_event_emitter(run_id, persist_fn=make_event_logger(run_dir))

    raw = _parse_proposal(outputs.get(STEP_PROPOSE_DECISION_OVERSIGHT))
    verdict = parse_oversight_verdict(raw)

    emit_event(on_event, "human_oversight", "decision_oversight_report", {"verdict": verdict["verdict"], "n_concerns": len(verdict["concerns"])})
    manifest = {**prep, **verdict}
    return verdict["verdict"], _write_manifest(_manifest_path(run_dir, STEP_DECISION_OVERSIGHT_DECIDE), manifest)
