"""
Evidence aggregation for the Optimization agent (agent #5). Unlike
every other agent in this project, Optimization doesn't read the task
table at all -- it reads the report JSON files agents #2-4's CLI
scripts already write to runs/<run_id>/, rather than requiring those
already-built scripts to also write to a separate shared log. Same
"harden the deterministic layer" philosophy as everywhere else: this
module only summarizes what already happened, it never re-decides
anything a prior report already decided, and no LLM involvement lives
here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from resource_scheduler.paths import runs_root

# The only knobs actually wired to a CLI flag across agents #2-4
# (--queue-size, --snapshot-window, --slice-capacity in
# run_task_prioritization_agent.py / run_resource_allocation_agent.py /
# run_failure_recovery_agent.py). The Optimization agent may only
# propose adjustments to real, already-read parameters -- never an
# invented knob nothing consumes.
ALLOWED_POLICY_KEYS = {"queue_size", "snapshot_window", "slice_capacity"}


def list_recent_runs(n: int = 10, runs_dir: Optional[Path] = None) -> list[Path]:
    """Most recent N run directories under runs/, sorted by directory
    name -- run ids are timestamp-prefixed (see cli_common.make_run_dir),
    so lexical sort is chronological."""
    base = runs_dir or runs_root()
    if not base.exists():
        return []
    run_dirs = sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name)
    return run_dirs[-n:]


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def collect_policy_evidence(run_dirs: list[Path]) -> dict:
    """Reads whichever of task_prioritization_report.json /
    resource_allocation_report.json / failure_recovery_report.json exist
    in each run dir (a run may only have exercised some agents) and
    tallies outcomes. Missing/malformed files are skipped, not fatal --
    a pass over partial history is still useful, and a single corrupt
    report must not take down the whole aggregation."""
    n_runs_scanned = 0
    ranking_valid = ranking_total = 0
    ranking_score_inconsistent = 0
    allocation_accepted = allocation_environment_rejected = allocation_agent_rejected = 0
    reroute_accepted = reroute_environment_rejected = 0

    for run_dir in run_dirs:
        touched = False

        tp = _load_json(run_dir / "task_prioritization_report.json")
        if tp is not None:
            touched = True
            ranking_total += 1
            if tp.get("valid"):
                ranking_valid += 1
            if tp.get("score_inconsistent"):
                ranking_score_inconsistent += 1

        ra = _load_json(run_dir / "resource_allocation_report.json")
        if ra is not None:
            touched = True
            allocation_accepted += len(ra.get("accepted_assignments") or [])
            allocation_environment_rejected += len(ra.get("environment_rejected") or [])
            allocation_agent_rejected += len(ra.get("agent_rejected") or [])

        fr = _load_json(run_dir / "failure_recovery_report.json")
        if fr is not None:
            touched = True
            reroute_validation = fr.get("reroute_validation") or {}
            reroute_accepted += len(reroute_validation.get("accepted_reroutes") or [])
            reroute_environment_rejected += len(reroute_validation.get("environment_rejected") or [])

        if touched:
            n_runs_scanned += 1

    total_allocation_proposed = allocation_accepted + allocation_environment_rejected
    total_reroutes_proposed = reroute_accepted + reroute_environment_rejected

    return {
        "n_runs_scanned": n_runs_scanned,
        "ranking_valid_rate": round(ranking_valid / ranking_total, 4) if ranking_total else None,
        "ranking_score_inconsistent_rate": round(ranking_score_inconsistent / ranking_total, 4) if ranking_total else None,
        "allocation_acceptance_rate": round(allocation_accepted / total_allocation_proposed, 4) if total_allocation_proposed else None,
        "allocation_accepted": allocation_accepted,
        "allocation_environment_rejected": allocation_environment_rejected,
        "allocation_agent_rejected": allocation_agent_rejected,
        "reroute_acceptance_rate": round(reroute_accepted / total_reroutes_proposed, 4) if total_reroutes_proposed else None,
        "reroute_accepted": reroute_accepted,
        "reroute_environment_rejected": reroute_environment_rejected,
    }


def validate_policy_proposal(proposal: object) -> list[str]:
    """Hard structural gate, same shape as the other agents' validate_*
    functions: policy_updates keys must be a subset of
    ALLOWED_POLICY_KEYS and numeric; evidence a string; recommend_apply
    a bool. Does not check whether the proposed VALUE is a good idea --
    that judgment is Human Oversight's job, not a structural check."""
    errors: list[str] = []
    if not isinstance(proposal, dict):
        return ["proposal is not a JSON object"]

    updates = proposal.get("policy_updates")
    if not isinstance(updates, dict):
        errors.append("policy_updates is missing or not an object")
    else:
        for key, value in updates.items():
            if key not in ALLOWED_POLICY_KEYS:
                errors.append(f"policy_updates has an unknown key {key!r} -- must be one of {sorted(ALLOWED_POLICY_KEYS)}")
            elif not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"policy_updates[{key!r}] must be a number")

    if not isinstance(proposal.get("evidence"), str):
        errors.append("evidence is missing or not a string")
    if not isinstance(proposal.get("recommend_apply"), bool):
        errors.append("recommend_apply is missing or not a boolean")

    return errors
