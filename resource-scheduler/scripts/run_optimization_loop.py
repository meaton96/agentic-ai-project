#!/usr/bin/env python3
"""
Agent #5, continuous mode: the actual "background loop... to tune
allocation policies over time" the Use Case II spec asks for.
scripts/run_optimization_agent.py (the original one-shot mode) is a
single LLM-reasoned recommendation from existing run history -- useful,
but not a loop, and not RL or greedy search. This is.

Each iteration:
  1. environment/policy_search.py::propose_next_candidate() picks a
     parameter setting to try next (deterministic, no LLM -- greedy
     hill-climbing from the best setting evaluated so far).
  2. evaluate_candidate() actually runs Task Prioritization + Resource
     Allocation with that setting, using a fresh throwaway Mailbox (a
     probe, not real production traffic) -- needs a live model unless
     --synthetic is given.
  3. compute_score() turns the resulting acceptance/validity rates into
     one scalar.
  4. The (params, score) pair is appended to a persisted history file
     so the search actually improves ACROSS invocations of this script,
     not just within one run.
  5. Sleep --sleep-seconds, repeat, up to --max-iterations.

If the search converges on a setting that's a genuine improvement over
the very first evaluated point (not just noise), that improvement is
sent to Human Oversight via the SAME policy_update_proposal path
scripts/run_optimization_agent.py already uses (run_oversight_step) --
reused, not duplicated. This loop never applies a parameter change
itself, same "propose, never apply" principle as everywhere else in
this project.

Costs real tokens on every non-synthetic iteration. Given this
session's own experience with the RIT endpoint's rate limits and
intermittent 504s, --max-iterations defaults small (5) and
--sleep-seconds defaults to a real pause (10s) between iterations --
this is deliberately conservative, not tuned for speed.

Usage:
    set -a; source .env; set +a
    python scripts/run_optimization_loop.py \\
        --data datasets/raw/industrial_scheduling_dataset.csv \\
        --max-iterations 5

    # exercise the search mechanism itself with zero tokens/API calls:
    python scripts/run_optimization_loop.py \\
        --data datasets/raw/industrial_scheduling_dataset.csv \\
        --synthetic --max-iterations 8 --sleep-seconds 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resource_scheduler.a2a.mailbox import Mailbox
from resource_scheduler.cli_common import make_retry_logger, make_run_dir, make_tracer, make_transcript_writer, resolve_model_endpoint
from resource_scheduler.environment.policy_search import DEFAULT_PARAMS, compute_score, propose_next_candidate
from resource_scheduler.environment.queue import compute_pending_queue
from resource_scheduler.environment.state import load_task_table
from resource_scheduler.model_client import ModelClient
from resource_scheduler.paths import artifacts_root
from resource_scheduler.steps.oversight_step import run_oversight_step
from resource_scheduler.steps.resource_allocation_step import run_resource_allocation_step
from resource_scheduler.steps.task_prioritization_step import run_task_prioritization_step

HISTORY_PATH = artifacts_root() / "policy_search_history.json"
IMPROVEMENT_THRESHOLD = 0.02  # ignore anything smaller as noise, not a real improvement


def load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def save_history(history: list[dict]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2, default=str))


def build_synthetic_ranking(df, queue_size: int) -> tuple[list[str], dict]:
    """Same placeholder ranking every other script's --synthetic-*
    flag already uses -- lets the search LOOP be exercised for free,
    even though the resulting scores don't reflect real LLM behavior
    (see this file's own docstring on that tradeoff)."""
    tasks = compute_pending_queue(df, queue_size=queue_size)
    ranked_task_ids = [t["task_id"] for t in tasks]
    score_breakdown = {
        tid: {"final_score": round(1.0 - i * (1.0 / max(len(ranked_task_ids), 1)), 4)}
        for i, tid in enumerate(ranked_task_ids)
    }
    return ranked_task_ids, score_breakdown


def synthetic_allocate(df, variance_injected, ranked_task_ids, score_breakdown, snapshot_window, slice_capacity) -> tuple[int, int]:
    """Deterministic stand-in for Resource Allocation's LLM proposal in
    --synthetic mode. NOT a simulation of what an LLM would choose --
    just greedily round-robins each ranked task across the available
    machines/slices -- but it's checked against the exact same
    check_constraints gate a real run uses, so the resulting acceptance
    rate is genuine, not fabricated. This is what makes --synthetic
    mode actually exercise the real constraint logic (including the
    slice-capacity-vs-window scale issue found earlier this session)
    while costing zero tokens."""
    from resource_scheduler.environment.allocation import check_constraints
    from resource_scheduler.tools.resource_allocation_tool import build_allocation_context_fact

    facts = build_allocation_context_fact(
        df, variance_injected, ranked_task_ids, score_breakdown,
        snapshot_window=snapshot_window, slice_capacity=slice_capacity,
    )
    machine_status = {m["machine_id"]: m["status"] for m in facts["available_machines"]}
    machine_ids = list(machine_status.keys())
    slice_ids = [s["slice_id"] for s in facts["available_slices"]]
    running_load = {s["slice_id"]: s["current_load"] for s in facts["available_slices"]}

    accepted = rejected = 0
    for i, task_id in enumerate(ranked_task_ids):
        assignment = [{
            "task_id": task_id,
            "machine_id": machine_ids[i % len(machine_ids)],
            "network_slice_id": slice_ids[i % len(slice_ids)],
        }]
        violations = check_constraints(assignment, machine_status, running_load, slice_capacity=slice_capacity)
        if violations:
            rejected += 1
        else:
            accepted += 1
            running_load[assignment[0]["network_slice_id"]] = running_load.get(assignment[0]["network_slice_id"], 0) + 1
    return accepted, rejected


def evaluate_candidate(df, variance_injected, client, params: dict, synthetic: bool, model, trace, prompt_override_dir) -> dict:
    """Runs one probe of Task Prioritization + Resource Allocation at a
    given parameter setting and returns a policy_evidence-shaped dict
    for compute_score(). In live mode, uses a fresh throwaway Mailbox --
    this is a search probe, not real production traffic, and must never
    leak into a real orchestrator run's mailbox."""
    if synthetic:
        ranked_task_ids, score_breakdown = build_synthetic_ranking(df, params["queue_size"])
        accepted, rejected = synthetic_allocate(
            df, variance_injected, ranked_task_ids, score_breakdown,
            params["snapshot_window"], params["slice_capacity"],
        )
        total = accepted + rejected
        return {
            "ranking_valid_rate": 1.0,
            "ranking_score_inconsistent_rate": 0.0,
            "allocation_acceptance_rate": (accepted / total) if total else None,
            "reroute_acceptance_rate": None,
        }

    probe_mailbox = Mailbox()
    tp_result = run_task_prioritization_step(
        df, variance_injected, client, queue_size=params["queue_size"],
        snapshot_window=params["snapshot_window"], model=model,
        trace_fn=lambda record: trace(**record), mailbox=probe_mailbox,
        prompt_override_dir=prompt_override_dir,
    )
    ranking_valid_rate = 1.0 if tp_result.valid else 0.0
    ranking_score_inconsistent_rate = 1.0 if tp_result.score_inconsistent else 0.0
    if not tp_result.sent_to_resource_allocation:
        return {
            "ranking_valid_rate": ranking_valid_rate,
            "ranking_score_inconsistent_rate": ranking_score_inconsistent_rate,
            "allocation_acceptance_rate": None,
            "reroute_acceptance_rate": None,
        }

    ra_result = run_resource_allocation_step(
        df, variance_injected, client, probe_mailbox,
        snapshot_window=params["snapshot_window"], slice_capacity=params["slice_capacity"],
        model=model, trace_fn=lambda record: trace(**record),
        prompt_override_dir=prompt_override_dir,
    )
    total_proposed = len(ra_result.accepted_assignments) + len(ra_result.environment_rejected)
    allocation_acceptance_rate = (len(ra_result.accepted_assignments) / total_proposed) if total_proposed else None

    return {
        "ranking_valid_rate": ranking_valid_rate,
        "ranking_score_inconsistent_rate": ranking_score_inconsistent_rate,
        "allocation_acceptance_rate": allocation_acceptance_rate,
        "reroute_acceptance_rate": None,  # not evaluated per-probe -- needs prior committed state, out of scope for a fast iteration
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=10.0)
    parser.add_argument("--no-inject-variance", action="store_true")
    parser.add_argument("--synthetic", action="store_true",
                         help="exercise the search loop with zero LLM calls -- scores won't reflect real model behavior")
    parser.add_argument("--reset-history", action="store_true", help="discard any existing policy_search_history.json first")
    parser.add_argument("--model", default=None)
    parser.add_argument("--use-gateway", action="store_true")
    parser.add_argument("--use-local", action="store_true")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id, run_dir = make_run_dir(args.run_id)
    trace = make_tracer(run_dir / "trace.jsonl")
    write_transcript = make_transcript_writer(run_dir)

    df, variance_injected = load_task_table(args.data, inject_variance=not args.no_inject_variance)
    print(f"Loaded task table: {len(df)} rows")

    client: Optional[ModelClient] = None
    default_model = None
    if not args.synthetic:
        base_url, api_key, default_model = resolve_model_endpoint(
            args.use_gateway, args.model, "qwen3:8b", "rit-qwen3-8b", use_local=args.use_local,
        )
        client = ModelClient(base_url=base_url, api_key=api_key, default_model=default_model, on_retry=make_retry_logger())

    history = [] if args.reset_history else load_history()
    print(f"Starting from {len(history)} prior history entr{'y' if len(history) == 1 else 'ies'} "
          f"({HISTORY_PATH})")

    first_score: Optional[float] = history[0]["score"] if history else None
    best_params, best_score = DEFAULT_PARAMS, first_score

    for i in range(args.max_iterations):
        candidate, converged = propose_next_candidate(history)
        print(f"\nIteration {i + 1}/{args.max_iterations}: evaluating {candidate}"
              + (" (search converged on a local optimum)" if converged else ""))

        try:
            evidence = evaluate_candidate(df, variance_injected, client, candidate, args.synthetic, default_model, trace, None)
            score = compute_score(evidence)
            print(f"  score={score} (evidence: {evidence})")
        except Exception as e:
            # A background loop meant to run unattended must survive one
            # bad iteration -- confirmed live: an exhausted-retries 504
            # (ModelClient already tried 3 times) previously crashed the
            # whole process here instead of just failing this one probe.
            # Recorded as score=None -- exactly what propose_next_candidate
            # already treats as "no evidence either way", not "this
            # candidate is bad" -- so the search moves on to a different
            # neighbor next iteration instead of either crashing or
            # looping forever re-attempting the same failing request.
            print(f"  evaluation failed ({type(e).__name__}: {e}) -- recording as inconclusive, moving on.")
            score = None

        history.append({"params": candidate, "score": score, "iteration": len(history)})
        save_history(history)

        if score is not None and (best_score is None or score > best_score):
            best_params, best_score = candidate, score
        if first_score is None and score is not None:
            first_score = score

        if converged:
            print("  stopping: search has converged, no untried neighbors of the current best remain.")
            break
        if i < args.max_iterations - 1 and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    print(f"\nBest setting found: {best_params} (score={best_score})")

    improved = (
        best_score is not None and first_score is not None
        and best_score - first_score >= IMPROVEMENT_THRESHOLD
        and best_params != DEFAULT_PARAMS
    )
    if not improved:
        print("No improvement over the starting point large enough to report -- nothing sent to Human Oversight.")
        return

    print(f"Improvement found (+{round(best_score - first_score, 4)}) -- sending to Human Oversight...")
    mailbox = Mailbox(on_event=lambda e: trace(e["type"], **e["payload"]))
    mailbox.send(
        sender="optimization_loop", recipient="human_oversight", message_type="policy_update_proposal",
        payload={
            "policy_updates": best_params,
            "evidence": f"Greedy search over {len(history)} iteration(s) found this setting scores "
                        f"{best_score} vs. {first_score} at the starting point.",
            "recommend_apply": True,
            "underlying_evidence": {"n_runs_scanned": len(history), "n_run_dirs_considered": len(history)},
        },
    )
    if client is None:
        print("(--synthetic mode: skipping the actual Human Oversight review, which needs a live model)")
        return

    ho_result = run_oversight_step(client, mailbox, model=default_model, trace_fn=lambda record: trace(**record))
    write_transcript("human_oversight", ho_result.messages)
    print(f"Human Oversight verdict: {ho_result.verdict}")
    if ho_result.concerns:
        print(f"  concerns: {ho_result.concerns}")


if __name__ == "__main__":
    main()
