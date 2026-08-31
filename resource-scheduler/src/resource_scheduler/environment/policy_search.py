"""
Deterministic greedy search over the resource-scheduler's tunable
parameters (queue_size, snapshot_window, slice_capacity) -- the actual
tuning mechanism behind "runs a background loop... to tune allocation
policies over time" from the Use Case II spec, which the original
Optimization agent (steps/optimization_step.py) never was: that's a
single LLM-reasoned recommendation from existing history, not a loop
that runs repeatedly and climbs toward a better setting.

Greedy hill-climbing, not full RL: with only three small-integer
parameters and a well-behaved (if noisy) objective, a full RL
formulation (state/action/value functions, an exploration policy) would
be substantially more machinery than the problem calls for. Greedy
search is the spec's own named alternative ("RL or greedy search"), and
it's the proportionate choice here -- the same kind of judgment call
environment/queue.py already made choosing documented heuristic proxies
over a trained ranking model.

Pure functions, no LLM, no I/O -- fully unit-testable. The background
loop that actually RUNS the pipeline to evaluate a candidate (which
does need a live model, unless run in --synthetic mode) lives in
scripts/run_optimization_loop.py, not here.
"""
from __future__ import annotations

from typing import Optional

# (min, max, step) per parameter -- the same three parameters
# environment/policy_evidence.py::ALLOWED_POLICY_KEYS already restricts
# the one-shot Optimization agent to proposing, because they're the
# only ones actually wired to a CLI flag across agents #2-4.
SEARCH_SPACE: dict[str, dict[str, int]] = {
    "queue_size": {"min": 4, "max": 15, "step": 1},
    "snapshot_window": {"min": 50, "max": 200, "step": 25},
    "slice_capacity": {"min": 4, "max": 20, "step": 1},
}

# Matches every script's own --queue-size/--snapshot-window/--slice-capacity
# defaults -- the search starts from the setting already in production use,
# not an arbitrary point.
DEFAULT_PARAMS: dict[str, int] = {"queue_size": 8, "snapshot_window": 200, "slice_capacity": 8}


def compute_score(evidence: dict) -> Optional[float]:
    """A single scalar objective from environment/policy_evidence.py's
    aggregated rates: the mean of whichever rates are actually
    available (None means "no data for that metric in this window",
    not "zero" -- excluded from the mean rather than penalized as a
    failure), minus a penalty for inconsistent rankings. Returns None
    if there's no rate data at all to score against -- e.g. a probe
    that never got as far as producing a ranking."""
    rates = [
        evidence.get("ranking_valid_rate"),
        evidence.get("allocation_acceptance_rate"),
        evidence.get("reroute_acceptance_rate"),
    ]
    available = [r for r in rates if r is not None]
    if not available:
        return None
    score = sum(available) / len(available)
    inconsistent_penalty = 0.5 * (evidence.get("ranking_score_inconsistent_rate") or 0.0)
    return round(score - inconsistent_penalty, 4)


def clamp(value: int, param_name: str) -> int:
    bounds = SEARCH_SPACE[param_name]
    return max(bounds["min"], min(bounds["max"], value))


def neighbors(params: dict[str, int]) -> list[dict[str, int]]:
    """One-parameter-at-a-time +-step neighbors -- the standard
    hill-climbing neighborhood, not the full cross product of all three
    parameters moving at once, which would grow the candidate set
    combinatorially for no real benefit at this scale (8 candidates
    either way covers the same directions). A neighbor already at its
    parameter's bound in that direction is simply omitted, not clamped
    to a duplicate of the current point."""
    result = []
    for name, bounds in SEARCH_SPACE.items():
        for delta in (-bounds["step"], bounds["step"]):
            new_value = clamp(params[name] + delta, name)
            if new_value == params[name]:
                continue
            candidate = dict(params)
            candidate[name] = new_value
            result.append(candidate)
    return result


def _params_key(params: dict[str, int]) -> tuple:
    return tuple(sorted(params.items()))


def propose_next_candidate(history: list[dict]) -> tuple[dict[str, int], bool]:
    """history: a list of {"params": {...}, "score": float | None}
    entries, in the order they were tried (score=None means the probe
    at that setting produced no usable rate data). Returns
    (next_params_to_try, converged).

    Greedy hill-climbing: find the best-scoring entry tried so far
    (entries with score=None are ignored -- not evidence against a
    setting, just missing data), then propose the first untried
    neighbor of it. If every neighbor of the current best has already
    been tried, the search has converged on a local optimum -- returns
    the current best again with converged=True rather than looping
    forever proposing the same exhausted neighborhood."""
    scored = [h for h in history if h.get("score") is not None]
    if not scored:
        return dict(DEFAULT_PARAMS), False

    best = max(scored, key=lambda h: h["score"])
    tried = {_params_key(h["params"]) for h in history}

    for candidate in neighbors(best["params"]):
        if _params_key(candidate) not in tried:
            return candidate, False

    return dict(best["params"]), True
