"""
Tests for environment/policy_search.py -- no LLM, no I/O. The greedy
hill-climbing search is the actual mechanism behind Optimization's
background-loop mode (scripts/run_optimization_loop.py); everything
about whether it converges correctly, respects bounds, and handles
missing data is provable here without ever running the pipeline.
"""
from resource_scheduler.environment.policy_search import (
    DEFAULT_PARAMS,
    SEARCH_SPACE,
    clamp,
    compute_score,
    neighbors,
    propose_next_candidate,
)


def test_compute_score_averages_available_rates():
    evidence = {"ranking_valid_rate": 1.0, "allocation_acceptance_rate": 0.5, "reroute_acceptance_rate": None}
    # mean of 1.0 and 0.5 only -- reroute_acceptance_rate excluded, not treated as 0
    assert compute_score(evidence) == 0.75


def test_compute_score_returns_none_when_nothing_available():
    assert compute_score({"ranking_valid_rate": None, "allocation_acceptance_rate": None, "reroute_acceptance_rate": None}) is None


def test_compute_score_penalizes_score_inconsistency():
    base = {"ranking_valid_rate": 1.0, "allocation_acceptance_rate": 1.0, "reroute_acceptance_rate": None}
    consistent = compute_score(base)
    inconsistent = compute_score({**base, "ranking_score_inconsistent_rate": 1.0})
    assert inconsistent < consistent
    assert consistent - inconsistent == 0.5


def test_clamp_respects_bounds():
    assert clamp(SEARCH_SPACE["queue_size"]["max"] + 5, "queue_size") == SEARCH_SPACE["queue_size"]["max"]
    assert clamp(SEARCH_SPACE["queue_size"]["min"] - 5, "queue_size") == SEARCH_SPACE["queue_size"]["min"]
    assert clamp(9, "queue_size") == 9


def test_neighbors_moves_one_param_at_a_time():
    params = {"queue_size": 8, "snapshot_window": 100, "slice_capacity": 8}
    ns = neighbors(params)
    for n in ns:
        differing = [k for k in params if n[k] != params[k]]
        assert len(differing) == 1


def test_neighbors_omits_out_of_bounds_directions():
    params = {"queue_size": SEARCH_SPACE["queue_size"]["max"], "snapshot_window": 100, "slice_capacity": 8}
    ns = neighbors(params)
    # no neighbor should push queue_size further past its max
    assert all(n["queue_size"] <= SEARCH_SPACE["queue_size"]["max"] for n in ns)
    queue_size_increases = [n for n in ns if n["queue_size"] > params["queue_size"]]
    assert queue_size_increases == []


def test_propose_next_candidate_starts_at_defaults_with_empty_history():
    candidate, converged = propose_next_candidate([])
    assert candidate == DEFAULT_PARAMS
    assert converged is False


def test_propose_next_candidate_ignores_entries_with_no_score():
    history = [{"params": DEFAULT_PARAMS, "score": None}]
    candidate, converged = propose_next_candidate(history)
    assert candidate == DEFAULT_PARAMS  # falls back to defaults, same as empty history
    assert converged is False


def test_propose_next_candidate_proposes_an_untried_neighbor_of_the_best():
    history = [{"params": DEFAULT_PARAMS, "score": 0.5}]
    candidate, converged = propose_next_candidate(history)
    assert candidate != DEFAULT_PARAMS
    assert converged is False
    assert candidate in neighbors(DEFAULT_PARAMS)


def test_propose_next_candidate_picks_the_best_scoring_entry_not_the_most_recent():
    history = [
        {"params": DEFAULT_PARAMS, "score": 0.9},
        {"params": {**DEFAULT_PARAMS, "queue_size": 9}, "score": 0.1},
    ]
    candidate, converged = propose_next_candidate(history)
    # should explore around the 0.9-scoring DEFAULT_PARAMS, not the poor 0.1 one
    assert candidate in neighbors(DEFAULT_PARAMS)


def test_propose_next_candidate_converges_when_all_neighbors_tried():
    best = dict(DEFAULT_PARAMS)
    history = [{"params": best, "score": 1.0}]
    history += [{"params": n, "score": 0.1} for n in neighbors(best)]
    candidate, converged = propose_next_candidate(history)
    assert converged is True
    assert candidate == best


def test_search_space_matches_the_shared_default_params_keys():
    assert set(SEARCH_SPACE.keys()) == set(DEFAULT_PARAMS.keys())
