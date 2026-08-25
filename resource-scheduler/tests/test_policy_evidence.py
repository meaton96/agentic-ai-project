"""
Tests for environment/policy_evidence.py -- no LLM involved. Builds
fake run directories with the same report JSON shapes agents #2-4's
CLI scripts actually write, to confirm aggregation reads them correctly
without needing a real end-to-end run.
"""
import json

from resource_scheduler.environment.policy_evidence import (
    ALLOWED_POLICY_KEYS,
    collect_policy_evidence,
    list_recent_runs,
    validate_policy_proposal,
)


def make_run_dir(base, name, tp=None, ra=None, fr=None):
    run_dir = base / name
    run_dir.mkdir()
    if tp is not None:
        (run_dir / "task_prioritization_report.json").write_text(json.dumps(tp))
    if ra is not None:
        (run_dir / "resource_allocation_report.json").write_text(json.dumps(ra))
    if fr is not None:
        (run_dir / "failure_recovery_report.json").write_text(json.dumps(fr))
    return run_dir


def test_list_recent_runs_returns_most_recent_n_sorted(tmp_path):
    for name in ["run_260101_000000_a", "run_260102_000000_b", "run_260103_000000_c"]:
        (tmp_path / name).mkdir()
    dirs = list_recent_runs(n=2, runs_dir=tmp_path)
    assert [d.name for d in dirs] == ["run_260102_000000_b", "run_260103_000000_c"]


def test_list_recent_runs_nonexistent_dir_returns_empty(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert list_recent_runs(n=5, runs_dir=missing) == []


def test_collect_policy_evidence_aggregates_across_runs(tmp_path):
    make_run_dir(
        tmp_path, "run1",
        tp={"valid": True, "score_inconsistent": False},
        ra={"accepted_assignments": [1, 2], "environment_rejected": [1], "agent_rejected": []},
    )
    make_run_dir(
        tmp_path, "run2",
        tp={"valid": False, "score_inconsistent": True},
        ra={"accepted_assignments": [1], "environment_rejected": [], "agent_rejected": [1]},
        fr={"reroute_validation": {"accepted_reroutes": [1], "environment_rejected": []}},
    )
    run_dirs = list_recent_runs(n=10, runs_dir=tmp_path)
    evidence = collect_policy_evidence(run_dirs)

    assert evidence["n_runs_scanned"] == 2
    assert evidence["ranking_valid_rate"] == 0.5  # 1 of 2
    assert evidence["ranking_score_inconsistent_rate"] == 0.5
    assert evidence["allocation_accepted"] == 3  # 2 + 1
    assert evidence["allocation_environment_rejected"] == 1
    assert evidence["allocation_agent_rejected"] == 1
    assert evidence["allocation_acceptance_rate"] == 0.75  # 3 / (3+1)
    assert evidence["reroute_accepted"] == 1
    assert evidence["reroute_acceptance_rate"] == 1.0


def test_collect_policy_evidence_skips_malformed_report(tmp_path):
    run_dir = tmp_path / "run_bad"
    run_dir.mkdir()
    (run_dir / "task_prioritization_report.json").write_text("{not valid json")
    evidence = collect_policy_evidence([run_dir])
    assert evidence["n_runs_scanned"] == 0  # malformed file skipped, not fatal
    assert evidence["ranking_valid_rate"] is None


def test_collect_policy_evidence_empty_history_returns_null_rates():
    evidence = collect_policy_evidence([])
    assert evidence["n_runs_scanned"] == 0
    assert evidence["ranking_valid_rate"] is None
    assert evidence["allocation_acceptance_rate"] is None
    assert evidence["reroute_acceptance_rate"] is None


def test_collect_policy_evidence_ignores_run_with_no_relevant_report(tmp_path):
    empty_run = tmp_path / "run_empty"
    empty_run.mkdir()
    (empty_run / "some_other_file.json").write_text("{}")
    evidence = collect_policy_evidence([empty_run])
    assert evidence["n_runs_scanned"] == 0


def valid_proposal():
    return {"policy_updates": {"slice_capacity": 10}, "evidence": "acceptance rate is low", "recommend_apply": True}


def test_validate_policy_proposal_accepts_valid_proposal():
    assert validate_policy_proposal(valid_proposal()) == []


def test_validate_policy_proposal_accepts_empty_updates_with_recommend_false():
    proposal = {"policy_updates": {}, "evidence": "no change warranted", "recommend_apply": False}
    assert validate_policy_proposal(proposal) == []


def test_validate_policy_proposal_rejects_non_dict():
    assert validate_policy_proposal("nope")


def test_validate_policy_proposal_rejects_unknown_key():
    proposal = valid_proposal()
    proposal["policy_updates"]["made_up_param"] = 5
    errors = validate_policy_proposal(proposal)
    assert any("unknown key" in e for e in errors)


def test_validate_policy_proposal_rejects_non_numeric_value():
    proposal = valid_proposal()
    proposal["policy_updates"]["slice_capacity"] = "ten"
    errors = validate_policy_proposal(proposal)
    assert any("must be a number" in e for e in errors)


def test_validate_policy_proposal_rejects_bool_as_numeric_value():
    """bool is a subclass of int in Python -- isinstance(True, int) is
    True -- so this must be checked explicitly or a bool would silently
    pass as a valid numeric policy value."""
    proposal = valid_proposal()
    proposal["policy_updates"]["slice_capacity"] = True
    errors = validate_policy_proposal(proposal)
    assert any("must be a number" in e for e in errors)


def test_validate_policy_proposal_rejects_missing_evidence():
    proposal = valid_proposal()
    del proposal["evidence"]
    errors = validate_policy_proposal(proposal)
    assert any("evidence" in e for e in errors)


def test_validate_policy_proposal_rejects_non_bool_recommend_apply():
    proposal = valid_proposal()
    proposal["recommend_apply"] = "yes"
    errors = validate_policy_proposal(proposal)
    assert any("recommend_apply" in e for e in errors)


def test_allowed_policy_keys_matches_actual_cli_flags():
    assert ALLOWED_POLICY_KEYS == {"queue_size", "snapshot_window", "slice_capacity"}
