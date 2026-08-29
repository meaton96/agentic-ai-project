"""
Tests for environment/oversight.py -- no LLM involved. Covers review
bundle assembly and, most importantly, the verdict-normalization rule:
an unparseable or malformed LLM response must degrade to "flagged",
never silently pass as "approved".
"""
from resource_scheduler.environment.oversight import build_oversight_review_bundle, parse_oversight_verdict


def test_build_oversight_review_bundle_carries_both_proposal_and_evidence():
    proposal = {"policy_updates": {"slice_capacity": 10}, "evidence": "low acceptance rate", "recommend_apply": True}
    evidence = {"n_runs_scanned": 5, "allocation_acceptance_rate": 0.4}
    bundle = build_oversight_review_bundle(proposal, evidence)
    assert bundle["policy_updates"] == {"slice_capacity": 10}
    assert bundle["proposal_evidence_summary"] == "low acceptance rate"
    assert bundle["recommend_apply"] is True
    assert bundle["underlying_evidence"] == evidence


def test_parse_oversight_verdict_accepts_approved():
    result = parse_oversight_verdict({"verdict": "approved", "concerns": [], "reasoning": "looks solid"})
    assert result["verdict"] == "approved"


def test_parse_oversight_verdict_accepts_rejected():
    result = parse_oversight_verdict({"verdict": "rejected", "concerns": ["thin evidence"], "reasoning": "not enough runs"})
    assert result["verdict"] == "rejected"
    assert result["concerns"] == ["thin evidence"]


def test_parse_oversight_verdict_degrades_none_to_flagged():
    result = parse_oversight_verdict(None)
    assert result["verdict"] == "flagged"
    assert result["concerns"]


def test_parse_oversight_verdict_degrades_non_dict_to_flagged():
    result = parse_oversight_verdict("not a dict")
    assert result["verdict"] == "flagged"


def test_parse_oversight_verdict_degrades_unrecognized_verdict_to_flagged():
    """The critical case: an LLM inventing a verdict string outside the
    allowed enum (e.g. "maybe", "approved_with_conditions") must not be
    trusted at face value -- it degrades to flagged, same as no verdict
    at all."""
    result = parse_oversight_verdict({"verdict": "maybe", "concerns": [], "reasoning": "unsure"})
    assert result["verdict"] == "flagged"


def test_parse_oversight_verdict_degrades_missing_verdict_key_to_flagged():
    result = parse_oversight_verdict({"concerns": [], "reasoning": "forgot the verdict field"})
    assert result["verdict"] == "flagged"


def test_parse_oversight_verdict_defaults_missing_concerns_to_empty_list():
    result = parse_oversight_verdict({"verdict": "approved", "reasoning": "fine"})
    assert result["concerns"] == []


def test_parse_oversight_verdict_defaults_non_list_concerns_to_empty_list():
    result = parse_oversight_verdict({"verdict": "approved", "concerns": "not a list", "reasoning": "fine"})
    assert result["concerns"] == []


def test_parse_oversight_verdict_defaults_missing_reasoning_to_empty_string():
    result = parse_oversight_verdict({"verdict": "approved", "concerns": []})
    assert result["reasoning"] == ""
