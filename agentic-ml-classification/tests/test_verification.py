"""
Phase 4 verification: the second-opinion audit layer, plus the new
feature-correlation gate added to modeling_step.py. Two things worth
proving here, not just exercising:

1. run_verification_step's verdict handling — including that a
   malformed/unparseable LLM response degrades to "flagged" (proceed,
   but recorded) rather than silently "approved" or a hard "rejected".
2. modeling_step's new feature-correlation gate actually catches a
   raw-feature-copies-the-target leak that label_permutation_test alone
   would NOT reliably catch — see priors/general/leakage_rules.md Rule
   4 for why these two checks are complementary, not redundant. This
   test is the empirical proof of that claim, not just a repeat of it.
"""
import json

import numpy as np
import pandas as pd
import pytest

from agentic_ml.harness.verification import build_review_bundle
from agentic_ml.model_client import ModelResponse
from agentic_ml.steps.modeling_step import run_modeling_step
from agentic_ml.steps.verification_step import run_verification_step

SAMPLE_BUNDLE = build_review_bundle(
    candidate_id="candidate_x",
    template_id="sklearn_mixed_pipeline",
    template_description="Mixed numeric/categorical sklearn pipeline.",
    template_when_to_use="General-purpose default for mixed tabular data.",
    config={"numeric_cols": ["age"], "categorical_cols": ["plan"]},
    explanation="Cheap baseline.",
    metrics={"roc_auc": {"value": 0.82, "ci_low": 0.75, "ci_high": 0.89}},
    label_permutation_check={"check": "label_permutation_test", "passed": True, "detail": "chance-level"},
    feature_correlation_check={"check": "suspicious_feature_correlation", "passed": True, "detail": "clean"},
    profiler_report={"is_imbalanced": False, "class_imbalance_ratio": 0.9, "leakage_risk_flags": []},
)


def test_build_review_bundle_shape():
    assert SAMPLE_BUNDLE["candidate_id"] == "candidate_x"
    assert SAMPLE_BUNDLE["template_id"] == "sklearn_mixed_pipeline"
    assert SAMPLE_BUNDLE["dataset_is_imbalanced"] is False
    assert SAMPLE_BUNDLE["validation_metrics"]["roc_auc"]["value"] == 0.82
    assert SAMPLE_BUNDLE["label_permutation_check"]["passed"] is True
    assert SAMPLE_BUNDLE["feature_correlation_check"]["passed"] is True


def _resp(text=None, tool_calls=None):
    return ModelResponse(
        text=text, tool_calls=tool_calls or [], raw=None, latency_seconds=0.01,
        model="fake-model", input_tokens=1, output_tokens=1,
    )


def _make_fake_client(final_text):
    call_count = {"n": 0}

    class FakeClient:
        def call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _resp(tool_calls=[{"id": "v1", "name": "get_candidate_review_bundle", "arguments": "{}"}])
            return _resp(text=final_text)

    return FakeClient()


def test_run_verification_step_approved():
    client = _make_fake_client(json.dumps({"verdict": "approved", "concerns": [], "reasoning": "looks fine"}))
    result = run_verification_step(SAMPLE_BUNDLE, client)
    assert result.ok
    assert result.verdict == "approved"
    assert result.concerns == []


def test_run_verification_step_flagged():
    client = _make_fake_client(json.dumps({
        "verdict": "flagged", "concerns": ["metric seems high"], "reasoning": "worth a second look",
    }))
    result = run_verification_step(SAMPLE_BUNDLE, client)
    assert result.ok
    assert result.verdict == "flagged"
    assert result.concerns == ["metric seems high"]


def test_run_verification_step_rejected():
    client = _make_fake_client(json.dumps({
        "verdict": "rejected", "concerns": ["explanation doesn't match config"], "reasoning": "mismatch",
    }))
    result = run_verification_step(SAMPLE_BUNDLE, client)
    assert result.ok
    assert result.verdict == "rejected"


def test_run_verification_step_unparseable_defaults_to_flagged():
    """A formatting glitch shouldn't silently promote (approved) or hard-block
    (rejected) a candidate that already passed two deterministic gates — the
    conservative middle ground is "flagged" so a human sees why."""
    client = _make_fake_client("this is not json")
    result = run_verification_step(SAMPLE_BUNDLE, client)
    assert result.ok is False
    assert result.verdict == "flagged"
    assert "did not parse" in result.concerns[0]


def test_run_verification_step_invalid_verdict_value_defaults_to_flagged():
    client = _make_fake_client(json.dumps({"verdict": "maybe", "concerns": [], "reasoning": "unsure"}))
    result = run_verification_step(SAMPLE_BUNDLE, client)
    assert result.ok is False
    assert result.verdict == "flagged"
    assert "invalid verdict" in result.concerns[0]


# --- modeling_step's new feature-correlation gate ---

def _fake_modeling_client(candidate_json):
    call_count = {"n": 0}

    class FakeClient:
        def call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return _resp(tool_calls=[{"id": "1", "name": "get_dataset_profile", "arguments": "{}"}])
            if n == 2:
                return _resp(tool_calls=[{"id": "2", "name": "list_templates", "arguments": "{}"}])
            return _resp(text=candidate_json)

    return FakeClient()


@pytest.fixture
def leaky_df():
    rng = np.random.RandomState(0)
    n = 300
    y = rng.randint(0, 2, size=n)
    return pd.DataFrame({
        "noise": rng.normal(size=n),
        "leak": y,  # near-perfect proxy for the target
        "target": y,
    })


def test_modeling_step_rejects_candidate_selecting_a_leaky_column(leaky_df):
    candidate_json = json.dumps({
        "candidate_id": "candidate_leaky", "template_id": "logistic_numeric",
        "config": {"numeric_cols": ["noise", "leak"]},
        "explanation": "Uses all available numeric columns.",
    })
    train_idx, val_idx = list(range(0, 200)), list(range(200, 300))

    result = run_modeling_step(
        full_df=leaky_df, X=leaky_df[["noise", "leak"]], y=leaky_df["target"],
        target_column="target", group_column=None, time_column=None,
        train_idx=train_idx, val_idx=val_idx, client=_fake_modeling_client(candidate_json),
    )

    assert result.feature_correlation_check is not None
    assert result.feature_correlation_check["passed"] is False
    assert "leak" in result.feature_correlation_check["detail"]
    assert result.ok is False
    assert "failed feature-correlation leakage gate" in result.errors


def test_modeling_step_accepts_candidate_without_leaky_column(leaky_df):
    candidate_json = json.dumps({
        "candidate_id": "candidate_clean", "template_id": "logistic_numeric",
        "config": {"numeric_cols": ["noise"]},
        "explanation": "Uses only the non-leaky numeric column.",
    })
    train_idx, val_idx = list(range(0, 200)), list(range(200, 300))

    result = run_modeling_step(
        full_df=leaky_df, X=leaky_df[["noise", "leak"]], y=leaky_df["target"],
        target_column="target", group_column=None, time_column=None,
        train_idx=train_idx, val_idx=val_idx, client=_fake_modeling_client(candidate_json),
    )

    assert result.feature_correlation_check is not None
    assert result.feature_correlation_check["passed"] is True
    assert result.ok is True
