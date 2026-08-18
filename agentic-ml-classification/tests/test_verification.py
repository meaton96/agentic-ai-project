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

from agentic_ml.ablation import AblationConfig
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

def _fake_modeling_client(candidate_json, max_tokens_seen=None):
    call_count = {"n": 0}

    class FakeClient:
        def call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
            call_count["n"] += 1
            n = call_count["n"]
            if max_tokens_seen is not None:
                max_tokens_seen.append(max_tokens)
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
    # errors now carries the check's numeric detail appended, not just the
    # bare gate name (see modeling_step.py) — a real diagnosability gap
    # this closes: state.last_action / history[i]["errors"] are the only
    # place a failure's specifics reach a caller that doesn't wire
    # on_event (e.g. a notebook), and the bare name alone wasn't enough to
    # tell a genuine leak apart from small-sample noise.
    matching_errors = [e for e in result.errors if e.startswith("failed feature-correlation leakage gate")]
    assert matching_errors
    assert "leak" in matching_errors[0]


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


# --- ablation: does disabling one leakage gate let the leaky candidate through? ---

def _run_leaky_candidate(leaky_df, ablation=None):
    candidate_json = json.dumps({
        "candidate_id": "candidate_leaky", "template_id": "logistic_numeric",
        "config": {"numeric_cols": ["noise", "leak"]},
        "explanation": "Uses all available numeric columns.",
    })
    train_idx, val_idx = list(range(0, 200)), list(range(200, 300))
    return run_modeling_step(
        full_df=leaky_df, X=leaky_df[["noise", "leak"]], y=leaky_df["target"],
        target_column="target", group_column=None, time_column=None,
        train_idx=train_idx, val_idx=val_idx, client=_fake_modeling_client(candidate_json),
        ablation=ablation,
    )


def test_ablation_none_matches_default_behavior(leaky_df):
    """ablation=None must be byte-for-byte the same as omitting the
    argument entirely — every existing call site that doesn't pass
    ablation= is relying on this."""
    result = _run_leaky_candidate(leaky_df, ablation=None)
    assert result.ok is False
    assert result.feature_correlation_check["passed"] is False


def test_ablation_disabling_correlation_gate_lets_the_leak_through(leaky_df):
    """The permutation gate alone does NOT catch this fixture (shuffling
    labels makes the proxy column just as useless as noise — see the
    complementary-gates claim in harness_constraints.md §Step 5). With
    only the correlation gate disabled, nothing left is checking the
    columns this candidate actually selected, so the leaky candidate is
    accepted."""
    result = _run_leaky_candidate(leaky_df, ablation=AblationConfig(skip_feature_correlation_gate=True))
    assert result.feature_correlation_check["passed"] is True
    assert "SKIPPED" in result.feature_correlation_check["detail"]
    assert result.label_permutation_check["passed"] is True  # unaffected, still runs for real
    assert result.ok is True  # the leak is now silently accepted


def test_ablation_disabling_permutation_gate_still_catches_this_leak(leaky_df):
    """The correlation gate alone is sufficient for THIS fixture (a raw
    near-duplicate column) — disabling only the permutation gate should
    not change the outcome, demonstrating the two gates are not
    symmetric for every fault: this one is only a Track-A (content)
    leak, not a process leak."""
    result = _run_leaky_candidate(leaky_df, ablation=AblationConfig(skip_label_permutation_gate=True))
    assert result.label_permutation_check["passed"] is True
    assert "SKIPPED" in result.label_permutation_check["detail"]
    assert result.feature_correlation_check["passed"] is False  # unaffected, still runs for real
    assert result.ok is False  # still correctly rejected


def test_modeling_step_requests_a_larger_token_budget_than_the_default(leaky_df):
    """Real incident: a candidate enumerating many individual derived-stat
    columns for a wide rolled-up table got cut off mid-column-name at
    exactly 1024 output tokens (ModelClient.call's own default) — modeling
    is uniquely at risk of this among the agents here (numeric_cols/
    categorical_cols can legitimately need to list many names), so it now
    requests a much larger budget than ToolCallingAgent's own default."""
    candidate_json = json.dumps({
        "candidate_id": "candidate_clean", "template_id": "logistic_numeric",
        "config": {"numeric_cols": ["noise"]},
        "explanation": "Uses only the non-leaky numeric column.",
    })
    train_idx, val_idx = list(range(0, 200)), list(range(200, 300))
    max_tokens_seen = []

    run_modeling_step(
        full_df=leaky_df, X=leaky_df[["noise", "leak"]], y=leaky_df["target"],
        target_column="target", group_column=None, time_column=None,
        train_idx=train_idx, val_idx=val_idx,
        client=_fake_modeling_client(candidate_json, max_tokens_seen=max_tokens_seen),
    )

    assert max_tokens_seen  # the fake client was actually called
    assert all(mt > 1024 for mt in max_tokens_seen)


# --- ablation: Modeling Candidate structural checks (Phase 1d) ---
# See docs/ablation_study_report.md and scripts/run_ablation_study.py
# for the full study; these lock in the most significant findings.

def test_ablation_disabling_column_check_lets_an_id_column_through_undetected():
    """Unlike selecting the target column as a feature (caught downstream
    by the correlation gate, corr=1.0), a likely-ID column carries no
    real correlation with the target and slips through every leakage
    gate completely undetected once the column check is disabled — a
    genuine gap, not backstopped anywhere."""
    rng = np.random.RandomState(0)
    n = 300
    df = pd.DataFrame({
        "age": rng.normal(40, 10, n),
        "row_num": range(n),  # numeric, unique per row -- classic ID column
        "target": rng.randint(0, 2, n),
    })
    train_idx, val_idx = list(range(200)), list(range(200, 300))
    candidate_json = json.dumps({
        "candidate_id": "c1", "template_id": "logistic_numeric",
        "config": {"numeric_cols": ["age", "row_num"]}, "explanation": "x",
    })

    baseline = run_modeling_step(
        full_df=df, X=df[["age", "row_num"]], y=df["target"], target_column="target",
        group_column=None, time_column=None, train_idx=train_idx, val_idx=val_idx,
        client=_fake_modeling_client(candidate_json),
    )
    assert baseline.ok is False
    assert any("likely-ID" in e for e in baseline.errors)

    ablated = run_modeling_step(
        full_df=df, X=df[["age", "row_num"]], y=df["target"], target_column="target",
        group_column=None, time_column=None, train_idx=train_idx, val_idx=val_idx,
        client=_fake_modeling_client(candidate_json),
        ablation=AblationConfig(skip_candidate_column_check=True),
    )
    assert ablated.ok is True, "an uncorrelated ID column has nothing for any leakage gate to catch"
    assert ablated.errors == []


def test_ablation_ast_check_disabled_lets_forbidden_import_execute():
    """A template importing os is blocked pre-execution when the AST
    check is active; disabled, it genuinely executes inside the
    sandboxed subprocess (proven by it correctly reporting its own
    tempdir cwd) -- containment of adversarial code, not a statistical
    leak, and qualitatively different from every other rule in this
    study. The fault template only proves capability; it never touches
    anything outside its own subprocess tempdir."""
    from agentic_ml.harness.sandbox import run_candidate_build

    malicious_source = (
        "import os\n\n"
        "def build_pipeline(config):\n"
        "    raise RuntimeError('forbidden import executed; os.getcwd()=' + os.getcwd())\n"
    )

    pipeline, err = run_candidate_build(malicious_source, {}, timeout_seconds=10)
    assert pipeline is None
    assert err.startswith("static check failed")
    assert "forbidden import: os" in err

    pipeline, err = run_candidate_build(
        malicious_source, {}, timeout_seconds=10, ablation=AblationConfig(skip_ast_check=True),
    )
    assert pipeline is None  # the candidate raises rather than returning a pipeline
    assert "forbidden import executed" in err
    assert "os.getcwd()=" in err  # proof the import genuinely worked


def test_ablation_ignoring_build_error_crashes_on_none_pipeline():
    """If both the template-config check and the build-error check are
    disabled, a candidate with a genuinely broken config reaches
    pipeline.fit() with pipeline=None and crashes with AttributeError --
    the deliberate consequence of not heeding a sandbox failure."""
    rng = np.random.RandomState(0)
    n = 300
    df = pd.DataFrame({
        "age": rng.normal(40, 10, n), "plan": rng.choice(["a", "b"], size=n),
        "target": rng.randint(0, 2, n),
    })
    train_idx, val_idx = list(range(200)), list(range(200, 300))
    # sklearn_mixed_pipeline requires categorical_cols; omitted here
    missing_key = json.dumps({
        "candidate_id": "c1", "template_id": "sklearn_mixed_pipeline", "config": {"numeric_cols": ["age"]},
    })

    with pytest.raises(AttributeError):
        run_modeling_step(
            full_df=df, X=df[["age", "plan"]], y=df["target"], target_column="target",
            group_column=None, time_column=None, train_idx=train_idx, val_idx=val_idx,
            client=_fake_modeling_client(missing_key),
            ablation=AblationConfig(skip_template_config_check=True, skip_build_error_check=True),
        )
