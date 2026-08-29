#!/usr/bin/env python3
"""
Ablation study, Phase 1: leave-one-out testing on the two candidate-
scoped leakage gates in modeling_step.py (label_permutation_test,
check_suspicious_feature_correlation). Working first pass of the study
proposed alongside docs/harness_pseudocode.md — see that file's rule
inventory and phasing plan for what isn't covered yet.

Deliberately does not touch templates/registry.py or any production
entry point. All fixtures live only in this script (Scenario 1 reuses
the leaky_df pattern from tests/test_verification.py).

Scenario 1 — content leak: a candidate selects a raw column that is a
near-perfect copy of the target. Runs through the real production path
(run_modeling_step) with a stubbed model client, under four ablation
configs. The two gates should behave asymmetrically here: this is a
raw-feature leak, so only the correlation gate catches it.

Scenario 2 — process leak: a hand-built encoder whose fit() ignores the
labels it's given and instead recomputes its mapping from the full
(train+val) dataset every time — the textbook "preprocessing fit
outside its proper training fold" bug. The leaking column is a
moderate-cardinality categorical id, so its raw values are non-numeric
strings the correlation gate structurally cannot inspect (pd.to_numeric
coerces the whole column to NaN and the check skips it entirely) — but
empirically, the permutation gate does NOT reliably catch it either: it
refits the WHOLE pipeline including the downstream classifier, which
legitimately relearns from whatever labels it's given each permutation
round and washes the leak's exploitability out (confirmed: mean
shuffled AUC = 1 - real AUC, a coin-flip on coefficient sign, not a
reliable catch). This is a genuine gap, not a fixture bug — see
Scenario 3 for what the permutation gate DOES reliably catch.

Scenario 3 — bypass leak: a classifier whose predict_proba() answers
from a closed-over reference to the true labels regardless of what
fit() was called with, modeling a prediction pathway that bypasses the
nominal training process entirely (e.g. a feature/join that already
baked in outcome-derived information through a channel fit(X, y) never
sees). Unlike Scenario 2, NOTHING downstream legitimately depends on
the labels fit() receives, so permutation cleanly catches it — this is
the actual shape of leak this gate's implementation is sensitive to.

Usage:
    python scripts/run_ablation_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_ml.ablation import AblationConfig
from agentic_ml.harness.dataset import DatasetSpec, LoadedDataset
from agentic_ml.harness.feature_engineering import apply_feature_op, validate_feature_proposal
from agentic_ml.harness.intake import validate_dataset_spec_proposal
from agentic_ml.harness.leakage import (
    check_fold_class_presence,
    check_group_overlap,
    check_suspicious_feature_correlation,
    label_permutation_test,
    run_all_split_leakage_checks,
)
from agentic_ml.harness.metrics import roc_auc_any
from agentic_ml.harness.profiler import profile_dataset
from agentic_ml.harness.sandbox import run_candidate_build
from agentic_ml.harness.splits import make_split, resolve_split_columns
from agentic_ml.model_client import ModelResponse
from agentic_ml.steps.modeling_step import run_modeling_step

RESULTS_DIR = Path(__file__).resolve().parent.parent / "runs" / "ablation"

ABLATION_CONFIGS = {
    "both_gates_on (baseline)": AblationConfig(),
    "permutation_gate_off": AblationConfig(skip_label_permutation_gate=True),
    "correlation_gate_off": AblationConfig(skip_feature_correlation_gate=True),
    "both_gates_off": AblationConfig(skip_label_permutation_gate=True, skip_feature_correlation_gate=True),
}


# --------------------------------------------------------------------------
# Scenario 1 — content leak, through the real production path
# --------------------------------------------------------------------------

class _FakeClient:
    """Deterministic stub, no real model call — matches this project's
    testing standard (CLAUDE.md: never call a real endpoint from tests)."""

    def __init__(self, candidate_json: str):
        self._candidate_json = candidate_json
        self._n = 0

    def call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
        self._n += 1
        if self._n == 1:
            return ModelResponse(text=None, tool_calls=[{"id": "1", "name": "get_dataset_profile", "arguments": "{}"}],
                                  raw=None, latency_seconds=0.0, model="fake", input_tokens=0, output_tokens=0)
        if self._n == 2:
            return ModelResponse(text=None, tool_calls=[{"id": "2", "name": "list_templates", "arguments": "{}"}],
                                  raw=None, latency_seconds=0.0, model="fake", input_tokens=0, output_tokens=0)
        return ModelResponse(text=self._candidate_json, tool_calls=[],
                              raw=None, latency_seconds=0.0, model="fake", input_tokens=0, output_tokens=0)


def run_scenario1() -> list[dict]:
    rng = np.random.RandomState(0)
    n = 300
    y = rng.randint(0, 2, size=n)
    df = pd.DataFrame({"noise": rng.normal(size=n), "leak": y, "target": y})

    candidate_json = json.dumps({
        "candidate_id": "candidate_leaky", "template_id": "logistic_numeric",
        "config": {"numeric_cols": ["noise", "leak"]},
        "explanation": "Uses all available numeric columns.",
    })
    train_idx, val_idx = list(range(0, 200)), list(range(200, 300))

    rows = []
    for name, ablation in ABLATION_CONFIGS.items():
        result = run_modeling_step(
            full_df=df, X=df[["noise", "leak"]], y=df["target"],
            target_column="target", group_column=None, time_column=None,
            train_idx=train_idx, val_idx=val_idx,
            client=_FakeClient(candidate_json), ablation=ablation,
        )
        rows.append({
            "scenario": "1_content_leak (raw near-duplicate column)",
            "ablation": name,
            "candidate_accepted": result.ok,
            "permutation_passed": result.label_permutation_check["passed"],
            "correlation_passed": result.feature_correlation_check["passed"],
            "val_roc_auc": round(result.metrics["roc_auc"]["value"], 4) if result.metrics else None,
        })
    return rows


# --------------------------------------------------------------------------
# Scenario 2 — process leak: encoder that ignores its own training fold
# --------------------------------------------------------------------------

class _LeakyMeanEncoder(BaseEstimator, TransformerMixin):
    """Deliberately broken, and a realistic bug rather than a contrived
    one: fit() ignores the (X, y) it's actually given and always
    recomputes its category -> mean(y) mapping from the FULL dataset —
    the classic 'encoder fit before/outside the proper split' mistake.
    Not a real project template; used only in this scenario, never
    registered in templates/registry.py."""

    def __init__(self, full_entity_ref: pd.Series, full_y_ref: pd.Series):
        self.full_entity_ref = full_entity_ref
        self.full_y_ref = full_y_ref

    def fit(self, X, y=None):
        merged = pd.DataFrame({"entity_id": self.full_entity_ref, "_y": self.full_y_ref})
        self.mapping_ = merged.groupby("entity_id")["_y"].mean().to_dict()
        self.global_mean_ = float(self.full_y_ref.mean())
        return self

    def transform(self, X):
        col = X["entity_id"] if hasattr(X, "columns") else pd.Series(np.ravel(X))
        return col.map(self.mapping_).fillna(self.global_mean_).to_numpy(dtype=float).reshape(-1, 1)


def run_scenario2() -> list[dict]:
    rng = np.random.RandomState(1)
    n, n_categories = 300, 75  # ~4 rows/category -> strong self-leakage signal, still not a bare id column
    y = rng.randint(0, 2, size=n)  # pure noise, no real relationship to entity_id
    entity_id = pd.Series(rng.choice([f"ent_{i}" for i in range(n_categories)], size=n), name="entity_id")
    df = pd.DataFrame({"entity_id": entity_id, "target": y})
    X, y = df[["entity_id"]], df["target"]
    train_idx, val_idx = list(range(0, 200)), list(range(200, 300))

    rows = []
    for name, ablation in ABLATION_CONFIGS.items():
        encoder = _LeakyMeanEncoder(full_entity_ref=df["entity_id"], full_y_ref=y)
        pipeline = Pipeline([
            ("encode", ColumnTransformer([("cat", encoder, ["entity_id"])])),
            ("clf", LogisticRegression(max_iter=1000)),
        ])

        def fit_and_score(X_tr, y_tr, X_va, y_va, _pipeline=pipeline):
            candidate = clone(_pipeline)
            candidate.fit(X_tr, y_tr)
            proba = candidate.predict_proba(X_va)
            return roc_auc_any(y_va, proba)

        real_val_auc = fit_and_score(X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx], y.iloc[val_idx])

        if ablation.skip_label_permutation_gate:
            perm_passed, perm_detail = True, "SKIPPED (ablation)"
        else:
            perm_check = label_permutation_test(
                fit_and_score, X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx], y.iloc[val_idx],
                metric_name="roc_auc", seed=42,
            )
            perm_passed, perm_detail = perm_check.passed, perm_check.detail

        if ablation.skip_feature_correlation_gate:
            corr_passed, corr_detail = True, "SKIPPED (ablation)"
        else:
            corr_check = check_suspicious_feature_correlation(X[["entity_id"]].iloc[train_idx], y.iloc[train_idx])
            corr_passed, corr_detail = corr_check.passed, corr_check.detail

        rows.append({
            "scenario": "2_process_leak (encoder fit outside its training fold)",
            "ablation": name,
            "candidate_accepted": perm_passed and corr_passed,
            "permutation_passed": perm_passed,
            "correlation_passed": corr_passed,
            "val_roc_auc": round(real_val_auc, 4),
            "permutation_detail": perm_detail,
            "correlation_detail": corr_detail,
        })
    return rows


# --------------------------------------------------------------------------
# Scenario 3 — a prediction pathway that bypasses fit() entirely
# --------------------------------------------------------------------------
# Scenario 2 established that an intermediate component (an encoder) that
# ignores its own y_train does NOT reliably get caught, because the
# downstream classifier is still legitimately refit on whatever labels it's
# given, which washes out the leak's exploitability under permutation
# (confirmed empirically: mean shuffled AUC = 1 - real AUC, i.e. the fitted
# coefficient's sign flips — not a reliable catch). This scenario asks: what
# DOES the permutation gate reliably catch? Empirically, only a leak where
# the FULL pipeline's output is independent of the y_train it receives at
# every stage, not just one intermediate step — modeled here as a
# classifier whose predict_proba() answers from a closed-over reference to
# the true labels regardless of what fit() was called with. This stands in
# for a real bug class: a prediction pathway that bypasses the nominal
# training process entirely (e.g. a feature/join that already baked in
# outcome-derived information through a channel fit(X, y) never sees).

class _GroundTruthLookupClassifier(BaseEstimator, ClassifierMixin):
    """Deliberately broken: fit() is a no-op with respect to its y
    argument; predict_proba() always answers from a closed-over reference
    to the true labels. Not a real project template; used only in this
    scenario."""

    def __init__(self, true_y_lookup: pd.Series):
        self.true_y_lookup = true_y_lookup

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        idx = X.index if hasattr(X, "index") else range(len(X))
        p1 = self.true_y_lookup.loc[idx].to_numpy(dtype=float)
        p1 = np.clip(p1 * 0.9 + 0.05, 0.01, 0.99)  # soften off exact 0/1
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def run_scenario3() -> list[dict]:
    rng = np.random.RandomState(4)
    n = 300
    y = pd.Series(rng.randint(0, 2, size=n))
    X = pd.DataFrame({"noise": rng.normal(size=n)})  # uninformative — the leak is not in any raw column
    train_idx, val_idx = list(range(200)), list(range(200, 300))

    rows = []
    for name, ablation in ABLATION_CONFIGS.items():
        clf = _GroundTruthLookupClassifier(true_y_lookup=y)

        def fit_and_score(X_tr, y_tr, X_va, y_va, _clf=clf):
            candidate = clone(_clf)
            candidate.fit(X_tr, y_tr)
            proba = candidate.predict_proba(X_va)
            return roc_auc_any(y_va, proba)

        real_val_auc = fit_and_score(X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx], y.iloc[val_idx])

        if ablation.skip_label_permutation_gate:
            perm_passed, perm_detail = True, "SKIPPED (ablation)"
        else:
            perm_check = label_permutation_test(
                fit_and_score, X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx], y.iloc[val_idx],
                metric_name="roc_auc", seed=42,
            )
            perm_passed, perm_detail = perm_check.passed, perm_check.detail

        if ablation.skip_feature_correlation_gate:
            corr_passed, corr_detail = True, "SKIPPED (ablation)"
        else:
            corr_check = check_suspicious_feature_correlation(X.iloc[train_idx], y.iloc[train_idx])
            corr_passed, corr_detail = corr_check.passed, corr_check.detail

        rows.append({
            "scenario": "3_bypass_leak (prediction pathway independent of fit's y)",
            "ablation": name,
            "candidate_accepted": perm_passed and corr_passed,
            "permutation_passed": perm_passed,
            "correlation_passed": corr_passed,
            "val_roc_auc": round(real_val_auc, 4),
            "permutation_detail": perm_detail,
            "correlation_detail": corr_detail,
        })
    return rows


# --------------------------------------------------------------------------
# Phase 1b — Feature Engineering: leave-one-out on each of the five
# structural checks in validate_feature_proposal, per docs/harness_pseudocode.md.
# For each rule: does disabling it let a bad proposal through, and what
# actually happens if that proposal is then applied for real?
# --------------------------------------------------------------------------

def _fe_fixture():
    df = pd.DataFrame({
        "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "category": ["a", "b", "a", "b", "a", "b"],
        "signup_date": pd.to_datetime(
            ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01", "2020-06-01"]
        ),
        "customer_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
        "target": [0, 1, 0, 1, 0, 1],
    })
    profile = profile_dataset(df, target_column="target").to_dict()
    return df, profile


FE_RULE_SCENARIOS = {
    "op_id_check": dict(
        ablation_field="skip_op_id_check",
        proposal={"derived_features": [{"op_id": "not_a_real_op", "params": {"col_a": "amount", "col_b": "amount"}}]},
    ),
    "target_column_check": dict(
        ablation_field="skip_target_column_check",
        proposal={"derived_features": [{"op_id": "ratio", "params": {"col_a": "target", "col_b": "amount"}}]},
    ),
    "numeric_dtype_check": dict(
        ablation_field="skip_numeric_dtype_check",
        proposal={"derived_features": [{"op_id": "ratio", "params": {"col_a": "category", "col_b": "amount"}}]},
    ),
    "datetime_dtype_check": dict(
        ablation_field="skip_datetime_dtype_check",
        proposal={"derived_features": [{"op_id": "datetime_parts", "params": {"col": "amount", "parts": ["year"]}}]},
    ),
    "protected_drop_check": dict(
        ablation_field="skip_protected_drop_check",
        proposal={"drop_columns": ["customer_id"]},  # customer_id is the group_column below
    ),
}


def _apply_and_observe(df, proposal) -> str:
    """Best-effort: actually apply what validation would have blocked, and
    report what really happens — a clean exception, silent garbage, or a
    downstream failure — rather than assuming "reject" and "accept" are
    the only two outcomes."""
    try:
        result_df = df.copy()
        for feat in proposal.get("derived_features", []):
            result_df, new_names = apply_feature_op(result_df, feat["op_id"], feat.get("params") or {})
            for name in new_names:
                corr = pd.to_numeric(result_df[name], errors="coerce").corr(pd.to_numeric(result_df["target"], errors="coerce"))
                return f"applied without error -> new column '{name}' (values: {result_df[name].tolist()}, corr with target: {corr:.3f})" if corr == corr else \
                       f"applied without error -> new column '{name}' (values: {result_df[name].tolist()})"
        if "drop_columns" in proposal:
            dropped = proposal["drop_columns"]
            result_df = result_df.drop(columns=dropped)
            try:
                make_split(result_df, target_column="target", data_hash="x", strategy="group",
                           group_column="customer_id", val_frac=0.34, test_frac=0.34, seed=0)
                return "dropped column, downstream group-split unexpectedly still succeeded"
            except Exception as e:
                return f"dropped column -> downstream make_split(strategy='group') failed: {type(e).__name__}: {e}"
        return "applied without error, no observable effect"
    except Exception as e:
        return f"CRASHED at apply-time: {type(e).__name__}: {e}"


def run_fe_scenarios() -> list[dict]:
    df, profile = _fe_fixture()
    rows = []
    for rule_name, spec in FE_RULE_SCENARIOS.items():
        for ablation_on in (False, True):
            ablation = AblationConfig(**{spec["ablation_field"]: ablation_on})
            errors = validate_feature_proposal(
                profile, target_column="target", group_column="customer_id", time_column="signup_date",
                proposal=spec["proposal"], ablation=ablation,
            )
            rejected = bool(errors)
            row = {
                "rule": rule_name,
                "ablation": "rule_on (baseline)" if not ablation_on else "rule_off",
                "rejected_at_validation": rejected,
                "validation_errors": errors,
            }
            if not rejected:
                row["what_actually_happens"] = _apply_and_observe(df, spec["proposal"])
            rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Phase 1c — Intake: leave-one-out on each of the six checks in
# validate_dataset_spec_proposal, per docs/harness_pseudocode.md /
# docs/rule_effects.md. Downstream consequences here turned out to vary
# a lot more than Feature Engineering's did — some are severe (an empty
# training set), some are structurally backstopped by an unrelated
# split-level check, and some are near-total no-ops because
# LoadedDataset.X already filters defensively. Each fixture below was
# verified interactively before being written down here, not assumed.
# --------------------------------------------------------------------------

def _intake_df():
    rng = np.random.RandomState(0)
    n = 200
    return pd.DataFrame({
        "x": rng.normal(size=n),
        "target": rng.randint(0, 2, size=n),
        "customer_id": rng.choice([f"c{i}" for i in range(50)], size=n),
    })


def run_intake_scenarios() -> list[dict]:
    df = _intake_df()
    rows = []

    def record(rule, ablation_on, rejected, errors, downstream):
        rows.append({
            "rule": rule,
            "ablation": "rule_on (baseline)" if not ablation_on else "rule_off",
            "rejected_at_validation": rejected,
            "validation_errors": errors,
            "what_actually_happens": downstream if not rejected else "-- correctly rejected --",
        })

    # 1. target existence check: propose a hallucinated target_column.
    # With the check disabled, the crash happens INSIDE validate_dataset_
    # spec_proposal itself (df[target] a few lines past where the check
    # used to short-circuit) — not something a caller downstream has any
    # chance to guard against separately.
    for ablation_on in (False, True):
        ablation = AblationConfig(skip_target_existence_check=ablation_on)
        proposal = {"target_column": "not_a_real_column"}
        try:
            errors = validate_dataset_spec_proposal(df, proposal, ablation=ablation)
            downstream = "-"
            rejected = bool(errors)
        except Exception as e:
            errors, downstream, rejected = [], f"CRASHED inside validate_dataset_spec_proposal itself: {type(e).__name__}: {e}", False
        record("target_existence_check", ablation_on, rejected, errors, downstream)

    # 2a. cardinality check, lower bound: a single-class target
    df_single = df.assign(target=1)
    for ablation_on in (False, True):
        ablation = AblationConfig(skip_cardinality_check=ablation_on)
        proposal = {"target_column": "target"}
        errors = validate_dataset_spec_proposal(df_single, proposal, ablation=ablation)
        downstream = "-"
        if not errors:
            m = make_split(df_single, target_column="target", data_hash="x", strategy="stratified",
                            val_frac=0.2, test_frac=0.2, seed=0)
            check = check_fold_class_presence(df_single["target"], m.train_idx, m.val_idx, m.test_idx)
            downstream = (
                f"split-level check_fold_class_presence still catches it: passed={check.passed} "
                f"({check.detail})"
            )
        record("cardinality_check (lower bound, n=1)", ablation_on, bool(errors), errors, downstream)

    # 2b. cardinality check, upper bound: a 30-class target (> MAX_CLASSES=20)
    df_wide = df.assign(target=np.random.RandomState(1).choice(range(30), size=len(df)))
    for ablation_on in (False, True):
        ablation = AblationConfig(skip_cardinality_check=ablation_on)
        proposal = {"target_column": "target"}
        errors = validate_dataset_spec_proposal(df_wide, proposal, ablation=ablation)
        downstream = "-"
        if not errors:
            m = make_split(df_wide, target_column="target", data_hash="x", strategy="stratified",
                            val_frac=0.2, test_frac=0.2, seed=0)
            check = check_fold_class_presence(df_wide["target"], m.train_idx, m.val_idx, m.test_idx)
            downstream = (
                f"split accepted (no crash); check_fold_class_presence passed={check.passed} — "
                "downstream multiclass metric computation grows more fragile to class/fold "
                "mismatches as cardinality rises relative to fold size (observed a ValueError "
                "in an adjacent probe with thin per-class val counts, not reproduced deterministically here)"
            )
        record("cardinality_check (upper bound, n=30)", ablation_on, bool(errors), errors, downstream)

    # 3. group/time existence check: propose a hallucinated group_column
    for ablation_on in (False, True):
        ablation = AblationConfig(skip_group_time_existence_check=ablation_on)
        proposal = {"target_column": "target", "group_column": "not_a_real_column"}
        errors = validate_dataset_spec_proposal(df, proposal, ablation=ablation)
        downstream = "-"
        if not errors:
            try:
                make_split(df, target_column="target", data_hash="x", strategy="group",
                           group_column=proposal["group_column"], val_frac=0.2, test_frac=0.2, seed=0)
                downstream = "no error (unexpected)"
            except Exception as e:
                downstream = f"CRASHED: {type(e).__name__}: {e}"
        record("group_time_existence_check", ablation_on, bool(errors), errors, downstream)

    # 4. group/time vs target collision check: propose group_column == target_column
    for ablation_on in (False, True):
        ablation = AblationConfig(skip_group_time_target_collision_check=ablation_on)
        proposal = {"target_column": "target", "group_column": "target"}
        errors = validate_dataset_spec_proposal(df, proposal, ablation=ablation)
        downstream = "-"
        if not errors:
            m = make_split(df, target_column="target", data_hash="x", strategy="group",
                            group_column="target", val_frac=0.2, test_frac=0.2, seed=0)
            fcp = check_fold_class_presence(df["target"], m.train_idx, m.val_idx, m.test_idx)
            overlap = check_group_overlap(df, "target", m.train_idx, m.val_idx, m.test_idx)
            downstream = (
                f"train fold size={len(m.train_idx)} (of {len(df)} rows) — "
                f"check_group_overlap passed={overlap.passed} (misses this entirely); "
                f"check_fold_class_presence passed={fcp.passed} ({fcp.detail})"
            )
        record("group_time_target_collision_check", ablation_on, bool(errors), errors, downstream)

    # 5. id_columns type check: propose id_columns as a STRING, not a list
    for ablation_on in (False, True):
        ablation = AblationConfig(skip_id_columns_type_check=ablation_on)
        proposal = {"target_column": "target", "id_columns": "xt"}  # 'x' is a real feature column
        errors = validate_dataset_spec_proposal(df, proposal, ablation=ablation)
        downstream = "-"
        if not errors:
            spec = DatasetSpec(path="fixture", target_column="target", id_columns=proposal["id_columns"])
            loaded = LoadedDataset(df=df, spec=spec, data_hash="x")
            downstream = (
                f"'{proposal['id_columns']}' unpacked character-by-character (Python string "
                f"iteration) into ['x', 't']; X columns = {list(loaded.X.columns)} — the genuine "
                "numeric feature 'x' silently vanishes because it happens to match a character, "
                "with no error at any layer"
            )
        record("id_columns_type_check", ablation_on, bool(errors), errors, downstream)

    # 6. id_columns existence/target-collision check: hallucinated + target-colliding entries
    for ablation_on in (False, True):
        ablation = AblationConfig(skip_id_columns_check=ablation_on)
        proposal = {"target_column": "target", "id_columns": ["not_a_real_column", "target"]}
        errors = validate_dataset_spec_proposal(df, proposal, ablation=ablation)
        downstream = "-"
        if not errors:
            spec = DatasetSpec(path="fixture", target_column="target", id_columns=proposal["id_columns"])
            loaded = LoadedDataset(df=df, spec=spec, data_hash="x")
            downstream = (
                f"X columns = {list(loaded.X.columns)}, y name = {loaded.y.name} — "
                "LoadedDataset.X's own 'if c in df.columns' filter already absorbs both faults; "
                "functionally a no-op, only the clean error message is lost"
            )
        record("id_columns_check (existence + target-collision)", ablation_on, bool(errors), errors, downstream)

    return rows


# --------------------------------------------------------------------------
# Phase 1d — Modeling Candidate structural checks: shape, columns,
# template/config, the static AST check, and whether a sandbox build
# failure is heeded. The AST check is qualitatively different from
# everything else in this study — it's the first fixture that
# demonstrates containment of actively adversarial code, not a
# statistical leak. The fault template used to test it only proves
# capability (reads its own sandboxed cwd) and never touches anything
# outside the subprocess's own tempdir.
# --------------------------------------------------------------------------

def _modeling_df():
    rng = np.random.RandomState(0)
    n = 300
    return pd.DataFrame({
        "age": rng.normal(40, 10, n),
        "plan": rng.choice(["a", "b"], size=n),
        "row_num": range(n),          # numeric, unique per row -- classic ID column
        "target": rng.randint(0, 2, n),
    })


def run_modeling_structural_scenarios() -> list[dict]:
    df = _modeling_df()
    train_idx, val_idx = list(range(200)), list(range(200, 300))
    rows = []

    def record(rule, ablation_on, ok, errors, downstream):
        rows.append({
            "rule": rule,
            "ablation": "rule_on (baseline)" if not ablation_on else "rule_off",
            "candidate_ok": ok,
            "errors": errors,
            "what_actually_happens": downstream,
        })

    def run(candidate_json, ablation, X_cols):
        try:
            r = run_modeling_step(
                full_df=df, X=df[X_cols], y=df["target"], target_column="target",
                group_column=None, time_column=None, train_idx=train_idx, val_idx=val_idx,
                client=_FakeClient(candidate_json), ablation=ablation,
            )
            if r.errors:
                return r.ok, r.errors, "-- correctly rejected --"
            return r.ok, r.errors, "no error (unexpected)"
        except Exception as e:
            return False, [], f"CRASHED: {type(e).__name__}: {e}"

    # 1. shape check: candidate missing 'template_id' entirely
    bad_shape = json.dumps({"candidate_id": "c1", "config": {"numeric_cols": ["age"]}})
    for ablation_on in (False, True):
        ok, errors, downstream = run(bad_shape, AblationConfig(skip_candidate_shape_check=ablation_on), ["age"])
        record("shape_check", ablation_on, ok, errors, downstream)

    # 2. column check, sub-case A: candidate selects the TARGET column as a feature
    leak_target = json.dumps({
        "candidate_id": "c1", "template_id": "logistic_numeric",
        "config": {"numeric_cols": ["age", "target"]}, "explanation": "x",
    })
    for ablation_on in (False, True):
        ok, errors, downstream = run(leak_target, AblationConfig(skip_candidate_column_check=ablation_on), ["age", "target"])
        if ablation_on and not ok and errors:
            downstream = "still caught downstream by the feature-correlation leakage gate (corr=1.0)"
        record("column_check (target as feature)", ablation_on, ok, errors, downstream)

    # 2b. column check, sub-case B: candidate selects a likely-ID column (row_num)
    leak_id = json.dumps({
        "candidate_id": "c1", "template_id": "logistic_numeric",
        "config": {"numeric_cols": ["age", "row_num"]}, "explanation": "x",
    })
    for ablation_on in (False, True):
        ok, errors, downstream = run(leak_id, AblationConfig(skip_candidate_column_check=ablation_on), ["age", "row_num"])
        if ablation_on and ok:
            downstream = "ACCEPTED with no errors -- unlike the target-column case, nothing downstream flags an uncorrelated ID column"
        record("column_check (likely-ID column as feature)", ablation_on, ok, errors, downstream)

    # 3. template/config check, sub-case A: unknown template_id
    unknown_template = json.dumps({
        "candidate_id": "c1", "template_id": "not_a_real_template", "config": {"numeric_cols": ["age"]},
    })
    for ablation_on in (False, True):
        ok, errors, downstream = run(unknown_template, AblationConfig(skip_template_config_check=ablation_on), ["age"])
        record("template_config_check (unknown template_id)", ablation_on, ok, errors, downstream)

    # 3b. template/config check, sub-case B: valid template, missing a required config key
    missing_key = json.dumps({
        "candidate_id": "c1", "template_id": "sklearn_mixed_pipeline", "config": {"numeric_cols": ["age"]},
    })
    for ablation_on in (False, True):
        ok, errors, downstream = run(missing_key, AblationConfig(skip_template_config_check=ablation_on), ["age", "plan"])
        if ablation_on:
            downstream = "backstopped -- sandbox catches the resulting KeyError and converts it to a normal fail(), not a crash"
        record("template_config_check (missing required config key)", ablation_on, ok, errors, downstream)

    # 4. static AST check: a template that imports os and proves it works,
    # entirely safely (only reads its own sandboxed tempdir, never writes
    # or reads anything outside it, never touches the network).
    malicious_source = (
        "import os\n\n"
        "def build_pipeline(config):\n"
        "    raise RuntimeError('forbidden import executed; os.getcwd()=' + os.getcwd())\n"
    )
    for ablation_on in (False, True):
        pipeline, err = run_candidate_build(malicious_source, {}, timeout_seconds=10,
                                             ablation=AblationConfig(skip_ast_check=ablation_on))
        rejected = pipeline is None and err is not None and err.startswith("static check failed")
        downstream = err if not rejected else "-"
        rows.append({
            "rule": "ast_check (forbidden import)",
            "ablation": "rule_on (baseline)" if not ablation_on else "rule_off",
            "candidate_ok": pipeline is not None,
            "errors": [err] if err else [],
            "what_actually_happens": downstream if downstream != "-" else "-- correctly blocked pre-execution --",
        })

    # 5. build_error_check: does the caller heed a sandbox build failure?
    # (template/config check also disabled here so the broken config
    # actually reaches the sandbox instead of being caught earlier.)
    for ablation_on in (False, True):
        ok, errors, downstream = run(
            missing_key,
            AblationConfig(skip_template_config_check=True, skip_build_error_check=ablation_on),
            ["age", "plan"],
        )
        record("build_error_check", ablation_on, ok, errors, downstream)

    return rows


# --------------------------------------------------------------------------
# Phase 1e — Split Resolution: make_split's own 3 checks, plus
# resolve_split_columns' auto-fill reconciliation (a recovery mechanism,
# not a reject gate -- ablating it means skipping the auto-fill, not
# skipping an error).
# --------------------------------------------------------------------------

def run_split_resolution_scenarios() -> list[dict]:
    rng = np.random.RandomState(0)
    n = 100
    df = pd.DataFrame({"x": rng.normal(size=n), "target": rng.randint(0, 2, size=n)})
    rows = []

    def record(rule, ablation_on, downstream):
        rows.append({
            "rule": rule,
            "ablation": "rule_on (baseline)" if not ablation_on else "rule_off",
            "what_actually_happens": downstream,
        })

    # 1. strategy validity check
    for ablation_on in (False, True):
        ablation = AblationConfig(skip_strategy_validity_check=ablation_on)
        try:
            make_split(df, target_column="target", data_hash="x", strategy="bogus_strategy",
                       val_frac=0.2, test_frac=0.2, seed=0, ablation=ablation)
            downstream = "no error (unexpected)"
        except Exception as e:
            downstream = f"-- correctly rejected: {type(e).__name__}: {e}" if not ablation_on else \
                f"CRASHED, different exception type/message than the normal check: {type(e).__name__}: {e!r}"
        record("strategy_validity_check", ablation_on, downstream)

    # 2. group_column required check
    for ablation_on in (False, True):
        ablation = AblationConfig(skip_group_required_check=ablation_on)
        try:
            make_split(df, target_column="target", data_hash="x", strategy="group", group_column=None,
                       val_frac=0.2, test_frac=0.2, seed=0, ablation=ablation)
            downstream = "no error (unexpected)"
        except Exception as e:
            downstream = f"-- correctly rejected: {type(e).__name__}: {e}" if not ablation_on else \
                f"CRASHED with a confusing exception (not the clear ValueError message): {type(e).__name__}: {e!r}"
        record("group_required_check", ablation_on, downstream)

    # 3. time_column required check
    for ablation_on in (False, True):
        ablation = AblationConfig(skip_time_required_check=ablation_on)
        try:
            make_split(df, target_column="target", data_hash="x", strategy="time", time_column=None,
                       val_frac=0.2, test_frac=0.2, seed=0, ablation=ablation)
            downstream = "no error (unexpected)"
        except Exception as e:
            downstream = f"-- correctly rejected: {type(e).__name__}: {e}" if not ablation_on else \
                f"CRASHED with a confusing exception (not the clear ValueError message): {type(e).__name__}: {e!r}"
        record("time_required_check", ablation_on, downstream)

    # 4. split-column reconciliation (recovery mechanism, not a gate):
    # profiler detects a likely group column, but neither intake nor the
    # CLI declared one.
    profile = {"likely_group_columns": ["cust_id"], "likely_datetime_columns": []}
    for ablation_on in (False, True):
        ablation = AblationConfig(skip_split_column_reconciliation=ablation_on)
        gc, tc, notes = resolve_split_columns("group", None, None, profile, ablation=ablation)
        if gc is not None:
            downstream = f"auto-filled group_column='{gc}' -- split proceeds transparently ({notes[0]})"
        else:
            try:
                make_split(df, target_column="target", data_hash="x", strategy="group", group_column=gc,
                           val_frac=0.2, test_frac=0.2, seed=0)
                downstream = "no error (unexpected)"
            except Exception as e:
                downstream = (
                    f"no safety loss -- make_split's own check still catches it cleanly: "
                    f"{type(e).__name__}: {e}"
                )
        record("split_column_reconciliation", ablation_on, downstream)

    return rows


# --------------------------------------------------------------------------
# Phase 1f — Split-Level Leakage Checks: the four checks
# run_all_split_leakage_checks actually calls. The rule inventory
# originally listed five, including a split-level correlation check
# that does not exist -- check_suspicious_feature_correlation is only
# ever called from modeling_step.py (candidate-scoped, Phase 1/2).
# --------------------------------------------------------------------------

def run_split_leakage_scenarios() -> list[dict]:
    rows = []

    def record(rule, ablation_on, check, other_fail, downstream=""):
        rows.append({
            "rule": rule,
            "ablation": "rule_on (baseline)" if not ablation_on else "rule_off",
            "check_passed": check.passed,
            "check_detail": check.detail,
            "other_checks_that_also_failed": other_fail,
            "what_actually_happens": downstream,
        })

    # 1. duplicate rows across splits: exact same row content in train AND val
    rng = np.random.RandomState(0)
    df1 = pd.DataFrame({"x": [1.0] * 2 + list(rng.normal(size=98)), "target": [0, 0] + list(rng.randint(0, 2, size=98))})
    train_idx1 = [0] + list(range(2, 70))
    val_idx1 = [1] + list(range(70, 85))  # index 1 duplicates index 0's row content exactly
    test_idx1 = list(range(85, 100))
    for ablation_on in (False, True):
        checks = run_all_split_leakage_checks(df1, "target", None, None, train_idx1, val_idx1, test_idx1,
                                               strategy="random", ablation=AblationConfig(skip_duplicate_rows_check=ablation_on))
        dup = next(c for c in checks if c.check_name == "duplicate_rows_across_splits")
        other_fail = [c.check_name for c in checks if not c.passed and c.check_name != "duplicate_rows_across_splits"]
        downstream = "no cross-check redundancy -- a genuine gap" if ablation_on and dup.passed else ""
        record("duplicate_rows_check", ablation_on, dup, other_fail, downstream)

    # 2. group overlap: group 'g1' (in train) also placed at the start of val,
    # with genuinely distinct x/target values -- no accidental exact-row duplicate
    n2 = 90
    df2 = pd.DataFrame({"grp": ["g1", "g2", "g3"] * 30, "x": rng.normal(size=n2), "target": rng.randint(0, 2, size=n2)})
    train_idx2, val_idx2, test_idx2 = list(range(0, 60)), list(range(60, 75)), list(range(75, 90))
    df2.loc[val_idx2[0], "grp"] = "g1"
    df2.loc[val_idx2[1], "grp"] = "g1"
    for ablation_on in (False, True):
        checks = run_all_split_leakage_checks(df2, "target", "grp", None, train_idx2, val_idx2, test_idx2,
                                               strategy="random", ablation=AblationConfig(skip_split_group_overlap_check=ablation_on))
        go = next(c for c in checks if c.check_name == "group_overlap")
        other_fail = [c.check_name for c in checks if not c.passed and c.check_name != "group_overlap"]
        downstream = "no cross-check redundancy -- a genuine gap" if ablation_on and go.passed else ""
        record("group_overlap_check", ablation_on, go, other_fail, downstream)

    # 3. time ordering: 'train' deliberately given the LATEST dates, 'val' the
    # EARLIEST -- violates the 'time' strategy's global ordering promise
    n3 = 90
    dates = pd.date_range("2020-01-01", periods=n3, freq="D")
    df3 = pd.DataFrame({"t": dates, "x": rng.normal(size=n3), "target": rng.randint(0, 2, size=n3)})
    train_idx3, val_idx3, test_idx3 = list(range(60, 90)), list(range(0, 30)), list(range(30, 60))
    for ablation_on in (False, True):
        checks = run_all_split_leakage_checks(df3, "target", None, "t", train_idx3, val_idx3, test_idx3,
                                               strategy="time", ablation=AblationConfig(skip_time_ordering_check=ablation_on))
        to = next(c for c in checks if c.check_name == "time_ordering")
        other_fail = [c.check_name for c in checks if not c.passed and c.check_name != "time_ordering"]
        downstream = "no cross-check redundancy -- a genuine gap" if ablation_on and to.passed else ""
        record("time_ordering_check", ablation_on, to, other_fail, downstream)

    # 4. fold class presence: val+test folds are single-class (all 0)
    n4 = 90
    df4 = pd.DataFrame({"x": rng.normal(size=n4), "target": [0] * n4})
    df4.loc[0:59, "target"] = rng.randint(0, 2, size=60)
    train_idx4, val_idx4, test_idx4 = list(range(0, 60)), list(range(60, 75)), list(range(75, 90))
    for ablation_on in (False, True):
        checks = run_all_split_leakage_checks(df4, "target", None, None, train_idx4, val_idx4, test_idx4,
                                               strategy="random", ablation=AblationConfig(skip_split_fold_class_presence_check=ablation_on))
        fc = next(c for c in checks if c.check_name == "fold_class_presence")
        other_fail = [c.check_name for c in checks if not c.passed and c.check_name != "fold_class_presence"]
        downstream = ""
        if ablation_on and fc.passed:
            # verify the emergent downstream backstop: NaN comparison semantics
            X4, y4 = df4[["x"]], df4["target"]
            clf = LogisticRegression()

            def fit_and_score(X_tr, y_tr, X_va, y_va, _clf=clf):
                c = clone(_clf); c.fit(X_tr, y_tr)
                return roc_auc_any(y_va, c.predict_proba(X_va))

            perm = label_permutation_test(fit_and_score, X4.iloc[train_idx4], y4.iloc[train_idx4],
                                           X4.iloc[val_idx4], y4.iloc[val_idx4], metric_name="roc_auc", seed=42)
            downstream = (
                f"a single-class val fold reaching modeling produces score=NaN; the downstream "
                f"permutation gate rejects it anyway (passed={perm.passed}), but only as an "
                f"EMERGENT side effect of 'NaN <= threshold' always being False in Python -- not "
                f"a designed backstop for this specific fault"
            )
        record("fold_class_presence_check", ablation_on, fc, other_fail, downstream)

    return rows


# --------------------------------------------------------------------------
# Phase 1g — Dynamic Planner + Verification + Finalize. "Finalize"
# doesn't have an independent check at all -- steps/finalize_step.py has
# no internal guard against being called twice; the one-shot promise is
# entirely validate_plan's required_state precondition on the "finalize"
# registry entry. And this phase found something none of the prior ones
# did: a REAL, currently-shipped gap, not a hypothetical ablation. An
# explicit args={"candidate_id": ...} in a planner proposal bypasses
# best_unverified_candidate_id()'s gate-status filter entirely, and
# nothing downstream re-checked it before this study — confirmed to let
# a gate-failed candidate reach the verification LLM and be approved.
# Fixed directly in orchestrator/dynamic_loop.py; the fix itself is what
# skip_verification_gate_status_check ablates back off, to keep this
# finding reproducible in the same before/after shape as everything else.
# --------------------------------------------------------------------------

VERIFICATION_APPROVED_JSON = json.dumps({"verdict": "approved", "concerns": [], "reasoning": "looks fine to me"})


def run_dynamic_planner_scenarios() -> list[dict]:
    from types import SimpleNamespace

    from agentic_ml.orchestrator.dynamic_loop import execute_agent_step, validate_plan
    from agentic_ml.orchestrator.run_state import CandidateSummary, DynamicRunContext, RunStateSummary

    rows = []

    def record(rule, ablation_on, downstream):
        rows.append({
            "rule": rule,
            "ablation": "rule_on (baseline)" if not ablation_on else "rule_off",
            "what_actually_happens": downstream,
        })

    # 1. planner registry check: hallucinated agent_id
    state1 = RunStateSummary(goal="predict x")
    proposal1 = {"action": "run_agent", "agent_id": "not_a_real_agent", "args": {}, "reasoning": "x"}
    for ablation_on in (False, True):
        try:
            errors = validate_plan(proposal1, state1, capabilities=set(), ablation=AblationConfig(skip_planner_registry_check=ablation_on))
            downstream = "-- correctly rejected --" if errors else "no error (unexpected)"
        except Exception as e:
            downstream = f"CRASHED inside validate_plan itself: {type(e).__name__}: {e}"
        record("planner_registry_check", ablation_on, downstream)

    # 2. planner precondition check: propose 'finalize' a second time, after
    # the test set has already been touched once this run. Note: this IS
    # the Finalize one-shot guard -- steps/finalize_step.py has no check of
    # its own; the entire guarantee lives in this precondition.
    state2 = RunStateSummary(goal="predict x")
    state2.candidates = [CandidateSummary(candidate_id="c1", template_id="logistic_numeric",
                                           metric_name="roc_auc", metric_value=0.9, passed_gate=True,
                                           verification_verdict="approved")]
    state2.final_test_metrics_present = True
    proposal2 = {"action": "run_agent", "agent_id": "finalize", "args": {}, "reasoning": "lock it in again"}
    for ablation_on in (False, True):
        errors = validate_plan(proposal2, state2, capabilities=set(), ablation=AblationConfig(skip_planner_precondition_check=ablation_on))
        downstream = "-- correctly rejected -- this IS the Finalize one-shot guard" if errors else \
            "no error -- the proposal would proceed to execute_agent_step, which re-runs run_finalize_step " \
            "and re-touches the test set (finalize_step.py has no guard of its own)"
        record("planner_precondition_check (= Finalize one-shot guard)", ablation_on, downstream)

    # 3. verification gate-status check: explicit candidate_id targeting a
    # candidate that FAILED its harness gates -- a real, previously-shipped
    # gap this study found (see module docstring), not a hypothetical.
    def _make_fixtures():
        passed = SimpleNamespace(candidate_id="c_passed", template_id="logistic_numeric",
                                  config={"numeric_cols": ["age"]}, explanation="ok",
                                  metrics={"roc_auc": {"value": 0.7}},
                                  label_permutation_check={"passed": True, "detail": "ok", "check": "x"},
                                  feature_correlation_check={"passed": True, "detail": "ok", "check": "y"},
                                  pipeline=None, ok=True)
        failed = SimpleNamespace(candidate_id="c_failed", template_id="logistic_numeric",
                                  config={"numeric_cols": ["age", "leak"]}, explanation="uses a leaky column",
                                  metrics={"roc_auc": {"value": 0.99}},
                                  label_permutation_check={"passed": True, "detail": "ok", "check": "x"},
                                  feature_correlation_check={"passed": False, "detail": "leak corr=1.0", "check": "y"},
                                  pipeline=None, ok=False)
        state = RunStateSummary(goal="predict x")
        state.candidates = [
            CandidateSummary(candidate_id="c_passed", template_id="logistic_numeric", metric_name="roc_auc", metric_value=0.7, passed_gate=True),
            CandidateSummary(candidate_id="c_failed", template_id="logistic_numeric", metric_name="roc_auc", metric_value=0.99, passed_gate=False),
        ]
        ctx = DynamicRunContext(data_path="x", goal="predict x")
        ctx.modeling_results = {"c_passed": passed, "c_failed": failed}
        ctx.profiler_report = {"is_imbalanced": False, "class_imbalance_ratio": 0.9, "leakage_risk_flags": []}
        ctx.run_id = "test"
        ctx.loaded = SimpleNamespace(data_hash="fakehash")
        return state, ctx

    class _VerificationApprovesEverythingClient:
        """Verification is a single-turn, no-tools call -- unlike
        _FakeClient (built for modeling's 2-tool-call pattern), this just
        returns 'approved' unconditionally."""
        def call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
            return ModelResponse(text=VERIFICATION_APPROVED_JSON, tool_calls=[], raw=None,
                                  latency_seconds=0.0, model="fake", input_tokens=0, output_tokens=0)

    for ablation_on in (False, True):
        state3, ctx3 = _make_fixtures()
        ok, errors = execute_agent_step(
            "verification", {"candidate_id": "c_failed"}, ctx3, state3, _VerificationApprovesEverythingClient(),
            None, None, None, lambda name, msgs: None, on_event=None,
            ablation=AblationConfig(skip_verification_gate_status_check=ablation_on),
        )
        failed_summary = next(c for c in state3.candidates if c.candidate_id == "c_failed")
        if ablation_on:
            downstream = (
                f"NOT blocked -- the gate-failed candidate's verification_verdict is now "
                f"'{failed_summary.verification_verdict}'. This is the real gap: reproduces the "
                f"exact bug this study found before it was fixed."
            )
        else:
            downstream = f"-- correctly rejected: {errors[0] if errors else ''}"
        record("verification_gate_status_check", ablation_on, downstream)

    return rows


# --------------------------------------------------------------------------

def main():
    rows = run_scenario1() + run_scenario2() + run_scenario3()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "leakage_gates_phase1.json"
    out_path.write_text(json.dumps(rows, indent=2))

    header = f"{'scenario':45} {'ablation':28} {'accepted':9} {'perm_ok':8} {'corr_ok':8} {'val_auc':8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['scenario']:45} {r['ablation']:28} {str(r['candidate_accepted']):9} "
              f"{str(r['permutation_passed']):8} {str(r['correlation_passed']):8} {str(r.get('val_roc_auc')):8}")
    print(f"\nWrote {len(rows)} results to {out_path}")

    fe_rows = run_fe_scenarios()
    fe_out_path = RESULTS_DIR / "feature_engineering_phase1b.json"
    fe_out_path.write_text(json.dumps(fe_rows, indent=2))

    print(f"\n{'rule':25} {'ablation':20} {'rejected':10} what_actually_happens")
    print("-" * 110)
    for r in fe_rows:
        happens = r.get("what_actually_happens", "-- correctly rejected, nothing applied --")
        print(f"{r['rule']:25} {r['ablation']:20} {str(r['rejected_at_validation']):10} {happens}")
    print(f"\nWrote {len(fe_rows)} results to {fe_out_path}")

    intake_rows = run_intake_scenarios()
    intake_out_path = RESULTS_DIR / "intake_phase1c.json"
    intake_out_path.write_text(json.dumps(intake_rows, indent=2))

    print(f"\n{'rule':45} {'ablation':20} {'rejected':10} what_actually_happens")
    print("-" * 130)
    for r in intake_rows:
        print(f"{r['rule']:45} {r['ablation']:20} {str(r['rejected_at_validation']):10} {r['what_actually_happens']}")
    print(f"\nWrote {len(intake_rows)} results to {intake_out_path}")

    modeling_rows = run_modeling_structural_scenarios()
    modeling_out_path = RESULTS_DIR / "modeling_structural_phase1d.json"
    modeling_out_path.write_text(json.dumps(modeling_rows, indent=2))

    print(f"\n{'rule':50} {'ablation':20} {'ok':7} what_actually_happens")
    print("-" * 140)
    for r in modeling_rows:
        print(f"{r['rule']:50} {r['ablation']:20} {str(r['candidate_ok']):7} {r['what_actually_happens']}")
    print(f"\nWrote {len(modeling_rows)} results to {modeling_out_path}")

    split_rows = run_split_resolution_scenarios()
    split_out_path = RESULTS_DIR / "split_resolution_phase1e.json"
    split_out_path.write_text(json.dumps(split_rows, indent=2))

    print(f"\n{'rule':30} {'ablation':20} what_actually_happens")
    print("-" * 120)
    for r in split_rows:
        print(f"{r['rule']:30} {r['ablation']:20} {r['what_actually_happens']}")
    print(f"\nWrote {len(split_rows)} results to {split_out_path}")

    leakage_rows = run_split_leakage_scenarios()
    leakage_out_path = RESULTS_DIR / "split_leakage_phase1f.json"
    leakage_out_path.write_text(json.dumps(leakage_rows, indent=2))

    print(f"\n{'rule':28} {'ablation':20} {'passed':8} {'other_fail':22} what_actually_happens")
    print("-" * 140)
    for r in leakage_rows:
        print(f"{r['rule']:28} {r['ablation']:20} {str(r['check_passed']):8} "
              f"{str(r['other_checks_that_also_failed']):22} {r['what_actually_happens']}")
    print(f"\nWrote {len(leakage_rows)} results to {leakage_out_path}")

    planner_rows = run_dynamic_planner_scenarios()
    planner_out_path = RESULTS_DIR / "dynamic_planner_phase1g.json"
    planner_out_path.write_text(json.dumps(planner_rows, indent=2))

    print(f"\n{'rule':55} {'ablation':20} what_actually_happens")
    print("-" * 140)
    for r in planner_rows:
        print(f"{r['rule']:55} {r['ablation']:20} {r['what_actually_happens']}")
    print(f"\nWrote {len(planner_rows)} results to {planner_out_path}")


if __name__ == "__main__":
    main()
