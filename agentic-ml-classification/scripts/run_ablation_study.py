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
strings the correlation gate structurally cannot inspect
(pd.to_numeric coerces the whole column to NaN and the check skips it
entirely) — only the permutation gate can catch this one. This is the
asymmetric counterpart Scenario 1 doesn't exercise.

Usage:
    python scripts/run_ablation_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_ml.ablation import AblationConfig
from agentic_ml.harness.leakage import check_suspicious_feature_correlation, label_permutation_test
from agentic_ml.harness.metrics import roc_auc_any
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

def main():
    rows = run_scenario1() + run_scenario2()

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


if __name__ == "__main__":
    main()
