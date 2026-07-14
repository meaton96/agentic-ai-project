"""
Phase 4 verification support: pure, deterministic assembly of the
review bundle the VerificationAgent sees. No LLM involvement here —
this module only shapes facts that other harness code already
computed (the modeling step's leakage checks, the profiler's report,
the template registry's metadata) into one dict.

Design note: the bundle is deliberately assembled AFTER a candidate has
already passed both of modeling_step.py's deterministic leakage gates
(label_permutation_test + check_suspicious_feature_correlation). The
VerificationAgent that reviews this bundle never sees a candidate that
failed one of those — it is a second, independent audit layer on top
of the harness's own gates, not a replacement for them.
"""
from __future__ import annotations


def build_review_bundle(
    candidate_id: str,
    template_id: str,
    template_description: str,
    template_when_to_use: str,
    config: dict,
    explanation: str,
    metrics: dict,
    label_permutation_check: dict,
    feature_correlation_check: dict,
    profiler_report: dict,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "template_id": template_id,
        "template_description": template_description,
        "template_when_to_use": template_when_to_use,
        "config": config,
        "explanation": explanation,
        "validation_metrics": metrics,
        "label_permutation_check": label_permutation_check,
        "feature_correlation_check": feature_correlation_check,
        "dataset_is_imbalanced": profiler_report.get("is_imbalanced"),
        "dataset_class_imbalance_ratio": profiler_report.get("class_imbalance_ratio"),
        "dataset_leakage_risk_flags": profiler_report.get("leakage_risk_flags"),
    }
