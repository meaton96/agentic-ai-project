"""
Hand-authored static-orchestrator topology for GET /api/workflow-catalog?type=static.

The static pipeline has no registry object to derive this from (unlike the
dynamic orchestrator's agent_registry.py, which this app's dynamic catalog
route reads directly) — this sequence is fixed and must be updated by hand
if PROJECT_OVERVIEW.md's §3 mermaid diagram ever changes.

Node `kind` follows that diagram's shading convention exactly: "gate" =
shaded/dark (deterministic, no LLM call), "agent" = unshaded/white (an LLM
call). The verification -> modeling loop-back (a rejected candidate falls
back to trying another modeling candidate, per §5.5) is represented as an
edge rather than a synthetic extra node, since it's a decision outcome, not
a step of its own.
"""
from __future__ import annotations


def build_static_topology() -> dict:
    nodes = [
        {
            "id": "intake", "label": "Intake Agent", "kind": "agent",
            "description": "Proposes which column is the prediction target and which are "
                           "identifiers/group/time columns, from raw schema facts alone.",
            "data": {},
        },
        {
            "id": "validate_intake", "label": "Harness validates", "kind": "gate",
            "description": "Deterministic check that intake's proposed DatasetSpec is "
                           "structurally valid before anything downstream sees it.",
            "data": {},
        },
        {
            "id": "feature_engineering", "label": "Feature Engineering Agent", "kind": "agent",
            "description": "Proposes columns to drop and stateless derived features from a "
                           "vetted catalog. Proposing zero changes is a legitimate outcome, "
                           "not a skipped step.",
            "data": {},
        },
        {
            "id": "validate_apply_fe", "label": "Harness validates + applies", "kind": "gate",
            "description": "Deterministic check and application of the feature engineering "
                           "proposal.",
            "data": {},
        },
        {
            "id": "profiler", "label": "Profiler Agent", "kind": "agent",
            "description": "Characterizes the dataset (types, missingness, cardinality, "
                           "imbalance, leakage risk) and recommends a split strategy.",
            "data": {},
        },
        {
            "id": "split_leakage", "label": "Harness: split + leakage checks", "kind": "gate",
            "description": "Deterministic: splits the data using the profiler's recommended "
                           "strategy and runs the independent leakage checks.",
            "data": {},
        },
        {
            "id": "modeling", "label": "Modeling Agent", "kind": "agent",
            "description": "Proposes one modeling candidate (template + config), which the "
                           "harness sandbox-builds, fits, scores, and gates. Can be called "
                           "more than once to try several candidates.",
            "data": {},
        },
        {
            "id": "leakage_gates", "label": "Harness: sandbox build/fit/score + 2 leakage gates",
            "kind": "gate",
            "description": "Sandbox-builds, fits, and scores the candidate; gated behind two "
                           "independent leakage checks before verification ever sees it.",
            "data": {},
        },
        {
            "id": "verification", "label": "Verification Agent", "kind": "agent",
            "description": "A second, independent LLM review of one gate-passing candidate. "
                           "Can only flag or reject it, never approve/unblock a candidate that "
                           "failed a deterministic gate.",
            "data": {"on_reject": "modeling"},
        },
        {
            "id": "finalize", "label": "Harness: refit on train+val, evaluate once", "kind": "gate",
            "description": "Deterministic: refits the best approved/flagged candidate on "
                           "train+val and evaluates it exactly once on the locked test set.",
            "data": {},
        },
        {
            "id": "summary", "label": "Narrated Summary", "kind": "agent",
            "description": "A plain-language, non-technical summary of the finished run. "
                           "Narrates already-computed facts only.",
            "data": {},
        },
    ]
    edges = [
        {"id": "intake->validate_intake", "source": "intake", "target": "validate_intake"},
        {"id": "validate_intake->feature_engineering", "source": "validate_intake", "target": "feature_engineering"},
        {"id": "feature_engineering->validate_apply_fe", "source": "feature_engineering", "target": "validate_apply_fe"},
        {"id": "validate_apply_fe->profiler", "source": "validate_apply_fe", "target": "profiler"},
        {"id": "profiler->split_leakage", "source": "profiler", "target": "split_leakage"},
        {"id": "split_leakage->modeling", "source": "split_leakage", "target": "modeling"},
        {"id": "modeling->leakage_gates", "source": "modeling", "target": "leakage_gates"},
        {"id": "leakage_gates->verification", "source": "leakage_gates", "target": "verification"},
        {"id": "verification->finalize", "source": "verification", "target": "finalize", "label": "approved / flagged"},
        {"id": "verification->modeling", "source": "verification", "target": "modeling",
         "label": "rejected: fall back to next-best candidate"},
        {"id": "finalize->summary", "source": "finalize", "target": "summary"},
    ]
    return {"nodes": nodes, "edges": edges}
