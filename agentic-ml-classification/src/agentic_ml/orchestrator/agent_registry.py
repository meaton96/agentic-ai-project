"""
Agent catalog: the fixed, auditable menu of steps the dynamic
orchestrator's planner agent (steps/planner_step.py) can choose from.
Mirrors templates/registry.py's exact shape and reasoning — the
planner's decision surface is "pick from this list and fill in args,"
never "invent an action" — the same way the modeling agent picks a
template_id rather than writing pipeline code.

`required_state` is the precondition gate: a dict of
RunStateSummary.to_planner_dict() keys to the value they must have
before this agent is eligible to run — a bool means "must be truthy/
falsy" (checked via `bool(actual) == required_val`), any other value
(e.g. None, or a specific string like "infer_only") means "must equal
this exact value" (checked via `actual == required_val`) — see
orchestrator/dynamic_loop.py::validate_plan. The planner is told about
these gates, but they are enforced there against the REAL state —
never trusted from the planner's own claim that a precondition holds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    title: str
    description: str
    when_to_use: str
    required_state: dict[str, object]  # bool -> truthiness check; anything else -> exact-match check
    arg_schema: dict[str, dict] = field(default_factory=dict)  # arg_name -> {"type": "string", "required": bool}
    requires_capability: Optional[str] = None  # None = always available; else gated by run capabilities

    def to_summary_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "title": self.title,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "required_state": self.required_state,
            "arg_schema": self.arg_schema,
        }


AGENTS: dict[str, AgentSpec] = {
    spec.agent_id: spec
    for spec in [
        AgentSpec(
            agent_id="featurize_timeseries",
            title="Featurize Time-Series",
            description="Deterministic (no LLM): rolls up a raw long-format time-series "
                        "CSV (many rows per example, e.g. one row per sensor reading per "
                        "timestep) into one row per example, using this run's configured "
                        "rollup engine. Neither intake nor any other agent can see this "
                        "dataset's schema until this has run.",
            when_to_use="Whenever the harness has detected this dataset is long-format "
                        "time-series data that hasn't been rolled up yet — always the "
                        "very first step for such a dataset, before intake.",
            required_state={"looks_long_format": True, "featurization_done": False},
        ),
        AgentSpec(
            agent_id="intake",
            title="Intake",
            description="Proposes which column is the prediction target and which are "
                        "identifiers/group/time columns, from raw schema facts alone.",
            when_to_use="First step, whenever the target column isn't already known "
                        "(no --target was given).",
            required_state={"target_known": False, "data_ready": True},
        ),
        AgentSpec(
            agent_id="feature_engineering",
            title="Feature Engineering",
            description="Proposes columns to drop and stateless derived features from a "
                        "vetted catalog, before the profiler/modeling agents see the data. "
                        "It's fine for this agent to propose zero changes if nothing looks "
                        "worth doing — that's a legitimate outcome, not a skipped step.",
            when_to_use="Immediately after the target is known, exactly once, even if it "
                        "ends up proposing no changes — profiler and everything downstream "
                        "need a settled column set to work from.",
            required_state={"target_known": True, "feature_engineering_done": False, "data_ready": True},
        ),
        AgentSpec(
            agent_id="profiler",
            title="Profiler",
            description="Characterizes the dataset (types, missingness, cardinality, "
                        "imbalance, leakage risk) and recommends a split strategy.",
            when_to_use="After feature engineering has run (even if it made no changes) — "
                        "profiling before that would recommend a strategy and flag risks "
                        "against a column set that's about to change.",
            required_state={"target_known": True, "feature_engineering_done": True, "profiler_done": False},
        ),
        AgentSpec(
            agent_id="split_and_check_leakage",
            title="Split + Leakage Checks",
            description="Deterministic (no LLM): splits the data using the profiler's "
                        "recommended strategy and runs the independent leakage checks.",
            when_to_use="Immediately after the profiler, exactly once.",
            required_state={"profiler_done": True, "split_done": False},
        ),
        AgentSpec(
            agent_id="modeling",
            title="Modeling",
            description="Proposes one modeling candidate (template + config), which the "
                        "harness sandbox-builds, fits, scores, and gates behind two "
                        "independent leakage checks. Can be called more than once to try "
                        "several candidates before moving on.",
            when_to_use="After the split passes its leakage checks. Call again for "
                        "another attempt if the current best candidate pool looks weak "
                        "or thin; move on to verification once satisfied.",
            required_state={"split_leakage_passed": True},
        ),
        AgentSpec(
            agent_id="verification",
            title="Verification",
            description="A second, independent LLM review of one gate-passing candidate. "
                        "Can only flag or reject it, never approve/unblock a candidate "
                        "that failed a deterministic gate.",
            when_to_use="Whenever a gate-passing candidate hasn't been reviewed yet. "
                        "Reviews the best-scoring unreviewed candidate by default.",
            required_state={"has_unverified_passing_candidate": True},
            arg_schema={"candidate_id": {"type": "string", "required": False}},
        ),
        AgentSpec(
            agent_id="finalize",
            title="Finalize",
            description="Deterministic (no LLM): refits the best approved/flagged "
                        "candidate on train+val, evaluates it exactly once on the locked "
                        "test set, and persists the model bundle for later deep-dive use.",
            when_to_use="Once at least one candidate has been verified (approved or "
                        "flagged) and you're ready to lock in a result. Touches the test "
                        "set — can only run once per run.",
            required_state={"has_verified_candidate": True, "final_test_metrics_present": False},
        ),
        # --- Phase 9: streaming ingestion + drift-triggered retraining ---
        # Same catalog-and-validator mechanism as every agent above, gated
        # behind requires_capability="streaming" so an ordinary (non-
        # streaming) dynamic-orchestrator run never even sees these three
        # in its catalog — see DynamicRunContext.capabilities() in
        # orchestrator/run_state.py, which only grants "streaming" once
        # ctx.accumulated_df has been established (i.e. after a cold-start
        # finalize has run under scripts/run_streaming_monitor.py).
        AgentSpec(
            agent_id="monitor_drift",
            title="Monitor Drift",
            description="Deterministic (no LLM): compares the newly arrived batch "
                        "against the data the current model was trained on and computes "
                        "a per-feature distribution-shift summary plus (since this is a "
                        "replay of historical, already-labeled data) the batch's own "
                        "performance metrics under the current model.",
            when_to_use="Whenever a new batch has arrived and hasn't been checked for "
                        "drift yet this cycle.",
            required_state={"new_batch_pending": True, "drift_checked": False},
            requires_capability="streaming",
        ),
        AgentSpec(
            agent_id="retrain_decision",
            title="Retrain Decision",
            description="Given the drift summary and how much has accumulated since the "
                        "model was last (re)trained, decides whether to do nothing, run "
                        "inference only on the new batch, or trigger a full retrain. "
                        "Never retrains anything itself — only proposes the action, the "
                        "same way a modeling candidate is only ever a proposal until the "
                        "harness's gates pass it.",
            when_to_use="Immediately after monitor_drift has run for this batch and no "
                        "decision has been made yet.",
            required_state={"drift_checked": True, "pending_retrain_action": None},
            requires_capability="streaming",
        ),
        AgentSpec(
            agent_id="infer_batch",
            title="Infer Batch",
            description="Deterministic (no LLM): scores the new batch with the current "
                        "model and logs the result — no retraining, no test-set exposure.",
            when_to_use="After retrain_decision has chosen infer_only for this batch, and "
                        "only once — running it again would just re-score the same batch.",
            required_state={"pending_retrain_action": "infer_only", "batch_action_completed": False},
            requires_capability="streaming",
        ),
        AgentSpec(
            agent_id="deep_dive",
            title="Deep-Dive",
            description="Explains why one specific already-flagged example was flagged, "
                        "using occlusion attribution and (for aviation flights) flight-"
                        "phase segmentation and cross-cylinder anomaly localization.",
            when_to_use="A different question than classification ('why', not 'is it') — "
                        "only meaningful once a model exists (this run's finalize, or a "
                        "pre-existing model bundle) and only for a specific example.",
            required_state={"model_path_available": True},
            arg_schema={"flight_id": {"type": "string", "required": True}},
            requires_capability="deep_dive",
        ),
        AgentSpec(
            agent_id="summarize",
            title="Summarize",
            description="A plain-language, non-technical summary of the finished run. "
                        "No tools, no decisions — narrates already-computed facts only.",
            when_to_use="After finalize, once — the natural last step before finishing.",
            required_state={"final_test_metrics_present": True, "summary_present": False},
        ),
    ]
}


def get_agent(agent_id: str) -> AgentSpec:
    if agent_id not in AGENTS:
        raise KeyError(f"Unknown agent_id '{agent_id}'. Available: {sorted(AGENTS)}")
    return AGENTS[agent_id]


def list_agent_summaries(capabilities: Optional[set[str]] = None) -> list[dict]:
    """Agents available for this run — excludes any agent whose
    requires_capability isn't in `capabilities` (e.g. deep_dive is
    omitted entirely unless the run was configured with raw-flight-data
    plumbing), the same way --skip-feature-engineering already omits a
    step from the static pipeline."""
    caps = capabilities or set()
    return [
        spec.to_summary_dict() for spec in AGENTS.values()
        if spec.requires_capability is None or spec.requires_capability in caps
    ]
