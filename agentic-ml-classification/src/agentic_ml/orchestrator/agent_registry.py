"""
Agent catalog: the fixed, auditable menu of steps the dynamic
orchestrator's planner agent (steps/planner_step.py) can choose from.
Mirrors templates/registry.py's exact shape and reasoning — the
planner's decision surface is "pick from this list and fill in args,"
never "invent an action" — the same way the modeling agent picks a
template_id rather than writing pipeline code.

`required_state` is the precondition gate: a dict of
RunStateSummary.to_planner_dict() keys to the truthiness they must have
before this agent is eligible to run. The planner is told about these
gates, but they are enforced by orchestrator/dynamic_loop.py's
validate_plan() against the REAL state — never trusted from the
planner's own claim that a precondition holds.
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
    required_state: dict[str, bool]
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
            agent_id="intake",
            title="Intake",
            description="Proposes which column is the prediction target and which are "
                        "identifiers/group/time columns, from raw schema facts alone.",
            when_to_use="First step, whenever the target column isn't already known "
                        "(no --target was given).",
            required_state={"target_known": False},
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
            required_state={"target_known": True, "feature_engineering_done": False},
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
