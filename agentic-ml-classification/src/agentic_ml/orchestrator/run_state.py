"""
Two kinds of state the dynamic orchestrator (orchestrator/dynamic_loop.py)
threads through its loop, deliberately kept separate:

- RunStateSummary: small, JSON-safe (bools/scalars/short lists only).
  This is what the planner agent's tool (tools/planner_tool.py) returns
  every turn — the ONLY view of run progress it ever gets. No
  dataframes, no fitted pipelines, no raw file paths — same reasoning
  as every other tool binding in this pipeline: the agent gets facts,
  never raw data.
- DynamicRunContext: the harness-side working state (loaded/engineered
  dataframes, the split manifest, full ModelingStepResult objects,
  the accepted pipeline, etc.) that orchestrator/dynamic_loop.py's
  execute_agent_step() actually reads and writes. Never serialized to
  the LLM.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class CandidateSummary:
    candidate_id: str
    template_id: str
    metric_name: str
    metric_value: Optional[float]
    passed_gate: bool
    verification_verdict: Optional[str] = None  # None until verification runs


@dataclass
class RunStateSummary:
    goal: str
    target_known: bool = False
    target_column: Optional[str] = None
    feature_engineering_done: bool = False
    profiler_done: bool = False
    recommended_split_strategy: Optional[str] = None
    split_done: bool = False
    split_leakage_passed: bool = False
    candidates: list[CandidateSummary] = field(default_factory=list)
    final_test_metrics_present: bool = False
    model_path_available: bool = False
    summary_present: bool = False
    deep_dive_completed_flight_ids: list[str] = field(default_factory=list)  # so the
                                        # planner can see it already explained a given
                                        # flight and knows to finish instead of repeating
                                        # deep_dive on the same one — deep_dive has no
                                        # "done" precondition since it's legitimately
                                        # repeatable across DIFFERENT flights in one run
    iteration: int = 0
    last_action: Optional[str] = None  # short human-readable outcome of the previous
                                        # execution, e.g. "modeling: FAILED - failed
                                        # label_permutation_test leakage gate" — gives
                                        # the planner textual feedback beyond just the
                                        # boolean state deltas

    # --- Phase 9: streaming ingestion + drift-triggered retraining ---
    # One outer driver call (scripts/run_streaming_monitor.py) handles ONE
    # incoming batch per run_dynamic_loop invocation; these fields are that
    # batch's per-cycle bookkeeping, reset by the driver each time it pushes
    # a new batch (see push_batch() there) — never touched by ordinary
    # (non-streaming) dynamic-orchestrator runs, where they stay at their
    # defaults for the whole run.
    new_batch_pending: bool = False
    drift_checked: bool = False
    drift_summary: Optional[dict] = None  # small, JSON-safe — see harness/drift.py
    pending_retrain_action: Optional[str] = None  # None | "no_action" | "infer_only" | "retrain"
    batch_action_completed: bool = False  # distinguishes "infer_only was chosen" from
                                        # "infer_batch has now actually run" — without this,
                                        # infer_batch's precondition (pending_retrain_action
                                        # == "infer_only") never goes false on its own once
                                        # satisfied, and a planner could propose it forever
                                        # for the same batch (the same repeatable-agent problem
                                        # deep_dive_completed_flight_ids solves for deep_dive,
                                        # here solved with a plain completion flag since
                                        # infer_batch has no per-call identity to track)
    model_version: int = 0  # bumped by finalize's execute_agent_step branch each time
                             # it runs — v1 after cold start, v(n+1) after each retrain
    n_batches_processed: int = 0

    @property
    def has_unverified_passing_candidate(self) -> bool:
        return any(c.passed_gate and c.verification_verdict is None for c in self.candidates)

    @property
    def has_verified_candidate(self) -> bool:
        return any(c.passed_gate and c.verification_verdict in ("approved", "flagged")
                   for c in self.candidates)

    def best_unverified_candidate_id(self) -> Optional[str]:
        unverified = [c for c in self.candidates if c.passed_gate and c.verification_verdict is None]
        if not unverified:
            return None
        return max(unverified, key=lambda c: (c.metric_value if c.metric_value is not None else -1)).candidate_id

    def best_verified_candidate_id(self) -> Optional[str]:
        verified = [c for c in self.candidates
                   if c.passed_gate and c.verification_verdict in ("approved", "flagged")]
        if not verified:
            return None
        return max(verified, key=lambda c: (c.metric_value if c.metric_value is not None else -1)).candidate_id

    def to_planner_dict(self) -> dict:
        """The exact facts shown to the planner and checked against by
        validate_plan() — every key an AgentSpec.required_state can
        reference must appear here."""
        return {
            "target_known": self.target_known,
            "target_column": self.target_column,
            "feature_engineering_done": self.feature_engineering_done,
            "profiler_done": self.profiler_done,
            "recommended_split_strategy": self.recommended_split_strategy,
            "split_done": self.split_done,
            "split_leakage_passed": self.split_leakage_passed,
            "n_candidates": len(self.candidates),
            "candidates": [asdict(c) for c in self.candidates],
            "has_unverified_passing_candidate": self.has_unverified_passing_candidate,
            "has_verified_candidate": self.has_verified_candidate,
            "final_test_metrics_present": self.final_test_metrics_present,
            "model_path_available": self.model_path_available,
            "summary_present": self.summary_present,
            "deep_dive_completed_flight_ids": self.deep_dive_completed_flight_ids,
            "iteration": self.iteration,
            "last_action": self.last_action,
            "new_batch_pending": self.new_batch_pending,
            "drift_checked": self.drift_checked,
            "drift_summary": self.drift_summary,
            "pending_retrain_action": self.pending_retrain_action,
            "batch_action_completed": self.batch_action_completed,
            "model_version": self.model_version,
            "n_batches_processed": self.n_batches_processed,
        }


class DynamicRunContext:
    """Harness-side working state. Plain mutable bag, not a dataclass —
    execute_agent_step() reads/writes these fields directly as each
    agent runs; nothing here is ever shown to an LLM."""

    def __init__(self, data_path: str, goal: str, seed: int = 42,
                target_column: Optional[str] = None,
                group_column: Optional[str] = None,
                time_column: Optional[str] = None,
                id_columns: Optional[list[str]] = None,
                strategy_override: Optional[str] = None,
                metric_names: Optional[list[str]] = None,
                raw_csv: Optional[str] = None,
                features_csv: Optional[str] = None,
                run_id: str = "run"):
        self.data_path = data_path
        self.goal = goal
        self.seed = seed
        self.run_id = run_id

        # populated as intake/feature-engineering run (or pre-seeded via --target)
        self.target_column = target_column
        self.group_column = group_column
        self.time_column = time_column
        self.id_columns = id_columns or []
        self.strategy_override = strategy_override
        self.metric_names = metric_names or ["roc_auc", "pr_auc", "f1", "accuracy"]

        self.raw_df = None            # set on first load
        self.raw_data_hash = None
        self.engineered_df = None     # raw_df, or FE-augmented if FE ran
        self.final_id_columns: list[str] = []

        self.profiler_report: Optional[dict] = None
        self.strategy_used: Optional[str] = None
        self.manifest = None          # split_manifest.SplitManifest-like object
        self.loaded_X = None
        self.loaded_y = None
        self.loaded = None            # LoadedDataset, for X/y access post-split

        self.modeling_results: dict[str, object] = {}   # candidate_id -> ModelingStepResult
        self.tried_template_ids: list[str] = []

        self.final_pipeline = None
        self.final_test_metrics: Optional[dict] = None
        self.model_path: Optional[str] = None
        self.model_feature_columns: Optional[list[str]] = None
        self.model_background = None

        self.summary_text: Optional[str] = None

        # deep-dive support: either populated by this run's own finalize,
        # or pre-seeded from --existing-model for an explain-only task
        self.raw_csv = raw_csv
        self.features_csv = features_csv
        self.deep_dive_results: dict[str, object] = {}  # flight_id -> DeepDiveStepResult

        # --- Phase 9: streaming ingestion + drift-triggered retraining ---
        # accumulated_df is None until the first finalize (cold start or
        # otherwise) establishes it from engineered_df — that's also what
        # capabilities() below uses to decide whether the streaming-only
        # catalog entries (monitor_drift/retrain_decision/infer_batch) are
        # even visible to the planner, so an ordinary classification run
        # never sees them. Once set, it always holds ALL data seen so far
        # (baseline pool + every batch processed), regardless of whether
        # each batch triggered a retrain — see dynamic_loop.py's
        # retrain_decision branch, the one place that grows it.
        self.accumulated_df: Optional[object] = None
        self.pending_batch_df: Optional[object] = None  # set by the streaming driver
                                                          # before each run_dynamic_loop call
        self.last_train_accumulated_len: int = 0  # len(accumulated_df) as of the CURRENT
                                                    # model's last (re)training — the
                                                    # prefix of accumulated_df monitor_drift
                                                    # treats as "what the model was trained on"
        self.n_batches_since_last_train: int = 0
        self.model_version: int = 0
        self.model_history: list[dict] = []  # one entry per finalize: version, model_path,
                                              # test_metrics, n_training_examples
        self.drift_report: Optional[dict] = None  # most recent monitor_drift output
        self.batch_inference_log: list[dict] = []  # one entry per infer_batch execution

    def capabilities(self) -> set[str]:
        caps = set()
        if self.raw_csv and self.features_csv:
            caps.add("deep_dive")
        if self.accumulated_df is not None:
            caps.add("streaming")
        return caps
