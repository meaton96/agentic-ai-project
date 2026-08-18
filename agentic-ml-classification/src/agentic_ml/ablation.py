"""
Ablation-study configuration for the harness's leave-one-out rule study
(see docs/harness_pseudocode.md for the full rule inventory this is
meant to eventually cover). Each flag disables exactly one deterministic
check so a research script can measure what gets through without it.

This module is never imported by run_orchestrator.py or
run_dynamic_orchestrator.py. Every flag defaults to False, so an
AblationConfig() — or the implicit `ablation or AblationConfig()` used
at every call site — is identical to current production behavior.
Passing a non-default config is only ever done explicitly, from
scripts/run_ablation_study.py or a test.

Phase 1 covers the two candidate-scoped leakage gates in
modeling_step.py (Track A — cheapest to ablate, already isolated pure
functions with a single call site each) plus the third gate added
alongside them (train-vs-CV consistency, see harness/leakage.py). Phase
1b covers the five structural checks in
harness/feature_engineering.py::validate_feature_proposal. Phase 1c
covers the checks in harness/intake.py::validate_dataset_spec_proposal.
Phase 1d covers the structural checks in
steps/modeling_step.py::run_modeling_step (shape/column/template-config
validation) and harness/sandbox.py (the static AST check and whether a
sandbox build failure is heeded). Phase 1e covers harness/splits.py
(make_split's own validation, and resolve_split_columns' auto-fill
reconciliation logic — the latter is a recovery mechanism, not a reject
gate, so "ablating" it means skipping the reconciliation rather than
skipping an error). Phase 1f covers the four checks
harness/leakage.py::run_all_split_leakage_checks actually calls (the
rule inventory originally listed five, including a split-level
correlation check that does not exist — check_suspicious_feature_
correlation is only ever called from modeling_step.py, already covered
in Phase 1/2). Phase 1g covers orchestrator/dynamic_loop.py: validate_
plan's registry + precondition checks (the latter also being the ONLY
implementation of the "Finalize" one-shot guard — there is no separate
check in steps/finalize_step.py), plus a check added directly to
execute_agent_step's "verification" branch after this study found it
was missing entirely: an explicit args={"candidate_id": ...} bypasses
best_unverified_candidate_id()'s gate-status filter, and nothing
downstream re-checked it — a real, previously-shipped gap, confirmed to
let a gate-failed candidate reach the verification LLM and be approved,
not a hypothetical ablation like everything else in this file.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationConfig:
    # modeling_step.py candidate-scoped leakage gates
    skip_label_permutation_gate: bool = False
    skip_feature_correlation_gate: bool = False
    skip_train_cv_consistency_gate: bool = False

    # harness/feature_engineering.py::validate_feature_proposal checks
    skip_op_id_check: bool = False
    skip_target_column_check: bool = False
    skip_numeric_dtype_check: bool = False
    skip_datetime_dtype_check: bool = False
    skip_protected_drop_check: bool = False

    # harness/intake.py::validate_dataset_spec_proposal checks
    skip_target_existence_check: bool = False
    skip_cardinality_check: bool = False
    skip_group_time_existence_check: bool = False
    skip_group_time_target_collision_check: bool = False
    skip_id_columns_type_check: bool = False
    skip_id_columns_check: bool = False

    # steps/modeling_step.py + harness/sandbox.py structural checks
    skip_candidate_shape_check: bool = False
    skip_candidate_column_check: bool = False
    skip_template_config_check: bool = False
    skip_ast_check: bool = False
    skip_build_error_check: bool = False

    # harness/splits.py checks
    skip_strategy_validity_check: bool = False
    skip_group_required_check: bool = False
    skip_time_required_check: bool = False
    skip_split_column_reconciliation: bool = False

    # harness/leakage.py::run_all_split_leakage_checks
    skip_duplicate_rows_check: bool = False
    skip_split_group_overlap_check: bool = False
    skip_time_ordering_check: bool = False
    skip_split_fold_class_presence_check: bool = False

    # orchestrator/dynamic_loop.py checks
    skip_planner_registry_check: bool = False
    skip_planner_precondition_check: bool = False
    skip_verification_gate_status_check: bool = False

    @property
    def any_active(self) -> bool:
        return any(
            [
                self.skip_label_permutation_gate,
                self.skip_feature_correlation_gate,
                self.skip_train_cv_consistency_gate,
                self.skip_op_id_check,
                self.skip_target_column_check,
                self.skip_numeric_dtype_check,
                self.skip_datetime_dtype_check,
                self.skip_protected_drop_check,
                self.skip_target_existence_check,
                self.skip_cardinality_check,
                self.skip_group_time_existence_check,
                self.skip_group_time_target_collision_check,
                self.skip_id_columns_type_check,
                self.skip_id_columns_check,
                self.skip_candidate_shape_check,
                self.skip_candidate_column_check,
                self.skip_template_config_check,
                self.skip_ast_check,
                self.skip_build_error_check,
                self.skip_strategy_validity_check,
                self.skip_group_required_check,
                self.skip_time_required_check,
                self.skip_split_column_reconciliation,
                self.skip_duplicate_rows_check,
                self.skip_split_group_overlap_check,
                self.skip_time_ordering_check,
                self.skip_split_fold_class_presence_check,
                self.skip_planner_registry_check,
                self.skip_planner_precondition_check,
                self.skip_verification_gate_status_check,
            ]
        )
