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

Only the two candidate-scoped leakage gates (Track A in the ablation
proposal — cheapest to ablate, since they're already isolated pure
functions with a single call site each) are wired up so far.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationConfig:
    skip_label_permutation_gate: bool = False
    skip_feature_correlation_gate: bool = False

    @property
    def any_active(self) -> bool:
        return self.skip_label_permutation_gate or self.skip_feature_correlation_gate
