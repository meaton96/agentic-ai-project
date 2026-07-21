"""
Split + leakage-check step: deterministic, no LLM. Extracted out of
run_orchestrator.py's inline logic (unchanged there — this module is
new, additive, used only by the dynamic orchestrator) so it's callable
as one unit, the same way every other phase's logic already lives in
steps/*.py rather than only inline in a script.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from agentic_ml.harness.leakage import run_all_split_leakage_checks
from agentic_ml.harness.splits import make_split, resolve_split_columns


@dataclass
class SplitStepResult:
    ok: bool  # True iff every leakage check passed
    strategy_used: Optional[str]
    group_column: Optional[str]
    time_column: Optional[str]
    manifest: Optional[object]
    leakage_checks: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def run_split_step(
    df: pd.DataFrame,
    target_column: str,
    data_hash: str,
    profiler_report: dict,
    group_column: Optional[str],
    time_column: Optional[str],
    seed: int,
    strategy_override: Optional[str] = None,
) -> SplitStepResult:
    strategy = strategy_override or profiler_report["recommended_split_strategy"]
    group_column, time_column, notes = resolve_split_columns(
        strategy, group_column, time_column, profiler_report,
    )
    manifest = make_split(
        df=df, target_column=target_column, data_hash=data_hash,
        strategy=strategy, seed=seed, group_column=group_column, time_column=time_column,
    )
    leakage_checks = run_all_split_leakage_checks(
        df=df, group_column=group_column, time_column=time_column,
        train_idx=manifest.train_idx, val_idx=manifest.val_idx, test_idx=manifest.test_idx,
        strategy=strategy,
    )
    ok = all(c.passed for c in leakage_checks)
    return SplitStepResult(
        ok=ok, strategy_used=strategy, group_column=group_column, time_column=time_column,
        manifest=manifest, leakage_checks=[c.to_dict() for c in leakage_checks], notes=notes,
        errors=[c.check_name for c in leakage_checks if not c.passed],
    )
