"""
Phase 9 streaming-ingestion simulator: deterministic, no LLM. Splits an
already-featurized flight-level table into a cold-start pool and a
sequence of arriving batches, grouped by `group_column` (e.g. plane_id)
so a single aircraft's flights never straddle the cold-start/batch
boundary or a batch-to-batch boundary — the same group-integrity
reasoning harness/splits.py's "group" strategy already uses for
train/val/test, applied here to simulated arrival order instead.

This is a replay of historical data, not a live feed: "arrival order"
is just a deterministic, seeded shuffle of group identity. Nothing here
looks at time_column — a real deployment's arrival order would be
actual wall-clock time, not something this harness needs to simulate.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def simulate_batches(
    df: pd.DataFrame,
    group_column: str,
    id_column: str,
    n_initial_groups: int,
    batch_size_groups: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Returns (initial_df, batches): initial_df is the cold-start pool,
    batches is the ordered sequence of subsequent arrivals. Every row of
    `df` appears in exactly one of these outputs, and every group's rows
    all land together — the same group never appears in more than one
    output. Row order within each output is stable-sorted by id_column
    (not the shuffled group order) purely so a given output's contents
    are reproducible and readable regardless of df's original row order.
    """
    if group_column not in df.columns:
        raise ValueError(f"group_column '{group_column}' not found in dataset columns")
    if id_column not in df.columns:
        raise ValueError(f"id_column '{id_column}' not found in dataset columns")
    if n_initial_groups < 1:
        raise ValueError(f"n_initial_groups must be >= 1, got {n_initial_groups}")
    if batch_size_groups < 1:
        raise ValueError(f"batch_size_groups must be >= 1, got {batch_size_groups}")

    unique_groups = pd.unique(df[group_column])
    if n_initial_groups >= len(unique_groups):
        raise ValueError(
            f"n_initial_groups ({n_initial_groups}) must leave at least one group for "
            f"batches, but the dataset only has {len(unique_groups)} unique '{group_column}' values"
        )

    rng = np.random.RandomState(seed)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)

    initial_groups = shuffled_groups[:n_initial_groups]
    remaining_groups = shuffled_groups[n_initial_groups:]

    def _subset(groups: np.ndarray) -> pd.DataFrame:
        subset = df.loc[df[group_column].isin(groups)]
        subset = subset.sort_values(id_column, kind="stable")
        return subset.reset_index(drop=True)

    initial_df = _subset(initial_groups)
    batches = [
        _subset(remaining_groups[start : start + batch_size_groups])
        for start in range(0, len(remaining_groups), batch_size_groups)
    ]
    return initial_df, batches
