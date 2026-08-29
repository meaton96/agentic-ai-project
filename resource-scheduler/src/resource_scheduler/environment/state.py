"""
Deterministic environment/state layer for the resource-scheduler MVP.

Plays the role harness/ plays in the sibling agentic-ml-classification
project: the single source of ground truth. Agents READ facts computed
here via tool calls and PROPOSE changes; nothing in this module is ever
computed or fabricated by an LLM, and no LLM call happens anywhere in
this file.

Data source note: industrial_scheduling_dataset.csv (datasets/raw/) is a
flat table of past task records, not a live stream -- there is no
timestamp column, so row order (Task_ID ascending) is used as a proxy
for time. compute_snapshot() replays the most recent `window` rows as
"what's happening right now" per machine/slice. This is a stand-in
until a real streaming feed exists -- swap load_task_table()'s source
for a live queue without touching callers or the snapshot schema.

Known data-quality issue: as shipped, Execution_Time, Latency_ms,
Sensor_Temp_C, and URLLC_Score are constant across every row (zero
variance -- confirmed by direct inspection, not assumed). A Load
Monitor agent has nothing to detect against a constant signal, so
load_task_table(inject_variance=True, the default) adds small, seeded
Gaussian jitter to those four columns purely for development. This is
never done silently: every load reports which columns were jittered via
its second return value, and that flows straight into the snapshot as
`synthetic_variance_injected` so a caller always knows whether the
numbers it's looking at are real. Set inject_variance=False the moment
a livelier feed is available.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

JITTERED_COLUMNS = ("Execution_Time", "Latency_ms", "Sensor_Temp_C", "URLLC_Score")
# Relative jitter std as a fraction of the column's (constant) value --
# small enough not to invent implausible readings, large enough that
# mean +/- k*std thresholding (see compute_thresholds) has something to
# key off of.
JITTER_RELATIVE_STD = 0.08


def load_task_table(
    path: str | Path,
    inject_variance: bool = True,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    """Reads the raw task-record CSV and optionally injects seeded jitter
    into whichever of JITTERED_COLUMNS are actually constant in this
    file (checked directly, not assumed -- a column that already varies
    is left untouched). Returns (df, variance_injected) where
    variance_injected maps column name -> bool for every column in
    JITTERED_COLUMNS present in the file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Task table not found: {path}")
    df = pd.read_csv(path)

    variance_injected: dict[str, bool] = {}
    rng = np.random.RandomState(seed)
    for col in JITTERED_COLUMNS:
        if col not in df.columns:
            continue
        is_constant = df[col].nunique(dropna=True) <= 1
        variance_injected[col] = bool(inject_variance and is_constant)
        if variance_injected[col]:
            base = float(df[col].iloc[0])
            std = abs(base) * JITTER_RELATIVE_STD or 0.01
            df[col] = base + rng.normal(loc=0.0, scale=std, size=len(df))

    return df, variance_injected


@dataclass
class MachineState:
    machine_id: str
    status: str
    queue_depth: int
    utilization_pct: float


@dataclass
class SliceState:
    slice_id: str
    latency_ms: float
    capacity_used_pct: float
    urllc_score: float


def compute_snapshot(df: pd.DataFrame, variance_injected: dict[str, bool], window: int = 200) -> dict:
    """Deterministically aggregates the most recent `window` rows (by
    row order, the closest thing this table has to recency) into a
    per-machine and per-slice snapshot. No LLM involvement.

    machine.status: the most frequent Machine_Status value in the
      window for that machine (a mode, not a live reading -- this is
      what "current state" means for a replayed static table).
    machine.queue_depth: count of that machine's rows in the window
      flagged Reallocation == "Yes" -- tasks still needing reassignment.
    machine.utilization_pct: share of that machine's rows in the window
      where status is Active or Overloaded (i.e. not Idle/Maintenance).
    slice.latency_ms / slice.urllc_score: mean over the window.
    slice.capacity_used_pct: that slice's share of all rows in the
      window, as a simple load proxy.
    """
    recent = df.tail(window)
    total = len(recent)

    machines = []
    for machine_id, g in recent.groupby("Machine_ID"):
        status = g["Machine_Status"].mode().iloc[0]
        queue_depth = int((g["Reallocation"] == "Yes").sum())
        utilization_pct = round(100.0 * g["Machine_Status"].isin(["Active", "Overloaded"]).mean(), 2)
        machines.append(asdict(MachineState(machine_id, status, queue_depth, utilization_pct)))

    slices = []
    for slice_id, g in recent.groupby("Network_Slice_ID"):
        latency_ms = round(float(g["Latency_ms"].mean()), 3)
        urllc_score = round(float(g["URLLC_Score"].mean()), 4)
        capacity_used_pct = round(100.0 * len(g) / total, 2) if total else 0.0
        slices.append(asdict(SliceState(slice_id, latency_ms, capacity_used_pct, urllc_score)))

    return {
        "window_rows": total,
        "machines": sorted(machines, key=lambda m: m["machine_id"]),
        "slices": sorted(slices, key=lambda s: s["slice_id"]),
        "synthetic_variance_injected": variance_injected,
    }


def compute_thresholds(df: pd.DataFrame) -> dict:
    """Baseline mean/std over the FULL table (not just the window) for
    each metric the Load Monitor agent is asked to flag, using the same
    mean + k*std shape the Farahani et al. paper uses for its
    warning/critical cutoffs (Eqn. 1-2): warning at k=2, critical at
    k=3 standard deviations above the mean. Machine utilization and
    slice capacity are plain percentages, so those two use fixed,
    domain-reasonable cutoffs instead (percent-of-capacity thresholds
    don't need to be derived from this table's particular baseline).
    Computed once per snapshot call, deterministically, from data --
    never asserted by an LLM.

    Degenerate case: with inject_variance=False against the current CSV
    (Latency_ms constant), there's no real anomaly threshold to compute,
    which would otherwise put warning == critical == mean -- every
    slice's identical reading would then satisfy value >= critical and
    get spuriously flagged critical, a false alarm manufactured purely
    from the absence of signal rather than any real anomaly. There is
    no valid anomaly threshold against a zero-variance baseline, so this
    falls back to None bounds (no threshold defined) instead of a
    degenerate real one -- _flag() treats None bounds as "cannot flag",
    so no flags fire on latency until real variance (injected or
    genuine) exists to threshold against. Deliberately None rather than
    float("inf"): the latter is a valid JSON number in Python's
    json.dumps but serializes as the non-standard `Infinity` token,
    which a strict downstream JSON parser (a future frontend, jq, ...)
    reading a persisted report would fail on.

    Detected via nunique() <= 1, NOT std() > 0: confirmed by direct
    inspection that a genuinely bit-identical constant column's std()
    is NOT exactly 0.0 here (pandas 3.0.5 measured ~4.4e-16) -- mean()
    itself accumulates floating-point rounding error summing 1000
    identical values, so (x - mean) isn't exactly zero even though every
    x is bit-identical. nunique() does exact equality on the raw values
    with no arithmetic in between, so it's the reliable signal; the same
    check load_task_table() already uses to decide whether to jitter a
    column in the first place."""
    latency = df["Latency_ms"]
    if latency.nunique(dropna=True) > 1:
        latency_std = float(latency.std())
        latency_thresholds = {
            "warning": round(float(latency.mean() + 2 * latency_std), 3),
            "critical": round(float(latency.mean() + 3 * latency_std), 3),
        }
    else:
        latency_thresholds = {"warning": None, "critical": None}
    return {
        "machine_utilization_pct": {"warning": 80.0, "critical": 95.0},
        "slice_latency_ms": latency_thresholds,
        "slice_capacity_used_pct": {"warning": 60.0, "critical": 85.0},
    }


def _flag(scope: str, id_: str, metric: str, value: float, bounds: dict) -> Optional[dict]:
    if bounds["critical"] is None or bounds["warning"] is None:
        return None  # no valid threshold to flag against (see compute_thresholds' degenerate case)
    if value >= bounds["critical"]:
        severity = "critical"
    elif value >= bounds["warning"]:
        severity = "warning"
    else:
        return None
    return {
        "scope": scope, "id": id_, "severity": severity, "metric": metric,
        "value": value, "threshold": bounds[severity],
    }


def compute_flags(snapshot: dict, thresholds: dict) -> list[dict]:
    """The authoritative flags for this snapshot, computed the same way
    every time from snapshot + thresholds. Load Monitor's job is to
    narrate these, not invent its own -- steps/load_monitor_step.py
    checks the agent's reported flags against this list rather than
    trusting them outright."""
    flags = []
    for m in snapshot["machines"]:
        flag = _flag("machine", m["machine_id"], "utilization_pct",
                      m["utilization_pct"], thresholds["machine_utilization_pct"])
        if flag:
            flags.append(flag)
    for s in snapshot["slices"]:
        flag = _flag("slice", s["slice_id"], "latency_ms",
                      s["latency_ms"], thresholds["slice_latency_ms"])
        if flag:
            flags.append(flag)
        flag = _flag("slice", s["slice_id"], "capacity_used_pct",
                      s["capacity_used_pct"], thresholds["slice_capacity_used_pct"])
        if flag:
            flags.append(flag)
    return flags


def hash_table(df: pd.DataFrame) -> str:
    hasher = hashlib.sha256()
    hasher.update(",".join(sorted(df.columns.astype(str))).encode("utf-8"))
    hasher.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return hasher.hexdigest()
