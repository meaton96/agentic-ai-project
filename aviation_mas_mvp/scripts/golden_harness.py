"""
golden_harness.py
=================
Phase-0 gate: freeze the pipeline's DETERMINISTIC outputs and fail loudly if any
later refactor changes them. Pipeline-agnostic -- it snapshots whatever
run_pipeline returns (dicts, DataFrames, arrays, floats, nested).

ML + LLM determinism strategy:
  - Deterministic outputs (feature tables, predictions, tool/evidence results,
    plan structure) are canonicalized and frozen.
  - Non-deterministic LLM free-text is quarantined via `exclude_keys` (dropped
    anywhere in the tree before comparison). Run the pipeline with a mocked/None
    chat_fn for the golden so even upstream control flow is deterministic.
  - Floats compared with absolute+relative tolerance; NaN normalized to None.

Usage:
    from golden_harness import save_golden, compare_to_golden
    out = run_pipeline(dataset, chat_fn=None)            # deterministic run
    save_golden(out, "golden/pipeline.json", exclude_keys={"explanation"})   # freeze once
    diffs = compare_to_golden(out, "golden/pipeline.json", exclude_keys={"explanation"})
    assert not diffs, diffs                               # gate
"""
from __future__ import annotations
import json
import math
import numpy as np
import pandas as pd

ROUND = 8   # store floats at this precision; compare with tol below


def canonicalize(obj, exclude_keys: set[str] | None = None):
    """Recursively convert to a JSON-safe, order-stable form."""
    ex = exclude_keys or set()
    if isinstance(obj, pd.DataFrame):
        return {"__df__": {
            "columns": [str(c) for c in obj.columns],
            "index": [canonicalize(i, ex) for i in obj.index.tolist()],
            "data": [[canonicalize(v, ex) for v in row] for row in obj.to_numpy().tolist()]}}
    if isinstance(obj, pd.Series):
        return {"__series__": {str(k): canonicalize(v, ex) for k, v in obj.to_dict().items()}}
    if isinstance(obj, np.ndarray):
        return [canonicalize(v, ex) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, ROUND)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): canonicalize(v, ex) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
                if str(k) not in ex}
    if isinstance(obj, (list, tuple)):
        return [canonicalize(v, ex) for v in obj]
    return obj   # str / int / None


def save_golden(obj, path: str, exclude_keys: set[str] | None = None) -> None:
    with open(path, "w") as f:
        json.dump(canonicalize(obj, exclude_keys), f, indent=2, sort_keys=True)


def _diff(cur, gold, tol, path, out):
    if isinstance(gold, dict) and isinstance(cur, dict):
        for k in sorted(set(gold) | set(cur)):
            if k not in cur:
                out.append(f"{path}/{k}: missing in current")
            elif k not in gold:
                out.append(f"{path}/{k}: unexpected in current")
            else:
                _diff(cur[k], gold[k], tol, f"{path}/{k}", out)
    elif isinstance(gold, list) and isinstance(cur, list):
        if len(gold) != len(cur):
            out.append(f"{path}: length {len(cur)} != golden {len(gold)}")
        else:
            for i, (a, b) in enumerate(zip(cur, gold)):
                _diff(a, b, tol, f"{path}[{i}]", out)
    elif isinstance(gold, (int, float)) and isinstance(cur, (int, float)) \
            and not isinstance(gold, bool) and not isinstance(cur, bool):
        a, b = float(cur), float(gold)
        if abs(a - b) > tol + tol * abs(b):
            out.append(f"{path}: {a} != golden {b} (|d|={abs(a-b):.2e})")
    else:
        if cur != gold:
            out.append(f"{path}: {cur!r} != golden {gold!r}")
    return out


def compare_objects(cur, gold, exclude_keys: set[str] | None = None,
                    tol: float = 1e-6) -> list[str]:
    """Diff two live objects (canonicalizing both). Empty list = match."""
    return _diff(canonicalize(cur, exclude_keys), canonicalize(gold, exclude_keys), tol, "", [])


def compare_to_golden(obj, path: str, exclude_keys: set[str] | None = None,
                      tol: float = 1e-6) -> list[str]:
    """Return a list of human-readable diffs; empty list = pass (golden reproduced)."""
    cur = canonicalize(obj, exclude_keys)
    with open(path) as f:
        gold = json.load(f)
    return _diff(cur, gold, tol, "", [])