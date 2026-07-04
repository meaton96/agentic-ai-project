"""
golden_harness.py
=================
Regression testing harness for pipeline outputs.

Provides utilities to snapshot complex, deterministic data structures (including
pandas DataFrames, NumPy arrays, and nested dictionaries) into JSON 'golden' 
files. Future runs can be compared against these snapshots to detect regressions 
within a specified float tolerance. Non-deterministic fields (e.g., LLM text) 
can be excluded from the snapshot and comparison.
"""
from __future__ import annotations
import json
import math
import numpy as np
import pandas as pd

ROUND = 8   # store floats at this precision; compare with tol below


def canonicalize(obj, exclude_keys: set[str] | None = None):
    """
    Recursively converts complex data structures into a JSON-serializable, 
    order-stable format.
    
    Normalizes pandas DataFrames, pandas Series, and NumPy arrays into standard 
    lists/dicts. Handles numeric edge cases (inf/NaN) and sorts dictionary keys 
    to ensure deterministic serialization.
    
    Args:
        obj: The Python object to canonicalize.
        exclude_keys (set[str] | None, optional): Keys to explicitly drop during 
            traversal (e.g., non-deterministic LLM output keys). Defaults to None.
            
    Returns:
        A JSON-safe, primitive representation of the input object.
    """
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
    """
    Canonicalizes and saves a pipeline output to a JSON file as a golden snapshot.
    
    Args:
        obj: The output object to save.
        path (str): Filepath where the golden JSON will be written.
        exclude_keys (set[str] | None, optional): Keys to omit from the snapshot. Defaults to None.
    """
    with open(path, "w") as f:
        json.dump(canonicalize(obj, exclude_keys), f, indent=2, sort_keys=True)


def _diff(cur, gold, tol, path, out):
    """
    Recursive helper function to identify differences between two canonicalized objects.
    
    Args:
        cur: The current object being evaluated.
        gold: The golden object being compared against.
        tol (float): Combined absolute and relative tolerance for float comparisons.
        path (str): The current JSON path traversal string (used for error reporting).
        out (list[str]): Mutable list accumulating diff messages.
        
    Returns:
        list[str]: The accumulated list of difference strings.
    """
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


def compare_to_golden(obj, path: str, exclude_keys: set[str] | None = None,
                      tol: float = 1e-6) -> list[str]:
    """
    Compares a new pipeline output against a saved golden JSON snapshot.
    
    Args:
        obj: The new output object to evaluate.
        path (str): Filepath to the existing golden JSON snapshot.
        exclude_keys (set[str] | None, optional): Keys to ignore during comparison. Defaults to None.
        tol (float, optional): Float tolerance for numeric comparisons. Defaults to 1e-6.
        
    Returns:
        list[str]: A list of human-readable differences. An empty list indicates
            a successful match (no regressions found).
    """
    cur = canonicalize(obj, exclude_keys)
    with open(path) as f:
        gold = json.load(f)
    return _diff(cur, gold, tol, "", [])
