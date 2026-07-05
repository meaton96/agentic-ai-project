"""
tools_lib.py
============
Per-tool contract invariants for inspect / featurize / classify / recommend.
These check properties that must hold regardless of the frozen golden values, so
they catch behavior changes the golden might not distinguish -- especially the
featurize->model column contract, which the Phase 1-4 refactors will churn.

Each helper returns a list of human-readable error strings (empty = pass).
Field names follow orchestrator_sim.build_hitl_summary:
  inspect  : n_flights, label_balance, n_channels, n_planes
  classify : n_scored, n_high
  recommend: n_queued, queue, items[{flight, p_maintenance, priority, ...}]
  preds.json: predictions[{flight, p_maintenance, urgency}]
"""
from __future__ import annotations
import json
from pathlib import Path

EXPECTED_CHANNELS = 23          # C28 sensor count; adjust per dataset
PRIORITIES = {"HIGH", "MEDIUM", "LOW"}


def load_artifacts(workdir, model_path) -> dict:
    import pandas as pd, joblib
    wd = Path(workdir)
    feats = pd.read_csv(wd / "feats.csv")
    preds = json.loads((wd / "preds.json").read_text())["predictions"]
    ql = (wd / "maintenance_queue.jsonl").read_text().splitlines()
    queue = [json.loads(l) for l in ql if l.strip()]
    model_feats = joblib.load(model_path)["feature_columns"]
    return {"feats": feats, "preds": preds, "queue": queue, "model_feats": model_feats}


def inspect_invariants(inspect_result: dict, n_meta_rows: int | None = None) -> list[str]:
    e = []
    if inspect_result.get("n_channels") != EXPECTED_CHANNELS:
        e.append(f"n_channels {inspect_result.get('n_channels')} != {EXPECTED_CHANNELS}")
    bal = {int(k): int(v) for k, v in inspect_result.get("label_balance", {}).items()}
    if not set(bal) <= {0, 1}:
        e.append(f"label_balance has non-binary keys: {set(bal)}")
    if sum(bal.values()) != inspect_result.get("n_flights"):
        e.append(f"label_balance sum {sum(bal.values())} != n_flights {inspect_result.get('n_flights')}")
    if n_meta_rows is not None and inspect_result.get("n_flights") != n_meta_rows:
        e.append(f"n_flights {inspect_result.get('n_flights')} != metadata rows {n_meta_rows}")
    return e


def featurize_contract(feats_df, model_feats: list[str], n_flights: int) -> list[str]:
    """THE key contract: every model feature must be present in the featurizer
    output, or classify's reindex silently zero-fills -> wrong-but-not-crashing
    predictions. This is the exact break the generic refactor risks."""
    e = []
    missing = [c for c in model_feats if c not in feats_df.columns]
    if missing:
        e.append(f"featurize missing {len(missing)} model columns, e.g. {missing[:3]}")
    if len(feats_df) != n_flights:
        e.append(f"feats rows {len(feats_df)} != n_flights {n_flights}")
    return e


def classify_invariants(preds: list[dict], n_flights: int) -> list[str]:
    e = []
    if len(preds) != n_flights:
        e.append(f"scored {len(preds)} != n_flights {n_flights} (not all flights scored)")
    for p in preds:
        pm = p.get("p_maintenance")
        if pm is None or not (0.0 <= float(pm) <= 1.0):
            e.append(f"p_maintenance out of [0,1] or missing: {p.get('flight')}={pm}")
            break
    urg = {p.get("urgency") for p in preds}
    if not urg <= PRIORITIES:
        e.append(f"unexpected urgency labels: {urg - PRIORITIES}")
    if len({p.get("flight") for p in preds}) != len(preds):
        e.append("duplicate flight ids in predictions")
    return e


def recommend_invariants(queue: list[dict], preds: list[dict], top_k: int) -> list[str]:
    e = []
    if len(queue) > top_k:
        e.append(f"queued {len(queue)} > top_k {top_k}")
    ps = [float(it["p_maintenance"]) for it in queue]
    if ps != sorted(ps, reverse=True):
        e.append("queue not sorted by p_maintenance descending")
    scored_ids = {p.get("flight") for p in preds}
    stray = [it["flight"] for it in queue if it["flight"] not in scored_ids]
    if stray:
        e.append(f"queued flights not in scored set: {stray[:3]}")
    bad_pri = {it.get("priority") for it in queue} - PRIORITIES
    if bad_pri:
        e.append(f"unexpected priority labels: {bad_pri}")
    return e