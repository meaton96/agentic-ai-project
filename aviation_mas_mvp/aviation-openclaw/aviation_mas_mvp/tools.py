"""
tools.py
========
The deterministic tools the single orchestrator agent calls, one per future
agent in the full MAS. Each returns a JSON-serialisable dict (the agent reads
the result and decides the next step). NONE of these tools contain LLM calls:
the LLM orchestrates; the tools compute. That split is the whole point.

Tool -> future agent mapping
  inspect_data        -> Perception Agent      (read-only sensor data)
  featurize_flights   -> Perception/Preproc    (feature extraction)
  classify_flights    -> Analysis Agent        (failure likelihood + urgency)
  recommend_maintenance -> Optimization Agent  (write-only maintenance queue)

CLI usage (so an OpenClaw agent can invoke via a single exec/shell call):
  python -m scripts.tools inspect       --flight-dir DIR --metadata META.csv
  python -m scripts.tools featurize      --flight-dir DIR --metadata META.csv --out feats.parquet
  python -m scripts.tools classify       --feats feats.parquet --model model.joblib --out preds.json
  python -m scripts.tools recommend      --preds preds.json --queue maintenance_queue.jsonl
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

try:
    from .featurize import build_feature_table, detect_channels
except ImportError:  # allow running as a plain script (smoke test, CLI from inside dir)
    from featurize import build_feature_table, detect_channels


# ---------- Perception ----------
def inspect_data(flight_dir: str, metadata_path: str, file_col="filename",
                 label_col="label", group_col="plane_id", n_preview=3) -> dict:
    md = pd.read_csv(metadata_path)
    flight_dir = Path(flight_dir)
    sample_channels, sample_steps = None, None
    for fn in md[file_col].head(n_preview):
        fp = flight_dir / str(fn)
        if fp.exists():
            df = pd.read_csv(fp)
            sample_channels = detect_channels(df)
            sample_steps = len(df)
            break
    return {
        "tool": "inspect_data",
        "n_flights": int(len(md)),
        "label_balance": {int(k): int(v) for k, v in
                          zip(*np.unique(md[label_col], return_counts=True))},
        "n_planes": int(md[group_col].nunique()),
        "detected_channels": sample_channels,
        "n_channels": len(sample_channels) if sample_channels else None,
        "example_flight_steps": sample_steps,
        "notes": "Read-only. Flags class balance and channel inventory for the planner.",
    }


# ---------- Perception / Preprocessing ----------
def featurize_flights(flight_dir: str, metadata_path: str, out_path: str,
                      file_col="filename", label_col="label",
                      group_col="plane_id", max_flights=None) -> dict:
    md = pd.read_csv(metadata_path)
    table = build_feature_table(flight_dir, md, file_col=file_col,
                                label_col=label_col, group_col=group_col,
                                max_flights=max_flights)
    out = Path(out_path)
    table.to_parquet(out) if out.suffix == ".parquet" else table.to_csv(out, index=False)
    return {"tool": "featurize_flights", "rows": int(len(table)),
            "n_features": int(table.shape[1] - 3), "out": str(out)}


# ---------- Analysis ----------
def classify_flights(feats_path: str, model_path: str, out_path: str,
                     high_urgency_thresh=0.66, med_urgency_thresh=0.33) -> dict:
    table = (pd.read_parquet(feats_path) if feats_path.endswith(".parquet")
             else pd.read_csv(feats_path))
    bundle = joblib.load(model_path)
    clf, cols = bundle["model"], bundle["feature_columns"]
    X = table.reindex(columns=cols, fill_value=0.0).values
    proba = clf.predict_proba(X)[:, 1]

    def urgency(p):
        return "HIGH" if p >= high_urgency_thresh else ("MEDIUM" if p >= med_urgency_thresh else "LOW")

    preds = []
    for i, p in enumerate(proba):
        fn = table["filename"].iloc[i] if "filename" in table else f"flight_{i}"
        preds.append({"flight": str(fn), "p_maintenance": float(p),
                      "urgency": urgency(p)})
    preds.sort(key=lambda r: r["p_maintenance"], reverse=True)
    Path(out_path).write_text(json.dumps({"predictions": preds}, indent=2))
    return {"tool": "classify_flights", "n_scored": len(preds),
            "n_high": sum(p["urgency"] == "HIGH" for p in preds),
            "out": out_path}


# ---------- Optimization (write-only queue) ----------
def recommend_maintenance(preds_path: str, queue_path: str, top_k=10) -> dict:
    preds = json.loads(Path(preds_path).read_text())["predictions"]
    queue = Path(queue_path)
    written = []
    with queue.open("a") as fh:  # append-only: write-only maintenance queue
        for r in preds[:top_k]:
            item = {
                "flight": r["flight"],
                "priority": r["urgency"],
                "p_maintenance": round(r["p_maintenance"], 4),
                "recommended_action": (
                    "Schedule inspection within 2 days" if r["urgency"] == "HIGH"
                    else "Schedule routine inspection" if r["urgency"] == "MEDIUM"
                    else "Monitor; no action required"),
                "evidence": "Elevated maintenance probability from baseline classifier.",
            }
            fh.write(json.dumps(item) + "\n")
            written.append(item)
    return {"tool": "recommend_maintenance", "n_queued": len(written),
            "queue": str(queue), "items": written}


# ---------- Approval gate (Step 3) ----------
# This tool is intentionally NOT wired into the agent's TOOLS_SCHEMA
# (see agent_harness.py). The whole point of the gate is that the model never
# has the authority to commit its own recommendations -- approval requires a
# secret that lives ONLY in an environment variable the human/operator sets,
# never in any tool's JSON result, the HITL summary, or the model's context.
# Run this yourself, from a shell, after reading the agent's HITL summary.
APPROVAL_TOKEN_ENV = "MAINTENANCE_APPROVAL_TOKEN"


def approve_recommendations(queue_path: str, approved_path: str, token: str,
                            token_env: str = APPROVAL_TOKEN_ENV) -> dict:
    """
    Commit currently-queued-but-unapproved recommendations to approved_path,
    gated by a token that must match an operator-set environment variable.

    Idempotent and incremental: tracks how many lines of `queue_path` have
    already been approved (in a small sidecar `<approved_path>.cursor`), and
    on each successful call only appends the NEW lines since last approval --
    so re-running with the same token never double-writes, and a queue that
    grows across multiple agent runs gets approved incrementally, not as one
    all-or-nothing blob.
    """
    import os
    expected = os.environ.get(token_env)
    if not expected:
        return {"tool": "approve_recommendations", "approved": False,
                "error": f"no approval token configured: set ${token_env} "
                        f"in the operator's shell before approving anything."}
    if token != expected:
        return {"tool": "approve_recommendations", "approved": False,
                "error": "token mismatch; nothing written."}

    queue = Path(queue_path)
    if not queue.exists():
        return {"tool": "approve_recommendations", "approved": False,
                "error": f"queue file not found: {queue_path}"}
    lines = queue.read_text().splitlines()

    approved = Path(approved_path)
    cursor_path = Path(str(approved) + ".cursor")
    already = int(cursor_path.read_text()) if cursor_path.exists() else 0
    already = min(already, len(lines))  # tolerate a queue that was reset/shrunk

    new_lines = lines[already:]
    if new_lines:
        with approved.open("a") as fh:
            for line in new_lines:
                fh.write(line + "\n")
    cursor_path.write_text(str(len(lines)))

    return {"tool": "approve_recommendations", "approved": True,
            "n_newly_approved": len(new_lines), "n_total_approved": len(lines),
            "approved_queue": str(approved)}


# ---------- CLI ----------
def _main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("inspect")
    s.add_argument("--flight-dir", required=True); s.add_argument("--metadata", required=True)

    s = sub.add_parser("featurize")
    s.add_argument("--flight-dir", required=True); s.add_argument("--metadata", required=True)
    s.add_argument("--out", required=True); s.add_argument("--max-flights", type=int, default=None)

    s = sub.add_parser("classify")
    s.add_argument("--feats", required=True); s.add_argument("--model", required=True)
    s.add_argument("--out", required=True)

    s = sub.add_parser("recommend")
    s.add_argument("--preds", required=True); s.add_argument("--queue", required=True)
    s.add_argument("--top-k", type=int, default=10)

    s = sub.add_parser("approve")
    s.add_argument("--queue", required=True); s.add_argument("--approved", required=True)
    s.add_argument("--token", required=True,
                  help=f"must match ${APPROVAL_TOKEN_ENV} in your shell")

    a = ap.parse_args()
    if a.cmd == "inspect":
        r = inspect_data(a.flight_dir, a.metadata)
    elif a.cmd == "featurize":
        r = featurize_flights(a.flight_dir, a.metadata, a.out, max_flights=a.max_flights)
    elif a.cmd == "classify":
        r = classify_flights(a.feats, a.model, a.out)
    elif a.cmd == "recommend":
        r = recommend_maintenance(a.preds, a.queue, top_k=a.top_k)
    elif a.cmd == "approve":
        r = approve_recommendations(a.queue, a.approved, a.token)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    _main()