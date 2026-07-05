"""
golden_lib.py
=============
Reusable logic behind the aviation golden test. Factored out of the pytest module
so it can be exercised directly (and so a mock run_pipeline can validate it).

run_pipeline is fully deterministic (4 CLI tools + a deterministic HITL summary,
no LLM), so we freeze the whole substantive output. Volatile I/O PATHS carry no
behavioral signal and vary by workdir/machine, so they're excluded; the HITL
summary is checked structurally (section presence + numbers) rather than frozen
verbatim, because it embeds a path. The per-flight predictions from preds.json are
pulled into the snapshot -- that's the strongest regression signal (every score).
"""
from __future__ import annotations
import json
from pathlib import Path

# I/O path keys that add no behavioral signal (vary by workdir/machine)
EXCLUDE_KEYS = {"queue_path", "queue", "out", "model", "feats", "preds",
                "flight_dir", "metadata", "path", "hitl_summary"}
STEP_ORDER = ["inspect", "featurize", "classify", "recommend"]


def clean_workdir(workdir) -> None:
    """recommend_maintenance APPENDS to the queue; remove stale artifacts so a
    re-run starts fresh (otherwise the queue accumulates across runs)."""
    wd = Path(workdir)
    for name in ("maintenance_queue.jsonl", "preds.json", "feats.csv"):
        f = wd / name
        if f.exists():
            f.unlink()


def run_and_snapshot(run_pipeline, flight_dir, metadata, model, workdir, top_k=10, spec=None) -> dict:
    """Run the pipeline and assemble the deterministic snapshot to freeze/compare."""
    Path(workdir).mkdir(parents=True, exist_ok=True)
    clean_workdir(workdir)
    out = run_pipeline(flight_dir=flight_dir, metadata=metadata, model=model,
                       workdir=workdir, top_k=top_k, spec=spec)
    preds_path = Path(workdir) / "preds.json"
    predictions = json.loads(preds_path.read_text())["predictions"]
    # sort predictions by flight id for order-stability
    predictions = sorted(predictions, key=lambda p: str(p.get("flight", p.get("filename", ""))))
    return {"result": out, "predictions": predictions,
            "_hitl_summary_raw": out.get("hitl_summary", "")}   # kept out-of-band for structural checks


def check_invariants(snapshot: dict, top_k: int = 10) -> list[str]:
    """Structural correctness that must hold regardless of the frozen values.
    Catches bugs the golden alone could miss (e.g. a golden frozen with a bug)."""
    errs = []
    out = snapshot["result"]
    trace = out.get("trace", [])
    names = [s["step"] for s in trace]
    if names != STEP_ORDER:
        errs.append(f"trace steps {names} != {STEP_ORDER}")

    res = {s["step"]: s["result"] for s in trace}
    uc = out.get("urgency_counts", {})
    n_scored = res.get("classify", {}).get("n_scored")
    if n_scored is not None and sum(uc.values()) != n_scored:
        errs.append(f"urgency_counts sum {sum(uc.values())} != n_scored {n_scored}")

    n_queued = res.get("recommend", {}).get("n_queued")
    if n_queued is not None and n_queued > top_k:
        errs.append(f"n_queued {n_queued} > top_k {top_k}")

    for p in snapshot["predictions"]:
        pm = p.get("p_maintenance", p.get("p"))
        if pm is not None and not (0.0 <= float(pm) <= 1.0):
            errs.append(f"prediction p out of [0,1]: {pm}")
            break
    urgencies = {p.get("urgency") for p in snapshot["predictions"] if "urgency" in p}
    if urgencies and not urgencies <= {"HIGH", "MEDIUM", "LOW"}:
        errs.append(f"unexpected urgency labels: {urgencies}")

    summary = snapshot.get("_hitl_summary_raw", "")
    if "HUMAN-IN-THE-LOOP" not in summary:
        errs.append("hitl_summary missing header")
    for u in ("HIGH", "MEDIUM", "LOW"):
        if f"{u}=" not in summary:
            errs.append(f"hitl_summary missing {u}= count")
    return errs