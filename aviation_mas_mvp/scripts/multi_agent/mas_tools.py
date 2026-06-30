"""
mas_tools.py
============
New tools for Step 4 (MAS split), owned by the Optimization Agent. These
REPLACE the plain `recommend` step with a real negotiation: announce a task,
collect bids from the resource pool (contract_net.py), award one, write the
queue entry with the actual assigned bay/day instead of generic boilerplate
text. `tools.py` (Step 2/3) is untouched -- the single-orchestrator harness
still works exactly as before; this is an additive path for the MAS variant.
"""
from __future__ import annotations
import json
from pathlib import Path

from .contract_net import ResourcePool, generate_bids, award


def request_bids(pool_path: str, urgency: str, estimated_hours: float,
                 earliest_day: int = 0, task_id: str = "") -> dict:
    """Read-only: announce a task, return ranked bids. Books nothing."""
    pool = ResourcePool.load(pool_path)
    pool.save(pool_path)  # creates the file on first call if it didn't exist
    bids = generate_bids(pool, urgency, estimated_hours, earliest_day)
    return {"tool": "request_bids", "task_id": task_id, "urgency": urgency,
            "estimated_hours": estimated_hours, "n_bids": len(bids),
            "bids": bids[:5]}  # top 5 -- the agent shouldn't need the whole pool


def award_bid(pool_path: str, queue_path: str, task_id: str, flight: str,
             p_maintenance: float, urgency: str, bay: int, day: int) -> dict:
    """
    Commit the chosen bid (books the slot, refuses a double-booking) and, only
    on success, append the queue entry -- now carrying the actual assigned
    bay/day instead of the old generic "schedule within 2 days" text.
    """
    pool = ResourcePool.load(pool_path)
    result = award(pool, bay, day, task_id)
    if not result["awarded"]:
        return {"tool": "award_bid", "queued": False, **result}
    pool.save(pool_path)

    item = {
        "flight": flight, "priority": urgency,
        "p_maintenance": round(float(p_maintenance), 4),
        "recommended_action": f"Scheduled: bay {bay}, day {day} from now",
        "assigned_bay": bay, "scheduled_day": day,
        "evidence": "Elevated maintenance probability from baseline classifier; "
                    "slot awarded via Contract Net negotiation.",
    }
    with Path(queue_path).open("a") as fh:
        fh.write(json.dumps(item) + "\n")
    return {"tool": "award_bid", "queued": True, "bay": bay, "day": day,
            "task_id": task_id, "item": item}


def list_top_predictions(preds_path: str, top_k: int = 10) -> dict:
    """
    Read-only: surface the actual ranked predictions to the Analysis Agent.
    classify's own JSON result only returns counts (n_scored/n_high) -- by
    design, so the LLM never sees raw model internals it could fabricate
    around. This tool is the deliberate, narrow exception: it returns ranked
    (flight, probability, urgency) tuples so Analysis can decide WHICH flights
    to formally announce as tasks, without exposing anything beyond that.
    """
    preds = json.loads(Path(preds_path).read_text())["predictions"]
    top = preds[:top_k]
    return {"tool": "list_top_predictions", "n_total": len(preds),
            "n_returned": len(top),
            "predictions": [{"flight": p["flight"], "p_maintenance": p["p_maintenance"],
                            "urgency": p["urgency"]} for p in top]}


def announce_task(flight: str, urgency: str, estimated_hours: float, justification: str) -> dict:
    """
    The Analysis Agent's formal hand-off to Optimization: not a write to any
    file, just a structured, validated record of "this flight needs action."
    The MAS coordinator collects these from Analysis's trace and feeds each
    one to a separate Optimization Agent run (request_bids -> award_bid).
    Existing purely so the hand-off is an explicit, schema-validated tool call
    rather than the coordinator silently inferring intent from classify's
    output -- the Analysis Agent decides what counts as a task, not the glue
    code around it.
    """
    if urgency not in ("HIGH", "MEDIUM", "LOW"):
        return {"error": f"urgency must be HIGH/MEDIUM/LOW, got {urgency!r}"}
    if estimated_hours <= 0:
        return {"error": "estimated_hours must be positive"}
    return {"tool": "announce_task", "flight": flight, "urgency": urgency,
            "estimated_hours": float(estimated_hours), "justification": justification}


# ---------- CLI, for parity with tools.py ----------
def _main():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("request-bids")
    s.add_argument("--pool", required=True); s.add_argument("--urgency", required=True)
    s.add_argument("--hours", type=float, required=True); s.add_argument("--earliest-day", type=int, default=0)
    s.add_argument("--task-id", default="")

    s = sub.add_parser("award-bid")
    s.add_argument("--pool", required=True); s.add_argument("--queue", required=True)
    s.add_argument("--task-id", required=True); s.add_argument("--flight", required=True)
    s.add_argument("--p-maintenance", type=float, required=True); s.add_argument("--urgency", required=True)
    s.add_argument("--bay", type=int, required=True); s.add_argument("--day", type=int, required=True)

    a = ap.parse_args()
    if a.cmd == "request-bids":
        r = request_bids(a.pool, a.urgency, a.hours, a.earliest_day, a.task_id)
    else:
        r = award_bid(a.pool, a.queue, a.task_id, a.flight, a.p_maintenance, a.urgency, a.bay, a.day)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    _main()
