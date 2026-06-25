"""
contract_net.py
================
A minimal Contract Net Protocol (announce -> bid -> award) for negotiating
maintenance downtime windows. This is the A2A coordination mechanism named in
SPEC.md Step 4 ("reuse the Contract Net Protocol scaffolding from your DFJSP
MAS spec") -- reimplemented generically here since the resource type differs
(maintenance bays, not AGVs); the protocol shape (announce/bid/award) is the
thing being reused, not literal code.

WHY A SYNTHETIC RESOURCE LAYER
-------------------------------
NGAFID-MC has no maintenance-log, parts-inventory, or downtime-window data --
the professor's schema asks the Optimization Agent to check exactly those
things, but none of them exist in this dataset. Without SOME resource model,
"Contract Net Protocol" degenerates into a renamed `recommend` call with
nothing to actually negotiate over. This module is an explicit, clearly-
synthetic stand-in (bays x days, fixed capacity, rule-based cost) -- same
spirit as synthdata.py's synthetic flights: illustrative of the mechanism,
not derived from real facility data.

DETERMINISM
------------
Bidders are rule-based, not LLM-backed -- a bay doesn't reason, it just
computes a bid from fixed rules given a task announcement. This keeps the
same split as the rest of the codebase: deterministic tools compute, the
LLM-backed Optimization Agent decides which bid to accept (or reject all).
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

N_BAYS = 3
N_DAYS = 5            # planning horizon, days from "today" (day 0)
BASE_COST_PER_DAY_DELAY = 0   # bids don't get cheaper for waiting, just slower
BAY_HOURLY_RATE = {0: 120.0, 1: 95.0, 2: 150.0}  # synthetic, illustrative only


@dataclass
class Slot:
    bay: int
    day: int
    booked: bool = False
    booked_task: str | None = None


@dataclass
class ResourcePool:
    slots: list[Slot] = field(default_factory=list)

    @classmethod
    def fresh(cls, n_bays: int = N_BAYS, n_days: int = N_DAYS) -> "ResourcePool":
        return cls(slots=[Slot(bay=b, day=d) for b in range(n_bays) for d in range(n_days)])

    @classmethod
    def load(cls, path: str | Path) -> "ResourcePool":
        p = Path(path)
        if not p.exists():
            return cls.fresh()
        raw = json.loads(p.read_text())
        return cls(slots=[Slot(**s) for s in raw["slots"]])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"slots": [asdict(s) for s in self.slots]}, indent=2))

    def available(self) -> list[Slot]:
        return [s for s in self.slots if not s.booked]

    def find(self, bay: int, day: int) -> Slot | None:
        return next((s for s in self.slots if s.bay == bay and s.day == day), None)


def generate_bids(pool: ResourcePool, urgency: str, estimated_hours: float,
                  earliest_day: int = 0) -> list[dict]:
    """
    Rule-based bidding: every available slot on or after `earliest_day` submits
    a bid. Cost = hourly rate x estimated hours; bids are NOT discounted for
    waiting (a later day is strictly worse for urgent tasks, never better),
    so ranking naturally favors early + cheap. This is intentionally simple --
    the point is a real multi-bidder negotiation existing at all, not a
    sophisticated cost model.
    """
    bids = []
    for s in pool.available():
        if s.day < earliest_day:
            continue
        cost = round(BAY_HOURLY_RATE.get(s.bay, 100.0) * estimated_hours, 2)
        # urgency penalty for late slots, so the agent sees urgency reflected
        # in the bid itself, not just in metadata it has to remember separately
        urgency_penalty = {"HIGH": 50, "MEDIUM": 15, "LOW": 0}.get(urgency, 0) * s.day
        bids.append({"bay": s.bay, "day": s.day, "cost": cost,
                     "score": round(cost + urgency_penalty, 2)})
    return sorted(bids, key=lambda b: b["score"])


def award(pool: ResourcePool, bay: int, day: int, task_id: str) -> dict:
    """Book the chosen slot. Refuses if already booked (no double-booking)."""
    slot = pool.find(bay, day)
    if slot is None:
        return {"awarded": False, "error": f"no such slot bay={bay} day={day}"}
    if slot.booked:
        return {"awarded": False, "error": f"slot bay={bay} day={day} already "
                                           f"booked for {slot.booked_task!r}"}
    slot.booked = True
    slot.booked_task = task_id
    return {"awarded": True, "bay": bay, "day": day, "task_id": task_id}
