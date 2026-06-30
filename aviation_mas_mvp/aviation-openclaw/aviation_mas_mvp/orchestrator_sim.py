"""
orchestrator_sim.py
===================
A deterministic, hard-coded execution pipeline for the aviation maintenance agent.

This module simulates the expected path of an LLM orchestrator by sequentially 
invoking the required CLI tools (inspect -> featurize -> classify -> recommend). 
It serves as a testing oracle to validate tool inputs/outputs and acts as a 
baseline to compare against the non-deterministic behavior of a live LLM agent.
"""

from __future__ import annotations
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Step:
    name: str
    cmd: list[str]
    result: dict = field(default_factory=dict)


def _run(cmd: list[str], cwd: str | Path | None = None) -> dict:
    """Wrapper to execute CLI tools and parse their standard output as JSON."""
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"tool failed: {' '.join(cmd)}\n{proc.stderr.strip()}")
    return json.loads(proc.stdout)


def run_pipeline(flight_dir: str | Path, metadata: str | Path, model: str | Path,
                 workdir: str | Path, top_k: int = 10,
                 python: str = sys.executable) -> dict:
    """
    Executes the full pipeline via deterministic CLI calls and returns 
    a structured trace along with the generated Human-In-The-Loop (HITL) summary.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    
    # Define intermediate artifact paths
    feats = workdir / "feats.csv"
    preds = workdir / "preds.json"
    queue = workdir / "maintenance_queue.jsonl"

    base = [python, "-m", f"{__package__}.tools"]
    
    # Define the exact sequence of tool executions required for a successful run
    steps = [
        Step("inspect",   base + ["inspect", "--flight-dir", str(flight_dir),
                                  "--metadata", str(metadata)]),
        Step("featurize", base + ["featurize", "--flight-dir", str(flight_dir),
                                  "--metadata", str(metadata), "--out", str(feats)]),
        Step("classify",  base + ["classify", "--feats", str(feats),
                                  "--model", str(model), "--out", str(preds)]),
        Step("recommend", base + ["recommend", "--preds", str(preds),
                                  "--queue", str(queue), "--top-k", str(top_k)]),
    ]

    # Execute each step sequentially
    for s in steps:
        s.result = _run(s.cmd)

    inspect, classify, recommend = (steps[0].result, steps[2].result, steps[3].result)
    
    # Load raw predictions to calculate distribution of urgency levels
    preds_obj = json.loads(preds.read_text())["predictions"]
    by_urg = {u: sum(p["urgency"] == u for p in preds_obj)
              for u in ("HIGH", "MEDIUM", "LOW")}

    # Aggregate results for the final report
    summary = build_hitl_summary(inspect, classify, recommend, by_urg)
    return {
        "trace": [{"step": s.name, "result": s.result} for s in steps],
        "urgency_counts": by_urg,
        "hitl_summary": summary,
        "queue_path": str(queue),
    }


def build_hitl_summary(inspect: dict, classify: dict, recommend: dict,
                       by_urg: dict) -> str:
    """
    Assembles a readable, technician-facing report strictly from the JSON 
    outputs produced by the tool execution trace.
    """
    bal = {int(k): int(v) for k, v in inspect["label_balance"].items()}
    n = inspect["n_flights"]
    pos = bal.get(1, 0)
    pos_rate = pos / n if n else 0.0
    
    # Compute a confidence heuristic grounded in class distribution and channel counts
    if 0.25 <= pos_rate <= 0.75 and inspect["n_channels"]:
        conf = "MODERATE"
        why = (f"class balance is usable ({pos}/{n} positive) and "
               f"{inspect['n_channels']} channels were detected cleanly")
    else:
        conf = "LOW"
        why = (f"class balance is skewed ({pos}/{n} positive), which limits how "
               f"much weight to put on the ranking")

    # Format the final summary string
    lines = [
        "HUMAN-IN-THE-LOOP MAINTENANCE SUMMARY",
        "=" * 40,
        f"Run scope    : {n} flights inspected | class balance {bal} | "
        f"{inspect['n_channels']} channels ({inspect['n_planes']} tails)",
        f"Findings     : HIGH={by_urg['HIGH']}  MEDIUM={by_urg['MEDIUM']}  "
        f"LOW={by_urg['LOW']}  (scored {classify['n_scored']})",
        f"Queued       : top {recommend['n_queued']} -> {recommend['queue']}",
        "Top recommendations:",
    ]
    
    for it in recommend["items"][:5]:
        lines.append(f"  - {it['flight']}: p={it['p_maintenance']:.3f} "
                     f"[{it['priority']}] -> {it['recommended_action']}")
                     
    lines += [
        f"Confidence   : {conf} — {why}.",
        "Evidence trail:",
        f"  inspect   -> {inspect['n_flights']} flights, balance {bal}",
        f"  classify  -> {classify['n_scored']} scored, {classify['n_high']} HIGH",
        f"  recommend -> {recommend['n_queued']} queued",
        "",
        "Awaiting technician approval before committing recommendations.",
    ]
    
    return "\n".join(lines)