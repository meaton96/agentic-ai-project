"""
mas_coordinator.py
===================
Step 4: the single orchestrator is split into four agents, each with its own
LLM call, own system prompt, and own restricted tool subset. A deterministic
Coordinator sequences them and wires data between runs.

Agent boundaries (reconciling SPEC.md's tool table with the professor's
schema):
    Perception   : inspect, featurize           (read-only + the MTS->tabular
                   bridge our sequential data needs, that the schema's
                   snapshot framing didn't have to specify)
    Analysis     : classify, list_top_predictions, announce_task
    Optimization : request_bids, award_bid       (Contract Net Protocol --
                   replaces the old flat `recommend`)
    HITL         : no tools -- synthesises the final summary from what the
                   other three produced. Approval gate is unchanged: still a
                   separate, credential-gated step outside any agent's reach.

HONEST LIMITATION: the Coordinator (not the agents themselves) wires data
between runs -- it reads Perception's trace to find the feature-table path,
reads Analysis's trace to find announced tasks, etc. The LLM-to-LLM "talking"
is mediated by deterministic glue code, not raw agent-generated message text.
This is a deliberate choice for this MVP slice (keeps each agent's job to
"reason within my scope and call my tools," and keeps cross-agent data
wiring testable in isolation, the same "LLM reasons, tools compute" split as
everywhere else in this codebase) -- not a claim that this is full,
unmediated A2A communication.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import tools as T
from . import mas_tools as MT
from .agent_harness import run_agent, AgentRun

# ---------------------------------------------------------------------------
# Per-agent tool schemas
# ---------------------------------------------------------------------------
PERCEPTION_TOOLS = [
    {"type": "function", "function": {
        "name": "inspect",
        "description": "Read-only sensor inventory, class balance, channel count. Call first.",
        "parameters": {"type": "object", "properties": {
            "flight_dir": {"type": "string"}, "metadata": {"type": "string"},
        }, "required": ["flight_dir", "metadata"]}}},
    {"type": "function", "function": {
        "name": "featurize",
        "description": "Build the tabular feature table from the flight directory. Call after inspect.",
        "parameters": {"type": "object", "properties": {
            "flight_dir": {"type": "string"}, "metadata": {"type": "string"}, "out": {"type": "string"},
        }, "required": ["flight_dir", "metadata", "out"]}}},
]

ANALYSIS_TOOLS = [
    {"type": "function", "function": {
        "name": "classify",
        "description": "Score flights with the trained model: probability + urgency, ranked.",
        "parameters": {"type": "object", "properties": {
            "feats": {"type": "string"}, "model": {"type": "string"}, "out": {"type": "string"},
        }, "required": ["feats", "model", "out"]}}},
    {"type": "function", "function": {
        "name": "list_top_predictions",
        "description": "See the actual ranked (flight, probability, urgency) list, to decide which flights to announce as tasks.",
        "parameters": {"type": "object", "properties": {
            "preds_path": {"type": "string"}, "top_k": {"type": "integer", "default": 10},
        }, "required": ["preds_path"]}}},
    {"type": "function", "function": {
        "name": "announce_task",
        "description": "Formally hand off ONE flight to the Optimization Agent for scheduling. "
                        "Call once per flight you've decided needs action.",
        "parameters": {"type": "object", "properties": {
            "flight": {"type": "string"}, "urgency": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
            "estimated_hours": {"type": "number"}, "justification": {"type": "string"},
        }, "required": ["flight", "urgency", "estimated_hours", "justification"]}}},
]

OPTIMIZATION_TOOLS = [
    {"type": "function", "function": {
        "name": "request_bids",
        "description": "Announce a task to the maintenance-bay resource pool; returns ranked bids "
                        "(bay, day, cost). Read-only, books nothing.",
        "parameters": {"type": "object", "properties": {
            "pool_path": {"type": "string"}, "urgency": {"type": "string"},
            "estimated_hours": {"type": "number"}, "earliest_day": {"type": "integer", "default": 0},
            "task_id": {"type": "string"},
        }, "required": ["pool_path", "urgency", "estimated_hours", "task_id"]}}},
    {"type": "function", "function": {
        "name": "award_bid",
        "description": "Commit ONE chosen bid (bay, day) and write the queue entry. Refuses if the "
                        "slot was already booked -- if so, call request_bids again or pick a "
                        "different bid from the last response.",
        "parameters": {"type": "object", "properties": {
            "pool_path": {"type": "string"}, "queue_path": {"type": "string"}, "task_id": {"type": "string"},
            "flight": {"type": "string"}, "p_maintenance": {"type": "number"}, "urgency": {"type": "string"},
            "bay": {"type": "integer"}, "day": {"type": "integer"},
        }, "required": ["pool_path", "queue_path", "task_id", "flight", "p_maintenance",
                       "urgency", "bay", "day"]}}},
]

PERCEPTION_PROMPT = (
    "You are the Perception Agent. Call inspect, then featurize, in that order. "
    "Report channel count, flight count, and class balance, then stop.")
ANALYSIS_PROMPT = (
    "You are the Analysis Agent. Call classify, then list_top_predictions to see the "
    "actual ranked flights, then call announce_task once for each flight you judge needs "
    "action (HIGH urgency flights at minimum). Estimate hours by urgency: HIGH=4, MEDIUM=2, LOW=1. "
    "Stop once you've announced every flight you intend to.")
OPTIMIZATION_PROMPT = (
    "You are the Optimization Agent, negotiating ONE maintenance task via Contract Net. "
    "Call request_bids once to see ranked bids, then award_bid on the best one. If award_bid "
    "is refused (slot already booked), call request_bids again and pick the next-best bid. "
    "Prefer low cost and, for HIGH urgency, earlier days. Stop once a bid is successfully awarded.")
HITL_PROMPT = (
    "You are the HITL Agent. You receive a JSON bundle summarising what the Perception, "
    "Analysis, and Optimization Agents found and did. Produce the same structured report "
    "format as a single-orchestrator HITL summary (run scope, findings, top recommendations, "
    "confidence grounded in any validated metric provided -- never in raw prediction counts, "
    "evidence trail, approval gate). You have no tools; reason only from the bundle you're given. "
    "End with: 'Awaiting technician approval before committing recommendations.'")


@dataclass
class MASRun:
    perception: AgentRun | None = None
    analysis: AgentRun | None = None
    optimizations: list = field(default_factory=list)
    hitl: AgentRun | None = None
    tasks_announced: list = field(default_factory=list)
    awards: list = field(default_factory=list)


def _trace_result(run, tool_name: str):
    return next((t.result for t in run.trace if t.name == tool_name and t.ok), None)


def _trace_results(run, tool_name: str):
    return [t.result for t in run.trace if t.name == tool_name and t.ok]


def run_mas_pipeline(client_factory, flight_dir: str, metadata: str, model_path: str,
                     workdir: str, pool_path: str, queue_path: str,
                     max_turns_per_agent: int = 8, verbose: bool = True):
    """
    `client_factory()` -> a fresh client for each agent run (most clients are
    stateless/cheap to construct; pass e.g. `lambda: VLLMClient(...)` or, for
    testing, a factory that returns the next scripted client in a sequence).
    """
    Path(workdir).mkdir(parents=True, exist_ok=True)
    feats = str(Path(workdir) / "feats.csv")
    preds = str(Path(workdir) / "preds.json")
    mas = MASRun()

    if verbose: print("=== Perception Agent ===")
    perc_dispatch = {
        "inspect": lambda flight_dir, metadata: T.inspect_data(flight_dir, metadata),
        "featurize": lambda flight_dir, metadata, out: T.featurize_flights(flight_dir, metadata, out),
    }
    mas.perception = run_agent(client_factory(), PERCEPTION_PROMPT,
                               f"Flight directory: {flight_dir}\nMetadata: {metadata}\n"
                               f"Feature output: {feats}\nRun perception and report.",
                               max_turns=max_turns_per_agent, verbose=verbose,
                               tools_schema=PERCEPTION_TOOLS, dispatch=perc_dispatch)

    if verbose: print("\n=== Analysis Agent ===")
    analysis_dispatch = {
        "classify": lambda feats, model, out: T.classify_flights(feats, model, out),
        "list_top_predictions": lambda preds_path, top_k=10: MT.list_top_predictions(preds_path, top_k),
        "announce_task": lambda flight, urgency, estimated_hours, justification:
            MT.announce_task(flight, urgency, estimated_hours, justification),
    }
    mas.analysis = run_agent(client_factory(), ANALYSIS_PROMPT,
                             f"Features: {feats}\nModel: {model_path}\nPredictions output: {preds}\n"
                             "Run analysis and announce tasks for flights needing action.",
                             max_turns=max_turns_per_agent, verbose=verbose,
                             tools_schema=ANALYSIS_TOOLS, dispatch=analysis_dispatch)
    mas.tasks_announced = _trace_results(mas.analysis, "announce_task")

    if verbose: print(f"\n=== Optimization Agent ({len(mas.tasks_announced)} task(s)) ===")
    opt_dispatch = {
        "request_bids": lambda pool_path, urgency, estimated_hours, task_id, earliest_day=0:
            MT.request_bids(pool_path, urgency, estimated_hours, earliest_day, task_id),
        "award_bid": lambda pool_path, queue_path, task_id, flight, p_maintenance, urgency, bay, day:
            MT.award_bid(pool_path, queue_path, task_id, flight, p_maintenance, urgency, bay, day),
    }
    pred_lookup = {}
    for r in _trace_results(mas.analysis, "list_top_predictions"):
        for p in r["predictions"]:
            pred_lookup[p["flight"]] = p["p_maintenance"]

    for i, task in enumerate(mas.tasks_announced):
        if verbose: print(f"--- negotiating task {i+1}/{len(mas.tasks_announced)}: {task['flight']} ---")
        opt_run = run_agent(client_factory(), OPTIMIZATION_PROMPT,
                            f"Task: flight={task['flight']}, urgency={task['urgency']}, "
                            f"estimated_hours={task['estimated_hours']}, task_id=task_{i}\n"
                            f"Pool: {pool_path}\nQueue: {queue_path}\n"
                            f"p_maintenance for the award_bid call: {pred_lookup.get(task['flight'], 0.5)}",
                            max_turns=max_turns_per_agent, verbose=verbose,
                            tools_schema=OPTIMIZATION_TOOLS, dispatch=opt_dispatch)
        mas.optimizations.append(opt_run)
        awarded = _trace_result(opt_run, "award_bid")
        if awarded and awarded.get("queued"):
            mas.awards.append(awarded)

    if verbose: print("\n=== HITL Agent ===")
    bundle = {
        "perception": _trace_result(mas.perception, "inspect"),
        "n_features": (_trace_result(mas.perception, "featurize") or {}).get("n_features"),
        "n_scored": (_trace_result(mas.analysis, "classify") or {}).get("n_scored"),
        "n_high": (_trace_result(mas.analysis, "classify") or {}).get("n_high"),
        "tasks_announced": mas.tasks_announced,
        "awards": mas.awards,
    }
    mas.hitl = run_agent(client_factory(), HITL_PROMPT, json.dumps(bundle, indent=2),
                         max_turns=2, verbose=verbose, tools_schema=[], dispatch={})
    return mas
