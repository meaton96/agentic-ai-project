"""
All API routes. Every handler either reads filesystem state the pipeline
already produces (datasets root, runs/<run_id>/*, artifacts/reports/
leaderboard.jsonl) via agentic_ml.paths, or calls RunManager to launch a
run through agentic_ml's own entry-point scripts — no ML/agent logic
lives here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from agentic_ml.harness.leaderboard import read_leaderboard
from agentic_ml.orchestrator import agent_registry
from agentic_ml.paths import datasets_root
from agentic_ml.paths import leaderboard_path as resolve_leaderboard_path
from agentic_ml.paths import run_dir as resolve_run_dir

from server.events_io import stream_event_lines
from server.prompts import delete_override, get_prompt_info, list_known_agents, write_override
from server.run_manager import RunAlreadyActiveError, RunConfig, RunManager
from server.schemas import (
    DatasetInfo,
    LaunchRunRequest,
    LaunchRunResponse,
    PromptInfo,
    PromptOverrideRequest,
    WorkflowCatalogResponse,
)
from server.static_topology import build_static_topology
from server.summaries import assemble_run_summary, list_all_run_ids

router = APIRouter()


def _run_manager(request: Request) -> RunManager:
    return request.app.state.run_manager


# --- datasets ---------------------------------------------------------


@router.get("/api/datasets", response_model=list[DatasetInfo])
def list_datasets() -> list[DatasetInfo]:
    root = datasets_root()
    if not root.exists():
        return []
    out = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            out.append(DatasetInfo(
                filename=str(path.relative_to(root)),
                size=stat.st_size,
                modified=stat.st_mtime,
            ))
    return out


def _resolve_dataset_path(dataset: str) -> Path:
    root = datasets_root().resolve()
    candidate = (datasets_root() / dataset).resolve()
    if not candidate.is_relative_to(root):
        raise HTTPException(422, f"dataset {dataset!r} escapes the datasets root")
    if not candidate.is_file():
        raise HTTPException(422, f"dataset {dataset!r} does not exist under the datasets root")
    return candidate


# --- runs ---------------------------------------------------------------


@router.post("/api/runs", response_model=LaunchRunResponse, status_code=202)
def launch_run(body: LaunchRunRequest, request: Request) -> LaunchRunResponse:
    run_manager = _run_manager(request)
    dataset_path = _resolve_dataset_path(body.dataset)

    config = RunConfig(
        dataset=str(dataset_path),
        orchestrator=body.orchestrator,
        target=body.target,
        goal=body.goal,
        strategy=body.strategy,
        max_candidates=body.max_candidates,
        model_endpoint=body.model_endpoint,
        skip_feature_engineering=body.skip_feature_engineering,
        use_prompt_overrides=body.use_prompt_overrides,
    )
    try:
        tracked = run_manager.start_run(config)
    except RunAlreadyActiveError as exc:
        raise HTTPException(409, f"a run is already active: {exc.active_run_id}") from exc

    return LaunchRunResponse(run_id=tracked.run_id, status=tracked.status)


@router.get("/api/runs")
def list_runs(request: Request) -> list[dict]:
    run_manager = _run_manager(request)
    summaries = [assemble_run_summary(run_id, run_manager) for run_id in list_all_run_ids(run_manager)]
    summaries = [s for s in summaries if s is not None]
    summaries.sort(key=lambda s: s["run_id"])
    return summaries


@router.get("/api/runs/{run_id}")
def get_run(run_id: str, request: Request) -> dict:
    run_manager = _run_manager(request)
    summary = assemble_run_summary(run_id, run_manager)
    if summary is None:
        raise HTTPException(404, f"run {run_id!r} not found")
    return summary


@router.get("/api/runs/{run_id}/events")
def stream_run_events(run_id: str, request: Request) -> StreamingResponse:
    run_manager = _run_manager(request)
    run_dir = resolve_run_dir(run_id)
    if run_manager.get_run(run_id) is None and not run_dir.exists():
        raise HTTPException(404, f"run {run_id!r} not found")

    def generate():
        for line in stream_event_lines(run_dir, is_still_running=lambda: run_manager.is_running(run_id)):
            yield f"data: {line}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# --- transcripts ----------------------------------------------------------


@router.get("/api/runs/{run_id}/transcripts")
def list_transcripts(run_id: str) -> list[str]:
    run_dir = resolve_run_dir(run_id)
    if not run_dir.exists():
        raise HTTPException(404, f"run {run_id!r} not found")
    transcripts_dir = run_dir / "transcripts"
    if not transcripts_dir.exists():
        return []
    return sorted(p.name for p in transcripts_dir.glob("*.json"))


@router.get("/api/runs/{run_id}/transcripts/{name}")
def get_transcript(run_id: str, name: str) -> list:
    """make_transcript_writer (cli_common.py) always writes a JSON array of
    prettified messages, never a bare object — the return type here matches
    that, not a generic dict."""
    run_dir = resolve_run_dir(run_id)
    transcripts_dir = (run_dir / "transcripts").resolve()
    path = (transcripts_dir / name).resolve()
    if path.parent != transcripts_dir or not path.is_file():
        raise HTTPException(404, f"transcript {name!r} not found for run {run_id!r}")
    return json.loads(path.read_text())


# --- leaderboard ------------------------------------------------------------


@router.get("/api/leaderboard")
def get_leaderboard(run_id: Optional[str] = None) -> list[dict]:
    entries = read_leaderboard(resolve_leaderboard_path())
    if run_id is not None:
        entries = [e for e in entries if e.get("run_id") == run_id]
    return entries


# --- workflow catalog (read-only visualization, Builder v0) -------------
# Editing/swapping/saving workflows is out of scope for this pass — this
# just proves the graph can accurately represent what the pipeline does.


def _dynamic_catalog_node(spec: agent_registry.AgentSpec) -> dict:
    # Every deterministic (non-LLM) agent in the catalog says so verbatim
    # at the start of its description (see agent_registry.py) — deriving
    # `kind` from that instead of a hand-maintained id list keeps this in
    # sync with the registry the same way the node list itself is.
    kind = "gate" if spec.description.startswith("Deterministic") else "agent"
    return {
        "id": spec.agent_id,
        "label": spec.title,
        "kind": kind,
        "description": spec.description,
        "data": {"when_to_use": spec.when_to_use, "required_state": spec.required_state},
    }


def _build_dynamic_catalog() -> dict:
    # Unordered/loosely-connected on purpose: the dynamic orchestrator's
    # planner picks from this catalog one step at a time based on live run
    # state (agent_registry.py's required_state gates), so there is no
    # single fixed sequence to draw edges for — forcing one would
    # misrepresent how this orchestrator actually works. required_state is
    # surfaced per-node instead (rendered in the side panel).
    nodes = [_dynamic_catalog_node(spec) for spec in agent_registry.AGENTS.values()]
    return {"nodes": nodes, "edges": []}


@router.get("/api/workflow-catalog", response_model=WorkflowCatalogResponse)
def get_workflow_catalog(type: Literal["static", "dynamic"] = "static") -> WorkflowCatalogResponse:
    catalog = build_static_topology() if type == "static" else _build_dynamic_catalog()
    return WorkflowCatalogResponse(**catalog)


# --- prompts ------------------------------------------------------------
# Read-only against the pipeline's shipped prompts/*.md; writes only ever
# touch this app's own override directory (server/prompts.py), never
# ../agentic-ml-classification.


def _require_known_agent(agent: str) -> None:
    if agent not in list_known_agents():
        raise HTTPException(404, f"unknown agent {agent!r}")


@router.get("/api/prompts", response_model=list[PromptInfo])
def list_prompts() -> list[PromptInfo]:
    return [PromptInfo(**get_prompt_info(agent)) for agent in list_known_agents()]


@router.put("/api/prompts/{agent}", response_model=PromptInfo)
def put_prompt_override(agent: str, body: PromptOverrideRequest) -> PromptInfo:
    _require_known_agent(agent)
    return PromptInfo(**write_override(agent, body.content))


@router.delete("/api/prompts/{agent}", response_model=PromptInfo)
def delete_prompt_override(agent: str) -> PromptInfo:
    _require_known_agent(agent)
    delete_override(agent)
    return PromptInfo(**get_prompt_info(agent))
