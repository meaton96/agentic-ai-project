"""
Shared plumbing for scripts/*.py entry points: model endpoint
resolution, run-directory creation, trace-log writing, and per-agent
transcript writing. Pulled out once it was needed identically by
run_profiler_agent.py, run_modeling_agent.py, and run_orchestrator.py.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Callable, Optional


def resolve_model_endpoint(
    use_gateway: bool,
    model: Optional[str],
    default_direct_model: str,
    default_gateway_model: str,
) -> tuple[str, str, str]:
    """Returns (base_url, api_key, default_model)."""
    if use_gateway:
        base_url = os.environ.get("MODEL_GATEWAY_BASE_URL", "http://localhost:4000/v1")
        api_key = os.environ.get("LITELLM_MASTER_KEY", "")
        default_model = model or default_gateway_model
    else:
        base_url = os.environ["RIT_BASE_URL"]
        api_key = os.environ["RIT_API_KEY"]
        default_model = model or os.environ.get("RIT_DEFAULT_MODEL", default_direct_model)
    return base_url, api_key, default_model


def make_run_dir(run_id: Optional[str]) -> tuple[str, Path]:
    run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def make_tracer(trace_path: Path) -> Callable[..., None]:
    def trace(event: str, **fields):
        with open(trace_path, "a") as f:
            f.write(json.dumps({"ts": time.time(), "event": event, **fields}) + "\n")
    return trace


def _prettify_tool_call(tool_call: dict) -> dict:
    """Parses a tool call's JSON-string arguments back into a real nested
    object, so the transcript file reads as JSON, not escaped JSON-in-JSON."""
    out = dict(tool_call)
    func = dict(out.get("function", {}))
    args = func.get("arguments")
    if isinstance(args, str):
        try:
            func["arguments"] = json.loads(args)
        except json.JSONDecodeError:
            pass
    out["function"] = func
    return out


def _prettify_message(message: dict) -> dict:
    """Same idea as _prettify_tool_call, applied to a full message: a
    'tool' role message's content is the tool's JSON result as a string
    (that's the wire format agent_runtime.py uses); parse it back into a
    real object. Also opportunistically parses a plain assistant message's
    content if it looks like the JSON almost every agent in this pipeline
    is instructed to respond with — pure readability, never changes what
    was actually said."""
    out = dict(message)
    if "tool_calls" in out:
        out["tool_calls"] = [_prettify_tool_call(tc) for tc in out["tool_calls"]]
    content = out.get("content")
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                out["content"] = json.loads(stripped)
            except json.JSONDecodeError:
                pass
    return out


def make_transcript_writer(run_dir: Path) -> Callable[[str, list[dict]], Path]:
    """Returns write(agent_name, messages) -> Path. Each call writes a new
    file under runs/<run_id>/transcripts/, numbered per agent_name (so
    calling this twice for "modeling" — e.g. two candidates in one
    orchestrator run — produces modeling_01.json and modeling_02.json,
    not a clobber). This is the full conversation each agent actually
    had: system prompt, tool calls with real arguments, tool results, and
    the final response — not just the metadata trace.jsonl records."""
    transcripts_dir = run_dir / "transcripts"
    counters: dict[str, int] = {}

    def write(agent_name: str, messages: list[dict]) -> Path:
        counters[agent_name] = counters.get(agent_name, 0) + 1
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        path = transcripts_dir / f"{agent_name}_{counters[agent_name]:02d}.json"
        pretty = [_prettify_message(m) for m in messages]
        path.write_text(json.dumps(pretty, indent=2, default=str))
        return path

    return write
