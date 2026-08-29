"""
Shared plumbing for scripts/*.py entry points: model endpoint
resolution, run-directory creation, trace-log writing, and per-agent
transcript writing. Mirrors agentic_ml.cli_common in the sibling
agentic-ml-classification project.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from dotenv import load_dotenv

load_dotenv()

from resource_scheduler.paths import run_dir as resolve_run_dir


def resolve_model_endpoint(
    use_gateway: bool,
    model: Optional[str],
    default_direct_model: str,
    default_gateway_model: str,
    use_local: bool = False,
) -> tuple[str, str, str]:
    """Returns (base_url, api_key, default_model). Same three-endpoint
    resolution order as agentic_ml.cli_common.resolve_model_endpoint:
    local server > LiteLLM gateway > RIT directly."""
    if use_local:
        base_url = os.environ.get("LOCAL_MODEL_BASE_URL", "http://localhost:8000/v1")
        api_key = os.environ.get("LOCAL_MODEL_API_KEY", "not-needed")
        default_model = model or os.environ.get("LOCAL_DEFAULT_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    elif use_gateway:
        base_url = os.environ.get("MODEL_GATEWAY_BASE_URL", "http://localhost:4000/v1")
        api_key = os.environ.get("LITELLM_MASTER_KEY", "")
        default_model = model or default_gateway_model
    else:
        base_url = os.environ.get("RIT_BASE_URL")
        api_key = os.environ.get("RIT_API_KEY")

        if not base_url or not api_key:
            raise ValueError(
                "Missing RIT configuration! Ensure RIT_BASE_URL and RIT_API_KEY "
                "are set in your .env file or system environment."
            )
        default_model = model or os.environ.get("RIT_DEFAULT_MODEL", default_direct_model)
    return base_url, api_key, default_model


def make_run_dir(run_id: Optional[str]) -> tuple[str, Path]:
    run_id = run_id or f"run_{datetime.now().strftime('%y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
    run_dir = resolve_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def make_tracer(trace_path: Path) -> Callable[..., None]:
    def trace(event: str, **fields):
        with open(trace_path, "a") as f:
            f.write(json.dumps({"ts": time.time(), "event": event, **fields}) + "\n")
    return trace


def _prettify_tool_call(tool_call: dict) -> dict:
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
