"""
Shared plumbing for scripts/*.py entry points: model endpoint
resolution, run-directory creation, and trace-log writing. Pulled out
once it was needed identically by run_profiler_agent.py,
run_modeling_agent.py, and run_orchestrator.py.
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
