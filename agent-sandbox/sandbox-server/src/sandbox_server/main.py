"""FastAPI app factory. A thin wrapper over sandbox-core: this process owns
no agent-execution logic of its own beyond bridging agent_loop's events to
SSE (see run_manager.py) — routes call straight into sandbox_core.

    uvicorn sandbox_server.main:app --reload

Single-user local dev server — bind 127.0.0.1 only, same rationale as
agent-frontend's server (see its CLAUDE.md): this can trigger execution of
whatever MCP tools an AgentSpec wires up.
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .pipeline_run_manager import PipelineRunManager
from .routes.agents import router as agents_router
from .routes.credentials import router as credentials_router
from .routes.pipeline_runs import router as pipeline_runs_router
from .routes.pipelines import router as pipelines_router
from .routes.runs import router as runs_router
from .routes.stream import router as stream_router
from .run_manager import Runner, RunManager

_DEV_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:4173", "http://127.0.0.1:4173",
]


def create_app(
    *,
    specs_dir: Optional[Path] = None,
    pipelines_dir: Optional[Path] = None,
    output_root: Optional[Path] = None,
    pipeline_runs_root: Optional[Path] = None,
    runner: Optional[Runner] = None,
) -> FastAPI:
    """specs_dir/output_root/runner are exposed here (not just read from env
    at import time) so tests can point a whole app at a tmp_path and a fake
    agent_loop in one call, the same way agent-frontend's create_app(launch_fn=...)
    does for its RunManager."""
    app = FastAPI(title="sandbox-server")
    app.state.specs_dir = Path(specs_dir) if specs_dir is not None else config.specs_dir()
    app.state.pipelines_dir = Path(pipelines_dir) if pipelines_dir is not None else config.pipelines_dir()
    resolved_output_root = Path(output_root) if output_root is not None else config.output_root()

    run_manager_kwargs = {"output_root": resolved_output_root}
    if runner is not None:
        run_manager_kwargs["runner"] = runner
    app.state.run_manager = RunManager(**run_manager_kwargs)

    pipeline_run_manager_kwargs = {
        "specs_dir": app.state.specs_dir,
        "output_root": resolved_output_root,
        "pipeline_runs_root": Path(pipeline_runs_root) if pipeline_runs_root is not None else config.pipeline_runs_root(),
    }
    if runner is not None:
        pipeline_run_manager_kwargs["runner"] = runner
    app.state.pipeline_run_manager = PipelineRunManager(**pipeline_run_manager_kwargs)

    app.add_middleware(CORSMiddleware, allow_origins=_DEV_ORIGINS, allow_methods=["*"], allow_headers=["*"])
    app.include_router(agents_router)
    app.include_router(credentials_router)
    app.include_router(runs_router)
    app.include_router(stream_router)
    app.include_router(pipelines_router)
    app.include_router(pipeline_runs_router)
    return app


app = create_app()
