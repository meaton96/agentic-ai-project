"""
FastAPI app factory. This server can trigger execution of sandboxed
candidate code (via RunManager -> the pipeline's own agent loops and
sklearn fits) — it must never listen on the network. Run it with:

    uvicorn server.app:app --host 127.0.0.1 --reload

Do not pass --host 0.0.0.0 or omit --host: the default bind here is
explicit and intentional (see ../CLAUDE.md, "Server/execution
invariants").
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI

from server.routes import router
from server.run_manager import LaunchFn, RunManager, default_launch_run


def create_app(launch_fn: Optional[LaunchFn] = None) -> FastAPI:
    """launch_fn is exposed here (not just on RunManager) so tests can build
    a whole app wired to the stubbed-ModelClient pattern via a single call."""
    app = FastAPI(title="agentic-ml-classification frontend server")
    app.state.run_manager = RunManager(launch_fn=launch_fn or default_launch_run)
    app.include_router(router)
    return app


app = create_app()
