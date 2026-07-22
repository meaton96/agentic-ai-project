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
from fastapi.middleware.cors import CORSMiddleware

from server.routes import router
from server.run_manager import LaunchFn, RunManager, default_launch_run

# The Vite dev server (web/) runs on a different port, so it's a different
# origin as far as the browser's CORS check is concerned, even though both
# sides are on 127.0.0.1/localhost — this does NOT loosen the "bind
# 127.0.0.1 only" invariant above (that's about which network interface
# this process listens on; CORS is about which browser-tab origins may
# read the response), it just allows the local frontend to call this API.
_DEV_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:4173", "http://127.0.0.1:4173",
]


def create_app(launch_fn: Optional[LaunchFn] = None) -> FastAPI:
    """launch_fn is exposed here (not just on RunManager) so tests can build
    a whole app wired to the stubbed-ModelClient pattern via a single call."""
    app = FastAPI(title="agentic-ml-classification frontend server")
    app.state.run_manager = RunManager(launch_fn=launch_fn or default_launch_run)
    app.add_middleware(
        CORSMiddleware, allow_origins=_DEV_ORIGINS, allow_methods=["*"], allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
