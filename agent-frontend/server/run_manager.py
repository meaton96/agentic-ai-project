"""
RunManager: launches one agentic_ml orchestrator run at a time in a
separate OS process (never a thread, never inline in the API event
loop — model fits are CPU-bound and the pipeline's own sandbox forks
subprocesses of its own).

Launching goes through an injectable `launch_fn(run_id, config)`,
called as the target of a `multiprocessing.Process` using the
"forkserver" context (not plain "fork", and not "spawn"):

  - Plain `fork()` from a process that already has background threads
    running — which a live FastAPI/uvicorn app always does (the ASGI
    server's own event loop plus its sync-route threadpool) — is
    unsafe: CPython 3.12+ emits "DeprecationWarning: this process is
    multi-threaded, use of fork() may lead to deadlocks in the child"
    because a thread that isn't the one calling fork() can vanish
    while holding a lock the child then blocks on forever. That's what
    the earlier fork-based version of this file hit under both
    uvicorn and pytest's TestClient.
  - "forkserver" avoids that: a small, dedicated, always-single-
    threaded server process is started once (lazily, before the first
    real run), and every actual run process is fork()'d from *that*
    process, never from this multi-threaded one. `set_forkserver_preload`
    below has it import `agentic_ml` up front, so the (heavy: sklearn,
    xgboost, lightgbm, pandas...) import cost is paid once for the
    forkserver's lifetime, not per run — unlike plain "spawn", which
    would re-pay that cost importing everything from scratch on every
    single `POST /api/runs`.
  - The tradeoff: unlike plain "fork", forkserver does NOT give a
    launched process a copy of this process's live memory — no
    monkeypatch or closure applied in the parent (e.g. pytest's
    `monkeypatch.setattr(ModelClient, "call", stub)`) is visible in the
    child. Everything the child needs — the launch function itself,
    and any stub — must be picklable and passed explicitly.
    `default_launch_run` below takes an optional `model_client_call`
    for exactly this: tests bind a module-level stub function via
    `functools.partial(default_launch_run, model_client_call=stub)`
    and hand that partial to `RunManager(launch_fn=...)` — the patch is
    applied inside the child itself, right before the real entry point
    runs, instead of being inherited from the parent's memory.
  - Same non-obvious consequence for os.environ: the forkserver
    controller process is started lazily, once, and kept alive — every
    process it forks afterward inherits *its* environment as of that
    first start, not whatever os.environ holds in the caller at the
    moment of each individual start_run() call. Concretely: a test that
    does `monkeypatch.setenv("AGENTIC_ML_DATA_ROOT", ...)` per-test
    would have every run after the first one silently land in the
    *first* test's data root instead of its own — confirmed by hand
    before landing this fix. start_run() below snapshots os.environ
    itself and every launch goes through _run_with_env(), which
    replaces the child's environ from that snapshot before anything
    else runs. This isn't specific to the stub — it would just as
    happily bite a real deployment that rotates RIT_API_KEY at runtime.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Callable, Literal, Optional

from agentic_ml.paths import run_dir as resolve_run_dir
from agentic_ml.prompt_loader import PROMPT_OVERRIDE_DIR_ENV_VAR

from server.mcp_config import mcp_server_url, resolve_mcp_settings
from server.prompts import resolve_override_dir

Orchestrator = Literal["static", "dynamic"]
ModelEndpoint = Literal["rit", "gateway", "local"]
RunStatus = Literal["pending", "running", "completed", "failed"]

_SCRIPT_NAMES = {"static": "run_orchestrator.py", "dynamic": "run_dynamic_orchestrator.py"}


@dataclass(frozen=True)
class RunConfig:
    """The union of what run_orchestrator.py and run_dynamic_orchestrator.py
    accept on the CLI that the API exposes (see server/schemas.py). `dataset`
    is already resolved to an absolute path by the route handler — RunManager
    and the launch function don't know about the datasets root."""

    dataset: str
    orchestrator: Orchestrator
    target: Optional[str] = None
    goal: Optional[str] = None
    strategy: Optional[str] = None
    max_candidates: Optional[int] = None
    model_endpoint: ModelEndpoint = "rit"
    skip_feature_engineering: bool = False
    use_prompt_overrides: bool = False
    use_mcp: bool = False


class RunAlreadyActiveError(Exception):
    def __init__(self, active_run_id: str):
        super().__init__(f"a run is already active: {active_run_id}")
        self.active_run_id = active_run_id


@dataclass
class TrackedRun:
    run_id: str
    config: RunConfig
    status: RunStatus
    started_at: float
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    process: Optional[multiprocessing.Process] = field(default=None, repr=False)


def build_argv(run_id: str, config: RunConfig) -> list[str]:
    script_name = _SCRIPT_NAMES[config.orchestrator]
    argv = [script_name, "--data", config.dataset, "--run-id", run_id]

    if config.target:
        argv += ["--target", config.target]
    if config.goal:
        argv += ["--goal", config.goal]
    if config.strategy:
        argv += ["--strategy", config.strategy]
    if config.model_endpoint == "gateway":
        argv += ["--use-gateway"]
    elif config.model_endpoint == "local":
        argv += ["--use-local"]

    if config.orchestrator == "static":
        if config.max_candidates is not None:
            argv += ["--max-candidates", str(config.max_candidates)]
        if config.skip_feature_engineering:
            argv += ["--skip-feature-engineering"]

    if config.orchestrator == "dynamic" and config.use_mcp:
        # run_orchestrator.py (static) has no --use-mcp flag at all — this
        # is deliberately gated on orchestrator, not just the checkbox, so
        # a stray use_mcp=True on a static launch is silently a no-op rather
        # than a CLI argparse error inside the launched child. The URL is
        # resolved from this app's own configs/mcp_server.json read (see
        # mcp_config.py) rather than relying on run_dynamic_orchestrator.
        # py's own --mcp-url default, so a deployment that points that
        # config file at a different host/port is honored automatically.
        argv += ["--use-mcp", "--mcp-url", mcp_server_url(resolve_mcp_settings())]

    if config.use_prompt_overrides:
        # Explicit path, always — see _run_with_env's PROMPT_OVERRIDE_DIR_ENV_VAR
        # stripping below for why this must never rely on the child
        # inheriting the env var instead.
        argv += ["--prompt-override-dir", str(resolve_override_dir())]

    return argv


def default_launch_run(
    run_id: str, config: RunConfig, model_client_call: Optional[Callable] = None,
) -> None:
    """Runs inside the forkserver-spawned child process. Imports the real
    CLI script as a module and calls its main() — the exact same entry
    point `python scripts/run_orchestrator.py ...` uses, including
    events.jsonl persistence via make_event_logger. A server-launched run
    and a manual CLI run are the same code path; nothing pipeline-specific
    lives here.

    model_client_call, if given, replaces agentic_ml.model_client.
    ModelClient.call for the lifetime of this child process — this is the
    stubbed-ModelClient pattern's hook point now that the child doesn't
    inherit the parent's memory (see module docstring). Must be a
    module-level (picklable-by-reference) function; a closure or lambda
    can't cross the forkserver boundary."""
    from server.pipeline_paths import pipeline_repo_root

    scripts_dir = str(pipeline_repo_root() / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    if model_client_call is not None:
        from agentic_ml.model_client import ModelClient
        ModelClient.call = model_client_call

    if config.orchestrator == "static":
        import run_orchestrator as entry
    else:
        import run_dynamic_orchestrator as entry

    sys.argv = build_argv(run_id, config)
    entry.main()


LaunchFn = Callable[[str, RunConfig], None]


def _run_with_env(env: dict[str, str], launch_fn: LaunchFn, run_id: str, config: RunConfig) -> None:
    """The actual Process target for every run, regardless of which
    launch_fn is configured: restores this child's os.environ from a
    snapshot taken in the parent at start_run() time (see module
    docstring — forkserver children otherwise inherit whatever env the
    forkserver controller process happened to have when *it* started),
    then delegates to launch_fn.

    PROMPT_OVERRIDE_DIR_ENV_VAR is dropped unconditionally, even though
    the snapshot is otherwise passed through as-is: agentic_ml.prompt_
    loader.resolve_prompt_override_dir() falls back to that env var
    whenever no explicit override_dir is passed, which would silently
    turn use_prompt_overrides=False into "whatever this server process's
    own environment happens to have" if it were ever set there (e.g.
    because this server also serves /api/prompts from that same
    directory). build_argv() above passes the directory explicitly via
    --prompt-override-dir when a run opts in — that's the only channel;
    the env var is never a valid way for a launched run to pick one up."""
    env = dict(env)
    env.pop(PROMPT_OVERRIDE_DIR_ENV_VAR, None)
    os.environ.clear()
    os.environ.update(env)
    launch_fn(run_id, config)


_process_context: Optional[multiprocessing.context.BaseContext] = None


def _launch_context() -> multiprocessing.context.BaseContext:
    """A single, lazily-created "forkserver" context shared by every
    RunManager in this process — the preloaded forkserver controller
    process is expensive enough to start (it imports agentic_ml) that it
    should only happen once, not per RunManager instance. Falls back to
    "spawn" on platforms without forkserver (Windows); fork is never used
    (see module docstring)."""
    global _process_context
    if _process_context is not None:
        return _process_context
    if "forkserver" in multiprocessing.get_all_start_methods():
        ctx = multiprocessing.get_context("forkserver")
        ctx.set_forkserver_preload(["agentic_ml"])
    else:
        ctx = multiprocessing.get_context("spawn")
    _process_context = ctx
    return ctx


class RunManager:
    """Tracks at most one active run. A second start_run() while one is
    active raises RunAlreadyActiveError — the route layer turns that into
    a 409. Historical runs (directories on disk this instance never
    launched) are intentionally NOT this class's concern; see
    server/summaries.py for how the two are merged for the read side."""

    def __init__(self, launch_fn: LaunchFn = default_launch_run):
        self._launch_fn = launch_fn
        self._lock = Lock()
        self._runs: dict[str, TrackedRun] = {}
        self._active_run_id: Optional[str] = None

    def start_run(self, config: RunConfig) -> TrackedRun:
        with self._lock:
            self._reap_active_locked()
            if self._active_run_id is not None:
                raise RunAlreadyActiveError(self._active_run_id)

            run_id = f"run_{uuid.uuid4().hex[:8]}"
            resolve_run_dir(run_id).mkdir(parents=True, exist_ok=True)

            env_snapshot = dict(os.environ)
            process = _launch_context().Process(
                target=_run_with_env, args=(env_snapshot, self._launch_fn, run_id, config), daemon=True,
            )

            tracked = TrackedRun(
                run_id=run_id, config=config, status="pending", started_at=time.time(), process=process,
            )
            self._runs[run_id] = tracked
            self._active_run_id = run_id

            process.start()
            tracked.status = "running"
            return tracked

    def _reap_active_locked(self) -> None:
        if self._active_run_id is None:
            return
        tracked = self._runs.get(self._active_run_id)
        if tracked is None or tracked.process is None:
            self._active_run_id = None
            return
        if tracked.process.is_alive():
            return

        tracked.finished_at = time.time()
        tracked.exit_code = tracked.process.exitcode
        tracked.status = "completed" if tracked.exit_code == 0 else "failed"
        if tracked.exit_code != 0:
            tracked.error = f"run process exited with code {tracked.exit_code}"
        self._active_run_id = None

    def get_run(self, run_id: str) -> Optional[TrackedRun]:
        with self._lock:
            if run_id == self._active_run_id:
                self._reap_active_locked()
            return self._runs.get(run_id)

    def list_tracked_runs(self) -> list[TrackedRun]:
        with self._lock:
            self._reap_active_locked()
            return list(self._runs.values())

    def is_running(self, run_id: str) -> bool:
        tracked = self.get_run(run_id)
        return tracked is not None and tracked.status in ("pending", "running")
