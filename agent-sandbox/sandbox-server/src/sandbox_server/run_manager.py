"""Tracks in-flight and historical runs, and bridges sandbox-core's
agent_loop to the SSE endpoint in routes/stream.py.

Design: a run is launched as an `asyncio.create_task` on the server's own
event loop (not a thread or subprocess — sandbox-core's runtime is all
async I/O, httpx + the mcp SDK, so there's no CPU-bound work to isolate).
Because everything shares one event loop, the bridge to SSE doesn't need
any cross-thread synchronization: `StreamingEventLog` below subclasses
sandbox_core's `EventLog` and overrides `append()` to call the real
`EventLog.append()` (so the on-disk events.jsonl stays the single source of
truth sandbox-core always produced) and then hand the same event to a
callback that fans it out to every subscriber queue for that run_id. Both
`append()` and `subscribe()`/`unsubscribe()` below are plain synchronous
functions with no `await` in them, which is what makes the fan-out
race-free under asyncio's cooperative scheduling: nothing else can run
between "snapshot the events seen so far" and "register this new
subscriber queue", so a client that connects mid-run can never miss an
event or see one twice.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Literal

from sandbox_core.runtime.agent_loop import execute_run
from sandbox_core.runtime.credential_store import YamlCredentialStore
from sandbox_core.runtime.event_log import EventLog, read_events
from sandbox_core.schemas.agent_spec import AgentSpec
from sandbox_core.schemas.events import AgentResultEvent, ErrorEvent, Event
from sandbox_core.schemas.run_spec import RunSpec

RunStatus = Literal["running", "completed", "errored", "truncated"]

Runner = Callable[..., Awaitable[Event]]


def _is_terminal(event: Event) -> bool:
    return isinstance(event, (AgentResultEvent, ErrorEvent))


def derive_status(events: list[Event]) -> RunStatus:
    for event in reversed(events):
        if isinstance(event, AgentResultEvent):
            return "completed"
        if isinstance(event, ErrorEvent):
            return "truncated" if (event.context or {}).get("phase") == "max_turns" else "errored"
    # a run with no terminal event and nothing tracking it live (server
    # restarted, or the CLI process that wrote it died before ever
    # appending anything) — "errored" is the least misleading of the four
    # states available; it is never reported for a run this process is
    # still actively running (see TrackedRun.status, set independently).
    return "errored"


@dataclass
class TrackedRun:
    run_id: str
    agent_id: str
    task: str
    created_at: datetime
    status: RunStatus = "running"
    events: list[Event] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)


class StreamingEventLog(EventLog):
    """EventLog with a hook: every event still lands in events.jsonl exactly
    as sandbox-core's own EventLog would write it, but also gets handed to
    `on_event` synchronously, in the same append() call. This is the whole
    bridge from the (unmodified) agent_loop to the SSE layer — agent_loop
    only ever calls `event_log.append(...)`, and has no idea this subclass
    exists."""

    def __init__(self, run_id: str, output_root: Path, on_event: Callable[[Event], None]):
        super().__init__(run_id, output_root=output_root)
        self._on_event = on_event

    def append(self, event: Event) -> Event:
        event = super().append(event)
        self._on_event(event)
        return event


class RunManager:
    def __init__(
        self,
        *,
        output_root: Path,
        runner: Runner = execute_run,
        resolver_factory: Callable[[], object] = YamlCredentialStore,
    ):
        self.output_root = Path(output_root)
        self._runner = runner
        self._resolver_factory = resolver_factory
        self._tracked: dict[str, TrackedRun] = {}

    # -- launching -----------------------------------------------------

    def launch_run(self, agent: AgentSpec, task: str) -> str:
        run = RunSpec(agent_id=agent.id, task=task)
        tracked = TrackedRun(run_id=run.run_id, agent_id=agent.id, task=task, created_at=run.created_at)
        self._tracked[run.run_id] = tracked

        event_log = StreamingEventLog(
            run.run_id, output_root=self.output_root, on_event=lambda e: self._broadcast(tracked, e)
        )
        asyncio.create_task(self._execute(agent, run, event_log, tracked))
        return run.run_id

    async def _execute(self, agent: AgentSpec, run: RunSpec, event_log: StreamingEventLog, tracked: TrackedRun):
        try:
            resolver = self._resolver_factory()
            await self._runner(agent, run, resolver=resolver, event_log=event_log)
        except Exception as exc:
            # execute_run can raise before ever touching event_log (a
            # missing credential, an MCP server that refuses to connect) —
            # those failures still need to reach the UI as a terminal event,
            # so synthesize one and push it through the same append() path
            # every other event takes (persists it too, not just broadcasts).
            event_log.append(
                ErrorEvent(
                    run_id=run.run_id,
                    seq=0,
                    ts=datetime.now(timezone.utc),
                    agent_id=agent.id,
                    message=str(exc),
                    context={"phase": "launch"},
                )
            )

    def _broadcast(self, tracked: TrackedRun, event: Event) -> None:
        tracked.events.append(event)
        if isinstance(event, AgentResultEvent):
            tracked.status = "completed"
        elif isinstance(event, ErrorEvent):
            tracked.status = "truncated" if (event.context or {}).get("phase") == "max_turns" else "errored"
        for queue in tuple(tracked.subscribers):
            queue.put_nowait(event)

    # -- streaming -------------------------------------------------------

    def subscribe(self, run_id: str) -> tuple[list[Event], asyncio.Queue | None, TrackedRun | None]:
        """Returns (events-so-far, live queue or None, tracked run or None).
        A None queue means: either the run isn't tracked by this process at
        all (caller should fall back to a one-shot disk replay), or it's
        tracked but already finished (the returned events list is already
        the complete log — nothing more will ever be broadcast)."""
        tracked = self._tracked.get(run_id)
        if tracked is None:
            return [], None, None
        queue: asyncio.Queue | None = None
        if tracked.status == "running":
            queue = asyncio.Queue()
            tracked.subscribers.append(queue)
        return list(tracked.events), queue, tracked

    def unsubscribe(self, tracked: TrackedRun, queue: asyncio.Queue) -> None:
        if queue in tracked.subscribers:
            tracked.subscribers.remove(queue)

    def read_disk_events(self, run_id: str) -> list[Event] | None:
        """For a run this process never tracked (server restarted, or it was
        launched by the CLI) — read once from events.jsonl. None means no
        such run exists on disk at all."""
        run_dir = self.output_root / run_id
        if not run_dir.exists():
            return None
        return read_events(run_dir / "events.jsonl")

    # -- listing / detail --------------------------------------------------

    def list_runs(self) -> list[dict]:
        run_ids = set(self._tracked)
        if self.output_root.exists():
            run_ids.update(p.name for p in self.output_root.iterdir() if p.is_dir())

        summaries = []
        for run_id in run_ids:
            summary = self._summarize(run_id)
            if summary is not None:
                summaries.append(summary)
        summaries.sort(key=lambda s: s["created_at"])
        return summaries

    def get_run_detail(self, run_id: str) -> dict | None:
        tracked = self._tracked.get(run_id)
        if tracked is not None:
            events = tracked.events
            agent_id, created_at, status = tracked.agent_id, tracked.created_at, tracked.status
        else:
            events = self.read_disk_events(run_id)
            if events is None:
                return None
            agent_id = events[0].agent_id if events else None
            created_at = events[0].ts if events else None
            status = derive_status(events)

        return {
            "run_id": run_id,
            "agent_id": agent_id,
            "created_at": created_at,
            "status": status,
            "events": [event.model_dump(mode="json") for event in events],
        }

    def _summarize(self, run_id: str) -> dict | None:
        tracked = self._tracked.get(run_id)
        if tracked is not None:
            turn_count = sum(1 for e in tracked.events if e.type == "llm_request")
            return {
                "run_id": run_id,
                "agent_id": tracked.agent_id,
                "created_at": tracked.created_at,
                "status": tracked.status,
                "turn_count": turn_count,
            }

        events = self.read_disk_events(run_id)
        if events is None:
            return None
        run_dir = self.output_root / run_id
        turn_count = sum(1 for e in events if e.type == "llm_request")
        created_at = events[0].ts if events else datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc)
        agent_id = events[0].agent_id if events else None
        return {
            "run_id": run_id,
            "agent_id": agent_id,
            "created_at": created_at,
            "status": derive_status(events),
            "turn_count": turn_count,
        }
