"""SSE endpoint: replay-then-tail for a run this server is actively
running, or a one-shot replay for a run it only knows about from disk
(historical, or launched by a previous server process / the CLI directly).
See run_manager.py's module docstring for how the replay-without-gaps
guarantee is made."""

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from sandbox_core.runtime.event_log import EventAdapter
from sandbox_core.schemas.events import AgentResultEvent, ErrorEvent, Event

router = APIRouter(prefix="/runs", tags=["runs"])

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
# How often the live branch below wakes up even with nothing in the queue,
# purely to re-check whether the client is still connected. Plain
# `await queue.get()` would block through a disconnect indefinitely — the
# ASGI server does not interrupt an in-progress generator on socket close,
# so a ~never-terminating tool call would otherwise leak this subscription.
POLL_INTERVAL = 1.0


def _sse_line(event: Event) -> str:
    return f"data: {EventAdapter.dump_json(event).decode('utf-8')}\n\n"


def _is_terminal(event: Event) -> bool:
    return isinstance(event, (AgentResultEvent, ErrorEvent))


@router.get("/{run_id}/stream")
async def stream_run(run_id: str, request: Request) -> StreamingResponse:
    run_manager = request.app.state.run_manager
    replay, queue, tracked = run_manager.subscribe(run_id)

    if tracked is None:
        events = run_manager.read_disk_events(run_id)
        if events is None:
            raise HTTPException(404, f"run {run_id!r} not found")

        async def replay_only():
            for event in events:
                yield _sse_line(event)

        return StreamingResponse(replay_only(), media_type="text/event-stream", headers=SSE_HEADERS)

    async def generate():
        try:
            for event in replay:
                yield _sse_line(event)
                if _is_terminal(event):
                    return
            if queue is None:
                return  # tracked but already finished; `replay` was the whole log
            while True:
                if await request.is_disconnected():
                    return  # client went away — the run itself keeps going untouched
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=POLL_INTERVAL)
                except asyncio.TimeoutError:
                    continue
                yield _sse_line(event)
                if _is_terminal(event):
                    return
        finally:
            if queue is not None:
                run_manager.unsubscribe(tracked, queue)

    return StreamingResponse(generate(), media_type="text/event-stream", headers=SSE_HEADERS)
