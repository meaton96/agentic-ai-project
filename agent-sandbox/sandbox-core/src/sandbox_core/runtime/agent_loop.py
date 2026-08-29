import asyncio
from contextlib import ExitStack
from datetime import datetime, timezone

from sandbox_core.schemas.agent_spec import AgentSpec
from sandbox_core.schemas.credentials import CredentialResolver
from sandbox_core.schemas.events import AgentResultEvent, ErrorEvent, Event
from sandbox_core.schemas.run_spec import RunSpec

from .event_log import EventLog
from .strands_adapter import build_agent, limits_for, max_turns_for

# stop_reasons a well-behaved run can end on without it being our own max_turns
# truncation; anything else unexpected becomes a generic ErrorEvent.
_SUCCESS_STOP_REASONS = {"end_turn", "tool_use", "stop_sequence"}


async def execute_run(agent: AgentSpec, run: RunSpec, *, resolver: CredentialResolver, event_log: EventLog) -> Event:
    """Full orchestration used by the CLI and sandbox-server: resolves
    credentials, builds a strands.Agent wired to every MCP server on the spec,
    runs it to completion (or to its turn budget), and guarantees the agent
    (and every MCP session it opened) is cleaned up on the way out — even on
    exception. All of the actual tool-calling loop, model retries, and MCP
    session lifecycle now live in Strands; this function's job is just
    building the agent and translating its terminal state into our Event
    union, exactly as the hand-rolled loop used to return one."""
    max_turns = max_turns_for(agent, run)

    with ExitStack() as stack:
        # Agent(tools=[MCPClient, ...]) synchronously loads tools from every
        # MCP server (via a background thread strands manages internally) —
        # offloaded to a worker thread so it can't block a shared asyncio
        # event loop (sandbox-server runs every run as a task on its own loop).
        strands_agent, hooks = await asyncio.to_thread(
            build_agent, agent, run, resolver=resolver, event_log=event_log, stack=stack
        )

        try:
            result = await strands_agent.invoke_async(run.task, limits=limits_for(max_turns))
        except Exception as exc:
            if hooks.captured_model_error is not None:
                # the model call itself already produced a typed ErrorEvent via
                # the AfterModelCallEvent hook; that's the terminal event we
                # return, matching the old ModelClient's "never raise, return
                # an ErrorEvent" contract.
                return hooks.captured_model_error
            return event_log.append(
                ErrorEvent(
                    run_id=run.run_id,
                    seq=0,
                    ts=datetime.now(timezone.utc),
                    agent_id=agent.id,
                    message=str(exc),
                    context={"phase": "error"},
                )
            )

        if result.stop_reason == "limit_turns":
            return event_log.append(
                ErrorEvent(
                    run_id=run.run_id,
                    seq=0,
                    ts=datetime.now(timezone.utc),
                    agent_id=agent.id,
                    message=f"run truncated: hit max_turns ({max_turns}) without a final answer",
                    context={"phase": "max_turns", "max_turns": max_turns},
                )
            )

        if result.stop_reason not in _SUCCESS_STOP_REASONS:
            return event_log.append(
                ErrorEvent(
                    run_id=run.run_id,
                    seq=0,
                    ts=datetime.now(timezone.utc),
                    agent_id=agent.id,
                    message=f"run stopped unexpectedly: stop_reason={result.stop_reason!r}",
                    context={"phase": "stop_reason", "stop_reason": result.stop_reason},
                )
            )

        text = "".join(block.get("text", "") for block in result.message.get("content", []))
        turns_used = len(strands_agent.event_loop_metrics.cycle_durations)
        return event_log.append(
            AgentResultEvent(
                run_id=run.run_id,
                seq=0,
                ts=datetime.now(timezone.utc),
                agent_id=agent.id,
                final_output=text,
                turns_used=turns_used,
            )
        )
