import json
import uuid
from datetime import datetime, timezone

from sandbox_core.schemas.agent_spec import AgentSpec
from sandbox_core.schemas.credentials import CredentialResolver
from sandbox_core.schemas.events import AgentResultEvent, ErrorEvent, ToolCallEvent, ToolResultEvent
from sandbox_core.schemas.run_spec import RunSpec

from .event_log import EventLog
from .mcp_client import connect_mcp_servers
from .model_client import ModelClient


def _parse_tool_arguments(function: dict) -> dict:
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        return json.loads(arguments) if arguments.strip() else {}
    return arguments or {}


def _tool_message_content(value) -> str:
    return value if isinstance(value, str) else json.dumps(value)


async def run_agent(agent: AgentSpec, run: RunSpec, *, model_client, servers, event_log: EventLog):
    """The tool-calling loop for one already-resolved agent + one already-connected
    set of MCP servers. Kept free of credential resolution / connection setup so
    tests can drive it with fake model_client / servers. Returns the terminal
    event: AgentResultEvent on success, ErrorEvent on failure or max_turns truncation."""
    max_turns = run.max_turns if run.max_turns is not None else agent.max_turns

    messages: list[dict] = [
        {"role": "system", "content": agent.system_prompt},
        {"role": "user", "content": run.task},
    ]

    tools = [spec for server in servers.values() for spec in server.tool_specs]
    tool_owner = {
        spec["function"]["name"]: name for name, server in servers.items() for spec in server.tool_specs
    }

    for turn in range(1, max_turns + 1):
        turn_result = await model_client.complete(messages, tools, run_id=run.run_id, agent_id=agent.id)
        event_log.append(turn_result.request)
        response = event_log.append(turn_result.response)

        if isinstance(response, ErrorEvent):
            return response

        if not response.tool_calls:
            return event_log.append(
                AgentResultEvent(
                    run_id=run.run_id,
                    seq=0,
                    ts=datetime.now(timezone.utc),
                    agent_id=agent.id,
                    final_output=response.content or "",
                    turns_used=turn,
                )
            )

        messages.append(
            {"role": "assistant", "content": response.content, "tool_calls": response.tool_calls}
        )

        for tool_call in response.tool_calls:
            call_id = tool_call.get("id") or str(uuid.uuid4())
            function = tool_call["function"]
            name = function["name"]
            args = _parse_tool_arguments(function)

            server_name = tool_owner.get(name)
            if server_name is None:
                message = f"model called unknown tool {name!r} (not exposed by any connected server)"
                event_log.append(
                    ToolCallEvent(
                        run_id=run.run_id,
                        seq=0,
                        ts=datetime.now(timezone.utc),
                        agent_id=agent.id,
                        call_id=call_id,
                        server="",
                        tool=name,
                        args=args,
                    )
                )
                event_log.append(
                    ToolResultEvent(
                        run_id=run.run_id,
                        seq=0,
                        ts=datetime.now(timezone.utc),
                        agent_id=agent.id,
                        call_id=call_id,
                        result=None,
                        error=message,
                        duration_ms=0.0,
                    )
                )
                raw_value = {"error": message}
            else:
                server = servers[server_name]
                call_event, result_event, raw_value = await server.call_tool(
                    tool=name, args=args, call_id=call_id, run_id=run.run_id, agent_id=agent.id
                )
                event_log.append(call_event)
                event_log.append(result_event)

            messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": _tool_message_content(raw_value)}
            )

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


async def execute_run(agent: AgentSpec, run: RunSpec, *, resolver: CredentialResolver, event_log: EventLog):
    """Full orchestration used by the CLI: resolves credentials, connects every
    MCP server on the spec, runs the loop, and guarantees connections close on
    the way out (even on exception) before returning."""
    api_key = resolver.resolve(agent.model.api_key_ref)
    model_client = ModelClient(agent.model, api_key)

    async with connect_mcp_servers(agent.mcp_servers, resolver) as servers:
        return await run_agent(agent, run, model_client=model_client, servers=servers, event_log=event_log)
