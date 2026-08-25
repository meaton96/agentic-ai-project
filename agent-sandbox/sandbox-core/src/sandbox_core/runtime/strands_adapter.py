"""Turns a validated AgentSpec into a running strands.Agent, and translates its
hook events into sandbox-core's own Event union as they fire.

This is the whole "custom extension": everything below `build_agent()` is
Strands' agent loop, model retries, and MCP session lifecycle, not ours. We
never adopt Strands' own session_manager/storage/checkpointing — EventLog
stays the single source of truth exactly as it was under the hand-rolled
runtime, so sandbox-server's SSE bridge needs no changes.
"""

import time
from contextlib import ExitStack
from datetime import datetime, timezone
from typing import Any

from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from strands import Agent
from strands.hooks import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
)
from strands.models.openai import OpenAIModel
from strands.tools.mcp import MCPClient
from strands.types.agent import Limits

from sandbox_core.schemas.agent_spec import AgentSpec, McpServerBinding
from sandbox_core.schemas.credentials import CredentialResolver
from sandbox_core.schemas.events import (
    ErrorEvent,
    LlmRequestEvent,
    LlmResponseEvent,
    TokenUsage,
    ToolCallEvent,
    ToolResultEvent,
    redact_result,
)
from sandbox_core.schemas.run_spec import RunSpec

from .event_log import EventLog


def _build_model(spec: AgentSpec, api_key: str) -> OpenAIModel:
    params: dict[str, Any] = {"temperature": spec.model.temperature}
    if spec.model.max_tokens is not None:
        params["max_tokens"] = spec.model.max_tokens
    # OpenAIModel talks to any OpenAI-compatible /chat/completions endpoint (RIT
    # gateway, local vLLM, ...) via client_args, same contract the old hand-rolled
    # ModelClient posted to directly.
    return OpenAIModel(
        client_args={"api_key": api_key, "base_url": spec.model.base_url},
        model_id=spec.model.model_name,
        params=params,
        # Our event schema only ever surfaces whole llm_response events (no
        # token-level UI), so plain request/response is simpler and easier to
        # test than reconstructing OpenAI's streaming-chunk format.
        stream=False,
    )


def _auth_headers(binding: McpServerBinding, credential: str | None) -> dict:
    headers = dict(binding.connection.get("headers", {}))
    if credential is not None:
        headers.setdefault("Authorization", f"Bearer {credential}")
    return headers


def _stdio_env(binding: McpServerBinding, credential: str | None) -> dict | None:
    env = dict(binding.connection.get("env", {})) or None
    if credential is not None:
        var_name = binding.connection.get("credential_env_var", "MCP_TOKEN")
        env = {**(env or {}), var_name: credential}
    return env


def _mcp_client_for(binding: McpServerBinding, resolver: CredentialResolver | None) -> MCPClient:
    credential = resolver.resolve(binding.credential_ref) if (binding.credential_ref and resolver) else None
    tool_filters = {"allowed": binding.allowed_tools} if binding.allowed_tools is not None else None

    if binding.transport in ("http", "sse"):
        return MCPClient(
            url=binding.connection["url"],
            headers=_auth_headers(binding, credential),
            tool_filters=tool_filters,
        )
    if binding.transport == "stdio":
        conn = binding.connection
        params = StdioServerParameters(
            command=conn["command"],
            args=conn.get("args", []),
            env=_stdio_env(binding, credential),
            cwd=conn.get("cwd"),
        )
        return MCPClient(transport_callable=lambda: stdio_client(params), tool_filters=tool_filters)
    raise ValueError(f"unknown mcp transport {binding.transport!r} on server {binding.name!r}")


def _flatten_tool_result_content(content: list[dict]) -> Any:
    """Collapses a Strands ToolResult's content list into a plain value for the
    event log — mirrors the old mcp_client._flatten_content: a bare value for
    the common single-block case, a list of blocks otherwise."""
    parts = []
    for block in content:
        if "text" in block:
            parts.append(block["text"])
        elif "json" in block:
            parts.append(block["json"])
        else:
            parts.append(block)
    if len(parts) == 1:
        return parts[0]
    return parts


class SandboxEventHooks(HookProvider):
    """Bridges Strands' hook events onto sandbox-core's Event union, appending
    each one to `event_log` as it fires — the same append() path the old
    hand-rolled agent_loop used, so EventLog/StreamingEventLog/SSE fan-out
    need no changes at all."""

    def __init__(self, *, event_log: EventLog, run: RunSpec, agent_id: str, mcp_bindings: dict[int, McpServerBinding]):
        self._event_log = event_log
        self._run = run
        self._agent_id = agent_id
        self._mcp_bindings = mcp_bindings  # id(MCPClient) -> owning McpServerBinding
        self._turn_started_at: float | None = None
        self.captured_model_error: ErrorEvent | None = None

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeModelCallEvent, self._before_model)
        registry.add_callback(AfterModelCallEvent, self._after_model)
        registry.add_callback(BeforeToolCallEvent, self._before_tool)
        registry.add_callback(AfterToolCallEvent, self._after_tool)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def _binding_for(self, selected_tool) -> McpServerBinding | None:
        client = getattr(selected_tool, "mcp_client", None)
        return self._mcp_bindings.get(id(client)) if client is not None else None

    # -- model call ------------------------------------------------------

    def _before_model(self, event: BeforeModelCallEvent) -> None:
        self._turn_started_at = time.monotonic()
        self._event_log.append(
            LlmRequestEvent(
                run_id=self._run.run_id,
                seq=0,
                ts=self._now(),
                agent_id=self._agent_id,
                messages=[dict(m) for m in event.agent.messages],
                model=event.agent.model.get_config().get("model_id", ""),
            )
        )

    def _after_model(self, event: AfterModelCallEvent) -> None:
        duration_ms = (time.monotonic() - self._turn_started_at) * 1000 if self._turn_started_at else 0.0

        if event.exception is not None:
            if event.retry:
                return  # Strands is about to retry; the eventual outcome gets its own Before/After pair
            error = ErrorEvent(
                run_id=self._run.run_id,
                seq=0,
                ts=self._now(),
                agent_id=self._agent_id,
                message=str(event.exception),
                context={"phase": "llm_request"},
            )
            self.captured_model_error = self._event_log.append(error)
            return

        if event.stop_response is None:
            return

        # event_loop_metrics.accumulated_usage isn't updated until after this
        # hook fires (strands/event_loop/event_loop.py applies it once the
        # message is appended to history), so a before/after diff on it is
        # always zero here. The per-turn usage is attached directly to the
        # response message's metadata before the hook runs — use that instead.
        usage = event.stop_response.message.get("metadata", {}).get("usage")
        token_usage = (
            TokenUsage(
                prompt_tokens=usage.get("inputTokens", 0),
                completion_tokens=usage.get("outputTokens", 0),
                total_tokens=usage.get("totalTokens", 0),
            )
            if usage
            else None
        )

        content_blocks = event.stop_response.message.get("content", [])
        text = "".join(block["text"] for block in content_blocks if "text" in block)
        tool_calls = [
            {
                "id": block["toolUse"]["toolUseId"],
                "type": "function",
                "function": {"name": block["toolUse"]["name"], "arguments": block["toolUse"]["input"]},
            }
            for block in content_blocks
            if "toolUse" in block
        ] or None

        self._event_log.append(
            LlmResponseEvent(
                run_id=self._run.run_id,
                seq=0,
                ts=self._now(),
                agent_id=self._agent_id,
                content=text or None,
                tool_calls=tool_calls,
                token_usage=token_usage,
                duration_ms=duration_ms,
            )
        )

    # -- tool call ---------------------------------------------------------

    def _before_tool(self, event: BeforeToolCallEvent) -> None:
        binding = self._binding_for(event.selected_tool)
        self._event_log.append(
            ToolCallEvent(
                run_id=self._run.run_id,
                seq=0,
                ts=self._now(),
                agent_id=self._agent_id,
                call_id=event.tool_use["toolUseId"],
                server=binding.name if binding else "",
                tool=event.tool_use["name"],
                args=event.tool_use.get("input") or {},
            )
        )

    def _after_tool(self, event: AfterToolCallEvent) -> None:
        binding = self._binding_for(event.selected_tool)
        args = event.tool_use.get("input") or {}
        raw_value = _flatten_tool_result_content(event.result.get("content", []))

        error = None
        if event.exception is not None:
            error = str(event.exception)
        elif event.result.get("status") == "error":
            error = str(raw_value)

        policy = binding.logging_policy if binding else "full"
        persisted = redact_result(raw_value, policy, tool=event.tool_use["name"], arg_keys=sorted(args.keys()))

        self._event_log.append(
            ToolResultEvent(
                run_id=self._run.run_id,
                seq=0,
                ts=self._now(),
                agent_id=self._agent_id,
                call_id=event.tool_use["toolUseId"],
                result=persisted,
                error=error,
                duration_ms=(event.duration or 0.0) * 1000,
            )
        )


def build_agent(
    spec: AgentSpec, run: RunSpec, *, resolver: CredentialResolver, event_log: EventLog, stack: ExitStack
) -> tuple[Agent, SandboxEventHooks]:
    """Resolves credentials, builds one MCPClient per McpServerBinding, and
    constructs the strands.Agent for this run. `stack` is an ExitStack the
    caller owns and closes on the way out (even on exception) — MCPClient
    lifecycle is otherwise managed by the Agent itself once passed into
    `tools=`, but we still track id(client) -> binding for event attribution."""
    api_key = resolver.resolve(spec.model.api_key_ref)
    model = _build_model(spec, api_key)

    mcp_bindings: dict[int, McpServerBinding] = {}
    tools: list[Any] = []
    for binding in spec.mcp_servers:
        client = _mcp_client_for(binding, resolver)
        mcp_bindings[id(client)] = binding
        tools.append(client)

    hooks = SandboxEventHooks(event_log=event_log, run=run, agent_id=spec.id, mcp_bindings=mcp_bindings)

    agent = Agent(
        model=model,
        system_prompt=spec.system_prompt,
        tools=tools,
        hooks=[hooks],
        callback_handler=None,  # suppress Strands' default stdout streaming printer
    )
    stack.callback(agent.cleanup)
    return agent, hooks


def max_turns_for(spec: AgentSpec, run: RunSpec) -> int:
    return run.max_turns if run.max_turns is not None else spec.max_turns


def limits_for(max_turns: int) -> Limits:
    return Limits(turns=max_turns)
