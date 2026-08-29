import hashlib
import json
from datetime import datetime
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class EventBase(BaseModel):
    run_id: str
    seq: int
    ts: datetime
    agent_id: str
    parent_call_id: str | None = None


class LlmRequestEvent(EventBase):
    type: Literal["llm_request"] = "llm_request"
    messages: list[dict]
    model: str


class LlmResponseEvent(EventBase):
    type: Literal["llm_response"] = "llm_response"
    content: str | None = None
    tool_calls: list[dict] | None = None
    token_usage: TokenUsage | None = None
    duration_ms: float


class ToolCallEvent(EventBase):
    type: Literal["tool_call"] = "tool_call"
    call_id: str
    server: str
    tool: str
    args: dict


class ToolResultEvent(EventBase):
    type: Literal["tool_result"] = "tool_result"
    call_id: str
    result: Any
    error: str | None = None
    duration_ms: float


class AgentSpawnEvent(EventBase):
    type: Literal["agent_spawn"] = "agent_spawn"
    child_agent_id: str
    spawned_via_tool: str


class AgentResultEvent(EventBase):
    type: Literal["agent_result"] = "agent_result"
    final_output: str
    turns_used: int


class ErrorEvent(EventBase):
    type: Literal["error"] = "error"
    message: str
    context: dict | None = None


Event = Annotated[
    Union[
        LlmRequestEvent,
        LlmResponseEvent,
        ToolCallEvent,
        ToolResultEvent,
        AgentSpawnEvent,
        AgentResultEvent,
        ErrorEvent,
    ],
    Field(discriminator="type"),
]


def _stable_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, default=str).encode("utf-8")


def _describe_shape(value: Any) -> dict:
    if isinstance(value, dict):
        return {"type": "dict", "keys": sorted(str(k) for k in value.keys())}
    if isinstance(value, list):
        return {"type": "list", "length": len(value)}
    return {"type": type(value).__name__}


def redact_result(
    result: Any,
    policy: Literal["full", "hashed", "metadata"],
    *,
    tool: str | None = None,
    arg_keys: list[str] | None = None,
) -> Any:
    """Applies an McpServerBinding.logging_policy to a tool result before persistence.

    `tool`/`arg_keys` are only used under the `metadata` policy, where the
    caller (which has the paired ToolCallEvent) supplies them since the
    metadata summary intentionally excludes the result payload itself.
    """
    if policy == "full":
        return result
    if policy == "hashed":
        payload = _stable_json_bytes(result)
        return {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_size": len(payload),
            "shape": _describe_shape(result),
        }
    if policy == "metadata":
        return {"tool": tool, "arg_keys": list(arg_keys) if arg_keys is not None else []}
    raise ValueError(f"unknown logging policy: {policy!r}")
