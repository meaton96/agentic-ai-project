"""
Minimal tool-calling agent loop. This is the entire "agent runtime"
this pipeline needs — no sessions, no transcript compaction, no
background daemon. Conversation state is just a Python list of
messages held by the caller for the duration of one task.

Each Tool is a plain Python callable plus a JSON-schema description.
The harness (not this module) decides which tools an agent is allowed
to call — see the MVP constraint that agents get no direct access to
test labels or final scoring logic.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .model_client import ModelClient


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema for the function's arguments
    handler: Callable[..., Any]

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class AgentRunResult:
    final_text: Optional[str]
    messages: list[dict]
    tool_call_log: list[dict] = field(default_factory=list)
    turns_used: int = 0
    stopped_reason: str = "completed"


class ToolCallingAgent:
    def __init__(
        self,
        model_client: ModelClient,
        tools: list[Tool],
        system_prompt: str,
        model: Optional[str] = None,
        max_turns: int = 8,
    ):
        self.client = model_client
        self.tools = {t.name: t for t in tools}
        self.system_prompt = system_prompt
        self.model = model
        self.max_turns = max_turns

    def run(self, user_message: str, trace_fn: Optional[Callable[[dict], None]] = None) -> AgentRunResult:
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        tool_schemas = [t.to_openai_schema() for t in self.tools.values()]
        tool_call_log = []

        def trace(event: str, **fields):
            record = {"ts": time.time(), "event": event, **fields}
            if trace_fn:
                trace_fn(record)

        for turn in range(self.max_turns):
            response = self.client.call(messages, model=self.model, tools=tool_schemas)
            trace(
                "model_call",
                turn=turn,
                model=response.model,
                latency_seconds=response.latency_seconds,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                had_tool_calls=bool(response.tool_calls),
            )

            if not response.tool_calls:
                messages.append({"role": "assistant", "content": response.text})
                return AgentRunResult(
                    final_text=response.text,
                    messages=messages,
                    tool_call_log=tool_call_log,
                    turns_used=turn + 1,
                    stopped_reason="completed",
                )

            # assistant turn requesting tool calls
            messages.append({
                "role": "assistant",
                "content": response.text,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in response.tool_calls
                ],
            })

            for tc in response.tool_calls:
                tool = self.tools.get(tc["name"])
                if tool is None:
                    result = {"error": f"unknown tool: {tc['name']}"}
                else:
                    try:
                        args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                        result = tool.handler(**args)
                    except Exception as e:
                        result = {"error": f"{type(e).__name__}: {e}"}

                tool_call_log.append({"tool": tc["name"], "arguments": tc["arguments"], "result": result})
                trace("tool_call", tool=tc["name"], turn=turn)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result, default=str),
                })

        return AgentRunResult(
            final_text=None,
            messages=messages,
            tool_call_log=tool_call_log,
            turns_used=self.max_turns,
            stopped_reason="max_turns_exceeded",
        )
