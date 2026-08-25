"""
ModelClient: a direct, stateless call to an OpenAI-compatible endpoint
(RIT directly, the LiteLLM gateway, or a local server). Identical in
shape to agentic_ml.model_client in the sibling agentic-ml-classification
project — no sessions, no transcripts, no compaction. Conversation state
is held explicitly by the caller (see agent_runtime.py), not implicitly
by this client.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI


@dataclass
class ModelResponse:
    text: Optional[str]
    tool_calls: list[dict]
    raw: Any
    latency_seconds: float
    model: str
    input_tokens: int
    output_tokens: int
    # Some reasoning-capable models served through this endpoint (confirmed
    # live against qwen3:latest on the RIT GenAI proxy) put chain-of-thought
    # in a non-standard `reasoning` field on the message, separate from
    # `content` -- but those tokens still count against max_tokens. Left
    # uncaptured, this is invisible AND dangerous: a verbose reasoning trace
    # can consume the entire token budget before the model ever emits its
    # actual answer, leaving `text` empty with no clue why. Captured here
    # for both diagnosis (transcripts) and because ToolCallingAgent needs
    # real headroom past it -- see agent_runtime.py's max_tokens default.
    reasoning: Optional[str] = None


class ModelClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_model: str,
        timeout_seconds: float = 120.0,
    ):
        self.base_url = base_url
        self.default_model = default_model
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)

    @classmethod
    def from_env(cls, prefix: str = "RIT", default_model: Optional[str] = None) -> "ModelClient":
        base_url = os.environ[f"{prefix}_BASE_URL"]
        api_key = os.environ[f"{prefix}_API_KEY"]
        model = default_model or os.environ.get(f"{prefix}_DEFAULT_MODEL", "")
        return cls(base_url=base_url, api_key=api_key, default_model=model)

    def call(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        model = model or self.default_model
        start = time.time()

        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        elapsed = time.time() - start

        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if getattr(message, "tool_calls", None):
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                })

        usage = getattr(response, "usage", None)
        return ModelResponse(
            text=message.content,
            tool_calls=tool_calls,
            raw=response,
            latency_seconds=round(elapsed, 3),
            model=model,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            reasoning=getattr(message, "reasoning", None),
        )
