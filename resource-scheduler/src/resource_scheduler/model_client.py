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

import openai
from openai import OpenAI

# Confirmed live against the RIT GenAI endpoint: 504 Gateway Time-out
# (surfaces as InternalServerError, since the openai SDK maps any >=500
# response to that class) hit every Task Prioritization call at
# queue_size=15, correlating with request size/generation time, not a
# one-off fluke. RateLimitError (429) and the connection/timeout errors
# below are the same story on a shared, rate-limited academic endpoint --
# all transient, all worth one more attempt. Deliberately NOT retrying
# anything else (bad model name, bad auth, malformed request): those are
# real problems retrying won't fix, and silently retrying them would just
# hide a bug behind extra latency.
_RETRYABLE_EXCEPTIONS = (
    openai.InternalServerError,
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
)


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
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        on_retry: Optional[Any] = None,
    ):
        self.base_url = base_url
        self.default_model = default_model
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        # max_retries=3 means up to 4 total attempts. Backoff doubles each
        # time (2s, 4s, 8s) -- long enough that a genuinely transient
        # gateway hiccup has a real chance to clear, short enough that a
        # live demo isn't stuck waiting minutes on an endpoint that's
        # actually down. on_retry, if given, is called with
        # (attempt, max_retries, wait_seconds, exception) before each
        # sleep -- lets a caller surface "retrying..." in its own trace/
        # print output instead of this class assuming stdout is the right
        # place for it.
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.on_retry = on_retry

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

        response = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                break
            except _RETRYABLE_EXCEPTIONS as e:
                if attempt == self.max_retries:
                    raise
                wait_seconds = self.retry_backoff_seconds * (2 ** attempt)
                if self.on_retry:
                    self.on_retry(attempt, self.max_retries, wait_seconds, e)
                time.sleep(wait_seconds)
        # Deliberately includes any retry backoff sleep time, not just the
        # final successful attempt -- latency_seconds is meant to answer
        # "how long did this call actually take from the caller's side",
        # and an unusually high value here is itself a useful signal that
        # retries happened, without needing to cross-reference on_retry
        # callback logs separately.
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
