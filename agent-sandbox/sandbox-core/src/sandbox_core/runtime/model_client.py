import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from sandbox_core.schemas.agent_spec import ModelConfig
from sandbox_core.schemas.events import ErrorEvent, LlmRequestEvent, LlmResponseEvent, TokenUsage

# Raw httpx rather than the `openai` package: every target here (RIT gateway,
# local vLLM, whatever a BYO agent points at) only needs to be an
# OpenAI-compatible /chat/completions endpoint, so a plain POST avoids pulling
# in and pinning to the openai SDK's client/version churn for no real benefit.

RETRYABLE_STATUS_CODES = {429}


@dataclass
class ModelTurn:
    request: LlmRequestEvent
    response: LlmResponseEvent | ErrorEvent


class ModelClient:
    def __init__(self, model: ModelConfig, api_key: str, *, timeout: float = 60.0, max_retries: int = 3):
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        run_id: str,
        agent_id: str,
        parent_call_id: str | None = None,
    ) -> ModelTurn:
        request_event = LlmRequestEvent(
            run_id=run_id,
            seq=0,
            ts=datetime.now(timezone.utc),
            agent_id=agent_id,
            parent_call_id=parent_call_id,
            messages=messages,
            model=self.model.model_name,
        )

        body: dict = {
            "model": self.model.model_name,
            "messages": messages,
            "temperature": self.model.temperature,
        }
        if self.model.max_tokens is not None:
            body["max_tokens"] = self.model.max_tokens
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        url = f"{self.model.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        started = time.monotonic()
        last_error: Exception | None = None
        attempt = 0

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = await client.post(url, json=body, headers=headers)
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    retryable = exc.response.status_code >= 500 or exc.response.status_code in RETRYABLE_STATUS_CODES
                    if not retryable or attempt == self.max_retries:
                        break
                    await asyncio.sleep(self._backoff_seconds(attempt))
                    continue
                except (httpx.TimeoutException, httpx.ConnectError) as exc:
                    last_error = exc
                    if attempt == self.max_retries:
                        break
                    await asyncio.sleep(self._backoff_seconds(attempt))
                    continue
                except httpx.HTTPError as exc:
                    last_error = exc
                    break
                else:
                    data = response.json()
                    choice = data["choices"][0]["message"]
                    usage = data.get("usage")
                    response_event = LlmResponseEvent(
                        run_id=run_id,
                        seq=0,
                        ts=datetime.now(timezone.utc),
                        agent_id=agent_id,
                        parent_call_id=parent_call_id,
                        content=choice.get("content"),
                        tool_calls=choice.get("tool_calls"),
                        token_usage=TokenUsage(**usage) if usage else None,
                        duration_ms=(time.monotonic() - started) * 1000,
                    )
                    return ModelTurn(request=request_event, response=response_event)

        error_event = ErrorEvent(
            run_id=run_id,
            seq=0,
            ts=datetime.now(timezone.utc),
            agent_id=agent_id,
            parent_call_id=parent_call_id,
            message=f"model request to {url} failed after {attempt} attempt(s): {last_error}",
            context={"phase": "llm_request", "attempts": attempt},
        )
        return ModelTurn(request=request_event, response=error_event)

    @staticmethod
    def _backoff_seconds(attempt: int) -> float:
        return min(2**attempt, 8.0)
