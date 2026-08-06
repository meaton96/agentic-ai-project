"""
ToolCallingAgent's trace_fn calls (agent_runtime.py). One thing worth
proving, not just exercising:

A real incident: running the dynamic orchestrator against a rolled-up
282-column aviation table, the intake agent's call to a 32K-context
local model was rejected outright (400, context length exceeded) —
and nothing in trace.jsonl or transcripts/ showed what had been sent,
because both are written only after a model call *succeeds*
(agent_runtime.py's "model_call" trace fires after client.call()
returns; write_transcript is called by the caller only once a step
function returns a result — neither happens if the call raises first).
"model_call_started"/"model_call_failed" close that gap: they fire
respectively before and (on an exception) instead of/around the
existing post-response trace, so a request-rejected-before-responding
failure still leaves a record of roughly how large the outgoing
request was — the exact fact that would have made this incident
diagnosable in seconds instead of requiring a manual reproduction.

A second real incident, same run: feature_engineering, retrying a
rejected proposal with the SAME rejection reason every time, produced
the exact same (still-rejected) output every time — because
temperature was silently pinned at ModelClient.call's own default
(0.0) with no way to override, so an unchanged prompt is mathematically
guaranteed to produce an unchanged response. ToolCallingAgent now
accepts a temperature it actually forwards.

A third real incident, same run: modeling, proposing a candidate that
enumerated many individual derived-stat columns for a wide rolled-up
table, got cut off mid-column-name at exactly 1024 output tokens —
max_tokens had the identical silently-pinned-with-no-override gap as
temperature did.
"""
from __future__ import annotations

import pytest

from agentic_ml.agent_runtime import Tool, ToolCallingAgent
from agentic_ml.model_client import ModelResponse


def _resp(text=None, tool_calls=None):
    return ModelResponse(
        text=text, tool_calls=tool_calls or [], raw=None, latency_seconds=0.01,
        model="fake-model", input_tokens=1, output_tokens=1,
    )


class _FakeClient:
    def __init__(self, responses=None, error=None):
        self._responses = responses or []
        self._error = error
        self.calls = 0
        self.temperatures_seen: list[float] = []
        self.max_tokens_seen: list[int] = []

    def call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
        self.calls += 1
        self.temperatures_seen.append(temperature)
        self.max_tokens_seen.append(max_tokens)
        if self._error is not None:
            raise self._error
        return self._responses.pop(0)


def test_model_call_started_and_completed_traced_on_success():
    client = _FakeClient(responses=[_resp(text="done")])
    agent = ToolCallingAgent(model_client=client, tools=[], system_prompt="You are a test agent.")
    events = []
    agent.run("hello", trace_fn=events.append)

    event_names = [e["event"] for e in events]
    assert event_names == ["model_call_started", "model_call"]
    assert events[0]["request_chars"] > 0
    assert events[0]["request_chars_est_tokens"] == events[0]["request_chars"] // 4


def test_model_call_failed_traced_and_exception_still_propagates():
    error = RuntimeError("BadRequestError: context length exceeded")
    client = _FakeClient(error=error)
    agent = ToolCallingAgent(model_client=client, tools=[], system_prompt="You are a test agent.")
    events = []

    with pytest.raises(RuntimeError, match="context length exceeded"):
        agent.run("hello", trace_fn=events.append)

    event_names = [e["event"] for e in events]
    assert event_names == ["model_call_started", "model_call_failed"]
    # the size estimate that would have made the real incident diagnosable
    # without a manual reproduction is present even though the call never
    # got a response back.
    assert events[1]["request_chars"] == events[0]["request_chars"]
    assert "context length exceeded" in events[1]["error"]


def test_temperature_defaults_to_zero_and_is_overridable():
    client = _FakeClient(responses=[_resp(text="done")])
    ToolCallingAgent(model_client=client, tools=[], system_prompt="You are a test agent.").run("hello")
    assert client.temperatures_seen == [0.0]

    client2 = _FakeClient(responses=[_resp(text="done")])
    ToolCallingAgent(
        model_client=client2, tools=[], system_prompt="You are a test agent.", temperature=0.4,
    ).run("hello")
    assert client2.temperatures_seen == [0.4]


def test_max_tokens_defaults_to_1024_and_is_overridable():
    client = _FakeClient(responses=[_resp(text="done")])
    ToolCallingAgent(model_client=client, tools=[], system_prompt="You are a test agent.").run("hello")
    assert client.max_tokens_seen == [1024]

    client2 = _FakeClient(responses=[_resp(text="done")])
    ToolCallingAgent(
        model_client=client2, tools=[], system_prompt="You are a test agent.", max_tokens=8192,
    ).run("hello")
    assert client2.max_tokens_seen == [8192]
