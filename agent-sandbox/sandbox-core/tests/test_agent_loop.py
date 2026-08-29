"""Exercises execute_run() end-to-end through a real strands.Agent, faking
only the HTTP transport underneath OpenAIModel (json responses shaped like a
real OpenAI-compatible /chat/completions reply) — this is a stronger
regression test than the old hand-written model_client/servers fakes since it
now also exercises Strands' own tool-calling loop and hook system, not just
our code."""

import json

import httpx
import pytest
from openai import AsyncOpenAI
from strands import tool
from strands.models.openai import OpenAIModel

import sandbox_core.runtime.strands_adapter as adapter
from sandbox_core.runtime.agent_loop import execute_run
from sandbox_core.runtime.event_log import EventLog
from sandbox_core.schemas.agent_spec import AgentSpec, ModelConfig
from sandbox_core.schemas.events import AgentResultEvent, ErrorEvent
from sandbox_core.schemas.run_spec import RunSpec

RUN_ID = "run-1"
AGENT_ID = "agent-1"


class FakeResolver:
    def resolve(self, ref: str) -> str:
        return "fake-api-key"


def make_agent(**overrides) -> AgentSpec:
    fields = dict(
        id=AGENT_ID,
        name="Test Agent",
        system_prompt="be helpful",
        model=ModelConfig(base_url="http://test-model", model_name="test-model", api_key_ref="ref"),
        max_turns=5,
    )
    fields.update(overrides)
    return AgentSpec(**fields)


def make_run(**overrides) -> RunSpec:
    fields = dict(run_id=RUN_ID, agent_id=AGENT_ID, task="write and confirm a file")
    fields.update(overrides)
    return RunSpec(**fields)


def chat_completion(*, content=None, tool_calls=None) -> dict:
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def mock_transport(responses: list[dict]) -> httpx.MockTransport:
    remaining = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if not remaining:
            raise AssertionError("model called more times than the test provided responses for")
        return httpx.Response(200, json=remaining.pop(0))

    return httpx.MockTransport(handler)


def patch_openai_http_client(monkeypatch, responses: list[dict]) -> None:
    """OpenAIModel(client=...) takes a pre-built AsyncOpenAI client and
    — unlike client_args, which gets wrapped/closed per call — promises not
    to close it, so one MockTransport-backed client can safely serve every
    model call in a run without one 'client has been closed' after the first."""

    def fake_build_model(spec, api_key):
        params: dict = {"temperature": spec.model.temperature}
        if spec.model.max_tokens is not None:
            params["max_tokens"] = spec.model.max_tokens
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=spec.model.base_url,
            http_client=httpx.AsyncClient(transport=mock_transport(responses)),
        )
        return OpenAIModel(client=client, model_id=spec.model.model_name, params=params, stream=False)

    monkeypatch.setattr(adapter, "_build_model", fake_build_model)


WRITE_FILE_TOOL_CALL = {
    "id": "call-1",
    "type": "function",
    "function": {"name": "write_file", "arguments": json.dumps({"path": "out.txt", "content": "hi"})},
}


@tool
def write_file(path: str, content: str) -> str:
    """Writes content to a file."""
    return f"wrote {len(content)} bytes to {path}"


def patch_build_agent_with_write_tool(monkeypatch) -> None:
    """The MCP-backed ConnectedMcpServer path is now entirely Strands' own
    MCPClient/tool-provider machinery (not ours to unit-test); a plain
    in-process @tool exercises the same BeforeToolCallEvent/AfterToolCallEvent
    hook mapping without needing a real MCP subprocess in these tests."""
    original_build_agent = adapter.build_agent

    def build_agent_with_tool(spec, run, *, resolver, event_log, stack):
        agent, hooks = original_build_agent(spec, run, resolver=resolver, event_log=event_log, stack=stack)
        agent.tool_registry.process_tools([write_file])
        return agent, hooks

    monkeypatch.setattr("sandbox_core.runtime.agent_loop.build_agent", build_agent_with_tool)


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_full_turn_with_tool_call_then_final_answer(tmp_path, monkeypatch):
    patch_openai_http_client(
        monkeypatch,
        [
            chat_completion(tool_calls=[WRITE_FILE_TOOL_CALL]),
            chat_completion(content="wrote out.txt successfully"),
        ],
    )
    event_log = EventLog(RUN_ID, output_root=tmp_path)
    patch_build_agent_with_write_tool(monkeypatch)

    result = await execute_run(make_agent(), make_run(), resolver=FakeResolver(), event_log=event_log)

    assert isinstance(result, AgentResultEvent)
    assert result.final_output == "wrote out.txt successfully"
    assert result.turns_used == 2

    events = _read_jsonl(event_log.path)
    types = [e["type"] for e in events]
    assert types == [
        "llm_request", "llm_response",
        "tool_call", "tool_result",
        "llm_request", "llm_response",
        "agent_result",
    ]
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    tool_result = next(e for e in events if e["type"] == "tool_result")
    assert "wrote 2 bytes to out.txt" in tool_result["result"]


@pytest.mark.asyncio
async def test_final_answer_without_any_tool_call(tmp_path, monkeypatch):
    patch_openai_http_client(monkeypatch, [chat_completion(content="no tools needed")])
    event_log = EventLog(RUN_ID, output_root=tmp_path)

    result = await execute_run(make_agent(), make_run(), resolver=FakeResolver(), event_log=event_log)

    assert isinstance(result, AgentResultEvent)
    assert result.final_output == "no tools needed"
    assert result.turns_used == 1


@pytest.mark.asyncio
async def test_model_error_becomes_error_event_not_a_raised_exception(tmp_path, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    def fake_build_model(spec, api_key):
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=spec.model.base_url,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        return OpenAIModel(client=client, model_id=spec.model.model_name, params={}, stream=False)

    monkeypatch.setattr(adapter, "_build_model", fake_build_model)

    event_log = EventLog(RUN_ID, output_root=tmp_path)

    result = await execute_run(make_agent(max_turns=1), make_run(), resolver=FakeResolver(), event_log=event_log)

    assert isinstance(result, ErrorEvent)
    assert result.context["phase"] == "llm_request"


@pytest.mark.asyncio
async def test_max_turns_truncation_emits_error_event(tmp_path, monkeypatch):
    # every turn calls a tool, so the loop never reaches a final answer
    responses = [chat_completion(tool_calls=[WRITE_FILE_TOOL_CALL]) for _ in range(3)]
    patch_openai_http_client(monkeypatch, responses)
    patch_build_agent_with_write_tool(monkeypatch)
    event_log = EventLog(RUN_ID, output_root=tmp_path)

    result = await execute_run(make_agent(max_turns=2), make_run(), resolver=FakeResolver(), event_log=event_log)

    assert isinstance(result, ErrorEvent)
    assert result.context["phase"] == "max_turns"
    assert result.context["max_turns"] == 2


@pytest.mark.asyncio
async def test_run_spec_max_turns_overrides_agent_spec_default(tmp_path, monkeypatch):
    responses = [chat_completion(tool_calls=[WRITE_FILE_TOOL_CALL]) for _ in range(2)]
    patch_openai_http_client(monkeypatch, responses)
    patch_build_agent_with_write_tool(monkeypatch)
    event_log = EventLog(RUN_ID, output_root=tmp_path)

    result = await execute_run(
        make_agent(max_turns=25), make_run(max_turns=1), resolver=FakeResolver(), event_log=event_log
    )

    assert isinstance(result, ErrorEvent)
    assert result.context["max_turns"] == 1
