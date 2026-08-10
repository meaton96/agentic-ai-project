import json
from datetime import datetime, timezone

import pytest

from sandbox_core.runtime.agent_loop import run_agent
from sandbox_core.runtime.event_log import EventLog
from sandbox_core.runtime.model_client import ModelTurn
from sandbox_core.schemas.agent_spec import AgentSpec, ModelConfig
from sandbox_core.schemas.events import (
    AgentResultEvent,
    ErrorEvent,
    LlmRequestEvent,
    LlmResponseEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from sandbox_core.schemas.run_spec import RunSpec

RUN_ID = "run-1"
AGENT_ID = "agent-1"
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_agent(**overrides) -> AgentSpec:
    fields = dict(
        id=AGENT_ID,
        name="Test Agent",
        system_prompt="be helpful",
        model=ModelConfig(base_url="http://localhost", model_name="test-model", api_key_ref="ref"),
        max_turns=5,
    )
    fields.update(overrides)
    return AgentSpec(**fields)


def make_run(**overrides) -> RunSpec:
    fields = dict(run_id=RUN_ID, agent_id=AGENT_ID, task="write and confirm a file")
    fields.update(overrides)
    return RunSpec(**fields)


def llm_response(*, content=None, tool_calls=None) -> ModelTurn:
    request = LlmRequestEvent(
        run_id=RUN_ID, seq=0, ts=NOW, agent_id=AGENT_ID, messages=[], model="test-model"
    )
    response = LlmResponseEvent(
        run_id=RUN_ID, seq=0, ts=NOW, agent_id=AGENT_ID, content=content, tool_calls=tool_calls, duration_ms=1.0
    )
    return ModelTurn(request=request, response=response)


class FakeModelClient:
    def __init__(self, turns: list[ModelTurn]):
        self._turns = iter(turns)
        self.calls: list[dict] = []

    async def complete(self, messages, tools, *, run_id, agent_id, parent_call_id=None):
        self.calls.append({"messages": [m.copy() for m in messages], "tools": tools})
        return next(self._turns)


class FakeServer:
    def __init__(self, name, tool_specs, result_text="wrote the file"):
        self.name = name
        self._tool_specs = tool_specs
        self.result_text = result_text
        self.calls: list[dict] = []

    @property
    def tool_specs(self):
        return self._tool_specs

    async def call_tool(self, *, tool, args, call_id, run_id, agent_id, parent_call_id=None):
        self.calls.append({"tool": tool, "args": args, "call_id": call_id})
        call_event = ToolCallEvent(
            run_id=run_id, seq=0, ts=NOW, agent_id=agent_id, parent_call_id=parent_call_id,
            call_id=call_id, server=self.name, tool=tool, args=args,
        )
        result_event = ToolResultEvent(
            run_id=run_id, seq=0, ts=NOW, agent_id=agent_id, parent_call_id=parent_call_id,
            call_id=call_id, result=self.result_text, error=None, duration_ms=2.0,
        )
        return call_event, result_event, self.result_text


WRITE_FILE_TOOL_CALL = {
    "id": "call-1",
    "type": "function",
    "function": {"name": "write_file", "arguments": json.dumps({"path": "out.txt", "content": "hi"})},
}


@pytest.mark.asyncio
async def test_full_turn_with_tool_call_then_final_answer(tmp_path):
    model_client = FakeModelClient(
        [
            llm_response(content=None, tool_calls=[WRITE_FILE_TOOL_CALL]),
            llm_response(content="wrote out.txt successfully", tool_calls=None),
        ]
    )
    server = FakeServer("fs", [{"type": "function", "function": {"name": "write_file", "description": "", "parameters": {}}}])
    event_log = EventLog(RUN_ID, output_root=tmp_path)

    result = await run_agent(
        make_agent(), make_run(), model_client=model_client, servers={"fs": server}, event_log=event_log
    )

    assert isinstance(result, AgentResultEvent)
    assert result.final_output == "wrote out.txt successfully"
    assert result.turns_used == 2
    assert server.calls == [{"tool": "write_file", "args": {"path": "out.txt", "content": "hi"}, "call_id": "call-1"}]

    events = _read_jsonl(event_log.path)
    types = [e["type"] for e in events]
    assert types == [
        "llm_request", "llm_response",
        "tool_call", "tool_result",
        "llm_request", "llm_response",
        "agent_result",
    ]
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))


@pytest.mark.asyncio
async def test_final_answer_without_any_tool_call(tmp_path):
    model_client = FakeModelClient([llm_response(content="no tools needed", tool_calls=None)])
    event_log = EventLog(RUN_ID, output_root=tmp_path)

    result = await run_agent(
        make_agent(), make_run(), model_client=model_client, servers={}, event_log=event_log
    )

    assert isinstance(result, AgentResultEvent)
    assert result.final_output == "no tools needed"
    assert result.turns_used == 1


@pytest.mark.asyncio
async def test_model_error_event_stops_the_run(tmp_path):
    error_response = ModelTurn(
        request=LlmRequestEvent(run_id=RUN_ID, seq=0, ts=NOW, agent_id=AGENT_ID, messages=[], model="test-model"),
        response=ErrorEvent(run_id=RUN_ID, seq=0, ts=NOW, agent_id=AGENT_ID, message="model request failed"),
    )
    model_client = FakeModelClient([error_response])
    event_log = EventLog(RUN_ID, output_root=tmp_path)

    result = await run_agent(
        make_agent(), make_run(), model_client=model_client, servers={}, event_log=event_log
    )

    assert isinstance(result, ErrorEvent)
    assert result.message == "model request failed"


@pytest.mark.asyncio
async def test_max_turns_truncation_emits_error_event(tmp_path):
    # every turn calls a tool, so the loop never reaches a final answer
    turns = [llm_response(content=None, tool_calls=[WRITE_FILE_TOOL_CALL]) for _ in range(2)]
    model_client = FakeModelClient(turns)
    server = FakeServer("fs", [])
    event_log = EventLog(RUN_ID, output_root=tmp_path)

    result = await run_agent(
        make_agent(max_turns=2), make_run(), model_client=model_client, servers={"fs": server}, event_log=event_log
    )

    assert isinstance(result, ErrorEvent)
    assert "max_turns" in result.context["phase"]
    assert result.context["max_turns"] == 2


@pytest.mark.asyncio
async def test_unknown_tool_name_is_handled_without_crashing(tmp_path):
    hallucinated_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "delete_everything", "arguments": "{}"},
    }
    model_client = FakeModelClient(
        [
            llm_response(content=None, tool_calls=[hallucinated_call]),
            llm_response(content="done", tool_calls=None),
        ]
    )
    event_log = EventLog(RUN_ID, output_root=tmp_path)

    result = await run_agent(
        make_agent(), make_run(), model_client=model_client, servers={}, event_log=event_log
    )

    assert isinstance(result, AgentResultEvent)
    events = _read_jsonl(event_log.path)
    result_events = [e for e in events if e["type"] == "tool_result"]
    assert len(result_events) == 1
    assert "delete_everything" in result_events[0]["error"]


@pytest.mark.asyncio
async def test_run_spec_max_turns_overrides_agent_spec_default(tmp_path):
    turns = [llm_response(content=None, tool_calls=[WRITE_FILE_TOOL_CALL])]
    model_client = FakeModelClient(turns)
    server = FakeServer("fs", [])
    event_log = EventLog(RUN_ID, output_root=tmp_path)

    result = await run_agent(
        make_agent(max_turns=25),
        make_run(max_turns=1),
        model_client=model_client,
        servers={"fs": server},
        event_log=event_log,
    )

    assert isinstance(result, ErrorEvent)
    assert result.context["max_turns"] == 1


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
