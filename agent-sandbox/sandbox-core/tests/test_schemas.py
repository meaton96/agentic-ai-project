from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter

from sandbox_core.schemas.agent_spec import AgentSpec, McpServerBinding, ModelConfig, SubAgentBinding
from sandbox_core.schemas.credentials import CredentialRef
from sandbox_core.schemas.events import (
    AgentResultEvent,
    AgentSpawnEvent,
    ErrorEvent,
    Event,
    LlmRequestEvent,
    LlmResponseEvent,
    TokenUsage,
    ToolCallEvent,
    ToolResultEvent,
    redact_result,
)
from sandbox_core.schemas.run_spec import RunSpec

EventAdapter = TypeAdapter(Event)


def roundtrip(model):
    cls = type(model)
    return cls.model_validate_json(model.model_dump_json())


def event_roundtrip(event):
    return EventAdapter.validate_json(EventAdapter.dump_json(event))


def make_agent_spec() -> AgentSpec:
    return AgentSpec(
        id="agent-1",
        name="Profiler",
        system_prompt="You are a helpful profiling agent.",
        model=ModelConfig(
            base_url="https://api.rit.example/v1",
            model_name="gpt-oss-120b",
            api_key_ref="rit-api-key",
        ),
        mcp_servers=[
            McpServerBinding(
                name="fs",
                transport="stdio",
                connection={"command": "mcp-server-fs", "args": ["--root", "/data"]},
                credential_ref=None,
                allowed_tools=["read_file", "list_dir"],
                logging_policy="hashed",
            )
        ],
        sub_agents=[
            SubAgentBinding(
                agent_id="agent-2",
                tool_name="delegate_to_profiler",
                tool_description="Delegate profiling subtasks to agent-2.",
            )
        ],
    )


def test_agent_spec_roundtrip():
    spec = make_agent_spec()
    result = roundtrip(spec)
    assert result == spec


def test_agent_spec_defaults():
    spec = AgentSpec(
        id="agent-3",
        name="Bare",
        system_prompt="minimal",
        model=ModelConfig(base_url="http://localhost:8000", model_name="local", api_key_ref="none"),
    )
    assert spec.mcp_servers == []
    assert spec.sub_agents == []
    assert spec.max_turns == 25
    assert roundtrip(spec) == spec


def test_run_spec_roundtrip():
    run = RunSpec(agent_id="agent-1", task="Summarize this dataset.")
    result = roundtrip(run)
    assert result == run
    # defaults are actually generated, not left as placeholders
    assert result.run_id
    assert result.created_at.tzinfo is not None


def test_run_spec_explicit_fields():
    run = RunSpec(
        run_id="fixed-id",
        agent_id="agent-1",
        task="task",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        max_turns=5,
    )
    assert roundtrip(run) == run


def test_credential_ref_roundtrip():
    ref = CredentialRef(ref="rit-api-key", description="RIT-hosted OpenAI-compatible endpoint key")
    assert roundtrip(ref) == ref


EVENT_COMMON = dict(run_id="run-1", seq=1, ts=datetime(2026, 1, 1, tzinfo=timezone.utc), agent_id="agent-1")


@pytest.mark.parametrize(
    "event",
    [
        LlmRequestEvent(**EVENT_COMMON, messages=[{"role": "user", "content": "hi"}], model="gpt-oss-120b"),
        LlmResponseEvent(
            **EVENT_COMMON,
            content="hello",
            tool_calls=None,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            duration_ms=123.4,
        ),
        ToolCallEvent(**EVENT_COMMON, call_id="call-1", server="fs", tool="read_file", args={"path": "/x"}),
        ToolResultEvent(**EVENT_COMMON, call_id="call-1", result={"content": "file contents"}, error=None, duration_ms=42.0),
        AgentSpawnEvent(**EVENT_COMMON, child_agent_id="agent-2", spawned_via_tool="delegate_to_profiler"),
        AgentResultEvent(**EVENT_COMMON, final_output="done", turns_used=3),
        ErrorEvent(**EVENT_COMMON, message="boom", context={"where": "tool_call"}),
    ],
)
def test_event_variant_roundtrip(event):
    result = event_roundtrip(event)
    assert result == event
    assert type(result) is type(event)


def test_event_discriminator_rejects_unknown_type():
    with pytest.raises(Exception):
        EventAdapter.validate_python({**EVENT_COMMON, "type": "not_a_real_type"})


def test_parent_call_id_defaults_to_none():
    event = AgentResultEvent(**EVENT_COMMON, final_output="done", turns_used=1)
    assert event.parent_call_id is None


def test_parent_call_id_links_delegation_chain():
    event = ToolCallEvent(
        **EVENT_COMMON, parent_call_id="call-0", call_id="call-1", server="fs", tool="read_file", args={}
    )
    assert event_roundtrip(event).parent_call_id == "call-0"


# --- redact_result ---

CANARY = "sk-super-secret-canary-value-do-not-leak"
PAYLOAD = {"content": CANARY, "nested": {"a": [1, 2, 3]}}


def test_redact_result_full_returns_raw_payload():
    assert redact_result(PAYLOAD, "full") is PAYLOAD


def test_redact_result_hashed_excludes_raw_payload():
    redacted = redact_result(PAYLOAD, "hashed")
    serialized = str(redacted)
    assert CANARY not in serialized
    assert set(redacted.keys()) == {"sha256", "byte_size", "shape"}
    assert isinstance(redacted["sha256"], str)
    assert len(redacted["sha256"]) == 64  # hex sha256


def test_redact_result_metadata_excludes_raw_payload():
    redacted = redact_result(PAYLOAD, "metadata", tool="read_file", arg_keys=["path"])
    serialized = str(redacted)
    assert CANARY not in serialized
    assert redacted == {"tool": "read_file", "arg_keys": ["path"]}


def test_redact_result_hashed_is_deterministic():
    first = redact_result(CANARY, "hashed")
    second = redact_result(CANARY, "hashed")
    assert first["sha256"] == second["sha256"]


def test_redact_result_hashed_differs_for_different_payloads():
    a = redact_result({"content": "one"}, "hashed")
    b = redact_result({"content": "two"}, "hashed")
    assert a["sha256"] != b["sha256"]


def test_redact_result_unknown_policy_raises():
    with pytest.raises(ValueError):
        redact_result(PAYLOAD, "bogus")
