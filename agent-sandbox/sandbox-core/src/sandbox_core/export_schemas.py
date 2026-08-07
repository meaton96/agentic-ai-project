"""Dumps JSON Schema for every sandbox_core model to agent-sandbox/schema/.

Run via: python -m sandbox_core.export_schemas
Other languages and the future UI validate against these files, so they are
versioned (`<model>.v1.json`) rather than overwritten in place on breaking changes.
"""

import json
from pathlib import Path

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
)
from sandbox_core.schemas.run_spec import RunSpec

# repo-relative: agent-sandbox/sandbox-core/src/sandbox_core/export_schemas.py -> agent-sandbox/schema
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "schema"

SCHEMA_VERSION = 1

MODELS = {
    "agent_spec": AgentSpec,
    "model_config": ModelConfig,
    "mcp_server_binding": McpServerBinding,
    "sub_agent_binding": SubAgentBinding,
    "run_spec": RunSpec,
    "credential_ref": CredentialRef,
    "token_usage": TokenUsage,
    "llm_request_event": LlmRequestEvent,
    "llm_response_event": LlmResponseEvent,
    "tool_call_event": ToolCallEvent,
    "tool_result_event": ToolResultEvent,
    "agent_spawn_event": AgentSpawnEvent,
    "agent_result_event": AgentResultEvent,
    "error_event": ErrorEvent,
}


def export_schemas(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for stem, model in MODELS.items():
        path = output_dir / f"{stem}.v{SCHEMA_VERSION}.json"
        path.write_text(json.dumps(model.model_json_schema(), indent=2) + "\n")
        written.append(path)

    # Event is a discriminated Union, not a BaseModel, so it needs a TypeAdapter.
    event_path = output_dir / f"event.v{SCHEMA_VERSION}.json"
    event_path.write_text(json.dumps(TypeAdapter(Event).json_schema(), indent=2) + "\n")
    written.append(event_path)

    return written


def main() -> None:
    written = export_schemas()
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
