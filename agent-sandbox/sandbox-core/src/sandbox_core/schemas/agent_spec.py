from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    base_url: str
    model_name: str
    api_key_ref: str
    temperature: float = 0.0
    max_tokens: int | None = None


class McpServerBinding(BaseModel):
    name: str
    transport: Literal["stdio", "http", "sse"]
    connection: dict
    credential_ref: str | None = None
    allowed_tools: list[str] | None = None
    logging_policy: Literal["full", "hashed", "metadata"] = "full"


class SubAgentBinding(BaseModel):
    agent_id: str
    tool_name: str
    tool_description: str


class AgentSpec(BaseModel):
    id: str
    name: str
    system_prompt: str
    model: ModelConfig
    mcp_servers: list[McpServerBinding] = Field(default_factory=list)
    sub_agents: list[SubAgentBinding] = Field(default_factory=list)
    max_turns: int = 25


@runtime_checkable
class AgentSpecLoader(Protocol):
    """Decouples pipeline execution from *how* agent specs are stored —
    sandbox-server's specs.py implements this over its YAML-file-per-id
    directory, cli.py's pipeline command over a sibling directory. Mirrors
    CredentialResolver's role for secrets."""

    def load(self, agent_id: str) -> AgentSpec: ...
