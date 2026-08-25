// Mirrors sandbox_core.schemas.* (agent_spec.py, events.py) and
// sandbox_server's run summary/detail response shapes. Kept as a hand-typed
// twin rather than generated, same as agent-frontend/web/src/api/types.ts.

export interface ModelConfig {
  base_url: string
  model_name: string
  api_key_ref: string
  temperature: number
  max_tokens: number | null
}

export type McpTransport = 'stdio' | 'http' | 'sse'
export type LoggingPolicy = 'full' | 'hashed' | 'metadata'

export interface McpServerBinding {
  name: string
  transport: McpTransport
  connection: Record<string, unknown>
  credential_ref: string | null
  allowed_tools: string[] | null
  logging_policy: LoggingPolicy
}

export interface SubAgentBinding {
  agent_id: string
  tool_name: string
  tool_description: string
}

export interface AgentSpec {
  id: string
  name: string
  system_prompt: string
  model: ModelConfig
  mcp_servers: McpServerBinding[]
  sub_agents: SubAgentBinding[]
  max_turns: number
}

export interface TokenUsage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

interface EventBase {
  run_id: string
  seq: number
  ts: string
  agent_id: string
  parent_call_id: string | null
}

export interface LlmRequestEvent extends EventBase {
  type: 'llm_request'
  messages: Record<string, unknown>[]
  model: string
}

export interface LlmResponseEvent extends EventBase {
  type: 'llm_response'
  content: string | null
  tool_calls: Record<string, unknown>[] | null
  token_usage: TokenUsage | null
  duration_ms: number
}

export interface ToolCallEvent extends EventBase {
  type: 'tool_call'
  call_id: string
  server: string
  tool: string
  args: Record<string, unknown>
}

export interface ToolResultEvent extends EventBase {
  type: 'tool_result'
  call_id: string
  result: unknown
  error: string | null
  duration_ms: number
}

export interface AgentSpawnEvent extends EventBase {
  type: 'agent_spawn'
  child_agent_id: string
  spawned_via_tool: string
}

export interface AgentResultEvent extends EventBase {
  type: 'agent_result'
  final_output: string
  turns_used: number
}

export interface ErrorEvent extends EventBase {
  type: 'error'
  message: string
  context: Record<string, unknown> | null
}

export type SandboxEvent =
  | LlmRequestEvent
  | LlmResponseEvent
  | ToolCallEvent
  | ToolResultEvent
  | AgentSpawnEvent
  | AgentResultEvent
  | ErrorEvent

export type RunStatus = 'running' | 'completed' | 'errored' | 'truncated'

export interface RunSummary {
  run_id: string
  agent_id: string | null
  created_at: string
  status: RunStatus
  turn_count: number
}

export interface RunDetail {
  run_id: string
  agent_id: string | null
  created_at: string | null
  status: RunStatus
  events: SandboxEvent[]
}

export interface LaunchRunRequest {
  agent_id: string
  task: string
}

export interface LaunchRunResponse {
  run_id: string
}

// FastAPI/pydantic's 422 error shape
export interface ValidationErrorDetail {
  loc: (string | number)[]
  msg: string
  type: string
}

// Mirrors sandbox_core.schemas.pipeline_spec / pipeline_run — a
// deterministic, harness-driven sequence of agent steps, not an
// LLM-orchestrated one. See docs/architecture.md.
export interface PipelineStep {
  step_id: string
  agent_id: string
  task_template: string
}

export interface PipelineSpec {
  id: string
  name: string
  steps: PipelineStep[]
}

export type PipelineStepStatus = 'completed' | 'errored' | 'truncated'
export type PipelineRunStatus = 'running' | 'completed' | 'errored'

export interface PipelineStepResult {
  step_id: string
  agent_id: string
  run_id: string
  status: PipelineStepStatus
  output: string | null
}

export interface PipelineRunRecord {
  pipeline_run_id: string
  pipeline_id: string
  task: string
  created_at: string
  status: PipelineRunStatus
  steps: PipelineStepResult[]
  error: string | null
}

export interface PipelineRunSummary {
  pipeline_run_id: string
  pipeline_id: string
  created_at: string
  status: PipelineRunStatus
  step_count: number
}

export interface LaunchPipelineRunRequest {
  pipeline_id: string
  task: string
}

export interface LaunchPipelineRunResponse {
  pipeline_run_id: string
}
