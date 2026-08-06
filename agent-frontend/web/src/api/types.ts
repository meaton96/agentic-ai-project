// Response/request shapes for server/ (agent-frontend's FastAPI app). These
// mirror the *actual* JSON confirmed by hitting a locally-running server
// (see M2 summary) — not assumed from reading server/schemas.py alone.

export type Orchestrator = 'static' | 'dynamic'
export type ModelEndpoint = 'rit' | 'gateway' | 'local'
export type SplitStrategy = 'random' | 'stratified' | 'group' | 'time' | 'group_time'

// RunManager only ever produces pending/running/completed/failed; "unknown"
// is server/summaries.py's fallback for a run directory with no terminal
// event yet (e.g. a stray/crashed run from before this server started).
export type RunStatus = 'pending' | 'running' | 'completed' | 'failed' | 'unknown'

export interface DatasetInfo {
  filename: string
  size: number
  modified: number
}

export interface LaunchRunRequest {
  dataset: string
  orchestrator: Orchestrator
  target?: string | null
  goal?: string | null
  strategy?: SplitStrategy | null
  max_candidates?: number | null
  model_endpoint?: ModelEndpoint
  skip_feature_engineering?: boolean
  use_prompt_overrides?: boolean
  // Dynamic-orchestrator only (run_orchestrator.py/static has no --use-mcp
  // flag at all) — fetches agent tool facts through the MCP fact server
  // instead of in-process closures. See GET /api/mcp/config for what that
  // server is currently configured to serve.
  use_mcp?: boolean
}

export interface LaunchRunResponse {
  run_id: string
  status: string
}

// One line of runs/<run_id>/events.jsonl, replayed or tailed verbatim over
// SSE — payload's shape depends entirely on `type` (see events.ts for the
// catalog this UI knows how to render); anything else is rendered as raw
// JSON in the side panel rather than assumed away.
export interface RunEvent {
  ts: number
  run_id: string
  phase: string
  type: string
  payload: Record<string, unknown>
}

// orchestrator_report.json / dynamic_orchestrator_report.json — pipeline-
// owned, deliberately loose here rather than re-declaring every field: the
// UI reads specific known keys defensively (see api/client.ts's helpers)
// instead of assuming the whole shape.
export type RunReport = Record<string, unknown>

export interface LeaderboardEntry {
  run_id: string
  candidate: string
  template_id: string
  source: string
  model: string
  split: string
  verification_verdict: string
  logged_at_utc: string
  metrics: Record<string, { metric: string; value: number; ci_low: number; ci_high: number; n_bootstrap: number }>
  strategy?: string
  data_hash?: string
  seed?: number
}

export interface RunSummary {
  run_id: string
  orchestrator: Orchestrator | null
  status: RunStatus
  // Preferred over first_event.payload.data: the dynamic orchestrator's
  // run_started event doesn't record the dataset path at all (a real
  // pipeline gap, confirmed on real data) — the server falls back to
  // RunManager's own launch config when it tracked this run, so this is
  // populated for any orchestrator type as long as this server launched
  // it, even though the event itself may have nothing.
  dataset: string | null
  started_at: number | null
  finished_at: number | null
  error: string | null
  n_events: number
  first_event: RunEvent | null
  last_event: RunEvent | null
  report: RunReport | null
  leaderboard_entries: LeaderboardEntry[]
}

// GET /api/prompts / PUT/DELETE /api/prompts/{agent} — default_content is
// always the pipeline's shipped prompts/<agent>.md, read-only from here;
// override_content is this app's own override file for that agent, or
// null if none has been saved.
export interface PromptInfo {
  agent: string
  default_content: string
  override_content: string | null
  has_override: boolean
}

// make_transcript_writer (cli_common.py) always writes a JSON array of
// prettified chat messages — role/content plus, for assistant turns that
// called a tool, tool_calls; for tool-result turns, tool_call_id.
export interface TranscriptMessage {
  role: string
  content: unknown
  tool_calls?: Array<{
    id: string
    type: string
    function: { name: string; arguments: unknown }
  }>
  tool_call_id?: string
}

// GET /api/workflow-catalog?type=static|dynamic — read-only topology for the
// Builder screen. "gate" = deterministic/no-LLM (dark, per PROJECT_OVERVIEW.md
// §3), "agent" = an LLM call. `data` carries type-specific extras: the
// dynamic catalog puts `when_to_use`/`required_state` there (straight off
// agentic_ml.orchestrator.agent_registry.AgentSpec), the static topology
// leaves it mostly empty.
export type WorkflowCatalogType = 'static' | 'dynamic'
export type WorkflowNodeKind = 'agent' | 'gate'

export interface WorkflowNode {
  id: string
  label: string
  kind: WorkflowNodeKind
  description: string
  data: Record<string, unknown>
}

export interface WorkflowEdge {
  id: string
  source: string
  target: string
  label?: string | null
}

export interface WorkflowCatalog {
  nodes: WorkflowNode[]
  edges: WorkflowEdge[]
}

// GET /api/mcp/config — read-only view of the pipeline's MCP fact server
// (agentic_ml.mcp_facts.server): configs/mcp_server.json plus the tool
// catalog that config would register, straight off the real FastMCP tool
// registry. `reachable` is a best-effort TCP check only (no MCP handshake) —
// this app never starts/stops that server process itself.
export interface McpToolInfo {
  name: string
  description: string
  input_schema: Record<string, unknown>
  enabled: boolean
}

export interface McpConfigResponse {
  config_path: string
  config_exists: boolean
  name: string
  host: string
  port: number
  url: string
  enabled_tools: string[]
  reachable: boolean
  tools: McpToolInfo[]
}
