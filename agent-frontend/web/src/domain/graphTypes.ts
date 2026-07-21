import type { RunEvent } from '../api/types'

export type NodeState = 'pending' | 'running' | 'passed' | 'failed' | 'vetoed'

// 'agent' = light node (an LLM call), 'harness' = dark node (deterministic
// code, no LLM) — matches PROJECT_OVERVIEW.md §3's mermaid diagram styling.
export type NodeKind = 'agent' | 'harness'

export interface GraphNodeData extends Record<string, unknown> {
  label: string
  kind: NodeKind
  state: NodeState
  /** Every event this node's state/details were derived from — shown in the
   * side panel for harness nodes (and as a fallback for agent nodes with no
   * transcript). Never re-interpreted into new decisions, just displayed. */
  events: RunEvent[]
  /** Transcript filename (from GET /api/runs/{id}/transcripts), if this
   * node represents an LLM call the pipeline wrote a transcript for. */
  transcriptName?: string
  /** Planner proposals validate_plan() rejected are rendered as their own
   * dimmed nodes rather than omitted — this flags that rendering mode. */
  rejected?: boolean
}

export interface GraphEdgeData {
  dashed?: boolean
}

export interface GraphNode {
  id: string
  position: { x: number; y: number }
  data: GraphNodeData
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  data?: GraphEdgeData
}

export interface GraphModel {
  nodes: GraphNode[]
  edges: GraphEdge[]
}
