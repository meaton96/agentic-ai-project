// Converts a GET /api/workflow-catalog response into React Flow's node/edge
// shape, with an initial layout position. This is display layout only —
// nothing here is persisted, and dragging a node in the Builder screen never
// writes back to the server (v0 is read-only: visualization, not editing).
import type { Edge, Node } from '@xyflow/react'
import type { WorkflowCatalog, WorkflowEdge, WorkflowNode } from '../api/types'

const COL = 260
const ROW = 150
const GRID_COLS = 3

export interface WorkflowFlowNodeData extends Record<string, unknown> {
  label: string
  kind: WorkflowNode['kind']
  description: string
  workflowData: Record<string, unknown>
}

export type WorkflowFlowNode = Node<WorkflowFlowNodeData>

/** Static topology has a fixed sequence (edges present) so nodes are laid out
 * in a single row in server-returned order. The dynamic catalog has no fixed
 * order (edges intentionally empty — see server/routes.py) so nodes are
 * arranged in a simple grid instead of implying a sequence that doesn't
 * exist. */
export function toFlowNodes(catalog: WorkflowCatalog): WorkflowFlowNode[] {
  const sequential = catalog.edges.length > 0
  return catalog.nodes.map((n, i) => ({
    id: n.id,
    type: 'workflowNode',
    position: sequential
      ? { x: i * COL, y: 0 }
      : { x: (i % GRID_COLS) * COL, y: Math.floor(i / GRID_COLS) * ROW },
    data: { label: n.label, kind: n.kind, description: n.description, workflowData: n.data },
  }))
}

export function toFlowEdges(edges: WorkflowEdge[]): Edge[] {
  return edges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    label: e.label ?? undefined,
    labelStyle: { fontSize: 11 },
    style: { stroke: 'var(--color-text-muted)' },
  }))
}
