import { describe, expect, it } from 'vitest'
import type { WorkflowCatalog } from '../api/types'
import { toFlowEdges, toFlowNodes } from './workflowGraph'

const STATIC_CATALOG: WorkflowCatalog = {
  nodes: [
    { id: 'intake', label: 'Intake Agent', kind: 'agent', description: 'd1', data: {} },
    { id: 'validate_intake', label: 'Harness validates', kind: 'gate', description: 'd2', data: {} },
    { id: 'finalize', label: 'Harness: refit + evaluate once', kind: 'gate', description: 'd3', data: {} },
  ],
  edges: [
    { id: 'intake->validate_intake', source: 'intake', target: 'validate_intake' },
    { id: 'validate_intake->finalize', source: 'validate_intake', target: 'finalize' },
  ],
}

const DYNAMIC_CATALOG: WorkflowCatalog = {
  nodes: [
    { id: 'intake', label: 'Intake', kind: 'agent', description: 'd1', data: { required_state: { target_known: false } } },
    { id: 'modeling', label: 'Modeling', kind: 'agent', description: 'd2', data: { required_state: { split_leakage_passed: true } } },
  ],
  edges: [],
}

describe('toFlowNodes / toFlowEdges', () => {
  it('produces one flow node/edge per catalog node/edge for the static (sequential) topology', () => {
    const nodes = toFlowNodes(STATIC_CATALOG)
    const edges = toFlowEdges(STATIC_CATALOG.edges)
    expect(nodes).toHaveLength(3)
    expect(edges).toHaveLength(2)
    expect(nodes.map((n) => n.id)).toEqual(['intake', 'validate_intake', 'finalize'])
    expect(edges.map((e) => e.id)).toEqual(['intake->validate_intake', 'validate_intake->finalize'])
    // sequential layout: distinct x positions, same row
    expect(new Set(nodes.map((n) => n.position.x)).size).toBe(3)
    expect(new Set(nodes.map((n) => n.position.y)).size).toBe(1)
  })

  it('produces one flow node per catalog node and zero edges for the unordered dynamic catalog', () => {
    const nodes = toFlowNodes(DYNAMIC_CATALOG)
    const edges = toFlowEdges(DYNAMIC_CATALOG.edges)
    expect(nodes).toHaveLength(2)
    expect(edges).toHaveLength(0)
  })
})
