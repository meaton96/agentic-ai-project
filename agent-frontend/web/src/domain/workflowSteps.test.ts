import { describe, expect, it } from 'vitest'
import type { WorkflowCatalog } from '../api/types'
import { buildStepViews } from './workflowSteps'

const STATIC_CATALOG: WorkflowCatalog = {
  nodes: [
    { id: 'intake', label: 'Intake Agent', kind: 'agent', description: 'd1', data: {} },
    { id: 'modeling', label: 'Modeling Agent', kind: 'agent', description: 'd2', data: {} },
    { id: 'verification', label: 'Verification Agent', kind: 'agent', description: 'd3', data: {} },
    { id: 'finalize', label: 'Harness: refit + evaluate once', kind: 'gate', description: 'd4', data: {} },
  ],
  edges: [
    { id: 'intake->modeling', source: 'intake', target: 'modeling' },
    { id: 'modeling->verification', source: 'modeling', target: 'verification' },
    { id: 'verification->finalize', source: 'verification', target: 'finalize', label: 'approved / flagged' },
    { id: 'verification->modeling', source: 'verification', target: 'modeling', label: 'rejected: try again' },
  ],
}

const DYNAMIC_CATALOG: WorkflowCatalog = {
  nodes: [
    { id: 'intake', label: 'Intake', kind: 'agent', description: 'd1', data: {} },
    { id: 'modeling', label: 'Modeling', kind: 'agent', description: 'd2', data: {} },
  ],
  edges: [],
}

describe('buildStepViews', () => {
  it('numbers steps and resolves single-target transitions in catalog order', () => {
    const order = STATIC_CATALOG.nodes.map((n) => n.id)
    const views = buildStepViews(STATIC_CATALOG, order)

    expect(views.map((v) => v.step)).toEqual([1, 2, 3, 4])
    expect(views[0].transitions).toEqual([
      { label: null, targetId: 'modeling', targetLabel: 'Modeling Agent', targetStep: 2 },
    ])
  })

  it('resolves a branching node into one transition per outgoing edge, in edge order', () => {
    const order = STATIC_CATALOG.nodes.map((n) => n.id)
    const views = buildStepViews(STATIC_CATALOG, order)
    const verification = views.find((v) => v.id === 'verification')!

    expect(verification.transitions).toEqual([
      { label: 'approved / flagged', targetId: 'finalize', targetLabel: 'Harness: refit + evaluate once', targetStep: 4 },
      { label: 'rejected: try again', targetId: 'modeling', targetLabel: 'Modeling Agent', targetStep: 2 },
    ])
  })

  it('recomputes step numbers and transition targets after the display order changes', () => {
    // Reversed from catalog order — simulates a drag-to-reorder.
    const reordered = ['finalize', 'verification', 'modeling', 'intake']
    const views = buildStepViews(STATIC_CATALOG, reordered)

    expect(views.map((v) => v.id)).toEqual(reordered)
    expect(views.map((v) => v.step)).toEqual([1, 2, 3, 4])

    const verification = views.find((v) => v.id === 'verification')!
    // finalize is now step 1, modeling is now step 3 — transitions must follow the new order.
    expect(verification.transitions).toEqual([
      { label: 'approved / flagged', targetId: 'finalize', targetLabel: 'Harness: refit + evaluate once', targetStep: 1 },
      { label: 'rejected: try again', targetId: 'modeling', targetLabel: 'Modeling Agent', targetStep: 3 },
    ])
  })

  it('gives every node a null step and no transitions for the unordered dynamic catalog', () => {
    const order = DYNAMIC_CATALOG.nodes.map((n) => n.id)
    const views = buildStepViews(DYNAMIC_CATALOG, order)

    expect(views.every((v) => v.step === null)).toBe(true)
    expect(views.every((v) => v.transitions.length === 0)).toBe(true)
  })
})
