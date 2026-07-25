// Builds the Builder screen's step-list view model from a GET
// /api/workflow-catalog response. Replaces the old React Flow canvas (see
// git history) with data for a plain draggable list — see
// components/builder/WorkflowStepList.tsx for the rendering, and
// BuilderPage.tsx for why `order` is separate, local, never-persisted
// state rather than something derived once from the catalog.
import type { WorkflowCatalog, WorkflowNode } from '../api/types'

export interface StepTransition {
  label: string | null
  targetId: string
  targetLabel: string
  targetStep: number | null
}

export interface WorkflowStepView {
  id: string
  step: number | null
  label: string
  kind: WorkflowNode['kind']
  description: string
  workflowData: Record<string, unknown>
  transitions: StepTransition[]
}

/** `order` is the current on-screen display order (drag-reorderable —
 * see WorkflowStepList), independent of catalog.nodes' own order. Step
 * numbers and "leads to" transition targets are computed from `order`,
 * not the catalog's original sequence, so they stay correct — pointing at
 * whichever row currently shows that step — even after a drag reorders
 * the list. Only the static (sequential) catalog gets step numbers and
 * transitions at all: the dynamic catalog's edges are intentionally empty
 * (no fixed order — see server/routes.py), so it renders as a plain
 * unordered/unnumbered list instead of implying a sequence that doesn't
 * exist. */
export function buildStepViews(catalog: WorkflowCatalog, order: string[]): WorkflowStepView[] {
  const sequential = catalog.edges.length > 0
  const nodesById = new Map(catalog.nodes.map((n) => [n.id, n]))
  const stepNumberOf = (id: string): number | null => {
    const i = order.indexOf(id)
    return i === -1 ? null : i + 1
  }

  return order
    .map((id) => nodesById.get(id))
    .filter((n): n is WorkflowNode => n != null)
    .map((n) => ({
      id: n.id,
      step: sequential ? stepNumberOf(n.id) : null,
      label: n.label,
      kind: n.kind,
      description: n.description,
      workflowData: n.data,
      transitions: sequential
        ? catalog.edges
            .filter((e) => e.source === n.id)
            .map((e) => ({
              label: e.label ?? null,
              targetId: e.target,
              targetLabel: nodesById.get(e.target)?.label ?? e.target,
              targetStep: stepNumberOf(e.target),
            }))
        : [],
    }))
}
