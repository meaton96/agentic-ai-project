import { useEffect, useMemo, useState } from 'react'
import { getWorkflowCatalog } from '../api/client'
import type { WorkflowCatalog, WorkflowCatalogType, WorkflowNode } from '../api/types'
import { buildStepViews } from '../domain/workflowSteps'
import { WorkflowStepList } from '../components/builder/WorkflowStepList'
import { WorkflowSidePanel } from '../components/builder/WorkflowSidePanel'

// Only the *outer* chrome (nav bar + app-main's own top/bottom padding) is
// subtracted from 100vh — this page's own header/description/radio row
// lives inside the flex column below and is sized by flexbox itself, so
// the step list/side panel row (flex: 1) always gets whatever's left,
// instead of a fixed-height canvas that cut off the bottom of a long list
// on any reasonably small window.
const CHROME_HEIGHT = 'calc(100vh - 96px)'

/** Workflow Builder v0: a read-only step-list visualization of the
 * pipeline topology, fetched from GET /api/workflow-catalog. Separate
 * screen from run detail's timeline (that renders one run's actual
 * progress; this renders the topology itself, independent of any run).
 * Editing, swapping candidates, and saving layouts are out of scope for
 * this pass — steps are draggable only so a user can reorder the list for
 * their own reading, never persisted, and no transitions can be created
 * or changed here. */
export function BuilderPage() {
  const [type, setType] = useState<WorkflowCatalogType>('static')
  const [catalog, setCatalog] = useState<WorkflowCatalog | null>(null)
  const [order, setOrder] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)

  useEffect(() => {
    setError(null)
    setSelectedId(null)
    // Cleared before the fetch, not left stale: `order` (which decides
    // step numbers/transition targets — see buildStepViews) belongs to
    // one specific catalog, and the previous type's order/catalog would
    // otherwise still be showing while the new one loads.
    setCatalog(null)
    getWorkflowCatalog(type)
      .then((c) => {
        setCatalog(c)
        setOrder(c.nodes.map((n) => n.id))
      })
      .catch((e: Error) => setError(e.message))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [type])

  const steps = useMemo(() => (catalog ? buildStepViews(catalog, order) : []), [catalog, order])
  const selectedStep = useMemo(() => steps.find((s) => s.id === selectedId) ?? null, [steps, selectedId])
  const selectedNode = useMemo<WorkflowNode | null>(
    () => catalog?.nodes.find((n) => n.id === selectedId) ?? null,
    [catalog, selectedId],
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: CHROME_HEIGHT, minHeight: 480 }}>
      <div style={{ display: 'flex', gap: 12, alignItems: 'baseline', marginBottom: 4 }}>
        <h2 style={{ margin: 0 }}>Workflow Builder</h2>
        <span className="muted">read-only visualization of the pipeline topology</span>
      </div>
      <p className="muted" style={{ marginTop: 0, marginBottom: 16 }}>
        Steps can be dragged to reorder the list for your own reading — nothing here edits, swaps, or saves a
        workflow.
      </p>

      <div className="field radio-group" style={{ marginBottom: 16 }}>
        <label>
          <input type="radio" checked={type === 'static'} onChange={() => setType('static')} />
          static
        </label>
        <label>
          <input type="radio" checked={type === 'dynamic'} onChange={() => setType('dynamic')} />
          dynamic
        </label>
      </div>

      {error && <div className="error-text">Failed to load workflow catalog: {error}</div>}
      {!catalog && !error && <div className="card muted">Loading…</div>}

      {catalog && (
        <div style={{ display: 'flex', gap: 24, flex: 1, minHeight: 0 }}>
          <div
            className="card"
            style={{ flex: 1, minWidth: 0, overflowY: 'auto', padding: 8 }}
            data-testid="workflow-step-list"
          >
            <WorkflowStepList steps={steps} selectedId={selectedId} onSelect={setSelectedId} onReorder={setOrder} />
          </div>

          <div style={{ width: 320, flexShrink: 0, overflowY: 'auto' }}>
            {selectedNode && selectedStep ? (
              <WorkflowSidePanel
                node={selectedNode}
                step={selectedStep.step}
                transitions={selectedStep.transitions}
                onClose={() => setSelectedId(null)}
              />
            ) : (
              <div className="card muted">
                Click a step to see its full description
                {type === 'dynamic' ? ', when_to_use, and required_state.' : ' and role in the sequence.'}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
