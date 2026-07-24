import type { WorkflowNode } from '../../api/types'

function formatRequiredState(requiredState: Record<string, unknown>): string {
  return Object.entries(requiredState)
    .map(([key, value]) => `${key} = ${JSON.stringify(value)}`)
    .join('\n')
}

export interface WorkflowSidePanelProps {
  node: WorkflowNode
  onClose: () => void
}

/** Side panel opened by clicking a node — modeled on the run detail
 * timeline's expandable-detail pattern (StepCard), but rendered as a fixed
 * panel since a canvas of draggable nodes doesn't have the vertical list's
 * "expand in place" affordance. Shows exactly what the catalog gave us:
 * full description plus, for the dynamic catalog, `when_to_use` and the
 * `required_state` precondition gate straight from agent_registry.py; for
 * the static topology, its role in the fixed sequence (and the
 * verification node's reject-loopback target, if present). */
export function WorkflowSidePanel({ node, onClose }: WorkflowSidePanelProps) {
  const whenToUse = typeof node.data.when_to_use === 'string' ? node.data.when_to_use : undefined
  const requiredState =
    node.data.required_state && typeof node.data.required_state === 'object'
      ? (node.data.required_state as Record<string, unknown>)
      : undefined
  const onReject = typeof node.data.on_reject === 'string' ? node.data.on_reject : undefined

  return (
    <div className="card" style={{ width: 320, position: 'sticky', top: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
        <div>
          <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--color-text-muted)' }}>
            {node.kind}
          </div>
          <h3 style={{ margin: '2px 0 0 0' }}>{node.label}</h3>
        </div>
        <button onClick={onClose} style={{ background: 'transparent', color: 'var(--color-text-muted)', padding: 4 }}>
          ✕
        </button>
      </div>

      <h4>Description</h4>
      <p style={{ fontSize: 13, margin: 0 }}>{node.description}</p>

      {whenToUse && (
        <>
          <h4>When to use</h4>
          <p style={{ fontSize: 13, margin: 0 }}>{whenToUse}</p>
        </>
      )}

      {requiredState && Object.keys(requiredState).length > 0 && (
        <>
          <h4>Required state (precondition gate)</h4>
          <pre
            style={{
              whiteSpace: 'pre-wrap', fontSize: 12, background: 'var(--color-surface-sunken)',
              padding: 8, borderRadius: 'var(--radius-sm)', margin: 0,
            }}
          >
            {formatRequiredState(requiredState)}
          </pre>
        </>
      )}

      {onReject && (
        <>
          <h4>On reject</h4>
          <p className="muted" style={{ fontSize: 13, margin: 0 }}>
            Falls back to the next-best candidate, looping back to <strong>{onReject}</strong>.
          </p>
        </>
      )}
    </div>
  )
}
