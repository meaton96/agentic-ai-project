import type { WorkflowNode } from '../../api/types'
import type { StepTransition } from '../../domain/workflowSteps'

function formatRequiredState(requiredState: Record<string, unknown>): string {
  return Object.entries(requiredState)
    .map(([key, value]) => `${key} = ${JSON.stringify(value)}`)
    .join('\n')
}

export interface WorkflowSidePanelProps {
  node: WorkflowNode
  step: number | null
  transitions: StepTransition[]
  onClose: () => void
}

/** Side panel opened by clicking a step in WorkflowStepList — full
 * description plus, for the dynamic catalog, `when_to_use` and the
 * `required_state` precondition gate straight from agent_registry.py; for
 * the static topology's steps, every outgoing transition (including
 * verification's branch: approved/flagged forward vs. rejected looping
 * back to modeling), derived from the catalog's own edges rather than a
 * single hand-picked `on_reject` field, so a branch with any number of
 * outcomes reads the same way. */
export function WorkflowSidePanel({ node, step, transitions, onClose }: WorkflowSidePanelProps) {
  const whenToUse = typeof node.data.when_to_use === 'string' ? node.data.when_to_use : undefined
  const requiredState =
    node.data.required_state && typeof node.data.required_state === 'object'
      ? (node.data.required_state as Record<string, unknown>)
      : undefined

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
        <div>
          <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--color-text-muted)' }}>
            {step !== null ? `Step ${step} · ${node.kind}` : node.kind}
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

      {transitions.length > 0 && (
        <>
          <h4>Leads to</h4>
          {transitions.map((t) => (
            <p key={`${t.targetId}-${t.label ?? ''}`} style={{ fontSize: 13, margin: '0 0 6px 0' }}>
              {t.label ? `${t.label} → ` : '→ '}
              <strong>{t.targetStep !== null ? `Step ${t.targetStep}: ` : ''}{t.targetLabel}</strong>
            </p>
          ))}
        </>
      )}
    </div>
  )
}
