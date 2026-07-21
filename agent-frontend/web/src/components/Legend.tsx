import type { NodeState } from '../domain/graphTypes'
import { STATE_COLOR } from '../domain/stateColors'

const STATES: { state: NodeState; description: string }[] = [
  { state: 'pending', description: "hasn't run (yet, or at all — e.g. a skipped step)" },
  { state: 'running', description: 'in progress' },
  { state: 'passed', description: 'completed / approved' },
  { state: 'failed', description: 'failed a gate, or rejected' },
  { state: 'vetoed', description: 'verification rejected it' },
]

/** Explains the two visual conventions every step card uses: dark vs.
 * light fill for harness (deterministic) vs. agent (LLM) steps, and the
 * left-edge color for state — same conventions PROJECT_OVERVIEW.md §3's
 * diagram uses (dark = no LLM call, light = an agent). */
export function Legend() {
  return (
    <div className="card" style={{ width: 220, position: 'sticky', top: 16, fontSize: 12 }}>
      <h4 style={{ marginTop: 0 }}>Legend</h4>

      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <div style={{ width: 20, height: 14, borderRadius: 3, background: '#1e293b' }} />
        <span>Harness step (deterministic, no LLM)</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
        <div style={{ width: 20, height: 14, borderRadius: 3, background: 'var(--color-surface)', border: '1px solid var(--color-border)' }} />
        <span>Agent step (LLM call)</span>
      </div>

      <div className="muted" style={{ textTransform: 'uppercase', fontSize: 10, letterSpacing: '0.04em', marginBottom: 6 }}>
        Left-edge color = state
      </div>
      {STATES.map(({ state, description }) => (
        <div key={state} style={{ display: 'flex', alignItems: 'flex-start', gap: 8, marginBottom: 6 }}>
          <div style={{ width: 4, height: 16, borderRadius: 2, background: STATE_COLOR[state], flexShrink: 0, marginTop: 1 }} />
          <span>
            <strong>{state}</strong> — {description}
          </span>
        </div>
      ))}

      <div className="muted" style={{ marginTop: 12, fontStyle: 'italic' }}>
        Dimmed, struck-through steps are planner proposals the harness rejected before they ran.
      </div>
    </div>
  )
}
