import type { DragEvent } from 'react'
import type { WorkflowStepView } from '../../domain/workflowSteps'

export interface WorkflowStepRowProps {
  step: WorkflowStepView
  selected: boolean
  onSelect: () => void
  onDragStart: () => void
  onDragOver: (e: DragEvent) => void
  onDrop: () => void
}

/** One row in the Builder's left-hand step list. Deliberately compact and
 * non-expanding (unlike the run timeline's StepCard) — clicking a row
 * shows its full detail in WorkflowSidePanel on the right instead. The
 * drag handle reorders rows for on-screen reading only; see
 * WorkflowStepList/BuilderPage for why that's never persisted. */
export function WorkflowStepRow({ step, selected, onSelect, onDragStart, onDragOver, onDrop }: WorkflowStepRowProps) {
  const dark = step.kind === 'gate'
  return (
    <div
      draggable
      onDragStart={onDragStart}
      onDragOver={onDragOver}
      onDrop={onDrop}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      data-testid={`step-row-${step.id}`}
      style={{
        display: 'flex',
        gap: 10,
        alignItems: 'flex-start',
        padding: '10px 12px',
        marginBottom: 4,
        borderRadius: 'var(--radius-sm)',
        cursor: 'pointer',
        background: selected ? 'var(--color-running-bg)' : 'transparent',
        borderLeft: `3px solid ${selected ? 'var(--color-primary)' : 'transparent'}`,
      }}
    >
      <span style={{ opacity: 0.35, fontSize: 13, lineHeight: '22px', cursor: 'grab' }} title="Drag to reorder">
        ⠿
      </span>

      {step.step !== null && (
        <span
          style={{
            flexShrink: 0,
            width: 22,
            height: 22,
            marginTop: 1,
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 11,
            fontWeight: 700,
            background: dark ? '#1e293b' : 'var(--color-surface-sunken)',
            color: dark ? '#ffffff' : 'var(--color-text)',
          }}
        >
          {step.step}
        </span>
      )}

      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', gap: 8, alignItems: 'baseline', flexWrap: 'wrap' }}>
          <span style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--color-text-muted)' }}>
            {step.kind}
          </span>
          <strong style={{ fontSize: 13 }}>{step.label}</strong>
        </div>

        {step.transitions.map((t) => (
          <div key={`${t.targetId}-${t.label ?? ''}`} className="muted" style={{ fontSize: 11, marginTop: 2 }}>
            {t.label ? `${t.label} → ` : '→ '}
            {t.targetStep !== null ? `Step ${t.targetStep}: ` : ''}
            {t.targetLabel}
          </div>
        ))}
      </div>
    </div>
  )
}
