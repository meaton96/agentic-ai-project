import { useState, type ReactNode } from 'react'

export interface CollapsibleProps {
  title: ReactNode
  subtitle?: ReactNode
  defaultOpen?: boolean
  dark?: boolean
  children: ReactNode
}

/** A small collapse/expand section, used for the pieces inside an
 * already-expanded step card (one transcript message, one raw event) —
 * distinct from StepCard's own top-level expand, which this nests inside.
 * Defaults come from the caller: a tool message's JSON result or a
 * tool_called event's payload is usually the "massive json blob" worth
 * hiding by default, while short prose (system/user/assistant text)
 * is more useful left open. */
export function Collapsible({ title, subtitle, defaultOpen = false, dark = false, children }: CollapsibleProps) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div
      style={{
        marginBottom: 8,
        border: `1px solid ${dark ? 'rgba(255,255,255,0.12)' : 'var(--color-border)'}`,
        borderRadius: 'var(--radius-sm)',
        overflow: 'hidden',
      }}
    >
      <button
        onClick={() => setOpen((v) => !v)}
        style={{
          all: 'unset', display: 'flex', alignItems: 'center', gap: 8, width: '100%', boxSizing: 'border-box',
          padding: '6px 10px', cursor: 'pointer',
          background: dark ? 'rgba(255,255,255,0.04)' : 'var(--color-surface-sunken)',
        }}
      >
        <span style={{ fontSize: 11, opacity: 0.6, width: 10 }}>{open ? '▾' : '▸'}</span>
        <span style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', color: dark ? '#cbd5e1' : 'var(--color-text-muted)' }}>
          {title}
        </span>
        {subtitle && !open && (
          <span
            style={{
              fontSize: 11, color: dark ? '#94a3b8' : 'var(--color-text-muted)', overflow: 'hidden',
              textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1, minWidth: 0, textAlign: 'left',
            }}
          >
            {subtitle}
          </span>
        )}
      </button>
      {open && <div style={{ padding: 10 }}>{children}</div>}
    </div>
  )
}
