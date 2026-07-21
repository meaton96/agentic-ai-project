import type { NodeState } from './graphTypes'

// Shared by StepCard (the actual left-edge color) and Legend (what that
// color means) — one source of truth so they can't drift apart.
export const STATE_COLOR: Record<NodeState, string> = {
  pending: 'var(--color-pending)',
  running: 'var(--color-running)',
  passed: 'var(--color-success)',
  failed: 'var(--color-danger)',
  vetoed: 'var(--color-vetoed)',
}
