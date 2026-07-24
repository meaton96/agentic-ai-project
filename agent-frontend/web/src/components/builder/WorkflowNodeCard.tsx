import { Handle, Position, type NodeProps } from '@xyflow/react'
import type { WorkflowFlowNode } from '../../domain/workflowGraph'

/** Node renderer for the Builder screen's React Flow canvas. Reuses the
 * same dark/light convention StepCard and Legend already use for the run
 * timeline: "gate" (deterministic, no LLM) = dark fill, "agent" (LLM call)
 * = light fill — see PROJECT_OVERVIEW.md §3. Nodes are draggable (layout
 * only, per React Flow's default node dragging) but not editable — no
 * inputs, no delete affordance, no persistence of position. */
export function WorkflowNodeCard({ data, selected }: NodeProps<WorkflowFlowNode>) {
  const dark = data.kind === 'gate'
  return (
    <div
      style={{
        minWidth: 200,
        maxWidth: 220,
        padding: '10px 14px',
        borderRadius: 'var(--radius)',
        background: dark ? '#1e293b' : 'var(--color-surface)',
        color: dark ? '#ffffff' : 'var(--color-text)',
        border: selected ? '2px solid var(--color-primary)' : '1px solid var(--color-border)',
        boxShadow: 'var(--shadow-sm)',
        cursor: 'pointer',
      }}
    >
      <Handle type="target" position={Position.Left} style={{ opacity: 0.4 }} />
      <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.04em', opacity: 0.7, marginBottom: 4 }}>
        {data.kind}
      </div>
      <div style={{ fontSize: 13, fontWeight: 600, lineHeight: 1.3 }}>{data.label}</div>
      <Handle type="source" position={Position.Right} style={{ opacity: 0.4 }} />
    </div>
  )
}
