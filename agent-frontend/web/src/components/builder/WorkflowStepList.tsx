import { useRef } from 'react'
import type { WorkflowStepView } from '../../domain/workflowSteps'
import { WorkflowStepRow } from './WorkflowStepRow'

export interface WorkflowStepListProps {
  steps: WorkflowStepView[]
  selectedId: string | null
  onSelect: (id: string) => void
  onReorder: (orderedIds: string[]) => void
}

/** The Builder screen's left-hand list — a plain scrollable, drag-to-
 * reorder list, in place of the old React Flow pan/zoom canvas. Matches
 * the run timeline's own choice (PipelineTimeline.tsx) of a scrollable
 * list over a canvas: it reads correctly at any window size without
 * panning/zooming, and a small window just means more scrolling, not
 * nodes falling off-screen. See workflowSteps.ts for how step numbers and
 * "leads to" transitions recompute after a reorder, and BuilderPage.tsx
 * for why reordering is local, display-only state — nothing here is
 * persisted, same as the old canvas's node dragging. */
export function WorkflowStepList({ steps, selectedId, onSelect, onReorder }: WorkflowStepListProps) {
  const dragIndex = useRef<number | null>(null)

  function handleDrop(dropIndex: number) {
    const from = dragIndex.current
    dragIndex.current = null
    if (from === null || from === dropIndex) return
    const ids = steps.map((s) => s.id)
    const [moved] = ids.splice(from, 1)
    ids.splice(dropIndex, 0, moved)
    onReorder(ids)
  }

  return (
    <div>
      {steps.map((step, i) => (
        <WorkflowStepRow
          key={step.id}
          step={step}
          selected={step.id === selectedId}
          onSelect={() => onSelect(step.id)}
          onDragStart={() => {
            dragIndex.current = i
          }}
          onDragOver={(e) => e.preventDefault()}
          onDrop={() => handleDrop(i)}
        />
      ))}
    </div>
  )
}
