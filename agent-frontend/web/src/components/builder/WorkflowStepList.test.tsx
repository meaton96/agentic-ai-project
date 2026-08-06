import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { WorkflowStepList } from './WorkflowStepList'
import type { WorkflowStepView } from '../../domain/workflowSteps'

const STEPS: WorkflowStepView[] = [
  { id: 'a', step: 1, label: 'Step A', kind: 'agent', description: '', workflowData: {}, transitions: [] },
  { id: 'b', step: 2, label: 'Step B', kind: 'gate', description: '', workflowData: {}, transitions: [] },
  { id: 'c', step: 3, label: 'Step C', kind: 'agent', description: '', workflowData: {}, transitions: [] },
]

describe('WorkflowStepList', () => {
  it('calls onSelect with the clicked step id', () => {
    const onSelect = vi.fn()
    render(<WorkflowStepList steps={STEPS} selectedId={null} onSelect={onSelect} onReorder={vi.fn()} />)

    fireEvent.click(screen.getByTestId('step-row-b'))
    expect(onSelect).toHaveBeenCalledWith('b')
  })

  it('reorders via drag-and-drop and reports the new id order, not a mutated original', () => {
    const onReorder = vi.fn()
    render(<WorkflowStepList steps={STEPS} selectedId={null} onSelect={vi.fn()} onReorder={onReorder} />)

    // drag row "c" (index 2) onto row "a" (index 0) — should become [c, a, b]
    fireEvent.dragStart(screen.getByTestId('step-row-c'))
    fireEvent.dragOver(screen.getByTestId('step-row-a'))
    fireEvent.drop(screen.getByTestId('step-row-a'))

    expect(onReorder).toHaveBeenCalledWith(['c', 'a', 'b'])
    expect(STEPS.map((s) => s.id)).toEqual(['a', 'b', 'c']) // original prop array untouched
  })

  it('does nothing when a row is dropped on itself', () => {
    const onReorder = vi.fn()
    render(<WorkflowStepList steps={STEPS} selectedId={null} onSelect={vi.fn()} onReorder={onReorder} />)

    fireEvent.dragStart(screen.getByTestId('step-row-b'))
    fireEvent.drop(screen.getByTestId('step-row-b'))

    expect(onReorder).not.toHaveBeenCalled()
  })
})
