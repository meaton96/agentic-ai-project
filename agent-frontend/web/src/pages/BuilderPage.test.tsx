import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { BuilderPage } from './BuilderPage'
import * as client from '../api/client'
import type { WorkflowCatalog } from '../api/types'

vi.mock('../api/client')

const STATIC_CATALOG: WorkflowCatalog = {
  nodes: [
    { id: 'intake', label: 'Intake Agent', kind: 'agent', description: 'Proposes the target column and identifiers.', data: {} },
    { id: 'validate_intake', label: 'Harness validates', kind: 'gate', description: 'Deterministic check of the proposed DatasetSpec.', data: {} },
    { id: 'finalize', label: 'Harness: refit + evaluate once', kind: 'gate', description: 'Deterministic refit and one-time test evaluation.', data: {} },
  ],
  edges: [
    { id: 'intake->validate_intake', source: 'intake', target: 'validate_intake' },
    { id: 'validate_intake->finalize', source: 'validate_intake', target: 'finalize' },
  ],
}

const DYNAMIC_CATALOG: WorkflowCatalog = {
  nodes: [
    {
      id: 'intake', label: 'Intake', kind: 'agent',
      description: 'Proposes which column is the prediction target.',
      data: { when_to_use: 'First step, whenever the target column is not already known.', required_state: { target_known: false } },
    },
    {
      id: 'modeling', label: 'Modeling', kind: 'agent',
      description: 'Proposes one modeling candidate.',
      data: { when_to_use: 'After the split passes its leakage checks.', required_state: { split_leakage_passed: true } },
    },
  ],
  edges: [],
}

describe('BuilderPage', () => {
  it('renders the static topology (agent + gate node kinds) and shows a node description on click', async () => {
    vi.mocked(client.getWorkflowCatalog).mockResolvedValue(STATIC_CATALOG)
    render(<BuilderPage />)

    expect(await screen.findByText('Intake Agent')).toBeInTheDocument()
    expect(screen.getByText('Harness validates')).toBeInTheDocument()
    expect(screen.getByText('Harness: refit + evaluate once')).toBeInTheDocument()
    expect(document.querySelectorAll('.react-flow__node')).toHaveLength(STATIC_CATALOG.nodes.length)

    fireEvent.click(screen.getByText('Intake Agent'))
    expect(await screen.findByText('Proposes the target column and identifiers.')).toBeInTheDocument()
  })

  it('renders the dynamic catalog (unordered) and shows when_to_use + required_state on click', async () => {
    vi.mocked(client.getWorkflowCatalog).mockResolvedValue(DYNAMIC_CATALOG)
    render(<BuilderPage />)

    fireEvent.click(screen.getByRole('radio', { name: 'dynamic' }))

    expect(await screen.findByText('Modeling')).toBeInTheDocument()
    expect(document.querySelectorAll('.react-flow__node')).toHaveLength(DYNAMIC_CATALOG.nodes.length)

    fireEvent.click(screen.getByText('Modeling'))
    expect(await screen.findByText('Proposes one modeling candidate.')).toBeInTheDocument()
    expect(screen.getByText('After the split passes its leakage checks.')).toBeInTheDocument()
    expect(screen.getByText(/split_leakage_passed = true/)).toBeInTheDocument()
  })

  it('shows an error if the workflow catalog request fails', async () => {
    vi.mocked(client.getWorkflowCatalog).mockRejectedValue(new Error('boom'))
    render(<BuilderPage />)

    await waitFor(() => expect(client.getWorkflowCatalog).toHaveBeenCalled())
    expect(await screen.findByText(/Failed to load workflow catalog: boom/)).toBeInTheDocument()
  })
})
