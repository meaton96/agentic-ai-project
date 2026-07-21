import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import { RunsListPage } from './RunsListPage'
import * as client from '../api/client'
import type { RunSummary } from '../api/types'

vi.mock('../api/client')

const RUN: RunSummary = {
  run_id: 'run_abc123',
  orchestrator: 'static',
  status: 'completed',
  started_at: null,
  finished_at: null,
  error: null,
  n_events: 5,
  first_event: {
    ts: 1700000000, run_id: 'run_abc123', phase: 'run', type: 'run_started',
    payload: { data: '/data/datasets/churn.csv', goal: '', target: 'churned' },
  },
  last_event: { ts: 1700000100, run_id: 'run_abc123', phase: 'run', type: 'run_completed', payload: { status: 'success' } },
  report: null,
  leaderboard_entries: [],
}

describe('RunsListPage', () => {
  it('renders a table row per run from GET /api/runs', async () => {
    vi.mocked(client.listRuns).mockResolvedValue([RUN])

    render(
      <MemoryRouter>
        <RunsListPage />
      </MemoryRouter>,
    )

    expect(await screen.findByText('run_abc123')).toBeInTheDocument()
    expect(screen.getByText('static')).toBeInTheDocument()
    expect(screen.getByText('churn.csv')).toBeInTheDocument()
    expect(screen.getByText('completed')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'run_abc123' })).toHaveAttribute('href', '/runs/run_abc123')
  })

  it('shows an empty state with no runs', async () => {
    vi.mocked(client.listRuns).mockResolvedValue([])

    render(
      <MemoryRouter>
        <RunsListPage />
      </MemoryRouter>,
    )

    expect(await screen.findByText(/No runs yet/)).toBeInTheDocument()
  })

  it('shows an error if the request fails', async () => {
    vi.mocked(client.listRuns).mockRejectedValue(new Error('boom'))

    render(
      <MemoryRouter>
        <RunsListPage />
      </MemoryRouter>,
    )

    expect(await screen.findByText(/Failed to load runs: boom/)).toBeInTheDocument()
  })
})
