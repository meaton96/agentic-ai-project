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
  dataset: '/data/datasets/churn.csv',
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

// The dynamic orchestrator's run_started event never records the dataset
// path at all (a real pipeline gap) — the server falls back to its own
// launch config for a run it tracked, so `dataset` can be populated even
// when first_event has nothing usable. This is exactly that shape.
const DYNAMIC_RUN_WITH_NO_DATA_IN_EVENT: RunSummary = {
  run_id: 'run_dyn456',
  orchestrator: 'dynamic',
  status: 'completed',
  dataset: '/data/datasets/titanic.csv',
  started_at: null,
  finished_at: null,
  error: null,
  n_events: 2,
  first_event: { ts: 1700000000, run_id: 'run_dyn456', phase: 'run', type: 'run_started', payload: { goal: '', max_iterations: 15 } },
  last_event: { ts: 1700000100, run_id: 'run_dyn456', phase: 'run', type: 'run_completed', payload: { status: 'success' } },
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

  it('shows the dataset for a dynamic run even though its own run_started event has no data field', async () => {
    vi.mocked(client.listRuns).mockResolvedValue([DYNAMIC_RUN_WITH_NO_DATA_IN_EVENT])

    render(
      <MemoryRouter>
        <RunsListPage />
      </MemoryRouter>,
    )

    expect(await screen.findByText('run_dyn456')).toBeInTheDocument()
    expect(screen.getByText('titanic.csv')).toBeInTheDocument()
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
