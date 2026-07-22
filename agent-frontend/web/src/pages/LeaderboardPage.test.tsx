import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import { LeaderboardPage } from './LeaderboardPage'
import * as client from '../api/client'
import type { LeaderboardEntry } from '../api/types'

vi.mock('../api/client')

const ENTRY: LeaderboardEntry = {
  run_id: 'run_abc123', candidate: 'candidate_a', template_id: 'sklearn_mixed_pipeline',
  source: 'orchestrator', model: 'qwen3-coder:30b', split: 'validation',
  verification_verdict: 'approved', logged_at_utc: '2026-07-21T20:41:03Z',
  metrics: { roc_auc: { metric: 'roc_auc', value: 0.5333, ci_low: 0.25, ci_high: 0.79, n_bootstrap: 200 } },
}

describe('LeaderboardPage', () => {
  it('renders entries from GET /api/leaderboard', async () => {
    vi.mocked(client.getLeaderboard).mockResolvedValue([ENTRY])
    render(
      <MemoryRouter>
        <LeaderboardPage />
      </MemoryRouter>,
    )
    expect(await screen.findByText('candidate_a')).toBeInTheDocument()
    expect(screen.getByText('0.533')).toBeInTheDocument()
    expect(client.getLeaderboard).toHaveBeenCalledWith(undefined)
  })

  it('filters by run id via the query param', async () => {
    vi.mocked(client.getLeaderboard).mockResolvedValue([ENTRY])
    render(
      <MemoryRouter initialEntries={['/leaderboard?run_id=run_abc123']}>
        <LeaderboardPage />
      </MemoryRouter>,
    )
    expect(await screen.findByText('candidate_a')).toBeInTheDocument()
    expect(client.getLeaderboard).toHaveBeenCalledWith('run_abc123')
  })
})
