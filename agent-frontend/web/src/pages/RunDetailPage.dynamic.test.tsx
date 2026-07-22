import { act, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { RunDetailPage } from './RunDetailPage'
import * as client from '../api/client'
import type { RunEvent, RunSummary } from '../api/types'
import { FakeEventSource } from '../test-utils/FakeEventSource'

vi.mock('../api/client')

const RUN_ID = 'run_dynamic1'

function summaryFor(status: RunSummary['status']): RunSummary {
  return {
    run_id: RUN_ID, orchestrator: 'dynamic', status, started_at: null, finished_at: null,
    error: null, n_events: 0, first_event: null, last_event: null, report: null, leaderboard_entries: [],
  }
}

// Mirrors the real event shapes confirmed against a live server for a
// dynamic-orchestrator run whose first planner proposal names an agent_id
// outside the real catalog (validate_plan() rejects it), then finishes.
const SCRIPTED_EVENTS: RunEvent[] = [
  { ts: 1, run_id: RUN_ID, phase: 'run', type: 'run_started', payload: { goal: 'predict churned', max_iterations: 15 } },
  { ts: 2, run_id: RUN_ID, phase: 'planner', type: 'agent_started', payload: { iteration: 0, previous_error: null } },
  {
    ts: 3, run_id: RUN_ID, phase: 'planner', type: 'planner_proposal_rejected',
    payload: {
      iteration: 0,
      proposal: { action: 'run_agent', agent_id: 'delete_all_data', args: {}, reasoning: '???' },
      errors: ["agent_id 'delete_all_data' is not a known/available agent for this run"],
    },
  },
  { ts: 4, run_id: RUN_ID, phase: 'planner', type: 'agent_started', payload: { iteration: 0, previous_error: 'not known' } },
  {
    ts: 5, run_id: RUN_ID, phase: 'planner', type: 'planner_proposal_accepted',
    payload: { iteration: 0, proposal: { action: 'finish', agent_id: null, args: {}, reasoning: 'done' } },
  },
  { ts: 6, run_id: RUN_ID, phase: 'run', type: 'run_completed', payload: { status: 'success', iteration: 0 } },
]

describe('RunDetailPage — dynamic orchestrator graph', () => {
  beforeEach(() => {
    FakeEventSource.reset()
    vi.stubGlobal('EventSource', FakeEventSource)
    vi.mocked(client.getRun).mockResolvedValue(summaryFor('running'))
    vi.mocked(client.listTranscripts).mockResolvedValue([])
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('renders a rejected planner proposal as its own distinct node, not omitted', async () => {
    render(
      <MemoryRouter initialEntries={[`/runs/${RUN_ID}`]}>
        <Routes>
          <Route path="/runs/:runId" element={<RunDetailPage />} />
        </Routes>
      </MemoryRouter>,
    )

    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1))
    const source = FakeEventSource.latest()

    await act(async () => {
      for (const event of SCRIPTED_EVENTS) source.emit(event)
    })

    const rejectedNode = await screen.findByText('rejected: delete_all_data')
    expect(rejectedNode).toBeInTheDocument()
    // dimmed/struck-through rendering, not just present in a list somewhere
    expect(rejectedNode).toHaveStyle({ textDecoration: 'line-through' })
    expect(rejectedNode.closest('[data-testid^="node-"]')).toHaveAttribute('data-state', 'failed')

    const finishNode = await screen.findByText('finish')
    expect(finishNode.closest('[data-testid^="node-"]')).toHaveAttribute('data-state', 'passed')
  })

  it('shows the full typical sequence as pending steps before the planner has decided anything', async () => {
    render(
      <MemoryRouter initialEntries={[`/runs/${RUN_ID}`]}>
        <Routes>
          <Route path="/runs/:runId" element={<RunDetailPage />} />
        </Routes>
      </MemoryRouter>,
    )
    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1))
    const source = FakeEventSource.latest()

    await act(async () => {
      source.emit(SCRIPTED_EVENTS[0]) // just run_started — planner hasn't proposed anything yet
    })

    for (const agentId of ['intake', 'feature_engineering', 'profiler', 'split_and_check_leakage', 'modeling', 'verification', 'finalize', 'summarize']) {
      expect(await screen.findByTestId(`node-step_${agentId}`)).toHaveAttribute('data-state', 'pending')
    }
  })

  it('renders a second modeling attempt after a rejected verdict as its own step', async () => {
    const retryEvents: RunEvent[] = [
      { ts: 1, run_id: RUN_ID, phase: 'run', type: 'run_started', payload: { goal: 'predict churned', max_iterations: 15 } },
      { ts: 2, run_id: RUN_ID, phase: 'planner', type: 'planner_proposal_accepted', payload: { iteration: 0, proposal: { action: 'run_agent', agent_id: 'modeling', args: {}, reasoning: '' } } },
      { ts: 3, run_id: RUN_ID, phase: 'modeling', type: 'candidate_scored', payload: { candidate_id: 'candidate_a', passed_gate: true, metrics: {} } },
      { ts: 4, run_id: RUN_ID, phase: 'planner', type: 'planner_proposal_accepted', payload: { iteration: 1, proposal: { action: 'run_agent', agent_id: 'verification', args: {}, reasoning: '' } } },
      { ts: 5, run_id: RUN_ID, phase: 'verification', type: 'verification_verdict', payload: { candidate_id: 'candidate_a', verdict: 'rejected', concerns: ['too good to be true'] } },
      { ts: 6, run_id: RUN_ID, phase: 'planner', type: 'planner_proposal_accepted', payload: { iteration: 2, proposal: { action: 'run_agent', agent_id: 'modeling', args: {}, reasoning: '' } } },
      { ts: 7, run_id: RUN_ID, phase: 'modeling', type: 'candidate_scored', payload: { candidate_id: 'candidate_b', passed_gate: true, metrics: {} } },
    ]
    render(
      <MemoryRouter initialEntries={[`/runs/${RUN_ID}`]}>
        <Routes>
          <Route path="/runs/:runId" element={<RunDetailPage />} />
        </Routes>
      </MemoryRouter>,
    )
    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1))
    const source = FakeEventSource.latest()
    await act(async () => {
      for (const event of retryEvents) source.emit(event)
    })

    expect(await screen.findByTestId('node-step_modeling')).toHaveAttribute('data-state', 'passed')
    expect(screen.getByTestId('node-step_verification')).toHaveAttribute('data-state', 'vetoed')
    expect(screen.getByText('modeling (attempt 2)')).toBeInTheDocument()
    expect(screen.getByText('modeling (attempt 2)').closest('[data-testid^="node-"]')).toHaveAttribute('data-state', 'passed')
  })
})
