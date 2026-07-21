import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { RunDetailPage } from './RunDetailPage'
import * as client from '../api/client'
import type { RunEvent, RunSummary } from '../api/types'
import { FakeEventSource } from '../test-utils/FakeEventSource'

vi.mock('../api/client')

const RUN_ID = 'run_static1'

function summaryFor(status: RunSummary['status']): RunSummary {
  return {
    run_id: RUN_ID, orchestrator: 'static', status, started_at: null, finished_at: null,
    error: null, n_events: 0, first_event: null, last_event: null, report: null, leaderboard_entries: [],
  }
}

// Mirrors the real event shapes confirmed by hitting a live server (see M2
// summary) for a --target-given, --skip-feature-engineering, one-candidate
// static-orchestrator run.
const SCRIPTED_EVENTS: RunEvent[] = [
  { ts: 1, run_id: RUN_ID, phase: 'run', type: 'run_started', payload: { data: '/x/churn.csv', goal: '', target: 'churned' } },
  { ts: 2, run_id: RUN_ID, phase: 'profiler', type: 'agent_started', payload: { target_column: 'churned' } },
  { ts: 3, run_id: RUN_ID, phase: 'profiler', type: 'profiler_report', payload: { ok: true, recommended_split_strategy: 'stratified', is_imbalanced: false, leakage_risk_flags: [] } },
  { ts: 4, run_id: RUN_ID, phase: 'split_and_check_leakage', type: 'leakage_gate_result', payload: { check: 'duplicate_rows_across_splits', passed: true, detail: 'ok' } },
  { ts: 5, run_id: RUN_ID, phase: 'split_and_check_leakage', type: 'split_completed', payload: { strategy_used: 'stratified', ok: true, n_train: 142, n_val: 29, n_test: 29 } },
  { ts: 6, run_id: RUN_ID, phase: 'modeling', type: 'agent_started', payload: { already_tried_template_ids: [] } },
  { ts: 7, run_id: RUN_ID, phase: 'modeling', type: 'leakage_gate_result', payload: { candidate_id: 'candidate_a', check: 'label_permutation_test', passed: true, detail: 'ok' } },
  { ts: 8, run_id: RUN_ID, phase: 'modeling', type: 'candidate_scored', payload: { candidate_id: 'candidate_a', template_id: 'sklearn_mixed_pipeline', metrics: {}, passed_gate: true } },
  { ts: 9, run_id: RUN_ID, phase: 'verification', type: 'agent_started', payload: { candidate_id: 'candidate_a' } },
  { ts: 10, run_id: RUN_ID, phase: 'verification', type: 'verification_verdict', payload: { candidate_id: 'candidate_a', verdict: 'approved', concerns: [], reasoning: 'fine', unparseable: false } },
  { ts: 11, run_id: RUN_ID, phase: 'finalize', type: 'final_test_metrics', payload: { test_metrics: {} } },
  { ts: 12, run_id: RUN_ID, phase: 'run', type: 'run_completed', payload: { status: 'success' } },
]

function renderDetail() {
  return render(
    <MemoryRouter initialEntries={[`/runs/${RUN_ID}`]}>
      <Routes>
        <Route path="/runs/:runId" element={<RunDetailPage />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('RunDetailPage — static orchestrator graph', () => {
  beforeEach(() => {
    FakeEventSource.reset()
    vi.stubGlobal('EventSource', FakeEventSource)
    vi.mocked(client.getRun).mockResolvedValue(summaryFor('running'))
    vi.mocked(client.listTranscripts).mockResolvedValue([])
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('reaches the correct node states for a full, successful run', async () => {
    renderDetail()

    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1))
    const source = FakeEventSource.latest()

    await act(async () => {
      for (const event of SCRIPTED_EVENTS) source.emit(event)
    })

    // skipped (no --target-less run here, no feature-engineering flag) —
    // both stay pending since no events ever arrive for those phases.
    await waitFor(() => {
      expect(screen.getByTestId('node-intake')).toHaveAttribute('data-state', 'pending')
      expect(screen.getByTestId('node-feature_engineering')).toHaveAttribute('data-state', 'pending')
    })

    expect(screen.getByTestId('node-profiler')).toHaveAttribute('data-state', 'passed')
    expect(screen.getByTestId('node-split_harness')).toHaveAttribute('data-state', 'passed')
    expect(screen.getByTestId('node-modeling_candidate_a')).toHaveAttribute('data-state', 'passed')
    expect(screen.getByTestId('node-modeling_harness_candidate_a')).toHaveAttribute('data-state', 'passed')
    expect(screen.getByTestId('node-verification_candidate_a')).toHaveAttribute('data-state', 'passed')
    expect(screen.getByTestId('node-finalize_harness')).toHaveAttribute('data-state', 'passed')
    expect(screen.getByTestId('node-summary')).toHaveAttribute('data-state', 'passed')
  })

  it('marks the verification node vetoed on a rejected verdict, not merely failed', async () => {
    renderDetail()
    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1))
    const source = FakeEventSource.latest()

    const rejectedEvents = SCRIPTED_EVENTS.map((e) =>
      e.type === 'verification_verdict' ? { ...e, payload: { ...e.payload, verdict: 'rejected' } } : e,
    ).filter((e) => e.type !== 'run_completed' && e.type !== 'final_test_metrics')

    await act(async () => {
      for (const event of rejectedEvents) source.emit(event)
    })

    await waitFor(() => {
      expect(screen.getByTestId('node-verification_candidate_a')).toHaveAttribute('data-state', 'vetoed')
    })
    // nothing to refit — finalize never ran for a wholly-rejected candidate pool
    expect(screen.getByTestId('node-finalize_harness')).toHaveAttribute('data-state', 'pending')
  })

  it('shows an inline summary while collapsed and fetches the transcript on expand', async () => {
    vi.mocked(client.listTranscripts).mockResolvedValue(['profiler_01.json'])
    vi.mocked(client.getTranscript).mockResolvedValue([{ role: 'system', content: 'you are the profiler' }])
    const user = userEvent.setup()
    renderDetail()

    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1))
    const source = FakeEventSource.latest()
    await act(async () => {
      for (const event of SCRIPTED_EVENTS) source.emit(event)
    })

    // inline summary visible without expanding
    await waitFor(() => {
      expect(screen.getByTestId('node-split_harness')).toHaveTextContent('train=142 val=29 test=29')
    })
    expect(client.getTranscript).not.toHaveBeenCalled()

    // expanding the profiler card fetches and renders its transcript
    await user.click(screen.getByText('Profiler Agent'))
    expect(await screen.findByText('you are the profiler')).toBeInTheDocument()
    expect(client.getTranscript).toHaveBeenCalledWith(RUN_ID, 'profiler_01.json')
  })
})
