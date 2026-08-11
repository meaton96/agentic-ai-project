import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { EventLogViewer } from './EventLogViewer'
import type { SandboxEvent } from '../api/types'

const BASE = { run_id: 'run-1', agent_id: 'agent-1', ts: '2026-01-01T12:00:00.500000+00:00', parent_call_id: null }

const FIXTURE_EVENTS: SandboxEvent[] = [
  { ...BASE, type: 'llm_request', seq: 1, messages: [{ role: 'user', content: 'hi' }], model: 'test-model' },
  { ...BASE, type: 'llm_response', seq: 2, content: null, tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'write_file', arguments: '{}' } }], token_usage: null, duration_ms: 12 },
  { ...BASE, type: 'tool_call', seq: 3, call_id: 'call-1', server: 'fs', tool: 'write_file', args: { path: 'out.txt' } },
  { ...BASE, type: 'tool_result', seq: 4, call_id: 'call-1', result: 'wrote out.txt', error: null, duration_ms: 5 },
  { ...BASE, type: 'agent_spawn', seq: 5, child_agent_id: 'sub-agent', spawned_via_tool: 'delegate', parent_call_id: 'call-1' },
  { ...BASE, type: 'agent_result', seq: 6, final_output: 'all done', turns_used: 3 },
  { ...BASE, type: 'error', seq: 7, message: 'model request failed', context: { phase: 'llm_request' } },
]

describe('EventLogViewer', () => {
  it('shows an empty state with no events', () => {
    render(<EventLogViewer events={[]} autoScroll={false} />)
    expect(screen.getByText(/no events yet/i)).toBeInTheDocument()
  })

  it('renders one collapsed row per event, with type/agent/summary visible but JSON hidden', () => {
    render(<EventLogViewer events={FIXTURE_EVENTS} autoScroll={false} />)
    const rows = screen.getAllByTestId('event-row')
    expect(rows).toHaveLength(FIXTURE_EVENTS.length)

    expect(screen.getByText('llm request')).toBeInTheDocument()
    expect(screen.getByText('tool call')).toBeInTheDocument()
    expect(screen.getByText(/write_file → fs/)).toBeInTheDocument()

    // collapsed: no raw JSON blob rendered anywhere yet
    expect(screen.queryByText(/"final_output"/)).not.toBeInTheDocument()
  })

  it('expands a row to show the full event JSON on click, and collapses again', async () => {
    const user = userEvent.setup()
    render(<EventLogViewer events={FIXTURE_EVENTS} autoScroll={false} />)

    const resultRow = screen.getAllByTestId('event-row')[5] // agent_result
    const header = within(resultRow).getByRole('button')
    expect(header).toHaveAttribute('aria-expanded', 'false')

    await user.click(header)
    expect(header).toHaveAttribute('aria-expanded', 'true')
    expect(within(resultRow).getByText(/"final_output": "all done"/)).toBeInTheDocument()

    await user.click(header)
    expect(header).toHaveAttribute('aria-expanded', 'false')
    expect(within(resultRow).queryByText(/"final_output"/)).not.toBeInTheDocument()
  })

  it('marks the error row distinctly so it is unmistakable', () => {
    render(<EventLogViewer events={FIXTURE_EVENTS} autoScroll={false} />)
    const errorRow = screen.getAllByTestId('event-row')[6]
    expect(errorRow.className).toContain('event-row-error')
    expect(within(errorRow).getByText('model request failed')).toBeInTheDocument()
  })

  it('nests an event under the event whose call_id matches its parent_call_id', () => {
    render(<EventLogViewer events={FIXTURE_EVENTS} autoScroll={false} />)
    const rows = screen.getAllByTestId('event-row')

    // tool_call (seq 3) established call_id "call-1" at depth 0
    expect(rows[2]).toHaveAttribute('data-depth', '0')
    // agent_spawn (seq 5) has parent_call_id "call-1" -> nested one level deeper
    expect(rows[4]).toHaveAttribute('data-depth', '1')
    // everything else has no parent_call_id -> depth 0
    expect(rows[0]).toHaveAttribute('data-depth', '0')
    expect(rows[5]).toHaveAttribute('data-depth', '0')
  })
})
