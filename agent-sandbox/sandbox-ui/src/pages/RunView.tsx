import { Link, useParams } from 'react-router-dom'
import { EventLogViewer } from '../components/EventLogViewer'
import { StatusBadge } from '../components/StatusBadge'
import { useRunStream } from '../hooks/useRunStream'
import type { RunStatus, SandboxEvent } from '../api/types'

/** Derives a RunStatus from the events seen so far on the live stream —
 * the same rule sandbox_server.run_manager.derive_status applies server-side,
 * kept in sync here so the badge updates the instant the terminal event
 * arrives rather than waiting on a separate poll. */
function statusFromEvents(events: SandboxEvent[]): RunStatus {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i]
    if (event.type === 'agent_result') return 'completed'
    if (event.type === 'error') {
      return event.context?.phase === 'max_turns' ? 'truncated' : 'errored'
    }
  }
  return 'running'
}

export function RunView() {
  const { runId } = useParams<{ runId: string }>()
  const { events, connectionState } = useRunStream(runId)
  const status = statusFromEvents(events)

  return (
    <div>
      <div className="row" style={{ justifyContent: 'space-between', marginBottom: 16 }}>
        <div>
          <h2 style={{ margin: 0 }}>
            Run <code>{runId}</code>
          </h2>
          {connectionState === 'error' && <span className="error-text">connection lost — retrying…</span>}
        </div>
        <div className="row">
          <StatusBadge status={status} />
          <Link to="/runs">
            <button className="secondary">all runs</button>
          </Link>
        </div>
      </div>

      <EventLogViewer events={events} autoScroll />
    </div>
  )
}
