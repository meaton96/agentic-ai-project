import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { getRun, listRuns } from '../api/client'
import type { RunDetail, RunSummary } from '../api/types'
import { EventLogViewer } from '../components/EventLogViewer'
import { StatusBadge } from '../components/StatusBadge'

function RunHistoryList() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listRuns()
      .then((list) => setRuns([...list].sort((a, b) => b.created_at.localeCompare(a.created_at))))
      .catch((e: Error) => setError(e.message))
  }, [])

  return (
    <div>
      <h2>Run history</h2>
      {error && <div className="error-text">{error}</div>}
      {!runs && !error && <div className="muted">loading…</div>}
      {runs && runs.length === 0 && <div className="muted">no runs yet</div>}
      {runs && runs.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Agent</th>
              <th>Created</th>
              <th>Turns</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.run_id}>
                <td>
                  <Link to={`/runs/${encodeURIComponent(run.run_id)}`}>
                    <code>{run.run_id}</code>
                  </Link>
                </td>
                <td>{run.agent_id ?? <span className="muted">unknown</span>}</td>
                <td className="muted">{new Date(run.created_at).toLocaleString()}</td>
                <td className="muted">{run.turn_count}</td>
                <td>
                  <StatusBadge status={run.status} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

function RunHistoryDetail({ runId }: { runId: string }) {
  const [detail, setDetail] = useState<RunDetail | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setDetail(null)
    getRun(runId)
      .then(setDetail)
      .catch((e: Error) => setError(e.message))
  }, [runId])

  return (
    <div>
      <div className="row" style={{ justifyContent: 'space-between', marginBottom: 16 }}>
        <div>
          <h2 style={{ margin: 0 }}>
            Run <code>{runId}</code>
          </h2>
          {detail?.agent_id && <span className="muted">agent: {detail.agent_id}</span>}
        </div>
        <div className="row">
          {detail && <StatusBadge status={detail.status} />}
          <Link to="/runs">
            <button className="secondary">all runs</button>
          </Link>
        </div>
      </div>

      {error && <div className="error-text">{error}</div>}
      {!detail && !error && <div className="muted">loading…</div>}
      {detail && <EventLogViewer events={detail.events} autoScroll={false} />}
    </div>
  )
}

export function RunHistory() {
  const { runId } = useParams<{ runId: string }>()
  return runId ? <RunHistoryDetail runId={runId} /> : <RunHistoryList />
}
