import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { listRuns } from '../api/client'
import type { RunSummary } from '../api/types'
import { StatusBadge } from '../components/StatusBadge'

function fmtTime(ts: number | null): string {
  if (ts == null) return '—'
  return new Date(ts * 1000).toLocaleString()
}

function datasetOf(run: RunSummary): string {
  const data = run.first_event?.payload.data as string | undefined
  if (data) return data.split('/').pop() ?? data
  return '—'
}

function startedAtOf(run: RunSummary): number | null {
  return run.started_at ?? run.first_event?.ts ?? null
}

export function RunsListPage() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listRuns().then(setRuns).catch((e: Error) => setError(e.message))
  }, [])

  if (error) return <div className="error-text">Failed to load runs: {error}</div>
  if (!runs) return <div className="muted">loading…</div>

  return (
    <div>
      <h2>Runs</h2>
      {runs.length === 0 && <div className="card muted">No runs yet. Launch one from the Launch page.</div>}
      {runs.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Run ID</th>
              <th>Orchestrator</th>
              <th>Dataset</th>
              <th>Status</th>
              <th>Started</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.run_id}>
                <td>
                  <Link to={`/runs/${run.run_id}`}>{run.run_id}</Link>
                </td>
                <td>{run.orchestrator ?? '—'}</td>
                <td>{datasetOf(run)}</td>
                <td>
                  <StatusBadge status={run.status} />
                </td>
                <td>{fmtTime(startedAtOf(run))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
