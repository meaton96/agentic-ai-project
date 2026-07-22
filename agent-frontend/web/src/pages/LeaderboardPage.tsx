import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { getLeaderboard } from '../api/client'
import type { LeaderboardEntry } from '../api/types'
import { StatusBadge } from '../components/StatusBadge'

export function LeaderboardPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const runIdFilter = searchParams.get('run_id') ?? ''
  const [entries, setEntries] = useState<LeaderboardEntry[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setEntries(null)
    getLeaderboard(runIdFilter || undefined)
      .then(setEntries)
      .catch((e: Error) => setError(e.message))
  }, [runIdFilter])

  return (
    <div>
      <h2>Leaderboard</h2>
      <div className="field" style={{ maxWidth: 320 }}>
        <label>
          Filter by run id
          <input
            value={runIdFilter}
            onChange={(e) => setSearchParams(e.target.value ? { run_id: e.target.value } : {})}
            placeholder="run_xxxxxxxx"
          />
        </label>
      </div>

      {error && <div className="error-text">{error}</div>}
      {!entries && !error && <div className="muted">loading…</div>}
      {entries && entries.length === 0 && <div className="card muted">No leaderboard entries.</div>}
      {entries && entries.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Candidate</th>
              <th>Template</th>
              <th>Split</th>
              <th>ROC AUC</th>
              <th>Verdict</th>
              <th>Logged</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((entry, i) => (
              <tr key={i}>
                <td>{entry.run_id}</td>
                <td>{entry.candidate}</td>
                <td>{entry.template_id}</td>
                <td>{entry.split}</td>
                <td>{entry.metrics.roc_auc?.value.toFixed(3) ?? '—'}</td>
                <td>
                  <StatusBadge status={entry.verification_verdict} />
                </td>
                <td>{entry.logged_at_utc}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
