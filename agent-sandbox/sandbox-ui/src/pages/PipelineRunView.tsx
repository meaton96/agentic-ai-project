import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { getPipelineRun, listPipelineRuns } from '../api/client'
import type { PipelineRunRecord, PipelineRunSummary } from '../api/types'
import { StatusBadge } from '../components/StatusBadge'

const POLL_INTERVAL_MS = 1000

function PipelineRunList() {
  const [runs, setRuns] = useState<PipelineRunSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listPipelineRuns()
      .then((list) => setRuns([...list].sort((a, b) => b.created_at.localeCompare(a.created_at))))
      .catch((e: Error) => setError(e.message))
  }, [])

  return (
    <div>
      <h2>Pipeline runs</h2>
      {error && <div className="error-text">{error}</div>}
      {!runs && !error && <div className="muted">loading…</div>}
      {runs && runs.length === 0 && <div className="muted">no pipeline runs yet</div>}
      {runs && runs.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Pipeline</th>
              <th>Created</th>
              <th>Steps</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.pipeline_run_id}>
                <td>
                  <Link to={`/pipeline-runs/${encodeURIComponent(run.pipeline_run_id)}`}>
                    <code>{run.pipeline_run_id}</code>
                  </Link>
                </td>
                <td>{run.pipeline_id}</td>
                <td className="muted">{new Date(run.created_at).toLocaleString()}</td>
                <td className="muted">{run.step_count}</td>
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

function PipelineRunDetail({ pipelineRunId }: { pipelineRunId: string }) {
  const [detail, setDetail] = useState<PipelineRunRecord | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined

    async function poll() {
      try {
        const record = await getPipelineRun(pipelineRunId)
        if (cancelled) return
        setDetail(record)
        if (record.status === 'running') {
          timer = setTimeout(poll, POLL_INTERVAL_MS)
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'failed to load pipeline run')
      }
    }

    poll()
    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
    }
  }, [pipelineRunId])

  return (
    <div>
      <div className="row" style={{ justifyContent: 'space-between', marginBottom: 16 }}>
        <div>
          <h2 style={{ margin: 0 }}>
            Pipeline run <code>{pipelineRunId}</code>
          </h2>
          {detail?.pipeline_id && <span className="muted">pipeline: {detail.pipeline_id}</span>}
        </div>
        <div className="row">
          {detail && <StatusBadge status={detail.status} />}
          <Link to="/pipeline-runs">
            <button className="secondary">all pipeline runs</button>
          </Link>
        </div>
      </div>

      {error && <div className="error-text">{error}</div>}
      {!detail && !error && <div className="muted">loading…</div>}

      {detail && (
        <>
          {detail.error && <div className="error-text" style={{ marginBottom: 12 }}>{detail.error}</div>}
          <table>
            <thead>
              <tr>
                <th>Step</th>
                <th>Agent</th>
                <th>Status</th>
                <th>Run</th>
              </tr>
            </thead>
            <tbody>
              {detail.steps.map((step) =>
                step.kind === 'gate' ? (
                  <tr key={step.step_id}>
                    <td>{step.step_id}</td>
                    <td className="muted">gate</td>
                    <td className="muted">decision: {step.decision}</td>
                    <td className="muted">
                      → routed to {step.routed_to ?? 'end'}
                      {step.output && (
                        <>
                          <br />
                          output: <code>{step.output}</code>
                        </>
                      )}
                    </td>
                  </tr>
                ) : (
                  <tr key={step.step_id}>
                    <td>{step.step_id}</td>
                    <td className="muted">{step.agent_id}</td>
                    <td>
                      <StatusBadge status={step.status} />
                    </td>
                    <td>
                      <Link to={`/runs/${encodeURIComponent(step.run_id)}/live`}>
                        <code>{step.run_id}</code>
                      </Link>
                    </td>
                  </tr>
                ),
              )}
              {detail.status === 'running' && (
                <tr>
                  <td colSpan={4} className="muted">
                    running next step…
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </>
      )}
    </div>
  )
}

export function PipelineRunView() {
  const { pipelineRunId } = useParams<{ pipelineRunId: string }>()
  return pipelineRunId ? <PipelineRunDetail pipelineRunId={pipelineRunId} /> : <PipelineRunList />
}
