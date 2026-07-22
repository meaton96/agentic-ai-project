import { useEffect, useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import { getRun, listTranscripts } from '../api/client'
import type { RunSummary } from '../api/types'
import { useRunEvents } from '../hooks/useRunEvents'
import { buildStaticGraph } from '../domain/staticGraph'
import { buildDynamicGraph } from '../domain/dynamicGraph'
import { PipelineTimeline } from '../components/timeline/PipelineTimeline'
import { StatusBadge } from '../components/StatusBadge'
import { Legend } from '../components/Legend'

export function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>()
  const [summary, setSummary] = useState<RunSummary | null>(null)
  const [summaryError, setSummaryError] = useState<string | null>(null)
  const [transcriptNames, setTranscriptNames] = useState<string[]>([])
  const { events, connectionState } = useRunEvents(runId)

  useEffect(() => {
    if (!runId) return
    getRun(runId).then(setSummary).catch((e: Error) => setSummaryError(e.message))
  }, [runId])

  // Re-poll the summary (status/orchestrator/report) as events arrive — the
  // event stream alone doesn't tell us the orchestrator type up front, and
  // this also keeps status/error/report current as the run finishes.
  useEffect(() => {
    if (!runId) return
    getRun(runId).then(setSummary).catch(() => undefined)
  }, [runId, events.length])

  useEffect(() => {
    if (!runId) return
    listTranscripts(runId).then(setTranscriptNames).catch(() => undefined)
  }, [runId, events.length])

  const orchestrator = summary?.orchestrator ?? (events.some((e) => e.phase === 'planner') ? 'dynamic' : null)

  const graph = useMemo(() => {
    if (orchestrator === 'dynamic') return buildDynamicGraph(events, transcriptNames)
    if (orchestrator === 'static') return buildStaticGraph(events, transcriptNames)
    return null
  }, [orchestrator, events, transcriptNames])

  if (!runId) return <div>no run id</div>
  if (summaryError) return <div className="error-text">Failed to load run: {summaryError}</div>

  return (
    <div>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 16 }}>
        <h2 style={{ margin: 0 }}>{runId}</h2>
        {summary && <StatusBadge status={summary.status} />}
        <span className="muted">{orchestrator ?? '…'} orchestrator</span>
        <span className="muted">· {connectionState}</span>
        {summary?.error && <span className="error-text">{summary.error}</span>}
      </div>

      {graph === null && <div className="card muted">Waiting for the first event to determine orchestrator type…</div>}

      {graph !== null && (
        <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start' }}>
          <div style={{ flex: 1, minWidth: 0 }}>
            <PipelineTimeline runId={runId} graph={graph} />
          </div>
          <Legend />
        </div>
      )}
    </div>
  )
}
