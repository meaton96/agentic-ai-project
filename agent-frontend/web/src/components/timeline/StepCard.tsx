import { useEffect, useState } from 'react'
import { getTranscript } from '../../api/client'
import type { TranscriptMessage } from '../../api/types'
import type { GraphNode } from '../../domain/graphTypes'
import { promptSourceForEvents, summarizeNodeEvents } from '../../domain/eventClassification'
import { STATE_COLOR } from '../../domain/stateColors'
import { StatusBadge } from '../StatusBadge'

function MessageView({ message }: { message: TranscriptMessage }) {
  return (
    <div style={{ marginBottom: 10, padding: 10, background: 'var(--color-surface-sunken)', borderRadius: 'var(--radius-sm)' }}>
      <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', color: 'var(--color-text-muted)' }}>
        {message.role}
      </div>
      {message.content != null && (
        <pre style={{ whiteSpace: 'pre-wrap', fontSize: 12, margin: '4px 0' }}>
          {typeof message.content === 'string' ? message.content : JSON.stringify(message.content, null, 2)}
        </pre>
      )}
      {message.tool_calls?.map((tc) => (
        <div key={tc.id} style={{ fontSize: 12, color: 'var(--color-text-muted)' }}>
          <strong>tool call:</strong> {tc.function.name}({JSON.stringify(tc.function.arguments)})
        </div>
      ))}
    </div>
  )
}

export interface StepCardProps {
  runId: string
  node: GraphNode
  indent?: boolean
}

/** One pipeline step, rendered full-width in the vertical timeline.
 * Collapsed by default: label, status, and a one-line summary of what
 * happened (see summarizeNodeEvents). Expanding fetches and shows the
 * full transcript (system prompt / tool calls+args / tool results / final
 * response) for agent steps, plus every raw event payload — leakage check
 * detail, metrics, verdicts — for harness steps and anything without a
 * transcript (e.g. a rejected planner proposal). */
export function StepCard({ runId, node, indent }: StepCardProps) {
  const [expanded, setExpanded] = useState(false)
  const [transcript, setTranscript] = useState<TranscriptMessage[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  const dark = node.data.kind === 'harness'
  const summary = summarizeNodeEvents(node.data.events)
  const promptSource = promptSourceForEvents(node.data.events)

  useEffect(() => {
    if (!expanded || !node.data.transcriptName || transcript) return
    getTranscript(runId, node.data.transcriptName)
      .then(setTranscript)
      .catch((e: Error) => setError(e.message))
  }, [expanded, runId, node.data.transcriptName, transcript])

  return (
    <div
      data-testid={`node-${node.id}`}
      data-state={node.data.state}
      style={{
        marginLeft: indent ? 32 : 0,
        marginBottom: 10,
        borderRadius: 'var(--radius)',
        background: dark ? '#1e293b' : 'var(--color-surface)',
        color: dark ? '#ffffff' : 'var(--color-text)',
        border: '1px solid var(--color-border)',
        borderLeft: `5px solid ${STATE_COLOR[node.data.state]}`,
        opacity: node.data.rejected ? 0.6 : 1,
        boxShadow: 'var(--shadow-sm)',
        overflow: 'hidden',
      }}
    >
      <button
        onClick={() => setExpanded((v) => !v)}
        style={{
          all: 'unset', display: 'flex', alignItems: 'center', gap: 12, width: '100%', boxSizing: 'border-box',
          padding: '14px 18px', cursor: 'pointer',
        }}
      >
        <span style={{ fontSize: 14, opacity: 0.6, width: 12 }}>{expanded ? '▾' : '▸'}</span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            style={{
              fontSize: 15, fontWeight: 600, textDecoration: node.data.rejected ? 'line-through' : 'none',
            }}
          >
            {node.data.label}
          </div>
          {summary && (
            <div style={{ fontSize: 12, opacity: 0.75, marginTop: 3, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {summary}
            </div>
          )}
        </div>
        {promptSource && (
          <span
            className={`badge ${promptSource.source === 'override' ? 'badge-override' : 'badge-pending'}`}
            title={promptSource.path}
          >
            prompt: {promptSource.source}
          </span>
        )}
        <StatusBadge status={node.data.state} />
      </button>

      {expanded && (
        <div style={{ padding: '0 18px 18px 18px', background: dark ? 'rgba(255,255,255,0.04)' : 'var(--color-surface)' }}>
          {node.data.transcriptName && (
            <>
              <h4 style={{ color: dark ? '#cbd5e1' : undefined }}>Transcript ({node.data.transcriptName})</h4>
              {error && <div className="error-text">{error}</div>}
              {!transcript && !error && <div className="muted">loading…</div>}
              {transcript?.map((m, i) => <MessageView key={i} message={m} />)}
            </>
          )}

          <h4 style={{ color: dark ? '#cbd5e1' : undefined }}>Events</h4>
          {node.data.events.length === 0 && <div className="muted">no events recorded for this step</div>}
          {node.data.events.map((e, i) => (
            <div key={i} style={{ marginBottom: 8 }}>
              <div style={{ fontSize: 11, color: dark ? '#94a3b8' : 'var(--color-text-muted)' }}>{e.type}</div>
              <pre
                style={{
                  whiteSpace: 'pre-wrap', fontSize: 11, background: dark ? 'rgba(255,255,255,0.06)' : 'var(--color-surface-sunken)',
                  color: dark ? '#e2e8f0' : undefined, padding: 8, borderRadius: 'var(--radius-sm)',
                }}
              >
                {JSON.stringify(e.payload, null, 2)}
              </pre>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
