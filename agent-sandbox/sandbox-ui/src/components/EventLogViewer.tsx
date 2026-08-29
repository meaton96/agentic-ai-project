import { useEffect, useRef, useState, type CSSProperties } from 'react'
import type { SandboxEvent } from '../api/types'

export interface EventLogViewerProps {
  events: SandboxEvent[]
  /** RunView (live) auto-scrolls to the newest event; RunHistory (static
   * browsing) should not — pass explicitly, no implicit default either way
   * matters here since silently guessing wrong is annoying either
   * direction. */
  autoScroll: boolean
}

const INDENT_PX = 20

interface EventMeta {
  label: string
  marker: string
  colorVar: string
  bgVar: string
}

// One family per row of the pair (out/back), so up-arrow/down-arrow reads
// as "sent"/"received" within a family, plus agent lifecycle (+/✓) and a
// deliberately different glyph+treatment for error (see .event-row-error in
// index.css — errors also get a full red-tinted row background, not just
// this marker, so they're unmistakable even skimming a long log).
const EVENT_META: Record<SandboxEvent['type'], EventMeta> = {
  llm_request: { label: 'llm request', marker: '↑', colorVar: '--color-llm', bgVar: '--color-llm-bg' },
  llm_response: { label: 'llm response', marker: '↓', colorVar: '--color-llm', bgVar: '--color-llm-bg' },
  tool_call: { label: 'tool call', marker: '→', colorVar: '--color-tool', bgVar: '--color-tool-bg' },
  tool_result: { label: 'tool result', marker: '←', colorVar: '--color-tool', bgVar: '--color-tool-bg' },
  agent_spawn: { label: 'agent spawn', marker: '+', colorVar: '--color-agent', bgVar: '--color-agent-bg' },
  agent_result: { label: 'agent result', marker: '✓', colorVar: '--color-agent', bgVar: '--color-agent-bg' },
  error: { label: 'error', marker: '!', colorVar: '--color-error', bgVar: '--color-error-bg' },
}

function truncate(text: string, max = 100): string {
  return text.length > max ? `${text.slice(0, max)}…` : text
}

function summarizeEvent(event: SandboxEvent): string {
  switch (event.type) {
    case 'llm_request':
      return `→ ${event.model} (${event.messages.length} message${event.messages.length === 1 ? '' : 's'})`
    case 'llm_response':
      if (event.tool_calls?.length) {
        const names = event.tool_calls
          .map((tc) => (tc as { function?: { name?: string } }).function?.name ?? '?')
          .join(', ')
        return `requested tool call(s): ${names}`
      }
      return truncate(event.content ?? '(empty response)')
    case 'tool_call':
      return `${event.tool} → ${event.server}(${truncate(JSON.stringify(event.args), 80)})`
    case 'tool_result':
      if (event.error) return `error: ${truncate(event.error)}`
      return truncate(typeof event.result === 'string' ? event.result : JSON.stringify(event.result))
    case 'agent_spawn':
      return `spawned ${event.child_agent_id} via ${event.spawned_via_tool}`
    case 'agent_result':
      return `${truncate(event.final_output)} (${event.turns_used} turn${event.turns_used === 1 ? '' : 's'})`
    case 'error':
      return event.message
  }
}

function formatTs(ts: string): string {
  const date = new Date(ts)
  if (Number.isNaN(date.getTime())) return ts
  return `${date.toLocaleTimeString(undefined, { hour12: false })}.${String(date.getMilliseconds()).padStart(3, '0')}`
}

/** Depth is resolved purely from `parent_call_id` chasing a preceding
 * event's own `call_id` (currently only tool_call/tool_result carry one) —
 * inert today since M3 never sets parent_call_id (no sub-agent delegation
 * yet), but correct now so nested sub-agent runs nest visually the moment
 * that lands, with no changes needed here. */
function computeDepths(events: SandboxEvent[]): number[] {
  const callIdToDepth = new Map<string, number>()
  const depths: number[] = []
  for (const event of events) {
    const parentId = event.parent_call_id
    const depth = parentId && callIdToDepth.has(parentId) ? callIdToDepth.get(parentId)! + 1 : 0
    depths.push(depth)
    if ('call_id' in event) callIdToDepth.set(event.call_id, depth)
  }
  return depths
}

export function EventLogViewer({ events, autoScroll }: EventLogViewerProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (autoScroll) bottomRef.current?.scrollIntoView({ block: 'nearest' })
  }, [events.length, autoScroll])

  if (events.length === 0) {
    return <div className="event-log-empty">no events yet</div>
  }

  const depths = computeDepths(events)

  return (
    <div>
      <div className="event-log-toolbar">{events.length} event{events.length === 1 ? '' : 's'}</div>
      <div className="event-log" data-testid="event-log">
        {events.map((event, i) => (
          <EventRow key={`${event.run_id}-${event.seq}-${i}`} event={event} depth={depths[i]} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

function EventRow({ event, depth }: { event: SandboxEvent; depth: number }) {
  const [open, setOpen] = useState(false)
  const meta = EVENT_META[event.type]
  const isError = event.type === 'error'

  const style: CSSProperties = {
    marginLeft: depth * INDENT_PX,
    '--event-color': `var(${meta.colorVar})`,
    '--event-bg': `var(${meta.bgVar})`,
  } as CSSProperties

  return (
    <div
      className={`event-row${isError ? ' event-row-error' : ''}`}
      style={style}
      data-testid="event-row"
      data-event-type={event.type}
      data-depth={depth}
    >
      <button className="event-row-header" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <span className="event-row-caret" aria-hidden="true">
          {open ? '▾' : '▸'}
        </span>
        <span className="event-marker" aria-hidden="true">
          {meta.marker}
        </span>
        <span className="event-type">{meta.label}</span>
        <span className="event-agent" title={event.agent_id}>
          {event.agent_id}
        </span>
        <span className="event-summary" title={summarizeEvent(event)}>
          {summarizeEvent(event)}
        </span>
        <span className="event-ts">{formatTs(event.ts)}</span>
      </button>
      {open && (
        <div className="event-row-body">
          <pre>{JSON.stringify(event, null, 2)}</pre>
        </div>
      )}
    </div>
  )
}
