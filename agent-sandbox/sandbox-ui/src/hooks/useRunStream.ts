import { useEffect, useState } from 'react'
import { runStreamUrl } from '../api/client'
import type { SandboxEvent } from '../api/types'

export type ConnectionState = 'connecting' | 'open' | 'closed' | 'error'

export interface UseRunStreamResult {
  events: SandboxEvent[]
  connectionState: ConnectionState
}

const TERMINAL_EVENT_TYPES = new Set(['agent_result', 'error'])
// GET /runs/{id}/stream replays-then-closes for a run the server isn't
// actively tracking (see sandbox_server/routes/stream.py) — if we never get
// a single message before the connection drops, EventSource's default
// auto-retry would otherwise hammer a dead/finished endpoint forever. Give
// up after a few attempts with zero messages ever received.
const MAX_RETRIES_BEFORE_FIRST_MESSAGE = 4

/**
 * Connects to sandbox-server's replay-then-tail SSE stream for one run and
 * accumulates every event it emits, in order. Works identically for a live
 * run (events keep arriving until agent_result/error) and a finished one
 * (replay arrives, stream closes, nothing further happens) — same code path
 * powers RunView (live) and RunHistory (static, via GET /runs/{id} instead
 * of this hook).
 *
 * Deliberately closes the connection once a terminal event is seen, rather
 * than relying on the server ending the HTTP response: EventSource treats
 * any connection close (including a clean one) as a drop and retries by
 * default, which would otherwise reconnect forever after a run finishes.
 * A genuine mid-run drop (network hiccup) is left to EventSource's own
 * automatic reconnect, which gives "reconnect-on-drop" for free.
 */
export function useRunStream(runId: string | undefined): UseRunStreamResult {
  const [events, setEvents] = useState<SandboxEvent[]>([])
  const [connectionState, setConnectionState] = useState<ConnectionState>('connecting')

  useEffect(() => {
    setEvents([])
    if (!runId) {
      setConnectionState('closed')
      return
    }

    setConnectionState('connecting')
    const source = new EventSource(runStreamUrl(runId))
    let receivedAnyMessage = false
    let retriesBeforeFirstMessage = 0
    let deliberatelyClosed = false

    source.onopen = () => {
      setConnectionState('open')
    }

    source.onmessage = (message) => {
      receivedAnyMessage = true
      const event = JSON.parse(message.data) as SandboxEvent
      setEvents((prev) => [...prev, event])
      if (TERMINAL_EVENT_TYPES.has(event.type)) {
        deliberatelyClosed = true
        source.close()
        setConnectionState('closed')
      }
    }

    source.onerror = () => {
      if (deliberatelyClosed) return
      if (!receivedAnyMessage) {
        retriesBeforeFirstMessage += 1
        if (retriesBeforeFirstMessage >= MAX_RETRIES_BEFORE_FIRST_MESSAGE) {
          source.close()
          setConnectionState('error')
          return
        }
      }
      setConnectionState('error')
    }

    return () => {
      deliberatelyClosed = true
      source.close()
    }
  }, [runId])

  return { events, connectionState }
}
