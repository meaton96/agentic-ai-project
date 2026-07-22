import { useEffect, useState } from 'react'
import { runEventsUrl } from '../api/client'
import type { RunEvent } from '../api/types'

export type ConnectionState = 'connecting' | 'open' | 'closed' | 'error'

export interface UseRunEventsResult {
  events: RunEvent[]
  connectionState: ConnectionState
}

const TERMINAL_EVENT_TYPES = new Set(['run_completed', 'run_failed'])
// GET /api/runs/{id}/events replays then closes for a run this server never
// tracked as live (see server/events_io.py) — if we never receive a single
// message before the connection drops, EventSource's default auto-retry
// would otherwise hammer a dead/finished endpoint forever. Give up after a
// few attempts with zero messages ever received.
const MAX_RETRIES_BEFORE_FIRST_MESSAGE = 4

/**
 * Connects to the server's replay-then-tail SSE stream for one run and
 * accumulates every event it emits, in order. Works identically for a live
 * run (events keep arriving until run_completed/run_failed) and a finished
 * one (replay arrives, stream closes, nothing further happens) — that's the
 * server's own unification (see server/events_io.py), this hook just
 * consumes it rather than re-deriving "is this run still going" itself.
 *
 * Deliberately closes the connection once a terminal event is seen, rather
 * than relying on the server ending the HTTP response: EventSource treats
 * any connection close (including a clean one) as a drop and retries by
 * default, which would otherwise reconnect forever after a run finishes.
 * A genuine mid-run drop (network hiccup) is left to EventSource's own
 * automatic reconnect, which is what gives us "reconnect-on-drop" for free.
 */
export function useRunEvents(runId: string | undefined): UseRunEventsResult {
  const [events, setEvents] = useState<RunEvent[]>([])
  const [connectionState, setConnectionState] = useState<ConnectionState>('connecting')

  useEffect(() => {
    setEvents([])
    if (!runId) {
      setConnectionState('closed')
      return
    }

    setConnectionState('connecting')
    const source = new EventSource(runEventsUrl(runId))
    let receivedAnyMessage = false
    let retriesBeforeFirstMessage = 0
    let deliberatelyClosed = false

    source.onopen = () => {
      setConnectionState('open')
    }

    source.onmessage = (message) => {
      receivedAnyMessage = true
      const event = JSON.parse(message.data) as RunEvent
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
      // otherwise let EventSource's built-in auto-reconnect keep trying
      setConnectionState('error')
    }

    return () => {
      deliberatelyClosed = true
      source.close()
    }
  }, [runId])

  return { events, connectionState }
}
