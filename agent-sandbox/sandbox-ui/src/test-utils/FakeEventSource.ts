// Minimal, scriptable stand-in for the browser's EventSource — jsdom has no
// SSE support. Tests install this via `vi.stubGlobal('EventSource',
// FakeEventSource)`, then grab the created instance to `.emit(...)` a
// scripted event sequence and assert on how the app reacts, with no real
// network/server involved.
import type { SandboxEvent } from '../api/types'

export class FakeEventSource {
  static instances: FakeEventSource[] = []

  url: string
  onopen: (() => void) | null = null
  onmessage: ((ev: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  closed = false

  constructor(url: string) {
    this.url = url
    FakeEventSource.instances.push(this)
  }

  emit(event: SandboxEvent) {
    this.onmessage?.({ data: JSON.stringify(event) })
  }

  triggerError() {
    this.onerror?.()
  }

  close() {
    this.closed = true
  }

  static reset() {
    FakeEventSource.instances = []
  }

  static latest(): FakeEventSource {
    const instance = FakeEventSource.instances.at(-1)
    if (!instance) throw new Error('no FakeEventSource has been constructed yet')
    return instance
  }
}
