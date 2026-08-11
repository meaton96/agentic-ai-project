// The one module that knows sandbox-server's routes and response shapes.
// Components call these functions, never `fetch` directly — see
// useRunStream.ts for the SSE counterpart of this file (which needs the
// raw stream URL, not a parsed response).
import type {
  AgentSpec,
  LaunchRunRequest,
  LaunchRunResponse,
  RunDetail,
  RunSummary,
  ValidationErrorDetail,
} from './types'

export const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://127.0.0.1:8000'

export class ApiError extends Error {
  status: number
  validationErrors?: ValidationErrorDetail[]
  constructor(status: number, message: string, validationErrors?: ValidationErrorDetail[]) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.validationErrors = validationErrors
  }
}

function summarizeDetail(detail: unknown): { message: string; validationErrors?: ValidationErrorDetail[] } {
  if (typeof detail === 'string') return { message: detail }
  if (Array.isArray(detail)) {
    const validationErrors = detail as ValidationErrorDetail[]
    const message = validationErrors
      .map((e) => `${e.loc.filter((p) => p !== 'body').join('.')}: ${e.msg}`)
      .join('; ')
    return { message, validationErrors }
  }
  return { message: 'request failed' }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!res.ok) {
    let message = res.statusText
    let validationErrors: ValidationErrorDetail[] | undefined
    try {
      const body = (await res.json()) as { detail?: unknown }
      const summarized = summarizeDetail(body?.detail)
      if (summarized.message) message = summarized.message
      validationErrors = summarized.validationErrors
    } catch {
      // non-JSON error body — keep statusText
    }
    throw new ApiError(res.status, message, validationErrors)
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

export function listAgents(): Promise<AgentSpec[]> {
  return request('/agents')
}

export function getAgent(id: string): Promise<AgentSpec> {
  return request(`/agents/${encodeURIComponent(id)}`)
}

export function createAgent(spec: AgentSpec): Promise<AgentSpec> {
  return request('/agents', { method: 'POST', body: JSON.stringify(spec) })
}

export function updateAgent(id: string, spec: AgentSpec): Promise<AgentSpec> {
  return request(`/agents/${encodeURIComponent(id)}`, { method: 'PUT', body: JSON.stringify(spec) })
}

export function deleteAgent(id: string): Promise<void> {
  return request(`/agents/${encodeURIComponent(id)}`, { method: 'DELETE' })
}

export function setCredential(ref: string, value: string): Promise<void> {
  return request(`/credentials/${encodeURIComponent(ref)}`, { method: 'POST', body: JSON.stringify({ value }) })
}

export function listCredentials(): Promise<string[]> {
  return request('/credentials')
}

export function launchRun(body: LaunchRunRequest): Promise<LaunchRunResponse> {
  return request('/runs', { method: 'POST', body: JSON.stringify(body) })
}

export function listRuns(): Promise<RunSummary[]> {
  return request('/runs')
}

export function getRun(runId: string): Promise<RunDetail> {
  return request(`/runs/${encodeURIComponent(runId)}`)
}

// Not fetched via `request` — handed to `new EventSource(...)` by
// useRunStream.ts, which needs the raw URL, not a parsed response.
export function runStreamUrl(runId: string): string {
  return `${API_BASE}/runs/${encodeURIComponent(runId)}/stream`
}
