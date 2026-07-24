// The one module that knows the server's response shapes and how to reach
// it. Components call these functions, never `fetch` directly — see
// CLAUDE.md-equivalent guidance for this app (keep API calls out of
// components) and useRunEvents.ts for the SSE counterpart of this file.
import type {
  DatasetInfo,
  LaunchRunRequest,
  LaunchRunResponse,
  LeaderboardEntry,
  PromptInfo,
  RunSummary,
  TranscriptMessage,
  WorkflowCatalog,
  WorkflowCatalogType,
} from './types'

export const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://127.0.0.1:8001'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!res.ok) {
    // FastAPI's default error body is {"detail": "..."} — fall back to
    // statusText if a route ever returns something else.
    let detail = res.statusText
    try {
      const body = (await res.json()) as { detail?: string }
      if (body?.detail) detail = body.detail
    } catch {
      // non-JSON error body — keep statusText
    }
    throw new ApiError(res.status, detail)
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

export function listDatasets(): Promise<DatasetInfo[]> {
  return request('/api/datasets')
}

export function launchRun(body: LaunchRunRequest): Promise<LaunchRunResponse> {
  return request('/api/runs', { method: 'POST', body: JSON.stringify(body) })
}

export function listRuns(): Promise<RunSummary[]> {
  return request('/api/runs')
}

export function getRun(runId: string): Promise<RunSummary> {
  return request(`/api/runs/${encodeURIComponent(runId)}`)
}

export function listTranscripts(runId: string): Promise<string[]> {
  return request(`/api/runs/${encodeURIComponent(runId)}/transcripts`)
}

export function getTranscript(runId: string, name: string): Promise<TranscriptMessage[]> {
  return request(`/api/runs/${encodeURIComponent(runId)}/transcripts/${encodeURIComponent(name)}`)
}

export function getLeaderboard(runId?: string): Promise<LeaderboardEntry[]> {
  const qs = runId ? `?run_id=${encodeURIComponent(runId)}` : ''
  return request(`/api/leaderboard${qs}`)
}

export function listPrompts(): Promise<PromptInfo[]> {
  return request('/api/prompts')
}

export function putPromptOverride(agent: string, content: string): Promise<PromptInfo> {
  return request(`/api/prompts/${encodeURIComponent(agent)}`, {
    method: 'PUT', body: JSON.stringify({ content }),
  })
}

export function deletePromptOverride(agent: string): Promise<PromptInfo> {
  return request(`/api/prompts/${encodeURIComponent(agent)}`, { method: 'DELETE' })
}

export function getWorkflowCatalog(type: WorkflowCatalogType): Promise<WorkflowCatalog> {
  return request(`/api/workflow-catalog?type=${encodeURIComponent(type)}`)
}

// Not fetched via `request` — this is handed to `new EventSource(...)` by
// useRunEvents.ts, which needs the raw URL, not a parsed response.
export function runEventsUrl(runId: string): string {
  return `${API_BASE}/api/runs/${encodeURIComponent(runId)}/events`
}
