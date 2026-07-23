import { useEffect, useState } from 'react'
import { listPrompts } from '../api/client'
import type { PromptInfo } from '../api/types'
import { PromptEditorCard } from '../components/PromptEditorCard'

export function PromptsPage() {
  const [prompts, setPrompts] = useState<PromptInfo[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listPrompts().then(setPrompts).catch((e: Error) => setError(e.message))
  }, [])

  function handleChange(updated: PromptInfo) {
    setPrompts((prev) => prev?.map((p) => (p.agent === updated.agent ? updated : p)) ?? prev)
  }

  if (error) return <div className="error-text">Failed to load prompts: {error}</div>
  if (!prompts) return <div className="muted">loading…</div>

  return (
    <div>
      <h2>Prompts</h2>
      <p className="muted" style={{ maxWidth: 700 }}>
        Each agent's shipped default prompt (left, read-only) and this app's own override for it
        (right, editable). Saving writes an override file this app owns — the pipeline's own
        <code> prompts/*.md</code> files are never modified. A run only uses these overrides if
        launched with "use prompt overrides" checked on the Launch page.
      </p>
      {prompts.map((prompt) => (
        <PromptEditorCard key={prompt.agent} prompt={prompt} onChange={handleChange} />
      ))}
    </div>
  )
}
