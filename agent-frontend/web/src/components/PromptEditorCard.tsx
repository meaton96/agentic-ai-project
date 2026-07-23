import { useState } from 'react'
import { ApiError, deletePromptOverride, putPromptOverride } from '../api/client'
import type { PromptInfo } from '../api/types'

export interface PromptEditorCardProps {
  prompt: PromptInfo
  onChange: (updated: PromptInfo) => void
}

const PANE_STYLE: React.CSSProperties = {
  flex: 1,
  minWidth: 0,
  height: 260,
  fontFamily: 'ui-monospace, SF Mono, Menlo, Consolas, monospace',
  fontSize: 12,
  padding: 10,
  borderRadius: 'var(--radius-sm)',
  border: '1px solid var(--color-border)',
  resize: 'vertical',
  whiteSpace: 'pre-wrap',
  overflow: 'auto',
}

/** Side-by-side comparison for one agent: the shipped default (read-only,
 * left) against this app's own override (editable, right) — starts as a
 * copy of the default so editing is always "start from what's currently
 * running" rather than a blank box. Save writes it as this agent's
 * override; Revert deletes the override file entirely, reverting the
 * agent to the shipped default (not just resetting the textarea). */
export function PromptEditorCard({ prompt, onChange }: PromptEditorCardProps) {
  const startingText = prompt.override_content ?? prompt.default_content
  const [draft, setDraft] = useState(startingText)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const dirty = draft !== startingText

  async function handleSave() {
    setSaving(true)
    setError(null)
    try {
      const updated = await putPromptOverride(prompt.agent, draft)
      onChange(updated)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  async function handleRevert() {
    setSaving(true)
    setError(null)
    try {
      const updated = await deletePromptOverride(prompt.agent)
      onChange(updated)
      setDraft(updated.default_content)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
        <h3 style={{ margin: 0 }}>{prompt.agent}</h3>
        {prompt.has_override && <span className="badge badge-override-active">override active</span>}
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          <button onClick={handleRevert} disabled={saving || !prompt.has_override}>
            Revert to default
          </button>
          <button onClick={handleSave} disabled={saving || !dirty}>
            {saving ? 'Saving…' : 'Save override'}
          </button>
        </div>
      </div>

      {error && <div className="error-text" style={{ marginBottom: 8 }}>{error}</div>}

      <div style={{ display: 'flex', gap: 12 }}>
        <div>
          <div className="muted" style={{ marginBottom: 4 }}>
            default
          </div>
          <pre
            data-testid={`${prompt.agent}-default`}
            style={{ ...PANE_STYLE, background: 'var(--color-surface-sunken)', margin: 0 }}
          >
            {prompt.default_content}
          </pre>
        </div>
        <div>
          <div className="muted" style={{ marginBottom: 4 }}>
            override {dirty && '(unsaved changes)'}
          </div>
          <textarea
            aria-label={`${prompt.agent} override`}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            style={{ ...PANE_STYLE, background: 'var(--color-surface)' }}
          />
        </div>
      </div>
    </div>
  )
}
