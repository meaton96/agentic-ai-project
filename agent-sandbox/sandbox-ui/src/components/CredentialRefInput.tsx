import { useState } from 'react'
import { setCredential } from '../api/client'

export interface CredentialRefInputProps {
  label: string
  value: string
  onChange: (ref: string) => void
  placeholder?: string
}

/** A credential ref name field plus a small inline affordance to actually
 * set that ref's value via POST /credentials/{ref} — the point is that a
 * user filling out an agent form never has to go hand-edit
 * ~/.sandbox/credentials.yaml to make api_key_ref/credential_ref resolve. */
export function CredentialRefInput({ label, value, onChange, placeholder }: CredentialRefInputProps) {
  const [expanded, setExpanded] = useState(false)
  const [secretValue, setSecretValue] = useState('')
  const [status, setStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [error, setError] = useState<string | null>(null)

  async function handleSave() {
    if (!value.trim()) {
      setStatus('error')
      setError('set a credential ref name first')
      return
    }
    setStatus('saving')
    setError(null)
    try {
      await setCredential(value.trim(), secretValue)
      setStatus('saved')
      setSecretValue('')
      setExpanded(false)
    } catch (e) {
      setStatus('error')
      setError(e instanceof Error ? e.message : 'failed to save credential')
    }
  }

  return (
    <label>
      {label}
      <div className="row">
        <input
          value={value}
          onChange={(e) => {
            onChange(e.target.value)
            setStatus('idle')
          }}
          placeholder={placeholder ?? 'credential ref name'}
          style={{ flex: 1 }}
        />
        <button type="button" className="secondary" onClick={() => setExpanded((v) => !v)}>
          {expanded ? 'cancel' : 'set credential'}
        </button>
      </div>
      {expanded && (
        <div className="row" style={{ marginTop: 6 }}>
          <input
            type="password"
            value={secretValue}
            onChange={(e) => setSecretValue(e.target.value)}
            placeholder="secret value"
            style={{ flex: 1 }}
          />
          <button type="button" onClick={handleSave} disabled={status === 'saving'}>
            {status === 'saving' ? 'saving…' : 'save'}
          </button>
        </div>
      )}
      {status === 'saved' && <div className="hint" style={{ color: 'var(--color-success)' }}>credential saved</div>}
      {error && <div className="error-text">{error}</div>}
    </label>
  )
}
