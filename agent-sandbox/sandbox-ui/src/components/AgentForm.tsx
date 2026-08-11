import { useState } from 'react'
import type { AgentSpec, LoggingPolicy, McpServerBinding, McpTransport } from '../api/types'
import { CredentialRefInput } from './CredentialRefInput'
import { TagInput } from './TagInput'

export interface AgentFormProps {
  initial?: AgentSpec
  /** false while editing — changing an id would desync it from the
   * PUT /agents/{id} path param and from the filename it's stored under. */
  idEditable: boolean
  submitLabel: string
  onSubmit: (spec: AgentSpec) => Promise<void>
  onCancel?: () => void
}

interface McpBindingForm {
  key: string
  name: string
  transport: McpTransport
  connectionText: string
  credential_ref: string
  allowed_tools: string[]
  logging_policy: LoggingPolicy
}

interface FormState {
  id: string
  name: string
  system_prompt: string
  base_url: string
  model_name: string
  api_key_ref: string
  temperature: string
  max_tokens: string
  max_turns: string
  mcp_servers: McpBindingForm[]
}

const CONNECTION_PLACEHOLDER: Record<McpTransport, string> = {
  stdio: '{\n  "command": "npx",\n  "args": ["-y", "@modelcontextprotocol/server-filesystem", "./workspace"]\n}',
  http: '{\n  "url": "https://example.com/mcp"\n}',
  sse: '{\n  "url": "https://example.com/mcp/sse"\n}',
}

let bindingKeySeq = 0
function newBindingKey(): string {
  bindingKeySeq += 1
  return `new-${bindingKeySeq}`
}

function emptyBinding(): McpBindingForm {
  return {
    key: newBindingKey(),
    name: '',
    transport: 'stdio',
    connectionText: '{}',
    credential_ref: '',
    allowed_tools: [],
    logging_policy: 'full',
  }
}

function bindingToForm(binding: McpServerBinding): McpBindingForm {
  return {
    key: newBindingKey(),
    name: binding.name,
    transport: binding.transport,
    connectionText: JSON.stringify(binding.connection, null, 2),
    credential_ref: binding.credential_ref ?? '',
    allowed_tools: binding.allowed_tools ?? [],
    logging_policy: binding.logging_policy,
  }
}

function specToForm(spec: AgentSpec): FormState {
  return {
    id: spec.id,
    name: spec.name,
    system_prompt: spec.system_prompt,
    base_url: spec.model.base_url,
    model_name: spec.model.model_name,
    api_key_ref: spec.model.api_key_ref,
    temperature: String(spec.model.temperature),
    max_tokens: spec.model.max_tokens != null ? String(spec.model.max_tokens) : '',
    max_turns: String(spec.max_turns),
    mcp_servers: spec.mcp_servers.map(bindingToForm),
  }
}

function emptyForm(): FormState {
  return {
    id: '',
    name: '',
    system_prompt: '',
    base_url: '',
    model_name: '',
    api_key_ref: '',
    temperature: '0',
    max_tokens: '',
    max_turns: '25',
    mcp_servers: [],
  }
}

/** Returns either a valid AgentSpec or a list of human-readable problems —
 * the only thing that can actually fail client-side is a binding's
 * connection JSON not parsing (everything else pydantic will reject with a
 * field-level 422 the caller surfaces separately). */
function formToSpec(form: FormState): { spec: AgentSpec } | { errors: string[] } {
  const errors: string[] = []
  const mcp_servers: McpServerBinding[] = []

  for (const binding of form.mcp_servers) {
    let connection: Record<string, unknown>
    try {
      connection = JSON.parse(binding.connectionText || '{}')
    } catch {
      errors.push(`MCP server "${binding.name || '(unnamed)'}": connection is not valid JSON`)
      continue
    }
    mcp_servers.push({
      name: binding.name,
      transport: binding.transport,
      connection,
      credential_ref: binding.credential_ref.trim() || null,
      allowed_tools: binding.allowed_tools.length > 0 ? binding.allowed_tools : null,
      logging_policy: binding.logging_policy,
    })
  }

  if (errors.length > 0) return { errors }

  const temperature = Number(form.temperature)
  const maxTurns = Number(form.max_turns)
  if (Number.isNaN(temperature)) errors.push('temperature must be a number')
  if (Number.isNaN(maxTurns)) errors.push('max turns must be a number')
  if (errors.length > 0) return { errors }

  return {
    spec: {
      id: form.id.trim(),
      name: form.name.trim(),
      system_prompt: form.system_prompt,
      model: {
        base_url: form.base_url.trim(),
        model_name: form.model_name.trim(),
        api_key_ref: form.api_key_ref.trim(),
        temperature,
        max_tokens: form.max_tokens.trim() ? Number(form.max_tokens) : null,
      },
      mcp_servers,
      sub_agents: [],
      max_turns: maxTurns,
    },
  }
}

export function AgentForm({ initial, idEditable, submitLabel, onSubmit, onCancel }: AgentFormProps) {
  const [form, setForm] = useState<FormState>(initial ? specToForm(initial) : emptyForm())
  const [errors, setErrors] = useState<string[]>([])
  const [submitting, setSubmitting] = useState(false)

  function updateBinding(key: string, patch: Partial<McpBindingForm>) {
    setForm((f) => ({
      ...f,
      mcp_servers: f.mcp_servers.map((b) => (b.key === key ? { ...b, ...patch } : b)),
    }))
  }

  function removeBinding(key: string) {
    setForm((f) => ({ ...f, mcp_servers: f.mcp_servers.filter((b) => b.key !== key) }))
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const result = formToSpec(form)
    if ('errors' in result) {
      setErrors(result.errors)
      return
    }
    setErrors([])
    setSubmitting(true)
    try {
      await onSubmit(result.spec)
    } catch (err) {
      setErrors([err instanceof Error ? err.message : 'failed to save agent'])
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <div className="field">
        <label>
          Agent ID
          <input
            value={form.id}
            onChange={(e) => setForm((f) => ({ ...f, id: e.target.value }))}
            disabled={!idEditable}
            placeholder="e.g. file-writer"
            required
          />
        </label>
      </div>

      <div className="field">
        <label>
          Name
          <input value={form.name} onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))} required />
        </label>
      </div>

      <div className="field">
        <label>
          System prompt
          <textarea
            value={form.system_prompt}
            onChange={(e) => setForm((f) => ({ ...f, system_prompt: e.target.value }))}
            rows={5}
            required
          />
        </label>
      </div>

      <h4>Model</h4>
      <div className="field">
        <label>
          Base URL
          <input
            value={form.base_url}
            onChange={(e) => setForm((f) => ({ ...f, base_url: e.target.value }))}
            placeholder="https://api.example.com/v1"
            required
          />
        </label>
      </div>
      <div className="field">
        <label>
          Model name
          <input value={form.model_name} onChange={(e) => setForm((f) => ({ ...f, model_name: e.target.value }))} required />
        </label>
      </div>
      <div className="field">
        <CredentialRefInput
          label="API key credential ref"
          value={form.api_key_ref}
          onChange={(v) => setForm((f) => ({ ...f, api_key_ref: v }))}
        />
      </div>
      <div className="field row">
        <label style={{ flex: 1 }}>
          Temperature
          <input
            type="number"
            step="0.1"
            value={form.temperature}
            onChange={(e) => setForm((f) => ({ ...f, temperature: e.target.value }))}
          />
        </label>
        <label style={{ flex: 1 }}>
          Max tokens <span className="hint">(optional)</span>
          <input
            type="number"
            value={form.max_tokens}
            onChange={(e) => setForm((f) => ({ ...f, max_tokens: e.target.value }))}
          />
        </label>
      </div>

      <div className="field">
        <label style={{ maxWidth: 200 }}>
          Max turns
          <input
            type="number"
            value={form.max_turns}
            onChange={(e) => setForm((f) => ({ ...f, max_turns: e.target.value }))}
          />
        </label>
      </div>

      <h4>MCP Servers</h4>
      {form.mcp_servers.length === 0 && <p className="muted">no MCP servers wired up yet</p>}
      {form.mcp_servers.map((binding) => (
        <div className="card" key={binding.key} style={{ marginBottom: 12 }}>
          <div className="field row">
            <label style={{ flex: 1 }}>
              Server name
              <input value={binding.name} onChange={(e) => updateBinding(binding.key, { name: e.target.value })} required />
            </label>
            <label style={{ flex: 1 }}>
              Transport
              <select
                value={binding.transport}
                onChange={(e) => updateBinding(binding.key, { transport: e.target.value as McpTransport })}
              >
                <option value="stdio">stdio</option>
                <option value="http">http</option>
                <option value="sse">sse</option>
              </select>
            </label>
            <label style={{ flex: 1 }}>
              Logging policy
              <select
                value={binding.logging_policy}
                onChange={(e) => updateBinding(binding.key, { logging_policy: e.target.value as LoggingPolicy })}
              >
                <option value="full">full</option>
                <option value="hashed">hashed</option>
                <option value="metadata">metadata</option>
              </select>
            </label>
          </div>

          <div className="field">
            <label>
              Connection (JSON)
              <textarea
                value={binding.connectionText}
                onChange={(e) => updateBinding(binding.key, { connectionText: e.target.value })}
                rows={4}
                placeholder={CONNECTION_PLACEHOLDER[binding.transport]}
              />
            </label>
          </div>

          <div className="field">
            <CredentialRefInput
              label="Credential ref (optional)"
              value={binding.credential_ref}
              onChange={(v) => updateBinding(binding.key, { credential_ref: v })}
            />
          </div>

          <div className="field">
            <TagInput
              label="Allowed tools"
              hint="leave empty to allow every tool this server exposes"
              value={binding.allowed_tools}
              onChange={(tags) => updateBinding(binding.key, { allowed_tools: tags })}
            />
          </div>

          <button type="button" className="secondary" onClick={() => removeBinding(binding.key)}>
            remove server
          </button>
        </div>
      ))}
      <button
        type="button"
        className="secondary"
        onClick={() => setForm((f) => ({ ...f, mcp_servers: [...f.mcp_servers, emptyBinding()] }))}
        style={{ marginBottom: 20 }}
      >
        + add MCP server
      </button>

      {errors.length > 0 && (
        <div className="field">
          {errors.map((err) => (
            <div className="error-text" key={err}>
              {err}
            </div>
          ))}
        </div>
      )}

      <div className="row-actions">
        <button type="submit" disabled={submitting}>
          {submitting ? 'saving…' : submitLabel}
        </button>
        {onCancel && (
          <button type="button" className="secondary" onClick={onCancel}>
            cancel
          </button>
        )}
      </div>
    </form>
  )
}
