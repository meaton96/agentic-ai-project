import { useEffect, useState } from 'react'
import { getMcpConfig } from '../api/client'
import type { McpConfigResponse } from '../api/types'
import { Collapsible } from '../components/Collapsible'

export function McpPage() {
  const [config, setConfig] = useState<McpConfigResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getMcpConfig()
      .then(setConfig)
      .catch((e: Error) => setError(e.message))
  }, [])

  if (error) return <div className="error-text">Failed to load MCP config: {error}</div>
  if (!config) return <div className="muted">loading…</div>

  return (
    <div>
      <h2>MCP server</h2>
      <p className="muted" style={{ maxWidth: 700 }}>
        Read-only view of the pipeline's MCP fact server (<code>agentic_ml.mcp_facts.server</code>): the
        config it would start with and the tools it registers. This page never starts or stops that
        server — <code>reachable</code> only reports whether something is already listening at the
        configured host/port, not a full MCP handshake.
      </p>

      <div className="card" style={{ marginBottom: 'var(--space-4)' }}>
        <h3>{config.name}</h3>
        <dl style={{ display: 'grid', gridTemplateColumns: 'max-content 1fr', gap: '4px 16px', margin: 0 }}>
          <dt className="muted">Status</dt>
          <dd style={{ margin: 0 }}>
            <span className={`badge ${config.reachable ? 'badge-completed' : 'badge-pending'}`}>
              {config.reachable ? 'reachable' : 'not reachable'}
            </span>
          </dd>

          <dt className="muted">URL</dt>
          <dd style={{ margin: 0 }}>
            <code>{config.url}</code>
          </dd>

          <dt className="muted">Config file</dt>
          <dd style={{ margin: 0 }}>
            <code>{config.config_path}</code>{' '}
            {config.config_exists ? (
              <span className="hint">(found)</span>
            ) : (
              <span className="hint">(not found — server.py's built-in defaults apply)</span>
            )}
          </dd>

          <dt className="muted">Enabled tools</dt>
          <dd style={{ margin: 0 }}>
            {config.enabled_tools.length} of {config.tools.length}
          </dd>
        </dl>
      </div>

      <h3>Tools</h3>
      {config.tools.map((tool) => (
        <div key={tool.name} className="card" style={{ marginBottom: 'var(--space-3)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
            <strong>{tool.name}</strong>
            <span className={`badge ${tool.enabled ? 'badge-completed' : 'badge-pending'}`}>
              {tool.enabled ? 'enabled' : 'disabled'}
            </span>
          </div>
          <p className="muted" style={{ margin: '0 0 8px 0' }}>
            {tool.description}
          </p>
          <Collapsible title="Input schema">
            <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{JSON.stringify(tool.input_schema, null, 2)}</pre>
          </Collapsible>
        </div>
      ))}
    </div>
  )
}
