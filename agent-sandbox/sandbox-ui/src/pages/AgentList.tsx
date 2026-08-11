import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { deleteAgent, listAgents } from '../api/client'
import type { AgentSpec } from '../api/types'

export function AgentList() {
  const [agents, setAgents] = useState<AgentSpec[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  function reload() {
    listAgents()
      .then(setAgents)
      .catch((e: Error) => setError(e.message))
  }

  useEffect(reload, [])

  async function handleDelete(id: string) {
    if (!confirm(`Delete agent "${id}"? This removes its spec file.`)) return
    try {
      await deleteAgent(id)
      reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'failed to delete agent')
    }
  }

  return (
    <div>
      <div className="row" style={{ justifyContent: 'space-between', marginBottom: 16 }}>
        <h2 style={{ margin: 0 }}>Agents</h2>
        <Link to="/agents/new">
          <button>+ new agent</button>
        </Link>
      </div>

      {error && <div className="error-text">{error}</div>}
      {!agents && !error && <div className="muted">loading…</div>}
      {agents && agents.length === 0 && <div className="muted">no agents yet — create one to get started</div>}

      {agents && agents.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Name</th>
              <th>Model</th>
              <th>MCP servers</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {agents.map((agent) => (
              <tr key={agent.id}>
                <td>
                  <code>{agent.id}</code>
                </td>
                <td>{agent.name}</td>
                <td className="muted">{agent.model.model_name}</td>
                <td className="muted">{agent.mcp_servers.length}</td>
                <td>
                  <div className="row-actions">
                    <Link to={`/launch?agent=${encodeURIComponent(agent.id)}`}>
                      <button className="secondary">run</button>
                    </Link>
                    <Link to={`/agents/${encodeURIComponent(agent.id)}`}>
                      <button className="secondary">edit</button>
                    </Link>
                    <button className="danger" onClick={() => handleDelete(agent.id)}>
                      delete
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
