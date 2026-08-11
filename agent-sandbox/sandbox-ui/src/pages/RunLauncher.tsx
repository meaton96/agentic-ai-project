import { useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { launchRun, listAgents } from '../api/client'
import type { AgentSpec } from '../api/types'

export function RunLauncher() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()

  const [agents, setAgents] = useState<AgentSpec[] | null>(null)
  const [agentId, setAgentId] = useState(searchParams.get('agent') ?? '')
  const [task, setTask] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [launching, setLaunching] = useState(false)

  useEffect(() => {
    listAgents()
      .then((list) => {
        setAgents(list)
        if (!agentId && list.length > 0) setAgentId(list[0].id)
      })
      .catch((e: Error) => setError(e.message))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!agentId || !task.trim()) return
    setError(null)
    setLaunching(true)
    try {
      const { run_id } = await launchRun({ agent_id: agentId, task })
      navigate(`/runs/${encodeURIComponent(run_id)}/live`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'failed to launch run')
      setLaunching(false)
    }
  }

  return (
    <div>
      <h2>Launch a run</h2>
      {agents && agents.length === 0 && (
        <p className="muted">
          No agents exist yet — create one on the <a href="/">Agents</a> page first.
        </p>
      )}
      <form onSubmit={handleSubmit} className="card" style={{ maxWidth: 640 }}>
        <div className="field">
          <label>
            Agent
            <select value={agentId} onChange={(e) => setAgentId(e.target.value)} required>
              <option value="" disabled>
                select an agent
              </option>
              {agents?.map((agent) => (
                <option key={agent.id} value={agent.id}>
                  {agent.name} ({agent.id})
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="field">
          <label>
            Task
            <textarea
              value={task}
              onChange={(e) => setTask(e.target.value)}
              rows={4}
              placeholder="What should this agent do?"
              required
            />
          </label>
        </div>
        {error && <div className="error-text">{error}</div>}
        <button type="submit" disabled={launching || !agentId}>
          {launching ? 'launching…' : 'launch run'}
        </button>
      </form>
    </div>
  )
}
