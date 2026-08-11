import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { createAgent, getAgent, updateAgent } from '../api/client'
import type { AgentSpec } from '../api/types'
import { AgentForm } from '../components/AgentForm'

export function AgentEditor() {
  const { agentId } = useParams<{ agentId: string }>()
  const navigate = useNavigate()
  const isEditing = Boolean(agentId)

  const [existing, setExisting] = useState<AgentSpec | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  useEffect(() => {
    if (!agentId) return
    getAgent(agentId)
      .then(setExisting)
      .catch((e: Error) => setLoadError(e.message))
  }, [agentId])

  async function handleSubmit(spec: AgentSpec) {
    if (isEditing && agentId) {
      await updateAgent(agentId, spec)
    } else {
      await createAgent(spec)
    }
    navigate('/')
  }

  return (
    <div>
      <h2>{isEditing ? `Edit agent: ${agentId}` : 'New agent'}</h2>
      {loadError && <div className="error-text">{loadError}</div>}
      {isEditing && !existing && !loadError && <div className="muted">loading…</div>}
      {(!isEditing || existing) && (
        <AgentForm
          initial={existing ?? undefined}
          idEditable={!isEditing}
          submitLabel={isEditing ? 'save changes' : 'create agent'}
          onSubmit={handleSubmit}
          onCancel={() => navigate('/')}
        />
      )}
    </div>
  )
}
