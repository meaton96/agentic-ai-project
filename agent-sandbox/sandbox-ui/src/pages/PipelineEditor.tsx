import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { createPipeline, getPipeline, updatePipeline } from '../api/client'
import type { PipelineSpec } from '../api/types'
import { PipelineForm } from '../components/PipelineForm'

export function PipelineEditor() {
  const { pipelineId } = useParams<{ pipelineId: string }>()
  const navigate = useNavigate()
  const isEditing = Boolean(pipelineId)

  const [existing, setExisting] = useState<PipelineSpec | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  useEffect(() => {
    if (!pipelineId) return
    getPipeline(pipelineId)
      .then(setExisting)
      .catch((e: Error) => setLoadError(e.message))
  }, [pipelineId])

  async function handleSubmit(spec: PipelineSpec) {
    if (isEditing && pipelineId) {
      await updatePipeline(pipelineId, spec)
    } else {
      await createPipeline(spec)
    }
    navigate('/pipelines')
  }

  return (
    <div>
      <h2>{isEditing ? `Edit pipeline: ${pipelineId}` : 'New pipeline'}</h2>
      {loadError && <div className="error-text">{loadError}</div>}
      {isEditing && !existing && !loadError && <div className="muted">loading…</div>}
      {(!isEditing || existing) && (
        <PipelineForm
          initial={existing ?? undefined}
          idEditable={!isEditing}
          submitLabel={isEditing ? 'save changes' : 'create pipeline'}
          onSubmit={handleSubmit}
          onCancel={() => navigate('/pipelines')}
        />
      )}
    </div>
  )
}
