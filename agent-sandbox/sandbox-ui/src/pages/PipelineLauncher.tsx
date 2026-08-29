import { useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { launchPipelineRun, listPipelines } from '../api/client'
import type { PipelineSpec } from '../api/types'

export function PipelineLauncher() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()

  const [pipelines, setPipelines] = useState<PipelineSpec[] | null>(null)
  const [pipelineId, setPipelineId] = useState(searchParams.get('pipeline') ?? '')
  const [task, setTask] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [launching, setLaunching] = useState(false)

  useEffect(() => {
    listPipelines()
      .then((list) => {
        setPipelines(list)
        if (!pipelineId && list.length > 0) setPipelineId(list[0].id)
      })
      .catch((e: Error) => setError(e.message))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!pipelineId || !task.trim()) return
    setError(null)
    setLaunching(true)
    try {
      const { pipeline_run_id } = await launchPipelineRun({ pipeline_id: pipelineId, task })
      navigate(`/pipeline-runs/${encodeURIComponent(pipeline_run_id)}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'failed to launch pipeline run')
      setLaunching(false)
    }
  }

  const selected = pipelines?.find((p) => p.id === pipelineId)

  return (
    <div>
      <h2>Launch a pipeline</h2>
      {pipelines && pipelines.length === 0 && (
        <p className="muted">No pipelines exist yet — add a PipelineSpec YAML under pipelines/ first.</p>
      )}
      <form onSubmit={handleSubmit} className="card" style={{ maxWidth: 640 }}>
        <div className="field">
          <label>
            Pipeline
            <select value={pipelineId} onChange={(e) => setPipelineId(e.target.value)} required>
              <option value="" disabled>
                select a pipeline
              </option>
              {pipelines?.map((pipeline) => (
                <option key={pipeline.id} value={pipeline.id}>
                  {pipeline.name} ({pipeline.id})
                </option>
              ))}
            </select>
          </label>
        </div>
        {selected && (
          <p className="muted" style={{ marginTop: -8 }}>
            steps: {selected.steps.map((s) => s.step_id).join(' → ')}
          </p>
        )}
        <div className="field">
          <label>
            Seed task
            <textarea
              value={task}
              onChange={(e) => setTask(e.target.value)}
              rows={4}
              placeholder="Available to the first step as {{task}}"
              required
            />
          </label>
        </div>
        {error && <div className="error-text">{error}</div>}
        <button type="submit" disabled={launching || !pipelineId}>
          {launching ? 'launching…' : 'launch pipeline'}
        </button>
      </form>
    </div>
  )
}
