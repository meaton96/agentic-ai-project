import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { deletePipeline, listPipelines } from '../api/client'
import type { PipelineSpec } from '../api/types'

export function PipelineList() {
  const [pipelines, setPipelines] = useState<PipelineSpec[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  function reload() {
    listPipelines()
      .then(setPipelines)
      .catch((e: Error) => setError(e.message))
  }

  useEffect(reload, [])

  async function handleDelete(id: string) {
    if (!confirm(`Delete pipeline "${id}"? This removes its spec file.`)) return
    try {
      await deletePipeline(id)
      reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'failed to delete pipeline')
    }
  }

  return (
    <div>
      <div className="row" style={{ justifyContent: 'space-between', marginBottom: 16 }}>
        <h2 style={{ margin: 0 }}>Pipelines</h2>
        <Link to="/pipelines/new">
          <button>+ new pipeline</button>
        </Link>
      </div>
      <p className="muted">
        A pipeline is a deterministic, ordered sequence of agent steps — the sandbox runs each
        step's agent in turn and substitutes its output into the next step's task.
      </p>

      {error && <div className="error-text">{error}</div>}
      {!pipelines && !error && <div className="muted">loading…</div>}
      {pipelines && pipelines.length === 0 && (
        <div className="muted">no pipelines yet — add a PipelineSpec YAML under pipelines/</div>
      )}

      {pipelines && pipelines.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Name</th>
              <th>Steps</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {pipelines.map((pipeline) => (
              <tr key={pipeline.id}>
                <td>
                  <code>{pipeline.id}</code>
                </td>
                <td>{pipeline.name}</td>
                <td className="muted">{pipeline.steps.map((s) => s.step_id).join(' → ')}</td>
                <td>
                  <div className="row-actions">
                    <Link to={`/pipelines/launch?pipeline=${encodeURIComponent(pipeline.id)}`}>
                      <button className="secondary">run</button>
                    </Link>
                    <Link to={`/pipelines/${encodeURIComponent(pipeline.id)}`}>
                      <button className="secondary">edit</button>
                    </Link>
                    <button className="danger" onClick={() => handleDelete(pipeline.id)}>
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
