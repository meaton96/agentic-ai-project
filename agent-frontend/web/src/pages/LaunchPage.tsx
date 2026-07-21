import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ApiError, launchRun, listDatasets } from '../api/client'
import type { DatasetInfo, ModelEndpoint, Orchestrator, SplitStrategy } from '../api/types'

const STRATEGIES: SplitStrategy[] = ['random', 'stratified', 'group', 'time', 'group_time']
const MODEL_ENDPOINTS: ModelEndpoint[] = ['rit', 'gateway', 'local']

export function LaunchPage() {
  const navigate = useNavigate()
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  const [dataset, setDataset] = useState('')
  const [orchestrator, setOrchestrator] = useState<Orchestrator>('static')
  const [target, setTarget] = useState('')
  const [goal, setGoal] = useState('')
  const [strategy, setStrategy] = useState<SplitStrategy | ''>('')
  const [maxCandidates, setMaxCandidates] = useState('')
  const [modelEndpoint, setModelEndpoint] = useState<ModelEndpoint>('rit')
  const [skipFeatureEngineering, setSkipFeatureEngineering] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listDatasets()
      .then((list) => {
        setDatasets(list)
        if (list.length > 0) setDataset(list[0].filename)
      })
      .catch((e: Error) => setError(e.message))
  }, [])

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setSubmitting(true)
    try {
      const result = await launchRun({
        dataset,
        orchestrator,
        target: target || undefined,
        goal: goal || undefined,
        strategy: strategy || undefined,
        max_candidates: orchestrator === 'static' && maxCandidates ? Number(maxCandidates) : undefined,
        model_endpoint: modelEndpoint,
        skip_feature_engineering: orchestrator === 'static' ? skipFeatureEngineering : false,
      })
      navigate(`/runs/${result.run_id}`)
    } catch (e) {
      setError(e instanceof ApiError ? `${e.status}: ${e.message}` : String(e))
      setSubmitting(false)
    }
  }

  return (
    <div style={{ maxWidth: 520 }}>
      <h2>Launch a run</h2>
      <form onSubmit={handleSubmit} className="card">
        <div className="field">
          <label>
            Dataset
            <select value={dataset} onChange={(e) => setDataset(e.target.value)} required>
              {datasets.length === 0 && <option value="">no datasets found</option>}
              {datasets.map((d) => (
                <option key={d.filename} value={d.filename}>
                  {d.filename}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="field radio-group">
          <label>
            <input type="radio" checked={orchestrator === 'static'} onChange={() => setOrchestrator('static')} />
            static
          </label>
          <label>
            <input type="radio" checked={orchestrator === 'dynamic'} onChange={() => setOrchestrator('dynamic')} />
            dynamic
          </label>
        </div>

        <div className="field">
          <label>
            Target column
            <input
              value={target}
              disabled={goal !== ''}
              onChange={(e) => setTarget(e.target.value)}
              placeholder="e.g. churned — skips intake"
            />
          </label>
        </div>
        <div className="field">
          <label>
            Goal (natural language)
            <input
              value={goal}
              disabled={target !== ''}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="e.g. predict whether a customer churns"
            />
          </label>
        </div>

        <div className="field">
          <label>
            Split strategy override
            <select value={strategy} onChange={(e) => setStrategy(e.target.value as SplitStrategy | '')}>
              <option value="">(profiler recommendation)</option>
              {STRATEGIES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="field">
          <label>
            Max candidates {orchestrator === 'dynamic' && <span className="hint">(static-orchestrator only)</span>}
            <input
              type="number"
              min={1}
              value={maxCandidates}
              disabled={orchestrator === 'dynamic'}
              onChange={(e) => setMaxCandidates(e.target.value)}
            />
          </label>
        </div>

        <div className="field">
          <label>
            Model endpoint
            <select value={modelEndpoint} onChange={(e) => setModelEndpoint(e.target.value as ModelEndpoint)}>
              {MODEL_ENDPOINTS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="field">
          <label className="inline-label">
            <input
              type="checkbox"
              checked={skipFeatureEngineering}
              disabled={orchestrator === 'dynamic'}
              onChange={(e) => setSkipFeatureEngineering(e.target.checked)}
            />
            Skip feature engineering {orchestrator === 'dynamic' && <span className="hint">&nbsp;(static-orchestrator only)</span>}
          </label>
        </div>

        {error && <div className="error-text field">{error}</div>}

        <button type="submit" disabled={submitting || !dataset}>
          {submitting ? 'Launching…' : 'Launch run'}
        </button>
      </form>
    </div>
  )
}
