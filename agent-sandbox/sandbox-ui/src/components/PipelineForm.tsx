import { useEffect, useState } from 'react'
import { listAgents } from '../api/client'
import type { AgentSpec, PipelineSpec, PipelineStep } from '../api/types'

export interface PipelineFormProps {
  initial?: PipelineSpec
  /** false while editing — changing an id would desync it from the
   * PUT /pipelines/{id} path param and from the filename it's stored under. */
  idEditable: boolean
  submitLabel: string
  onSubmit: (spec: PipelineSpec) => Promise<void>
  onCancel?: () => void
}

interface StepForm {
  key: string
  step_id: string
  agent_id: string
  task_template: string
}

interface FormState {
  id: string
  name: string
  steps: StepForm[]
}

let stepKeySeq = 0
function newStepKey(): string {
  stepKeySeq += 1
  return `new-${stepKeySeq}`
}

function emptyStep(): StepForm {
  return { key: newStepKey(), step_id: '', agent_id: '', task_template: '' }
}

function stepToForm(step: PipelineStep): StepForm {
  return { key: newStepKey(), step_id: step.step_id, agent_id: step.agent_id, task_template: step.task_template }
}

function specToForm(spec: PipelineSpec): FormState {
  return { id: spec.id, name: spec.name, steps: spec.steps.map(stepToForm) }
}

function emptyForm(): FormState {
  return { id: '', name: '', steps: [emptyStep()] }
}

/** Returns either a valid PipelineSpec or a list of human-readable problems.
 * Unlike AgentForm, most of what can go wrong here (empty steps, duplicate
 * step_ids) is a validation rule PipelineSpec.steps enforces server-side too
 * (a 422) — checking it client-side just gives a faster, friendlier message. */
function formToSpec(form: FormState): { spec: PipelineSpec } | { errors: string[] } {
  const errors: string[] = []

  if (form.steps.length === 0) errors.push('a pipeline needs at least one step')

  const seenIds = new Set<string>()
  const steps: PipelineStep[] = []
  for (const step of form.steps) {
    if (!step.step_id.trim()) errors.push('every step needs a step id')
    if (!step.agent_id) errors.push(`step "${step.step_id || '(unnamed)'}": select an agent`)
    if (!step.task_template.trim()) errors.push(`step "${step.step_id || '(unnamed)'}": task template is required`)
    if (seenIds.has(step.step_id)) errors.push(`duplicate step id "${step.step_id}"`)
    seenIds.add(step.step_id)
    steps.push({ step_id: step.step_id.trim(), agent_id: step.agent_id, task_template: step.task_template })
  }

  if (errors.length > 0) return { errors }

  return { spec: { id: form.id.trim(), name: form.name.trim(), steps } }
}

export function PipelineForm({ initial, idEditable, submitLabel, onSubmit, onCancel }: PipelineFormProps) {
  const [form, setForm] = useState<FormState>(initial ? specToForm(initial) : emptyForm())
  const [agents, setAgents] = useState<AgentSpec[]>([])
  const [errors, setErrors] = useState<string[]>([])
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    listAgents()
      .then(setAgents)
      .catch(() => setAgents([]))
  }, [])

  function updateStep(key: string, patch: Partial<StepForm>) {
    setForm((f) => ({ ...f, steps: f.steps.map((s) => (s.key === key ? { ...s, ...patch } : s)) }))
  }

  function removeStep(key: string) {
    setForm((f) => ({ ...f, steps: f.steps.filter((s) => s.key !== key) }))
  }

  function moveStep(index: number, direction: -1 | 1) {
    setForm((f) => {
      const target = index + direction
      if (target < 0 || target >= f.steps.length) return f
      const steps = [...f.steps]
      ;[steps[index], steps[target]] = [steps[target], steps[index]]
      return { ...f, steps }
    })
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
      setErrors([err instanceof Error ? err.message : 'failed to save pipeline'])
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <div className="field">
        <label>
          Pipeline ID
          <input
            value={form.id}
            onChange={(e) => setForm((f) => ({ ...f, id: e.target.value }))}
            disabled={!idEditable}
            placeholder="e.g. intake-demo"
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

      <h4>Steps</h4>
      <p className="muted" style={{ marginTop: -8 }}>
        Runs in order, top to bottom. A step's <code>task_template</code> can reference{' '}
        <code>{'{{task}}'}</code> (the pipeline's seed task) and{' '}
        <code>{'{{steps.<step_id>.output}}'}</code> (any earlier step's output).
      </p>

      {form.steps.map((step, index) => (
        <div className="card" key={step.key} style={{ marginBottom: 12 }}>
          <div className="field row">
            <label style={{ flex: 1 }}>
              Step ID
              <input
                value={step.step_id}
                onChange={(e) => updateStep(step.key, { step_id: e.target.value })}
                placeholder="e.g. summarize"
                required
              />
            </label>
            <label style={{ flex: 1 }}>
              Agent
              <select value={step.agent_id} onChange={(e) => updateStep(step.key, { agent_id: e.target.value })} required>
                <option value="" disabled>
                  select an agent
                </option>
                {agents.map((agent) => (
                  <option key={agent.id} value={agent.id}>
                    {agent.name} ({agent.id})
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="field">
            <label>
              Task template
              <textarea
                value={step.task_template}
                onChange={(e) => updateStep(step.key, { task_template: e.target.value })}
                rows={3}
                placeholder="e.g. Summarize this text: {{task}}"
                required
              />
            </label>
          </div>

          <div className="row-actions">
            <button type="button" className="secondary" onClick={() => moveStep(index, -1)} disabled={index === 0}>
              move up
            </button>
            <button
              type="button"
              className="secondary"
              onClick={() => moveStep(index, 1)}
              disabled={index === form.steps.length - 1}
            >
              move down
            </button>
            <button type="button" className="danger" onClick={() => removeStep(step.key)}>
              remove step
            </button>
          </div>
        </div>
      ))}
      <button
        type="button"
        className="secondary"
        onClick={() => setForm((f) => ({ ...f, steps: [...f.steps, emptyStep()] }))}
        style={{ marginBottom: 20 }}
      >
        + add step
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
