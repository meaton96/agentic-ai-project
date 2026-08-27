import { useEffect, useState } from 'react'
import { listAgents } from '../api/client'
import type { AgentSpec, GateStep, PipelineSpec, PipelineStep, Step } from '../api/types'

export interface PipelineFormProps {
  initial?: PipelineSpec
  /** false while editing — changing an id would desync it from the
   * PUT /pipelines/{id} path param and from the filename it's stored under. */
  idEditable: boolean
  submitLabel: string
  onSubmit: (spec: PipelineSpec) => Promise<void>
  onCancel?: () => void
}

interface OnResultEntry {
  key: string
  decision: string
  target: string
}

interface StepForm {
  key: string
  kind: 'agent' | 'gate'
  step_id: string
  // agent-step fields
  agent_id: string
  task_template: string
  // gate-step fields
  gate: string
  onResult: OnResultEntry[]
}

interface FormState {
  id: string
  name: string
  maxSteps: string
  steps: StepForm[]
}

let stepKeySeq = 0
function newStepKey(): string {
  stepKeySeq += 1
  return `new-${stepKeySeq}`
}

function emptyStep(): StepForm {
  return { key: newStepKey(), kind: 'agent', step_id: '', agent_id: '', task_template: '', gate: '', onResult: [] }
}

function emptyOnResultEntry(): OnResultEntry {
  return { key: newStepKey(), decision: '', target: '' }
}

function stepToForm(step: Step): StepForm {
  if (step.kind === 'gate') {
    return {
      key: newStepKey(),
      kind: 'gate',
      step_id: step.step_id,
      agent_id: '',
      task_template: '',
      gate: step.gate,
      onResult: Object.entries(step.on_result).map(([decision, target]) => ({ key: newStepKey(), decision, target })),
    }
  }
  return {
    key: newStepKey(),
    kind: 'agent',
    step_id: step.step_id,
    agent_id: step.agent_id,
    task_template: step.task_template,
    gate: '',
    onResult: [],
  }
}

function specToForm(spec: PipelineSpec): FormState {
  return { id: spec.id, name: spec.name, maxSteps: String(spec.max_steps ?? 50), steps: spec.steps.map(stepToForm) }
}

function emptyForm(): FormState {
  return { id: '', name: '', maxSteps: '50', steps: [emptyStep()] }
}

/** Returns either a valid PipelineSpec or a list of human-readable problems.
 * Unlike AgentForm, most of what can go wrong here (empty steps, duplicate
 * step_ids) is a validation rule PipelineSpec.steps enforces server-side too
 * (a 422) — checking it client-side just gives a faster, friendlier message. */
function formToSpec(form: FormState): { spec: PipelineSpec } | { errors: string[] } {
  const errors: string[] = []

  if (form.steps.length === 0) errors.push('a pipeline needs at least one step')

  const maxSteps = Number(form.maxSteps)
  if (!Number.isInteger(maxSteps) || maxSteps < 1) errors.push('max steps must be a positive integer')

  const seenIds = new Set<string>()
  const steps: Step[] = []
  for (const step of form.steps) {
    const label = step.step_id || '(unnamed)'
    if (!step.step_id.trim()) errors.push('every step needs a step id')
    if (seenIds.has(step.step_id)) errors.push(`duplicate step id "${step.step_id}"`)
    seenIds.add(step.step_id)

    if (step.kind === 'agent') {
      if (!step.agent_id) errors.push(`step "${label}": select an agent`)
      if (!step.task_template.trim()) errors.push(`step "${label}": task template is required`)
      const agentStep: PipelineStep = {
        kind: 'agent',
        step_id: step.step_id.trim(),
        agent_id: step.agent_id,
        task_template: step.task_template,
      }
      steps.push(agentStep)
    } else {
      if (!step.gate.includes(':')) errors.push(`step "${label}": gate must be "module.path:function_name"`)
      if (step.onResult.length === 0) errors.push(`step "${label}": add at least one decision → next step mapping`)
      const onResult: Record<string, string> = {}
      const seenDecisions = new Set<string>()
      for (const entry of step.onResult) {
        if (!entry.decision.trim()) errors.push(`step "${label}": every decision needs a name`)
        if (!entry.target.trim()) errors.push(`step "${label}": decision "${entry.decision || '(unnamed)'}" needs a target step id (or "__end__")`)
        if (seenDecisions.has(entry.decision)) errors.push(`step "${label}": duplicate decision "${entry.decision}"`)
        seenDecisions.add(entry.decision)
        onResult[entry.decision.trim()] = entry.target.trim()
      }
      const gateStep: GateStep = { kind: 'gate', step_id: step.step_id.trim(), gate: step.gate.trim(), on_result: onResult }
      steps.push(gateStep)
    }
  }

  if (errors.length > 0) return { errors }

  return { spec: { id: form.id.trim(), name: form.name.trim(), steps, max_steps: maxSteps } }
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

  function addOnResultEntry(stepKey: string) {
    setForm((f) => ({
      ...f,
      steps: f.steps.map((s) => (s.key === stepKey ? { ...s, onResult: [...s.onResult, emptyOnResultEntry()] } : s)),
    }))
  }

  function updateOnResultEntry(stepKey: string, entryKey: string, patch: Partial<OnResultEntry>) {
    setForm((f) => ({
      ...f,
      steps: f.steps.map((s) =>
        s.key === stepKey
          ? { ...s, onResult: s.onResult.map((e) => (e.key === entryKey ? { ...e, ...patch } : e)) }
          : s,
      ),
    }))
  }

  function removeOnResultEntry(stepKey: string, entryKey: string) {
    setForm((f) => ({
      ...f,
      steps: f.steps.map((s) => (s.key === stepKey ? { ...s, onResult: s.onResult.filter((e) => e.key !== entryKey) } : s)),
    }))
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

      <div className="field">
        <label>
          Max steps
          <input
            type="number"
            min={1}
            value={form.maxSteps}
            onChange={(e) => setForm((f) => ({ ...f, maxSteps: e.target.value }))}
            required
          />
        </label>
        <p className="muted" style={{ marginTop: 4 }}>
          Safety cap on total step executions — protects against a gate that always routes backward looping forever.
        </p>
      </div>

      <h4>Steps</h4>
      <p className="muted" style={{ marginTop: -8 }}>
        An agent step runs in list order unless a gate routes elsewhere. A step's <code>task_template</code> can
        reference <code>{'{{task}}'}</code> (the pipeline's seed task) and <code>{'{{steps.<step_id>.output}}'}</code>{' '}
        (any earlier step's output). A gate step calls a deterministic <code>module.path:function_name</code>{' '}
        with every completed step's output and jumps to whichever step id its decision maps to (use{' '}
        <code>__end__</code> as a target to end the pipeline there).
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
              Step type
              <select value={step.kind} onChange={(e) => updateStep(step.key, { kind: e.target.value as 'agent' | 'gate' })}>
                <option value="agent">agent</option>
                <option value="gate">gate</option>
              </select>
            </label>
          </div>

          {step.kind === 'agent' ? (
            <>
              <div className="field">
                <label>
                  Agent
                  <select
                    value={step.agent_id}
                    onChange={(e) => updateStep(step.key, { agent_id: e.target.value })}
                    required
                  >
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
            </>
          ) : (
            <>
              <div className="field">
                <label>
                  Gate function
                  <input
                    value={step.gate}
                    onChange={(e) => updateStep(step.key, { gate: e.target.value })}
                    placeholder="e.g. agentic_ml.harness.verification:check"
                    required
                  />
                </label>
              </div>

              <div className="field">
                <label>Decision → next step</label>
                {step.onResult.map((entry) => (
                  <div className="field row" key={entry.key} style={{ marginBottom: 4 }}>
                    <input
                      style={{ flex: 1 }}
                      value={entry.decision}
                      onChange={(e) => updateOnResultEntry(step.key, entry.key, { decision: e.target.value })}
                      placeholder="decision, e.g. approved"
                      required
                    />
                    <span className="muted">→</span>
                    <input
                      style={{ flex: 1 }}
                      value={entry.target}
                      onChange={(e) => updateOnResultEntry(step.key, entry.key, { target: e.target.value })}
                      placeholder="target step id, or __end__"
                      required
                    />
                    <button type="button" className="danger" onClick={() => removeOnResultEntry(step.key, entry.key)}>
                      remove
                    </button>
                  </div>
                ))}
                <button type="button" className="secondary" onClick={() => addOnResultEntry(step.key)}>
                  + add decision mapping
                </button>
              </div>
            </>
          )}

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
