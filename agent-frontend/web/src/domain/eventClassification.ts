// Classifies one already-computed pipeline event into a UI node state.
// This is deliberately a lookup table over event `type` (+ a payload flag
// the server already computed, e.g. passed_gate / verdict / ok) — it does
// not re-derive any pipeline decision. If a real bug determined a
// candidate should have failed a leakage gate, that determination already
// happened in agentic_ml.harness.leakage and is already reflected in the
// payload this function reads; this function never re-runs a check.
import type { RunEvent } from '../api/types'
import type { NodeState } from './graphTypes'

/**
 * Returns the new terminal state a node should move to given one event
 * belonging to it, or null if this event doesn't change the node's state
 * (e.g. agent_started / tool_called / an intermediate leakage_gate_result
 * that isn't the phase's actual terminal event — those still get recorded
 * in the node's `events` list for the side panel, just without flipping
 * pending/running -> a terminal state).
 */
export function terminalStateForEvent(eventType: string, payload: Record<string, unknown>): NodeState | null {
  switch (eventType) {
    case 'proposal_validated':
      return 'passed'
    case 'proposal_rejected':
    case 'candidate_rejected':
      return 'failed'
    case 'profiler_report':
      return payload.ok ? 'passed' : 'failed'
    case 'split_completed':
      return payload.ok ? 'passed' : 'failed'
    case 'candidate_scored':
      return payload.passed_gate ? 'passed' : 'failed'
    case 'verification_verdict': {
      const verdict = payload.verdict
      if (verdict === 'rejected') return 'vetoed'
      if (verdict === 'approved' || verdict === 'flagged') return 'passed'
      return null
    }
    case 'final_test_metrics':
    case 'deep_dive_hypothesis':
    case 'retrain_decision':
    case 'batch_inference_completed':
    case 'drift_report':
    case 'summary_produced':
      return 'passed'
    default:
      return null
  }
}

/** True for event types that start a fresh sub-agent turn within a phase —
 * used to segment a phase's flat event list into per-candidate/per-attempt
 * groups (e.g. one modeling agent_started per candidate). */
export function isAgentStartedEvent(eventType: string): boolean {
  return eventType === 'agent_started'
}

// Agent ids the planner can propose that run with NO LLM call (see
// PROJECT_OVERVIEW.md §4 and the planner's own tool descriptions,
// "Deterministic (no LLM)") — rendered as dark/harness nodes in the
// dynamic graph, same convention as the static graph's fixed harness boxes.
const DETERMINISTIC_AGENT_IDS = new Set(['split_and_check_leakage', 'finalize'])

export function nodeKindForAgentId(agentId: string): 'agent' | 'harness' {
  return DETERMINISTIC_AGENT_IDS.has(agentId) ? 'harness' : 'agent'
}

type Metric = { value: number }

function fmtMetric(metrics: unknown, name = 'roc_auc'): string | null {
  const m = (metrics as Record<string, Metric> | undefined)?.[name]
  return typeof m?.value === 'number' ? `${name}=${m.value.toFixed(3)}` : null
}

/** A one-line, human-readable summary of a single event — shown on a step
 * card even while collapsed, so "what happened here" doesn't require
 * opening the transcript. Every fact here is read directly off the
 * payload the server already computed (see module docstring); nothing is
 * inferred or re-decided. */
export function summarizeEvent(event: RunEvent): string | null {
  const p = event.payload
  switch (event.type) {
    case 'proposal_validated': {
      const spec = p.dataset_spec_proposal as { target_column?: string } | undefined
      if (spec?.target_column) return `target: ${spec.target_column}`
      const dropped = (p.drop_columns as unknown[] | undefined)?.length
      const added = (p.new_columns as unknown[] | undefined)?.length
      if (dropped != null || added != null) return `dropped ${dropped ?? 0} column(s), added ${added ?? 0}`
      return 'validated'
    }
    case 'proposal_rejected':
    case 'candidate_rejected': {
      const errors = p.errors as string[] | undefined
      return errors?.[0] ?? 'rejected'
    }
    case 'profiler_report':
      return `strategy: ${p.recommended_split_strategy ?? '?'}${p.is_imbalanced ? ' · imbalanced' : ''}`
    case 'split_completed':
      return `train=${p.n_train} val=${p.n_val} test=${p.n_test}`
    case 'leakage_gate_result':
      return `${p.check}: ${p.passed ? 'passed' : 'failed'} — ${p.detail}`
    case 'candidate_scored':
      return `${p.passed_gate ? 'passed both leakage gates' : 'failed a leakage gate'}${fmtMetric(p.metrics) ? ` · ${fmtMetric(p.metrics)}` : ''}`
    case 'verification_verdict': {
      const concerns = p.concerns as string[] | undefined
      return `verdict: ${p.verdict}${concerns?.length ? ` — ${concerns[0]}` : ''}`
    }
    case 'final_test_metrics':
      return fmtMetric(p.test_metrics) ?? 'final metrics recorded'
    case 'planner_proposal_accepted': {
      const proposal = p.proposal as { action?: string; agent_id?: string; reasoning?: string } | undefined
      return proposal?.reasoning ?? `${proposal?.action}${proposal?.agent_id ? `: ${proposal.agent_id}` : ''}`
    }
    case 'planner_proposal_rejected': {
      const errors = p.errors as string[] | undefined
      return errors?.[0] ?? 'proposal rejected'
    }
    case 'run_completed':
      return 'run completed successfully'
    case 'run_failed':
      return `run failed: ${p.status ?? 'unknown reason'}`
    default:
      return null
  }
}

/** The most recent summarizable event for a node, read in the same
 * last-wins order terminalStateForEvent uses for state. */
export function summarizeNodeEvents(events: RunEvent[]): string | null {
  for (let i = events.length - 1; i >= 0; i--) {
    const summary = summarizeEvent(events[i])
    if (summary) return summary
  }
  return null
}

export interface PromptSourceInfo {
  source: 'default' | 'override'
  path: string
}

/** Reads the phase's own prompt_loaded event (emitted by every
 * steps/*_step.py before it calls the model at all — see
 * agentic_ml.prompt_loader) — this is the audit trail for which prompt
 * text an agent step actually used, not a guess based on whether
 * use_prompt_overrides was requested for the run (a requested override
 * silently falls back to default per-agent if no override file exists
 * for that specific agent, so the request alone doesn't say what ran). */
export function promptSourceForEvents(events: RunEvent[]): PromptSourceInfo | null {
  const event = events.find((e) => e.type === 'prompt_loaded')
  if (!event) return null
  return { source: event.payload.source as 'default' | 'override', path: event.payload.path as string }
}
