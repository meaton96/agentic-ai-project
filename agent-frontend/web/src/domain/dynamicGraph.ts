// Builds the dynamic-orchestrator timeline. Unlike the static orchestrator,
// the dynamic one genuinely doesn't know its own future steps — the
// planner decides one at a time based on live state (orchestrator/
// dynamic_loop.py). So this pre-seeds the *typical* classification
// sequence (the same order the planner's own system prompt describes) as
// pending steps up front, then claims each one as the matching
// planner_proposal_accepted event arrives, rather than only ever showing
// steps after they've already been decided. A run that deviates from the
// typical order — a second modeling attempt after a rejected verdict, a
// streaming-only agent like monitor_drift, an explain-only goal that skips
// straight to deep_dive — still renders correctly: repeat or unplanned
// agents are inserted as their own extra steps at the point they actually
// happened, and any skeleton step nothing ever claims just stays pending
// (the same "never ran" convention the static graph already uses for a
// skipped intake/feature_engineering).
import type { RunEvent } from '../api/types'
import { nodeKindForAgentId, terminalStateForEvent } from './eventClassification'
import type { GraphModel, GraphNode } from './graphTypes'

// Mirrors planner_step.py's SYSTEM_PROMPT: "Typical order for a
// classification goal: intake (only if target isn't known yet) ->
// feature_engineering (optional) -> profiler -> split_and_check_leakage ->
// modeling (one or more times) -> verification -> finalize -> summarize."
const EXPECTED_SEQUENCE = [
  'intake', 'feature_engineering', 'profiler', 'split_and_check_leakage',
  'modeling', 'verification', 'finalize', 'summarize',
]

function pendingSkeletonNode(agentId: string): GraphNode {
  return {
    id: `step_${agentId}`,
    position: { x: 0, y: 0 },
    data: { label: agentId, kind: nodeKindForAgentId(agentId), state: 'pending', events: [] },
  }
}

export function buildDynamicGraph(events: RunEvent[], transcriptNames: string[]): GraphModel {
  const nodes: GraphNode[] = [
    { id: 'start', position: { x: 0, y: 0 }, data: { label: 'Run start', kind: 'harness', state: 'passed', events: [] } },
    ...EXPECTED_SEQUENCE.map(pendingSkeletonNode),
  ]

  let cursor = 1 // insertion point for anything not claiming a pre-seeded skeleton slot
  const occurrenceCount = new Map<string, number>()
  const transcriptCallCounts = new Map<string, number>()
  let activeAgentId: string | null = null
  let activeIndex: number | null = null

  function claimTranscript(agentId: string): string | undefined {
    const claimed = transcriptCallCounts.get(agentId) ?? 0
    transcriptCallCounts.set(agentId, claimed + 1)
    return transcriptNames.filter((n) => n.startsWith(`${agentId}_`)).sort()[claimed]
  }

  for (const event of events) {
    if (event.phase !== 'planner') {
      if (activeIndex != null && activeAgentId && event.phase === activeAgentId) {
        const node = nodes[activeIndex]
        node.data.events.push(event)
        const next = terminalStateForEvent(event.type, event.payload)
        if (next) node.data.state = next
      }
      continue
    }

    if (event.type === 'planner_proposal_rejected') {
      const proposal = event.payload.proposal as { agent_id?: string } | string | null | undefined
      const label = typeof proposal === 'object' && proposal?.agent_id ? proposal.agent_id : '(unparseable proposal)'
      nodes.splice(cursor, 0, {
        id: `rejected_${nodes.length}`,
        position: { x: 0, y: 0 },
        data: { label: `rejected: ${label}`, kind: 'agent', state: 'failed', events: [event], rejected: true },
      })
      cursor += 1
      activeAgentId = null
      activeIndex = null
      continue
    }

    if (event.type === 'planner_proposal_accepted') {
      const proposal = event.payload.proposal as { action?: string; agent_id?: string | null }

      if (proposal?.action === 'finish') {
        nodes.splice(cursor, 0, {
          id: 'finish',
          position: { x: 0, y: 0 },
          data: { label: 'finish', kind: 'harness', state: 'passed', events: [event] },
        })
        activeAgentId = null
        activeIndex = null
        continue
      }

      const agentId = proposal.agent_id as string
      const occurrence = occurrenceCount.get(agentId) ?? 0
      occurrenceCount.set(agentId, occurrence + 1)
      const skeletonIndex = nodes.findIndex((n) => n.id === `step_${agentId}`)

      let targetIndex: number
      if (occurrence === 0 && skeletonIndex !== -1) {
        targetIndex = skeletonIndex // claim the pre-seeded slot, no splice needed
      } else {
        // a repeat call (e.g. modeling again after a rejected verdict) or an
        // agent outside the typical sequence (deep_dive, monitor_drift,
        // retrain_decision, infer_batch) — its own step, at the point it
        // actually happened, not folded into/hidden by the original slot.
        targetIndex = cursor
        nodes.splice(targetIndex, 0, {
          id: `step_${agentId}_${occurrence}`,
          position: { x: 0, y: 0 },
          data: {
            label: occurrence > 0 ? `${agentId} (attempt ${occurrence + 1})` : agentId,
            kind: nodeKindForAgentId(agentId), state: 'pending', events: [],
          },
        })
      }

      nodes[targetIndex].data.transcriptName = claimTranscript(agentId)
      nodes[targetIndex].data.state = 'running'
      nodes[targetIndex].data.events.push(event)
      cursor = targetIndex + 1
      activeAgentId = agentId
      activeIndex = targetIndex
      continue
    }
    // agent_started / tool_called / planner_proposal (raw) for the
    // planner's own turn aren't rendered as separate steps — the
    // rejected/accepted events above already capture the outcome that
    // matters; intermediate planner chatter would just add noise here.
  }

  return { nodes, edges: [] }
}
