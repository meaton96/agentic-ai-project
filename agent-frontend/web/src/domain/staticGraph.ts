// Builds the fixed-layout graph for a static-orchestrator run, modeled on
// PROJECT_OVERVIEW.md §3's mermaid diagram: harness gates as dark nodes,
// agents as light nodes. Node state is purely a rendering of events the
// server already computed (see eventClassification.ts) — never a
// re-derivation of what the harness decided.
import type { RunEvent } from '../api/types'
import { isAgentStartedEvent, terminalStateForEvent } from './eventClassification'
import type { GraphEdge, GraphModel, GraphNode, GraphNodeData, NodeState } from './graphTypes'

const COL = 220
const CANDIDATE_COL = 660

function byPhase(events: RunEvent[], phase: string): RunEvent[] {
  return events.filter((e) => e.phase === phase)
}

function reduceState(events: RunEvent[]): NodeState {
  if (events.length === 0) return 'pending'
  let state: NodeState = 'running'
  for (const e of events) {
    const next = terminalStateForEvent(e.type, e.payload)
    if (next) state = next
  }
  return state
}

function simpleNode(
  id: string, x: number, label: string, kind: 'agent' | 'harness', events: RunEvent[], transcriptName?: string,
): GraphNode {
  return {
    id,
    position: { x, y: 0 },
    data: { label, kind, state: reduceState(events), events, transcriptName },
  }
}

/** Splits a phase's flat event list into per-candidate/per-attempt groups,
 * one group per agent_started event (each candidate/attempt starts with
 * its own agent_started — see steps/modeling_step.py, verification_step.py). */
function segmentByAgentStarted(events: RunEvent[]): RunEvent[][] {
  const segments: RunEvent[][] = []
  for (const e of events) {
    if (isAgentStartedEvent(e.type)) segments.push([e])
    else if (segments.length > 0) segments[segments.length - 1].push(e)
    else segments.push([e]) // defensive: events before any agent_started (shouldn't happen)
  }
  return segments
}

function candidateIdOf(segment: RunEvent[]): string | undefined {
  for (const e of segment) {
    const id = e.payload.candidate_id
    if (typeof id === 'string') return id
  }
  return undefined
}

/** i-th transcript whose filename starts with `prefix_` (cli_common.py's
 * make_transcript_writer numbers them prefix_01.json, prefix_02.json, ... in
 * call order — the same order events for that phase arrive in). */
function nthTranscript(transcriptNames: string[], prefix: string, index: number): string | undefined {
  const matches = transcriptNames.filter((n) => n.startsWith(`${prefix}_`)).sort()
  return matches[index]
}

export function buildStaticGraph(events: RunEvent[], transcriptNames: string[]): GraphModel {
  const nodes: GraphNode[] = []
  const edges: GraphEdge[] = []
  let x = 0
  let prevId: string | null = null

  function chain(node: GraphNode, dashed = false) {
    nodes.push(node)
    if (prevId) edges.push({ id: `${prevId}->${node.id}`, source: prevId, target: node.id, data: { dashed } })
    prevId = node.id
  }

  const intakeEvents = byPhase(events, 'intake')
  chain(simpleNode('intake', x, 'Intake Agent', 'agent', intakeEvents, nthTranscript(transcriptNames, 'intake', 0)))
  x += COL
  chain(simpleNode('intake_harness', x, 'Harness validates', 'harness', intakeEvents))
  x += COL

  const feEvents = byPhase(events, 'feature_engineering')
  chain(simpleNode('feature_engineering', x, 'Feature Engineering Agent', 'agent', feEvents, nthTranscript(transcriptNames, 'feature_engineering', 0)))
  x += COL
  chain(simpleNode('fe_harness', x, 'Harness validates + applies', 'harness', feEvents))
  x += COL

  const profilerEvents = byPhase(events, 'profiler')
  chain(simpleNode('profiler', x, 'Profiler Agent', 'agent', profilerEvents, nthTranscript(transcriptNames, 'profiler', 0)))
  x += COL

  const splitEvents = byPhase(events, 'split_and_check_leakage')
  chain(simpleNode('split_harness', x, 'Harness: split + leakage checks', 'harness', splitEvents))
  x += COL

  const modelingSegments = segmentByAgentStarted(byPhase(events, 'modeling'))
  const verificationSegments = segmentByAgentStarted(byPhase(events, 'verification'))
  const candidateCount = Math.max(modelingSegments.length, 1)

  const verificationByCandidateId = new Map<string, RunEvent[]>()
  verificationSegments.forEach((seg, i) => {
    const id = candidateIdOf(seg) ?? `#${i}`
    verificationByCandidateId.set(id, seg)
  })

  const candidateBaseX = x
  modelingSegments.forEach((seg, i) => {
    const candidateId = candidateIdOf(seg) ?? `#${i}`
    const colX = candidateBaseX + i * CANDIDATE_COL
    prevId = 'split_harness' // every candidate branches off the split gate, not off each other
    chain(simpleNode(`modeling_${candidateId}`, colX, `Modeling Agent (${candidateId})`, 'agent', seg, nthTranscript(transcriptNames, 'modeling', i)))
    chain(simpleNode(`modeling_harness_${candidateId}`, colX + COL, 'Harness: sandbox build/fit/score + 2 leakage gates', 'harness', seg))

    const vSeg = verificationByCandidateId.get(candidateId)
    if (vSeg) {
      const vIndex = [...verificationByCandidateId.keys()].indexOf(candidateId)
      chain(simpleNode(`verification_${candidateId}`, colX + 2 * COL, `Verification Agent (${candidateId})`, 'agent', vSeg, nthTranscript(transcriptNames, 'verification', vIndex)))
    }
  })

  const finalizeX = candidateBaseX + candidateCount * CANDIDATE_COL
  const finalizeEvents = byPhase(events, 'finalize')
  // finalize follows whichever candidate was actually approved/flagged, not
  // necessarily the last one tried — fall back to split_harness if nothing
  // passed verification yet (nothing to refit).
  const approvedCandidateId = [...verificationByCandidateId.entries()].find(([, seg]) =>
    seg.some((e) => e.type === 'verification_verdict' && e.payload.verdict !== 'rejected'),
  )?.[0]
  prevId = approvedCandidateId ? `verification_${approvedCandidateId}` : 'split_harness'
  chain(simpleNode('finalize_harness', finalizeX, 'Harness: refit on train+val, evaluate once', 'harness', finalizeEvents))

  const runCompleted = events.some((e) => e.phase === 'run' && e.type === 'run_completed')
  const runFailed = events.some((e) => e.phase === 'run' && e.type === 'run_failed')
  const summaryState: NodeState = runCompleted ? 'passed' : runFailed ? 'failed' : finalizeEvents.length > 0 ? 'running' : 'pending'
  const summaryData: GraphNodeData = {
    label: 'Narrated Summary', kind: 'agent', state: summaryState, events: [],
    transcriptName: nthTranscript(transcriptNames, 'analyst_summary', 0),
  }
  nodes.push({ id: 'summary', position: { x: finalizeX + COL, y: 0 }, data: summaryData })
  edges.push({ id: `${prevId}->summary`, source: 'finalize_harness', target: 'summary' })

  return { nodes, edges }
}
