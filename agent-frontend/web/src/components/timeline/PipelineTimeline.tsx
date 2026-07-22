import type { GraphModel } from '../../domain/graphTypes'
import { StepCard } from './StepCard'

export interface PipelineTimelineProps {
  runId: string
  graph: GraphModel
}

/** Renders a run's steps top-down, in the order the events actually
 * happened, as a plain scrollable list rather than a pan/zoom canvas —
 * each step is a full-width, independently expandable card. This
 * sidesteps the class of bug a canvas graph has here: nodes streaming in
 * live from SSE while the view is only fitted once on mount, which can
 * leave the viewport pointed at empty space by the time a run finishes.
 * Rejected planner proposals (dynamic runs only) are indented slightly
 * under the point they branched from, rather than occupying their own
 * row in the main sequence. */
export function PipelineTimeline({ runId, graph }: PipelineTimelineProps) {
  return (
    <div style={{ maxWidth: 860 }}>
      {graph.nodes.map((node) => (
        <StepCard key={node.id} runId={runId} node={node} indent={node.data.rejected} />
      ))}
    </div>
  )
}
