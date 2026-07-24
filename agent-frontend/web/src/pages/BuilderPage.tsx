import { useEffect, useMemo, useState } from 'react'
import { Background, Controls, ReactFlow, useEdgesState, useNodesState, type NodeMouseHandler, type NodeTypes } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { getWorkflowCatalog } from '../api/client'
import type { WorkflowCatalog, WorkflowCatalogType, WorkflowNode } from '../api/types'
import { toFlowEdges, toFlowNodes, type WorkflowFlowNode } from '../domain/workflowGraph'
import { WorkflowNodeCard } from '../components/builder/WorkflowNodeCard'
import { WorkflowSidePanel } from '../components/builder/WorkflowSidePanel'

const nodeTypes: NodeTypes = { workflowNode: WorkflowNodeCard }

/** Workflow Builder v0: a read-only node-graph visualization of the
 * pipeline topology, fetched from GET /api/workflow-catalog. Separate
 * screen from run detail's timeline (that renders one run's actual
 * progress; this renders the topology itself, independent of any run).
 * Editing, swapping candidates, and saving layouts are out of scope for
 * this pass — nodes are draggable only so a user can rearrange the view
 * for readability, never persisted, and no edges can be created here. */
export function BuilderPage() {
  const [type, setType] = useState<WorkflowCatalogType>('static')
  const [catalog, setCatalog] = useState<WorkflowCatalog | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const [nodes, setNodes, onNodesChange] = useNodesState<WorkflowFlowNode>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState(toFlowEdges([]))

  useEffect(() => {
    setError(null)
    setSelectedId(null)
    getWorkflowCatalog(type)
      .then((c) => {
        setCatalog(c)
        setNodes(toFlowNodes(c))
        setEdges(toFlowEdges(c.edges))
      })
      .catch((e: Error) => setError(e.message))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [type])

  const selectedNode = useMemo<WorkflowNode | null>(
    () => catalog?.nodes.find((n) => n.id === selectedId) ?? null,
    [catalog, selectedId],
  )

  const handleNodeClick: NodeMouseHandler = (_event, node) => setSelectedId(node.id)

  return (
    <div>
      <div style={{ display: 'flex', gap: 12, alignItems: 'baseline', marginBottom: 4 }}>
        <h2 style={{ margin: 0 }}>Workflow Builder</h2>
        <span className="muted">read-only visualization of the pipeline topology</span>
      </div>
      <p className="muted" style={{ marginTop: 0, marginBottom: 16 }}>
        Nodes can be dragged around for layout only — nothing here edits, swaps, or saves a workflow.
      </p>

      <div className="field radio-group" style={{ marginBottom: 16 }}>
        <label>
          <input type="radio" checked={type === 'static'} onChange={() => setType('static')} />
          static
        </label>
        <label>
          <input type="radio" checked={type === 'dynamic'} onChange={() => setType('dynamic')} />
          dynamic
        </label>
      </div>

      {error && <div className="error-text">Failed to load workflow catalog: {error}</div>}
      {!catalog && !error && <div className="card muted">Loading…</div>}

      {catalog && (
        <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start' }}>
          <div className="card" style={{ flex: 1, minWidth: 0, height: 560, padding: 0, overflow: 'hidden' }} data-testid="workflow-canvas">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              nodeTypes={nodeTypes}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onNodeClick={handleNodeClick}
              nodesConnectable={false}
              fitView
            >
              <Background />
              <Controls showInteractive={false} />
            </ReactFlow>
          </div>

          {selectedNode ? (
            <WorkflowSidePanel node={selectedNode} onClose={() => setSelectedId(null)} />
          ) : (
            <div className="card muted" style={{ width: 320 }}>
              Click a node to see its full description
              {type === 'dynamic' ? ', when_to_use, and required_state.' : ' and role in the sequence.'}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
