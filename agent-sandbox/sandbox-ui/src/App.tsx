import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import { AgentList } from './pages/AgentList'
import { AgentEditor } from './pages/AgentEditor'
import { RunLauncher } from './pages/RunLauncher'
import { RunView } from './pages/RunView'
import { RunHistory } from './pages/RunHistory'
import { PipelineList } from './pages/PipelineList'
import { PipelineEditor } from './pages/PipelineEditor'
import { PipelineLauncher } from './pages/PipelineLauncher'
import { PipelineRunView } from './pages/PipelineRunView'

function Nav() {
  return (
    <nav className="app-nav">
      <span className="brand">sandbox</span>
      <NavLink to="/" end className={({ isActive }) => (isActive ? 'active' : '')}>
        Agents
      </NavLink>
      <NavLink to="/launch" className={({ isActive }) => (isActive ? 'active' : '')}>
        Launch
      </NavLink>
      <NavLink to="/runs" className={({ isActive }) => (isActive ? 'active' : '')}>
        Runs
      </NavLink>
      <NavLink to="/pipelines" className={({ isActive }) => (isActive ? 'active' : '')}>
        Pipelines
      </NavLink>
      <NavLink to="/pipeline-runs" className={({ isActive }) => (isActive ? 'active' : '')}>
        Pipeline runs
      </NavLink>
    </nav>
  )
}

export function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <Nav />
        <main className="app-main">
          <Routes>
            <Route path="/" element={<AgentList />} />
            <Route path="/agents/new" element={<AgentEditor />} />
            <Route path="/agents/:agentId" element={<AgentEditor />} />
            <Route path="/launch" element={<RunLauncher />} />
            <Route path="/runs/:runId/live" element={<RunView />} />
            <Route path="/runs" element={<RunHistory />} />
            <Route path="/runs/:runId" element={<RunHistory />} />
            <Route path="/pipelines" element={<PipelineList />} />
            <Route path="/pipelines/new" element={<PipelineEditor />} />
            <Route path="/pipelines/launch" element={<PipelineLauncher />} />
            <Route path="/pipelines/:pipelineId" element={<PipelineEditor />} />
            <Route path="/pipeline-runs" element={<PipelineRunView />} />
            <Route path="/pipeline-runs/:pipelineRunId" element={<PipelineRunView />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
