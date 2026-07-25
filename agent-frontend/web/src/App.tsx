import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import { RunsListPage } from './pages/RunsListPage'
import { RunDetailPage } from './pages/RunDetailPage'
import { LaunchPage } from './pages/LaunchPage'
import { LeaderboardPage } from './pages/LeaderboardPage'
import { PromptsPage } from './pages/PromptsPage'
import { BuilderPage } from './pages/BuilderPage'
import { McpPage } from './pages/McpPage'

function Nav() {
  return (
    <nav className="app-nav">
      <span className="brand">agentic-ml-classification</span>
      <NavLink to="/" end className={({ isActive }) => (isActive ? 'active' : '')}>
        Runs
      </NavLink>
      <NavLink to="/launch" className={({ isActive }) => (isActive ? 'active' : '')}>
        Launch
      </NavLink>
      <NavLink to="/leaderboard" className={({ isActive }) => (isActive ? 'active' : '')}>
        Leaderboard
      </NavLink>
      <NavLink to="/prompts" className={({ isActive }) => (isActive ? 'active' : '')}>
        Prompts
      </NavLink>
      <NavLink to="/builder" className={({ isActive }) => (isActive ? 'active' : '')}>
        Builder
      </NavLink>
      <NavLink to="/mcp" className={({ isActive }) => (isActive ? 'active' : '')}>
        MCP
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
            <Route path="/" element={<RunsListPage />} />
            <Route path="/runs/:runId" element={<RunDetailPage />} />
            <Route path="/launch" element={<LaunchPage />} />
            <Route path="/leaderboard" element={<LeaderboardPage />} />
            <Route path="/prompts" element={<PromptsPage />} />
            <Route path="/builder" element={<BuilderPage />} />
            <Route path="/mcp" element={<McpPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
