import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import { RunsListPage } from './pages/RunsListPage'
import { RunDetailPage } from './pages/RunDetailPage'
import { LaunchPage } from './pages/LaunchPage'
import { LeaderboardPage } from './pages/LeaderboardPage'

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
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
