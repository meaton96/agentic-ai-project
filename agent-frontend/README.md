# agent-frontend

Frontend/server for [`agentic-ml-classification`](../agentic-ml-classification):
a FastAPI server that wraps the pipeline's existing agent/harness code and
launches, tracks, and streams runs, plus a React app for browsing runs,
watching one execute live, launching new ones, and comparing results on a
leaderboard.

This repo owns orchestration and visualization only — no ML/agent logic
lives here. Every route in `server/` either reads filesystem state the
pipeline already produces (`runs/`, `artifacts/`) or launches a run through
the pipeline's own `scripts/run_orchestrator.py` /
`scripts/run_dynamic_orchestrator.py` entry points.

For contributor/agent-session rules (process-launch invariants, the
pipeline-repo boundary, testing standards), see [`CLAUDE.md`](CLAUDE.md).

## Status

Both milestones are done and tested:

- **M1 — server** (`server/`): FastAPI app, `RunManager` (one run at a time,
  launched in a separate `forkserver`'d process), SSE event streaming with
  replay-then-tail, unified live/historical reads. 12 pytest tests.
- **M2 — frontend** (`web/`): 4 screens (runs list, run detail, launcher,
  leaderboard), a vertical step-by-step run timeline with inline summaries
  and expandable transcripts, live updates over SSE. 14 vitest tests.

Not yet built: authentication (this is a local dev tool, bound to
`127.0.0.1` only), parallel/queued runs (the pipeline doesn't support
concurrent candidate search yet either), and any visual polish beyond a
light styling pass.

## Layout

```
server/            FastAPI app
  app.py             create_app() factory, CORS for the Vite dev server, run instructions
  routes.py           all API routes
  run_manager.py       RunManager: launches/tracks runs
  events_io.py         events.jsonl reader — replay (summaries) + replay-then-tail (SSE)
  summaries.py          per-run summary assembly (live-tracked or on-disk, same code path)
  schemas.py             request/response Pydantic models
tests/              pytest suite for server/ (stubbed ModelClient, no live model calls)

web/                React (Vite + TypeScript) app
  src/api/            typed client (client.ts) + response shapes (types.ts) — the only
                       module that knows the server's JSON shapes
  src/hooks/           useRunEvents — the shared SSE connect/replay/tail/reconnect hook
  src/domain/           event → node-state classification, static/dynamic timeline builders
                       (pure functions, no rendering — this is what the tests target)
  src/components/        StatusBadge, Legend, timeline/{PipelineTimeline,StepCard}
  src/pages/              RunsListPage, RunDetailPage, LaunchPage, LeaderboardPage
                          (+ colocated *.test.tsx files)

runs/, artifacts/, datasets/   local data roots (gitignored except datasets/,
                                which may hold committed sample data)
```

## Running it locally

Backend (from `agent-frontend/`):

```bash
uvicorn server.app:app --host 127.0.0.1 --reload
```

Binds to `127.0.0.1` only, by design — this server can trigger execution of
sandboxed candidate code and must never listen on the network. By default it
resolves `runs/`, `artifacts/`, and `datasets/` relative to wherever it's
launched from; set `AGENTIC_ML_DATA_ROOT` (or the more specific
`AGENTIC_ML_{RUNS,ARTIFACTS,DATASETS}_DIR`) to point somewhere else — same
env vars the pipeline's own CLI scripts use.

To actually launch a run (not just browse existing ones) you also need a
model endpoint configured — `RIT_BASE_URL`/`RIT_API_KEY` for the default
`rit` endpoint, or `--use-local`/gateway equivalents; see the pipeline
repo's own docs.

Frontend (from `agent-frontend/web/`):

```bash
npm install   # first time only
npm run dev   # http://localhost:5173
```

If the backend isn't on the default `http://127.0.0.1:8000`, point the
frontend at it via `web/.env.local`:

```
VITE_API_BASE_URL=http://127.0.0.1:8001
```

## Tests

```bash
# backend
python -m pytest tests/ -v

# frontend
cd web && npm test          # vitest run
cd web && npm run test:watch
```

## API routes

| Method | Path | Notes |
|---|---|---|
| GET | `/api/datasets` | lists files under the datasets root |
| POST | `/api/runs` | launches a run; 202 + `run_id`, 409 if one's already active, 422 for a bad dataset |
| GET | `/api/runs` | all runs — server-tracked and on-disk-only, merged |
| GET | `/api/runs/{run_id}` | status, report, leaderboard entries, first/last event |
| GET | `/api/runs/{run_id}/events` | SSE: replay `events.jsonl`, then tail while the run is live |
| GET | `/api/runs/{run_id}/transcripts` | transcript filenames for the run |
| GET | `/api/runs/{run_id}/transcripts/{name}` | one transcript's full message list |
| GET | `/api/leaderboard` | optionally filtered by `?run_id=` |

## Frontend screens

| Route | Screen |
|---|---|
| `/` | Runs list |
| `/runs/:runId` | Run detail — the live/historical timeline (same rendering code for both) |
| `/launch` | Launch a new run |
| `/leaderboard` | Leaderboard, filterable by run id |

## Known limitation

For a run that finishes very early (e.g. a dynamic run whose planner
proposes `finish` almost immediately), the still-`pending` skeleton steps
render *after* the completed `finish` step rather than being hidden —
correct information, slightly odd read. Flagged, not yet fixed.
