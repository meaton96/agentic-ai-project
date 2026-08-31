# agent-sandbox

A local dev tool for authoring agents (system prompt, model endpoint, MCP
tool servers) as YAML specs, launching runs against them, and watching the
turn-by-turn trace live in the browser. Agents can also be chained into
**pipelines** — a deterministic, harness-driven sequence of agent steps and
gate (approve/reject/retry) steps.

For how it fits together under the hood, see [`docs/architecture.md`](docs/architecture.md).
This README is just "how do I get it running and click around."

## Prerequisites

- **Node.js** (with `npm`/`npx` on `PATH`) — needed for `sandbox-ui` and for
  the MCP filesystem server some example agents use.
- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** — used to create the venv. `pip`
  works too if you already have a venv with it activated; `start.sh` falls
  back to plain `pip install` first and only uses `uv` if `pip` isn't
  available.

## One-time setup

All commands below are run from `agent-sandbox/` unless noted otherwise.

### 1. Python venv

```bash
cd sandbox-core
uv venv --python 3.11
cd ..
```

This creates `sandbox-core/.venv`. `start.sh` will `pip install -e` both
`sandbox-core` and `sandbox-server` into whichever Python it's pointed at
the first time you run it — you don't need to install anything by hand,
just point `PYTHON` at this venv (see "Starting the servers" below).

### 2. Repo-root `.env`

Gate steps (and a couple of model-endpoint settings) are read from a
`.env` file at the **repo root** (`agentic-ai-project/.env`, one level
above `agent-sandbox/`) — not a `.env` inside `agent-sandbox/` itself. If
you've already set one up for `agentic-ml-classification`
(see [the root README](../README.md)'s "RIT API key" step), you likely
just need to add one line to it:

```bash
# agentic-ai-project/.env
RIT_API_KEY=...
RIT_BASE_URL=https://api.genai.gccis.rit.edu/v1
AGENTIC_ML_DATA_ROOT=../agentic-ml-classification
SANDBOX_GATES_PYTHONPATH=workspace:gates
```

- `SANDBOX_GATES_PYTHONPATH` is what makes a `GateStep`'s
  `"module.path:function_name"` importable — colon-separated directories,
  relative to `agent-sandbox/` unless absolute. `workspace` and `gates`
  cover the example gates already in this repo; add to it if your Phase 2
  gate code lives somewhere else.
- This file is picked up automatically (upward `.env` search) — no need to
  `export` anything by hand or `cd` to a specific place first.

### 3. Model API key (credential store)

The model API key itself is **not** read from `.env` — agent specs
reference it indirectly via `api_key_ref` (e.g. `rit-api` in
[`agents/summarizer.yaml`](agents/summarizer.yaml)), which resolves against
a separate flat YAML credential store at `~/.sandbox/credentials.yaml`:

```bash
mkdir -p ~/.sandbox
cat >> ~/.sandbox/credentials.yaml <<'EOF'
rit-api: <your RIT API key>
EOF
```

(You can also set/update a credential from the UI's agent editor — it
writes to this same file. There's deliberately no way to *read* a
credential's value back out through the API or UI.)

## Starting the servers

From `agent-sandbox/`:

```bash
PYTHON=sandbox-core/.venv/bin/python ./start.sh
```

This starts both dev servers and tails their logs to your terminal
(`server.log` / `ui.log` also get written to disk):

- **sandbox-server** (FastAPI/uvicorn) — `http://127.0.0.1:8000`
- **sandbox-ui** (Vite) — `http://localhost:5173`

Open **`http://localhost:5173`**. `Ctrl+C` stops both servers cleanly.

First run will take a little longer — it installs both Python packages
editable and runs `npm install` for the UI if `sandbox-ui/node_modules`
isn't there yet.

## Using the UI

Two top-level sections, both reachable from the nav:

**Agents**
- **Agent List / Agent Editor** — create/edit an `AgentSpec`: system
  prompt, model endpoint + `api_key_ref`, which MCP tool servers it can
  use (with a per-server tool allowlist), max turns. Saved as
  `agents/<id>.yaml`.
- **Run Launcher** — pick an agent, type a task, hit run.
- **Run View** — watch a run live: every LLM request/response and tool
  call/result streams in as it happens (SSE). Reopen a finished run to
  replay the same trace from disk.
- **Run History** — every past run (including ones launched from the
  CLI), with status (`running` / `completed` / `errored` / `truncated`).

**Pipelines**
- **Pipeline List** — pipelines are authored as YAML by hand for now (no
  visual editor for the step graph yet), under `pipelines/*.yaml`. Steps
  are either an agent step (`kind: agent`) or a gate step (`kind: gate`,
  a deterministic Python function that inspects prior step output and
  decides where to route next — including backward, for a reject→retry
  loop). See `pipelines/gate-loop-demo.yaml` for a working example of the
  latter.
- **Pipeline Launcher** — pick a pipeline, type a seed task, hit run.
- **Pipeline Run View** — polls for step-by-step progress (no live
  stream at the pipeline level); click into any step to see that step's
  own live/replayed Run View.

Quick smoke test once the servers are up: **Pipelines → sensor-report →
run**, paste the contents of `workspace/docs/sensor-log-02.md` as the seed
task, and watch both steps complete.

## Headless (no UI/server) — for scripting or debugging a single agent/gate

```bash
sandbox-core/.venv/bin/sandbox run agents/file-writer.yaml \
  --task "write a file that says hello"

sandbox-core/.venv/bin/sandbox pipeline run pipelines/sensor-report.yaml \
  --task "..." --agents-dir ./agents
```

## Where things live

| Path | What |
|---|---|
| `agents/*.yaml` | Agent specs |
| `pipelines/*.yaml` | Pipeline specs |
| `runs/<run_id>/events.jsonl` | Per-agent-run event log |
| `pipeline-runs/<id>/pipeline.json` | Per-pipeline-run manifest |
| `workspace/` | Scratch data + example gate modules the demo pipelines read/write |
| `gates/` | Gate function modules importable via `SANDBOX_GATES_PYTHONPATH` |
| `~/.sandbox/credentials.yaml` | Model API keys / other secrets (gitignored, outside the repo) |

All of the `agents`/`pipelines`/`runs`/`pipeline-runs` paths are
overridable via `SANDBOX_AGENTS_DIR` / `SANDBOX_PIPELINES_DIR` /
`SANDBOX_RUNS_DIR` / `SANDBOX_PIPELINE_RUNS_DIR` if you need them
somewhere else.

## More detail

- [`docs/architecture.md`](docs/architecture.md) — full system design,
  component diagram, run lifecycle, known gaps.
- [`docs/testing-guide.md`](docs/testing-guide.md) /
  [`docs/phase3-testing-guide.md`](docs/phase3-testing-guide.md) — worked
  examples for building and testing your own pipelines, including a
  reject→retry gate loop.
- [`docs/future-work-roadmap.md`](docs/future-work-roadmap.md) —
  what's planned next.
