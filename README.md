# agentic-ai-project

A monorepo for an agentic ML pipeline: LLM agents make judgment calls
(target selection, feature proposals, template/strategy choice), while
a fully deterministic harness makes every correctness-critical decision
(splits, metrics, leakage gates, sandboxed execution, promotion). See
each sub-project's own README/CLAUDE.md for the full design rationale.

## Sub-projects

| Path | What it is |
|---|---|
| [`agentic-ml-classification/`](agentic-ml-classification/) | The core pipeline: agents + deterministic harness, CLI scripts, notebooks. Start here. |
| [`agent-frontend/`](agent-frontend/) | FastAPI server + React app for launching, watching, and comparing pipeline runs. **Work in progress** — expect it to change a lot. |
| [`agent-testing/`](agent-testing/), [`openclaw-poc/`](openclaw-poc/), [`openclaw-docs-agent/`](openclaw-docs-agent/), [`openclaw-ml-pipe/`](openclaw-ml-pipe/), [`aviation_mas_mvp/`](aviation_mas_mvp/) | Older exploration/testing code (including early OpenClaw-based attempts — see `agentic-ml-classification/README.md`'s "Why there's no OpenClaw here" for why that path was dropped). Not under active development. |

## Local setup

These steps get `agentic-ml-classification` and `agent-frontend`
running locally. Do them in order.

### 0. Clone the repo (agent-frontend branch)

```bash
git clone -b agent-frontend git@github.com:meaton96/agentic-ai-project.git
```

### 1. Datasets

Download the following from Kaggle and drop the files into
[`agentic-ml-classification/datasets/`](agentic-ml-classification/datasets/)
(under `raw/`, matching what the scripts/notebooks expect — a Kaggle
account is required to download):

- [NGAFID MC (C37)](https://www.kaggle.com/datasets/hooong/ngafid-mc-20210917?resource=download&select=NGAFID_MC_C37.csv)
- [Titanic](https://www.kaggle.com/c/titanic/data)
- [Iris](https://www.kaggle.com/datasets/uciml/iris)

### 2. Python environment (root, [uv](https://docs.astral.sh/uv/))

One venv at the repo root covers both `agentic-ml-classification` and
`agent-frontend`'s Python code.

```bash
# install Python 3.12 and uv if you don't already have them
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e agentic-ml-classification   # editable install, so
                                               # agent-frontend can
                                               # `import agentic_ml`
```

### 3. RIT API key

```bash
cp agentic-ml-classification/.env.example agentic-ml-classification/.env
```

Edit `agentic-ml-classification/.env` and fill in `RIT_API_KEY` (and
check `RIT_BASE_URL`/`RIT_DEFAULT_MODEL` look right). See the comments
in `.env.example` for the optional gateway/local-model settings.

### 4. agent-frontend (optional — visual run viewer, WIP)

Backend, from `agent-frontend/`:

```bash
uvicorn server.app:app --host 127.0.0.1 --reload
```

Frontend — needs Node v22.22 (via [nvm](https://github.com/nvm-sh/nvm)
is easiest if Node isn't already installed):

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
# restart your shell, or `source ~/.bashrc` / `source ~/.zshrc`, then:
nvm install 22.22
nvm use 22.22
```

Then, from `agent-frontend/web/`:

```bash
npm install
npm run dev   # http://localhost:5173
```

See [`agent-frontend/README.md`](agent-frontend/README.md) for env
vars (`VITE_API_BASE_URL`, `AGENTIC_ML_DATA_ROOT`, etc.) and current
limitations.

## Seeing it in action

- [`agentic-ml-classification/notebooks/dynamic_orchestrator.ipynb`](agentic-ml-classification/notebooks/dynamic_orchestrator.ipynb) —
  a full run of the agentic pipeline, cell by cell: the planner
  sequencing through intake -> feature engineering -> profiler ->
  modeling -> verification -> finalize -> summarize, plus a second
  example showing a different goal routing to a different agent
  sequence. The best single entry point for understanding what this
  project actually does.
- `agent-frontend` (steps 4 above) — the same kind of run, watched
  live in the browser instead of read cell-by-cell. **This is a work
  in progress and will change dramatically** — treat it as a preview,
  not a stable tool.

## Running the pipeline's agent tools over MCP (optional)

By default, agents get their tools (dataset profile, template catalog,
etc.) from in-process Python closures — no server involved. As an
alternative, `agentic-ml-classification` also ships a standalone MCP
(Model Context Protocol) server that exposes those same tools over the
network, so the dynamic orchestrator can be pointed at it instead.
Nothing is running by default; you start it yourself, from
`agentic-ml-classification/`:

```bash
# terminal 1 — the MCP server (config: configs/mcp_server.json)
python scripts/run_mcp_server.py

# terminal 2 — the dynamic orchestrator, routed through it
python scripts/run_dynamic_orchestrator.py \
    --data datasets/raw/your_dataset.csv \
    --goal "predict your_target_column" \
    --use-mcp
```

Only `run_dynamic_orchestrator.py` supports `--use-mcp` today; the
static orchestrator and standalone single-agent scripts still use the
in-process path. See
[`agentic-ml-classification/README.md`](agentic-ml-classification/README.md)'s
"MCP fact server" build-log entry for how it's built (a strict fact
server — the harness computes every fact, the server only serves
already-persisted JSON, never a dataframe or fitted model) and
[`PROJECT_OVERVIEW.md` §8.6](agentic-ml-classification/PROJECT_OVERVIEW.md)
for the design rationale.
