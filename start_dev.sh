#!/usr/bin/env bash
#
# Starts everything local agent-frontend + MCP development needs, instead
# of three separate terminals:
#   1. the pipeline's MCP fact server     (agentic-ml-classification/scripts/run_mcp_server.py)
#   2. the agent-frontend FastAPI backend (127.0.0.1:8001 — matches agent-frontend/web/.env.local)
#   3. the Vite dev server                (agent-frontend/web, :5173)
#
# See this repo's README.md ("Local setup", "Running the pipeline's agent
# tools over MCP") for what each of these is on its own. Ctrl+C stops all
# three (and anything they spawned, e.g. Vite's own child process).

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$ROOT/agentic-ml-classification"
FRONTEND_DIR="$ROOT/agent-frontend"
WEB_DIR="$FRONTEND_DIR/web"

for d in "$PIPELINE_DIR" "$FRONTEND_DIR" "$WEB_DIR"; do
  if [ ! -d "$d" ]; then
    echo "error: expected directory not found: $d" >&2
    exit 1
  fi
done

# Prefer whatever python is already active (e.g. a conda env your shell
# profile already activates) and only fall back to this repo's own .venv
# (README.md's "Python environment" step) if agentic_ml isn't importable
# yet — never override an environment that already works.
if ! python -c "import agentic_ml" >/dev/null 2>&1 && [ -f "$ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi
if ! python -c "import agentic_ml" >/dev/null 2>&1; then
  echo "error: 'import agentic_ml' failed. Follow README.md's 'Python environment'" >&2
  echo "setup step first (uv venv + pip install -e agentic-ml-classification)." >&2
  exit 1
fi

if [ ! -d "$WEB_DIR/node_modules" ]; then
  echo "error: $WEB_DIR/node_modules not found. Run 'npm install' in agent-frontend/web/ first." >&2
  exit 1
fi

pids=()

cleanup() {
  echo ""
  echo "stopping mcp/api/web..."
  for pid in "${pids[@]:-}"; do
    [ -n "$pid" ] && kill -TERM -"$pid" >/dev/null 2>&1
  done
  wait >/dev/null 2>&1
}
trap cleanup EXIT INT TERM

# Job control: each job backgrounded below becomes its own process group,
# so cleanup()'s `kill -TERM -$pid` also reaches whatever it spawns (e.g.
# the Vite child process `npm run dev` launches), not just the immediate
# npm/python/uvicorn process itself.
set -m

echo "[1/3] MCP fact server   -> agentic-ml-classification/scripts/run_mcp_server.py"
( cd "$PIPELINE_DIR" && exec python scripts/run_mcp_server.py ) &
pids+=("$!")

echo "[2/3] FastAPI backend   -> http://127.0.0.1:8001"
( cd "$FRONTEND_DIR" && exec uvicorn server.app:app --host 127.0.0.1 --port 8001 --reload ) &
pids+=("$!")

echo "[3/3] Vite dev server   -> http://localhost:5173"
( cd "$WEB_DIR" && exec npm run dev ) &
pids+=("$!")

echo ""
echo "all three running (each prints its own logs below) — Ctrl+C to stop everything"
echo ""
wait
