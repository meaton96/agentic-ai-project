#!/usr/bin/env bash
# Starts both M3 dev servers:
#   - sandbox-server (FastAPI/uvicorn) on http://127.0.0.1:8000
#   - sandbox-ui     (Vite dev server) on http://localhost:5173
#
# Agent specs / run logs are written under ./agents and ./runs (relative to
# this script's directory), matching agent-sandbox/.gitignore's `runs/`
# entry. Override via SANDBOX_AGENTS_DIR / SANDBOX_RUNS_DIR /
# SANDBOX_CREDENTIALS_PATH before running this script if you want different
# locations.
#
# Ctrl+C stops both servers (and, since each is launched via `setsid`, their
# own child processes too — e.g. `npm run dev` spawns a separate `vite`
# process; killing just npm's PID leaves that one running). Logs go to
# server.log / ui.log in this directory and are also tailed to stdout.
#
# Requires: setsid (util-linux, present on any normal Linux install).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
SERVER_LOG="$SCRIPT_DIR/server.log"
UI_LOG="$SCRIPT_DIR/ui.log"

echo "==> checking sandbox-server is importable (installs editable if not)"
if ! "$PYTHON" -c "import sandbox_server" >/dev/null 2>&1; then
  if "$PYTHON" -m pip --version >/dev/null 2>&1; then
    "$PYTHON" -m pip install -e ./sandbox-core -e ./sandbox-server
  elif command -v uv >/dev/null 2>&1; then
    # uv-managed venvs (pyvenv.cfg has a `uv = ...` line) don't ship pip by
    # design — uv replaces it. `uv pip install` targets $VIRTUAL_ENV (or the
    # nearest .venv) the same way `python -m pip` would inside an activated
    # venv, so this is the equivalent install for that kind of environment.
    uv pip install -e ./sandbox-core -e ./sandbox-server
  else
    echo "error: $PYTHON has no pip, and 'uv' isn't on PATH either — can't install sandbox-core/sandbox-server." >&2
    echo "Fix one of: activate a venv that has pip, install uv (https://docs.astral.sh/uv/), or run:" >&2
    echo "  PYTHON=/path/to/a/python/with/pip bash $0" >&2
    exit 1
  fi
fi

if [ ! -d "sandbox-ui/node_modules" ]; then
  echo "==> installing sandbox-ui npm dependencies"
  npm --prefix sandbox-ui install
fi

: > "$SERVER_LOG"
: > "$UI_LOG"

SERVER_PID=""
UI_PID=""
TAIL_PID=""
CLEANED_UP=""

cleanup() {
  [ -n "$CLEANED_UP" ] && return 0
  CLEANED_UP=1
  echo ""
  echo "==> stopping servers"
  [ -n "$TAIL_PID" ] && kill "$TAIL_PID" >/dev/null 2>&1 || true
  # each server was launched via `setsid`, so its PID is also its process
  # group id — killing -PID kills that whole group (npm + the vite process
  # it spawns), not just the one PID `$!` gave us.
  [ -n "$SERVER_PID" ] && kill -TERM -- "-$SERVER_PID" >/dev/null 2>&1 || true
  [ -n "$UI_PID" ] && kill -TERM -- "-$UI_PID" >/dev/null 2>&1 || true
  sleep 1
  # safety net in case anything survived detached from its group
  pkill -f "uvicorn sandbox_server.main:app" >/dev/null 2>&1 || true
  pkill -f "vite --port 5173" >/dev/null 2>&1 || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> starting sandbox-server on http://127.0.0.1:8000 (log: server.log)"
setsid "$PYTHON" -m uvicorn sandbox_server.main:app --host 127.0.0.1 --port 8000 \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "==> starting sandbox-ui on http://localhost:5173 (log: ui.log)"
setsid npm --prefix sandbox-ui run dev -- --port 5173 \
  > "$UI_LOG" 2>&1 &
UI_PID=$!

tail -n +1 -f "$SERVER_LOG" "$UI_LOG" &
TAIL_PID=$!

echo "==> both servers starting — press Ctrl+C to stop"
wait "$SERVER_PID" "$UI_PID"
