#!/usr/bin/env bash
# Let the AGENT (not us) drive the four real tools against the real c28_demo
# data, using a FRESH session so the new AGENTS.md/TOOLS.md/SOUL.md actually
# get loaded (bootstrap files only inject on a session's first turn).
#
# Run AFTER:
#   1. Replacing workspace/{AGENTS,TOOLS,SOUL}.md with the new versions.
#   2. ./test_tools.sh has already passed (the oracle run we diff against).
set -euo pipefail
cd "$(dirname "$0")"

PASS="[ PASS ]"; FAIL="[ FAIL ]"; INFO="[ .... ]"
log()  { printf '%s %s\n' "$INFO" "$*"; }
ok()   { printf '%s %s\n' "$PASS" "$*"; }
die()  { printf '%s %s\n' "$FAIL" "$*" >&2; exit 1; }

if docker compose version >/dev/null 2>&1; then DC=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then DC=(docker-compose)
else die "Docker Compose not found."; fi

curl -fsS http://127.0.0.1:18789/healthz >/dev/null 2>&1 || die "Gateway not healthy."
ok "Gateway healthy."

for f in AGENTS.md TOOLS.md SOUL.md; do
  [[ -s "workspace/$f" ]] || die "workspace/$f missing or empty — copy the new version in first."
done
ok "AGENTS.md / TOOLS.md / SOUL.md present and non-empty."

WORKDIR=/home/node/.openclaw/workspace
RUN_DIR="$WORKDIR/agent_run"

log "Setting tools.profile=coding explicitly (belt-and-suspenders; should already be the default)..."
"${DC[@]}" exec -T openclaw-gateway node dist/index.js config set tools.profile "coding"
log "Restarting gateway to apply..."
"${DC[@]}" restart openclaw-gateway >/dev/null
for _ in $(seq 1 30); do
  curl -fsS http://127.0.0.1:18789/healthz >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS http://127.0.0.1:18789/healthz >/dev/null 2>&1 || die "Gateway didn't come back healthy."
ok "Gateway restarted with tools.profile=coding."

log "Pre-creating $RUN_DIR (pandas won't auto-create it for --out/--queue)..."
"${DC[@]}" exec -T openclaw-gateway mkdir -p "$RUN_DIR"

# Fresh session key every run (timestamped) so bootstrap files are guaranteed
# to be injected fresh -- reusing "main"'s existing session would NOT pick up
# the new AGENTS.md/TOOLS.md/SOUL.md (bootstrap only loads on a session's
# first turn).
SESSION_KEY="aviation-run-$(date +%s)"
log "Using fresh session key: $SESSION_KEY"

PROMPT="Run the predictive-maintenance workflow now using these exact paths (all relative to your workspace root):
  flight-dir : data/c28_demo/flights
  metadata   : data/c28_demo/metadata.csv
  model      : data/c28_model.joblib
  feats out  : agent_run/feats.csv
  preds out  : agent_run/preds.json
  queue      : agent_run/maintenance_queue.jsonl
  top-k      : 10
Follow AGENTS.md exactly: inspect -> featurize -> classify -> recommend, one tool
at a time, then produce the HITL summary and stop."

log "Sending the task to the agent (fresh session, this may take a bit)..."
REPLY_JSON="$("${DC[@]}" exec -T openclaw-gateway sh -c \
  "node dist/index.js agent --agent main --session-key '$SESSION_KEY' --message \"\$1\" --json" _ "$PROMPT" \
  2>/tmp/oc_agentrun_err.txt)" || {
    printf '%s Agent run failed:\n' "$FAIL" >&2; cat /tmp/oc_agentrun_err.txt >&2; exit 1; }

echo "$REPLY_JSON" > /tmp/oc_agentrun_reply.json
ok "Agent run completed. Full JSON reply saved to /tmp/oc_agentrun_reply.json"

echo
echo "===== Agent's reply ====="
python3 - "$REPLY_JSON" <<'PY' || echo "$REPLY_JSON"
import json, sys
try:
    d = json.loads(sys.argv[1])
    text = d.get("reply") or d.get("text") or d.get("message")
    print(text if isinstance(text, str) else json.dumps(d, indent=2))
except Exception:
    print(sys.argv[1])
PY
echo "=========================="

echo
log "Checking what landed on disk under ./workspace/agent_run/ ..."
"${DC[@]}" exec -T openclaw-gateway sh -c "ls -la '$RUN_DIR' 2>/dev/null || echo '(agent_run is empty or missing -- agent may not have completed the chain)'"

cat <<EOF

Next: diff what the agent actually did against the deterministic oracle:
  diff <(cat workspace/agent_run/maintenance_queue.jsonl 2>/dev/null) \\
       <(cat workspace/test_run/oracle/maintenance_queue.jsonl 2>/dev/null)

They won't be byte-identical (the agent chose its own tool-call phrasing/
ordering details and may summarize differently), but n_queued, the urgency
counts, and the top flight IDs should line up closely with the oracle run
from test_tools.sh. Large divergence is the actual signal worth digging into.
EOF