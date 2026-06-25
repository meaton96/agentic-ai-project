#!/usr/bin/env bash
# Manual, no-agent test of the four real tools + the orchestrator_sim oracle,
# run INSIDE the gateway container against the real c28_demo data + the real
# trained model. No LLM involved -- this only proves the tool contract works
# in-container before the agent ever touches it.
#
# Run AFTER: rebuilding the image (new deps) and confirming the gateway is
# healthy. From aviation-openclaw/:
#   docker compose build openclaw-gateway
#   docker compose up -d --force-recreate openclaw-gateway
#   ./test_tools.sh
set -euo pipefail
cd "$(dirname "$0")"

PASS="[ PASS ]"; FAIL="[ FAIL ]"; INFO="[ .... ]"
log()  { printf '%s %s\n' "$INFO" "$*"; }
ok()   { printf '%s %s\n' "$PASS" "$*"; }
die()  { printf '%s %s\n' "$FAIL" "$*" >&2; exit 1; }

if docker compose version >/dev/null 2>&1; then DC=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then DC=(docker-compose)
else die "Docker Compose not found."; fi

curl -fsS http://127.0.0.1:18789/healthz >/dev/null 2>&1 \
  || die "Gateway not healthy. Start it first."
ok "Gateway healthy."

FLIGHT_DIR=/home/node/.openclaw/workspace/data/c28_demo/flights
METADATA=/home/node/.openclaw/workspace/data/c28_demo/metadata.csv
MODEL=/home/node/.openclaw/workspace/data/c28_model.joblib
RUN_DIR=/home/node/.openclaw/workspace/test_run
# `python -m pkg.module` only finds the package if the package's PARENT dir is
# on sys.path -- and `-m` auto-prepends the process's cwd to sys.path. So
# every python3 -m / python3 -c call below needs --workdir set to the
# workspace root (aviation_mas_mvp's parent), or it's a ModuleNotFoundError
# exactly like running the script from the wrong directory on a normal
# machine. No pip install / no package install is needed at all.
WORKDIR=/home/node/.openclaw/workspace

for p in "$FLIGHT_DIR" "$METADATA" "$MODEL"; do
  "${DC[@]}" exec -T openclaw-gateway test -e "$p" \
    || die "Missing in container: $p (check the ../data mount and rebuild)."
done
ok "data/c28_demo + model.joblib visible inside the container."

log "Resetting ./workspace/test_run/ ..."
"${DC[@]}" exec -T openclaw-gateway rm -rf "$RUN_DIR"
"${DC[@]}" exec -T openclaw-gateway mkdir -p "$RUN_DIR"

log "inspect"
"${DC[@]}" exec -T --workdir "$WORKDIR" openclaw-gateway python3 -m aviation_mas_mvp.tools inspect \
  --flight-dir "$FLIGHT_DIR" --metadata "$METADATA" || die "inspect failed"

log "featurize"
"${DC[@]}" exec -T --workdir "$WORKDIR" openclaw-gateway python3 -m aviation_mas_mvp.tools featurize \
  --flight-dir "$FLIGHT_DIR" --metadata "$METADATA" --out "$RUN_DIR/feats.csv" \
  || die "featurize failed"

log "classify"
"${DC[@]}" exec -T --workdir "$WORKDIR" openclaw-gateway python3 -m aviation_mas_mvp.tools classify \
  --feats "$RUN_DIR/feats.csv" --model "$MODEL" --out "$RUN_DIR/preds.json" \
  || die "classify failed"

log "recommend"
"${DC[@]}" exec -T --workdir "$WORKDIR" openclaw-gateway python3 -m aviation_mas_mvp.tools recommend \
  --preds "$RUN_DIR/preds.json" --queue "$RUN_DIR/maintenance_queue.jsonl" --top-k 10 \
  || die "recommend failed"

ok "All four tools ran cleanly against the real data."

echo
log "Running orchestrator_sim oracle (deterministic baseline trace + HITL summary)..."
"${DC[@]}" exec -T --workdir "$WORKDIR" openclaw-gateway python3 -c "
from aviation_mas_mvp import orchestrator_sim
r = orchestrator_sim.run_pipeline(
    flight_dir='$FLIGHT_DIR',
    metadata='$METADATA',
    model='$MODEL',
    workdir='$RUN_DIR/oracle',
    top_k=10,
)
print(r['hitl_summary'])
" || die "orchestrator_sim oracle run failed"

echo
ok "Manual tool-chain test complete."
cat <<EOF

Outputs on the host under: ./workspace/test_run/
  feats.csv, preds.json, maintenance_queue.jsonl   <- direct tool calls above
  oracle/feats.csv, oracle/preds.json, oracle/...   <- orchestrator_sim run

Next: point the agent at agent_task_brief.md and let IT drive these same
four tools via exec, then diff its trace/queue against oracle/ above.
EOF