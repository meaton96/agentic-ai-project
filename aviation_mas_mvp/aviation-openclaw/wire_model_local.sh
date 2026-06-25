#!/usr/bin/env bash
# Switch the "vllm" OpenClaw provider from the RIT API to the LOCAL vLLM
# server already running on the host -- faster, and removes the Cloudflare/
# WAF flakiness while we verify the agentic workflow. Run instead of
# wire_model.sh. To switch back later, just run wire_model.sh again --
# whichever script ran most recently wins, nothing to manually revert.
#
# Usage: ./wire_model_local.sh
set -euo pipefail
cd "$(dirname "$0")"

PASS="[ PASS ]"; FAIL="[ FAIL ]"; INFO="[ .... ]"
log()  { printf '%s %s\n' "$INFO" "$*"; }
ok()   { printf '%s %s\n' "$PASS" "$*"; }
die()  { printf '%s %s\n' "$FAIL" "$*" >&2; exit 1; }

if docker compose version >/dev/null 2>&1; then DC=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then DC=(docker-compose)
else die "Docker Compose not found."; fi

curl -fsS http://127.0.0.1:18789/healthz >/dev/null 2>&1 || die "Gateway not healthy. Start it first."
ok "Gateway healthy."

# Default to the standard vLLM bind (host 0.0.0.0, default port 8000, no
# --port in the launch command). Override by adding LOCAL_VLLM_API_URL=...
# to .env if you ever run vLLM on a different port.
LOCAL_VLLM_API_URL="$(grep -E '^LOCAL_VLLM_API_URL=' .env 2>/dev/null | cut -d= -f2- || true)"
LOCAL_VLLM_API_URL="${LOCAL_VLLM_API_URL:-http://host.docker.internal:8000/v1}"
LOCAL_VLLM_API_URL="${LOCAL_VLLM_API_URL%/}"  # strip a trailing slash (avoids //models)
log "Using local vLLM endpoint: $LOCAL_VLLM_API_URL"

log "Checking reachability and asking vLLM what model it's actually serving..."
MODELS_JSON="$("${DC[@]}" exec -T openclaw-gateway curl -fsS "$LOCAL_VLLM_API_URL/models" \
  2>/tmp/oc_local_vllm_err.txt)" || {
    printf '%s Could not reach %s from inside the container:\n' "$FAIL" "$LOCAL_VLLM_API_URL" >&2
    cat /tmp/oc_local_vllm_err.txt >&2
    printf '   Check vllm is bound to 0.0.0.0 (not 127.0.0.1) and the port matches.\n' >&2
    exit 1
  }
MODEL_ID="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' <<<"$MODELS_JSON" 2>/dev/null)" \
  || die "Reached the endpoint but couldn't parse a model id from: $MODELS_JSON"
ok "vLLM is serving: $MODEL_ID"

log "Writing models.providers.vllm config (local vLLM, no auth needed)..."
"${DC[@]}" exec -T openclaw-gateway sh -c "
  set -e
  node dist/index.js config set models.providers.vllm.baseUrl '$LOCAL_VLLM_API_URL'
  node dist/index.js config set models.providers.vllm.api 'openai-completions'
  node dist/index.js config set models.providers.vllm.apiKey 'local-vllm-no-auth-required'
  MODEL_JSON=\$(printf '[{\"id\":\"%s\",\"name\":\"%s\",\"reasoning\":false,\"input\":[\"text\"],\"cost\":{\"input\":0,\"output\":0,\"cacheRead\":0,\"cacheWrite\":0},\"contextWindow\":16384,\"maxTokens\":8192}]' '$MODEL_ID' '$MODEL_ID')
  node dist/index.js config set models.providers.vllm.models \"\$MODEL_JSON\" --strict-json --replace
  node dist/index.js config set agents.defaults.model.primary 'vllm/$MODEL_ID'
" || die "Config write failed -- see output above."
ok "Provider config written -> local vLLM ($MODEL_ID)."

log "Restarting gateway to apply..."
"${DC[@]}" restart openclaw-gateway >/dev/null
for _ in $(seq 1 30); do
  curl -fsS http://127.0.0.1:18789/healthz >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS http://127.0.0.1:18789/healthz >/dev/null 2>&1 || die "Gateway didn't come back healthy after restart."
ok "Gateway restarted."

log "Sending one real message through the gateway (openclaw agent --message)..."
REPLY_JSON="$("${DC[@]}" exec -T openclaw-gateway sh -c \
  'node dist/index.js agent --agent main --message "Reply with exactly one word: PONG" --json' \
  2>/tmp/oc_local_agent_err.txt)" || {
    printf '%s LLM call failed:\n' "$FAIL" >&2; cat /tmp/oc_local_agent_err.txt >&2; exit 1; }
echo "$REPLY_JSON" > /tmp/oc_local_agent_reply.json

if echo "$REPLY_JSON" | grep -qi pong; then
  ok "Local vLLM call succeeded."
else
  printf '%s Got a response but no PONG -- check /tmp/oc_local_agent_reply.json\n' "$FAIL"
fi

echo
ok "Now wired to the LOCAL vLLM endpoint."
cat <<EOF

Model:    $MODEL_ID
Endpoint: $LOCAL_VLLM_API_URL

Next: re-run ./run_agent_test.sh for the real workflow test -- it should
be both faster and free of the Cloudflare/WAF flakiness from the RIT API.

To switch back to the RIT API later: ./wire_model.sh
EOF