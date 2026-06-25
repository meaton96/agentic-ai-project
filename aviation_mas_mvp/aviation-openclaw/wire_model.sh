#!/usr/bin/env bash
# Step 2b — wire the RIT API as a model provider and prove ONE real LLM call
# works end-to-end through the gateway. Run this AFTER smoke_test.sh passes.
#
# Known-working pattern (confirmed in a prior RIT fleet sweep with this exact
# API): the OpenClaw provider must be named `vllm` — that's the plugin with
# auto auth pickup (VLLM_API_KEY env var). A provider literally named `rit`
# has no plugin behind it and falls into an auth-profiles.json dead end.
#
# Usage: ./wire_model.sh
set -euo pipefail
cd "$(dirname "$0")"

GATEWAY_URL="http://127.0.0.1:18789"
PASS="[ PASS ]"; FAIL="[ FAIL ]"; INFO="[ .... ]"
log()  { printf '%s %s\n' "$INFO" "$*"; }
ok()   { printf '%s %s\n' "$PASS" "$*"; }
die()  { printf '%s %s\n' "$FAIL" "$*" >&2; exit 1; }

if docker compose version >/dev/null 2>&1; then DC=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then DC=(docker-compose)
else die "Docker Compose not found."; fi

# --- preflight: gateway up and the .env vars we need are present -----------
curl -fsS "${GATEWAY_URL}/healthz" >/dev/null 2>&1 \
  || die "Gateway not healthy. Run ./smoke_test.sh first."
ok "Gateway healthy."

for v in RIT_API_KEY API_URL API_MODEL_NAME; do
  grep -qE "^${v}=.+" .env 2>/dev/null \
    || die "${v} missing or empty in .env — add it, then re-run."
done
ok ".env has RIT_API_KEY, API_URL, API_MODEL_NAME."

# Re-create so the gateway process picks up the new VLLM_API_KEY env mapping
# added to docker-compose.yml (env changes need a container recreate, config
# changes hot-reload but env does not).
log "Recreating gateway to pick up VLLM_API_KEY mapping..."
"${DC[@]}" up -d --force-recreate openclaw-gateway >/dev/null
for _ in $(seq 1 30); do
  curl -fsS "${GATEWAY_URL}/healthz" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS "${GATEWAY_URL}/healthz" >/dev/null 2>&1 || die "Gateway didn't come back healthy after recreate."
ok "Gateway back up."

# --- write the provider config, reading values from the CONTAINER's own ----
# --- environment (so secrets never appear in this script's own argv/log) ---
log "Writing models.providers.vllm config (baseUrl=\$API_URL, model=\$API_MODEL_NAME)..."
"${DC[@]}" exec -T openclaw-gateway sh -c '
  set -e
  node dist/index.js config set models.providers.vllm.baseUrl "$API_URL"
  node dist/index.js config set models.providers.vllm.api "openai-completions"
  node dist/index.js config set models.providers.vllm.apiKey \
    "{\"source\":\"env\",\"provider\":\"default\",\"id\":\"VLLM_API_KEY\"}" --strict-json
  # IMPORTANT: write the whole model entry atomically, not field-by-field.
  # The config schema validates the full models array on every write — a
  # partial {"id": "..."} object (no name/cost/contextWindow/etc.) fails
  # validation immediately, before a follow-up .name call ever runs. This
  # exact shape (reasoning/input/cost/contextWindow/maxTokens all present)
  # matches the confirmed-working RIT-sweep config.
  MODEL_JSON=$(printf "[{\"id\":\"%s\",\"name\":\"%s\",\"reasoning\":false,\"input\":[\"text\"],\"cost\":{\"input\":0,\"output\":0,\"cacheRead\":0,\"cacheWrite\":0},\"contextWindow\":32768,\"maxTokens\":8192}]" "$API_MODEL_NAME" "$API_MODEL_NAME")
  node dist/index.js config set models.providers.vllm.models "$MODEL_JSON" --strict-json --replace
  node dist/index.js config set agents.defaults.model.primary "vllm/$API_MODEL_NAME"
' || die "Config write failed — see output above."
ok "Provider config written (provider id: vllm -> RIT API)."

# Every config-set line above said "Restart the gateway to apply" — these
# provider/model changes do not hot-reload the running process. Restart now.
log "Restarting gateway to apply the new provider/model config..."
"${DC[@]}" restart openclaw-gateway >/dev/null
for _ in $(seq 1 30); do
  curl -fsS "${GATEWAY_URL}/healthz" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS "${GATEWAY_URL}/healthz" >/dev/null 2>&1 || die "Gateway didn't come back healthy after config restart."
ok "Gateway restarted with new config applied."

# --- known gotcha check: a plugin-owned catalog file can silently override --
# --- the baseUrl we just set (bit us once before in the RIT sweep) ---------
CATALOG=./config/agents/main/agent/plugins/vllm/catalog.json
if [[ -f "$CATALOG" ]]; then
  log "Found $CATALOG — checking it isn't overriding baseUrl..."
  CONFIGURED_URL="$(grep -E '^API_URL=' .env 2>/dev/null | cut -d= -f2- || true)"
  if [[ -z "$CONFIGURED_URL" ]]; then
    log "API_URL not found in .env -- skipping catalog comparison (nothing to compare against)."
  # -F: fixed-string match. The URL contains '.' and '/', which are regex
  # metacharacters; treating it as a pattern risks false "matches".
  elif grep -q '"baseUrl"' "$CATALOG" && ! grep -qF "$CONFIGURED_URL" "$CATALOG"; then
    printf '%s %s overrides baseUrl and will win over openclaw.json. Fix:\n' "$FAIL" "$CATALOG" >&2
    printf '   edit %s and set "baseUrl" to your RIT API URL, then re-run.\n' "$CATALOG" >&2
    exit 1
  else
    ok "Catalog baseUrl matches — no override conflict."
  fi
else
  log "No plugin catalog file yet (expected on a fresh setup) — skipping check."
fi

# --- the actual test: one real LLM call, no channel, no delivery -----------
log "Sending one real message through the gateway (openclaw agent --message)..."
REPLY_JSON="$("${DC[@]}" exec -T openclaw-gateway sh -c \
  'node dist/index.js agent --agent main --message "Reply with exactly one word: PONG" --json' \
  2>/tmp/oc_agent_err.txt)" || {
    printf '%s LLM call failed:\n' "$FAIL" >&2; cat /tmp/oc_agent_err.txt >&2; exit 1; }

echo "$REPLY_JSON" > /tmp/oc_agent_reply.json
log "Raw reply saved to /tmp/oc_agent_reply.json"
python3 - "$REPLY_JSON" <<'PY' || true
import json, sys
try:
    d = json.loads(sys.argv[1])
    text = d.get("reply") or d.get("text") or d.get("message") or d
    print("[ .... ] Parsed reply:", text)
except Exception as e:
    print("[ .... ] (could not parse as JSON, printed raw above):", e)
PY

if echo "$REPLY_JSON" | grep -qi "pong"; then
  ok "LLM call succeeded — model responded through the gateway."
else
  printf '%s Got a response but it did not contain PONG — check /tmp/oc_agent_reply.json\n' "$FAIL"
  printf '   (this may still be fine — some models add extra text; read the reply above)\n'
fi

echo
ok "Model wiring step complete."
cat <<EOF

What just happened:
  * Provider "vllm" -> baseUrl \$API_URL (your RIT API), model \$API_MODEL_NAME
  * Auth via VLLM_API_KEY (aliased from RIT_API_KEY in docker-compose.yml)
  * One real agent turn ran through the Gateway with no channel/delivery
    (this is what avoided the old "Channel is required" error)

Next: drop the four real tools into aviation_mas_mvp/, point the agent at
agent_task_brief.md, and wire them as exec calls.
EOF