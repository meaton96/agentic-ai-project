#!/usr/bin/env bash
# Trim the tool surface down to what this agent actually needs: `exec` (the
# four pipeline tools, per AGENTS.md/TOOLS.md) and `read` (harmless, in case
# it wants to sanity-check a file). tools.profile=coding currently pulls in
# cron, sessions_*, web_search, web_fetch, memory_*, process, edit, write --
# none of which this single-purpose agent's contract uses. Per the actual
# systemPromptReport dump, those unused tool schemas alone account for
# ~13,000 of the system prompt's ~23,000 characters -- against a 16,384-token
# model, that's roughly half the context budget spent on nothing. This is
# almost certainly why compaction was firing on literally the first turn of
# a brand-new session, which is what exposed the separate "Already compacted"
# bug in OpenClaw's post-run CLI compaction pass.
#
# Run AFTER wire_model_local.sh (or wire_model.sh) has wired a provider.
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

DENY_LIST='["gateway","sessions_spawn","subagents","cron","sessions_list","sessions_history","sessions_send","sessions_yield","web_search","web_fetch","memory_search","memory_get","update_plan","process","edit","write"]'
# Keeping "read" available (cheap at 304 chars) in case the agent ever wants
# to sanity-check a file directly. Remove it from the list above too if you
# want the absolute minimum (exec only).

log "Trimming tools.deny to exclude everything this agent doesn't need..."
"${DC[@]}" exec -T openclaw-gateway node dist/index.js config set tools.deny "$DENY_LIST" --strict-json --replace \
  || die "Config write failed."
ok "tools.deny updated."

log "Restarting gateway to apply..."
"${DC[@]}" restart openclaw-gateway >/dev/null
for _ in $(seq 1 30); do
  curl -fsS http://127.0.0.1:18789/healthz >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS http://127.0.0.1:18789/healthz >/dev/null 2>&1 || die "Gateway didn't come back healthy."
ok "Gateway restarted."

log "Re-testing with a fresh session (same PONG test as before)..."
PING_SESSION="ping-trimmed-$(date +%s)"
REPLY_JSON="$("${DC[@]}" exec -T openclaw-gateway sh -c \
  "node dist/index.js agent --agent main --session-key '$PING_SESSION' --message \"Reply with exactly one word: PONG\" --json" \
  2>/tmp/oc_trimmed_err.txt)" || {
    printf '%s Still failing:\n' "$FAIL" >&2; cat /tmp/oc_trimmed_err.txt >&2; exit 1; }

echo "$REPLY_JSON" > /tmp/oc_trimmed_reply.json
if echo "$REPLY_JSON" | grep -qi pong; then
  ok "PONG succeeded with the trimmed tool set."
else
  printf '%s Got a response but no PONG -- check /tmp/oc_trimmed_reply.json\n' "$FAIL"
fi

echo
cat <<EOF
Next: pull the new systemPromptChars / tools.schemaChars to confirm the
shrink actually happened (sanity check, optional):
  cat config/agents/main/sessions/sessions.json | python3 -c \\
    "import json,sys; d=json.load(sys.stdin); [print(k, v.get('systemPromptReport',{}).get('systemPrompt',{}).get('chars'), v.get('systemPromptReport',{}).get('tools',{}).get('schemaChars')) for k,v in d.items()]"

If PONG succeeded above, re-run ./run_agent_test.sh for the real workflow test.
EOF