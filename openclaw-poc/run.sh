#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Fix permissions so node (uid=1000) can write to mounted dirs ────────────
echo "→ Setting ownership for uid=1000 (node user inside container)..."
sudo chown -R 1000:1000 config/ workspace/

# ── 2. Pull latest image ───────────────────────────────────────────────────────
echo "→ Pulling openclaw image..."
docker compose pull

# ── 3. Start gateway in background ────────────────────────────────────────────
echo "→ Starting openclaw-gateway..."
docker compose up -d openclaw-gateway

# ── 4. Wait for gateway to become healthy ─────────────────────────────────────
echo "→ Waiting for gateway to be ready..."
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:18789/healthz > /dev/null 2>&1; then
    echo "   Gateway is up!"
    break
  fi
  printf "   attempt %d/30...\r" "$i"
  sleep 2
done

if ! curl -sf http://127.0.0.1:18789/healthz > /dev/null 2>&1; then
  echo "ERROR: Gateway did not become healthy. Check logs:"
  docker compose logs openclaw-poc
  exit 1
fi

# ── 5. Fire the test agent message ────────────────────────────────────────────
echo ""
echo "→ Sending test message to agent..."
docker compose exec openclaw-poc node dist/index.js agent \
  --message "Write a file called hello.txt containing exactly this text: Hello from OpenClaw + vLLM! The agent is working."

# ── 6. Verify the output file was written ─────────────────────────────────────
echo ""
echo "→ Checking workspace/hello.txt..."
if [ -f workspace/hello.txt ]; then
  echo "SUCCESS! Contents:"
  echo "---"
  cat workspace/hello.txt
  echo "---"
else
  echo "WARNING: workspace/hello.txt was not created."
  echo "Check the agent output above — it may have used a different filename."
  echo "Workspace contents:"
  ls -la workspace/
fi
