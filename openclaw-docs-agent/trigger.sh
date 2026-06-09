#!/usr/bin/env bash
# trigger.sh — Request a codebase documentation scan
#
# Usage:
#   ./trigger.sh                 # incremental scan (default)
#   ./trigger.sh full            # full scan — document every file from scratch
#   ./trigger.sh incremental     # document only files changed since last scan
#   ./trigger.sh architecture    # rebuild ARCHITECTURE.md and FILE_INDEX.md only
#
# The OpenClaw agent picks up the request within 2 minutes (cron interval).
# Watch progress with: tail -f workspace/docs/.last_run.log

set -euo pipefail

SCAN_TYPE="${1:-incremental}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="${SCRIPT_DIR}/workspace/docs"
SENTINEL="${DOCS_DIR}/.scan_requested"

case "$SCAN_TYPE" in
    full|incremental|architecture) ;;
    *)
        echo "Usage: $0 [full|incremental|architecture]"
        echo ""
        echo "  full          Regenerate docs for all files (slow — use for first run or major refactor)"
        echo "  incremental   Document only files changed since last scan (git diff or mtime)"
        echo "  architecture  Rebuild ARCHITECTURE.md and FILE_INDEX.md from existing file docs"
        exit 1
        ;;
esac

# Check agent is running
if ! docker compose ps --status running 2>/dev/null | grep -q "docs-agent"; then
    echo "⚠  Warning: docs-agent container doesn't appear to be running."
    echo "   Start it with: docker compose up -d"
    echo "   (Writing sentinel anyway in case it starts later.)"
fi

mkdir -p "$DOCS_DIR"
printf '%s' "$SCAN_TYPE" > "$SENTINEL"

echo "✓ Requested: ${SCAN_TYPE} scan"
echo "  Agent will pick this up within ~2 minutes."
echo ""
echo "  Monitor:  tail -f ${DOCS_DIR}/.last_run.log"
echo "  History:  cat ${DOCS_DIR}/.scan_log.txt"
echo "  Docs out: ${DOCS_DIR}/"
