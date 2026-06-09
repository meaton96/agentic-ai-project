# OpenClaw Codebase Documentation Agent

An OpenClaw agent that maintains up-to-date markdown documentation for your codebase — per-file docs with AST-parsed class/method maps, plus a high-level architecture overview. Designed to be fed back to an LLM as context when working on new features, replacing the need to dump raw source files.

---

## Prerequisites

- Docker + Docker Compose
- Your codebase must be accessible on the host filesystem (git repo recommended but not required)
- A running vLLM instance serving your model (e.g. `Qwen/Qwen3-VL-32B-Instruct` on port 8000)

---

## Setup (5 steps)

### 1. Edit `docker-compose.yml`

Set your codebase path in the volumes section:

```yaml
volumes:
  - /absolute/path/to/your/project:/root/.openclaw/workspace/codebase:ro
```

If your model name differs from `Qwen/Qwen3-VL-32B-Instruct`, update:
- `DOC_MODEL` env var in `docker-compose.yml`
- The model `id` and `primary` keys in `openclaw.json`

### 2. Edit `openclaw.json` (if needed)

The model ID must match exactly what vLLM loaded. Check with:
```bash
curl http://localhost:8000/v1/models | python3 -m json.tool
```
Update `"id"` and `"primary"` in `openclaw.json` to match.

### 3. Set a gateway token

In `docker-compose.yml`, replace `your-local-token-here` with any string you like:
```yaml
- OPENCLAW_GATEWAY_TOKEN=my-secret-token
```

### 4. Build and start the agent

```bash
docker compose up -d --build
```

Verify it started cleanly:
```bash
docker compose logs -f
```

You should see:
```
✓  [gateway] agent model: vllm/... (thinking=off, fast=off)
✓  [agents/tool-policy] tool policy removed N tool(s) via tools.profile (coding): ...
✓  [skills] loaded 1 skill(s): docs-generator, ...
✓  [gateway] ready
```

If `[skills]` is missing, check that `workspace/skills/docs-generator/SKILL.md` exists inside the container:
```bash
docker compose exec docs-agent ls /root/.openclaw/workspace/skills/
```

### 5. Run your first scan

```bash
./trigger.sh full
```

Then watch progress:
```bash
tail -f workspace/docs/.last_run.log
```

The initial full scan will take a while — roughly 1–3 minutes per 10 files depending on your model speed and file sizes.

---

## Triggering Scans

The agent polls every 2 minutes for a sentinel file. You trigger it by running:

```bash
./trigger.sh full            # Document everything from scratch
./trigger.sh incremental     # Only files changed since last scan
./trigger.sh architecture    # Rebuild ARCHITECTURE.md and FILE_INDEX.md only
```

Or write the sentinel directly:
```bash
echo "incremental" > workspace/docs/.scan_requested
```

**Incremental scan logic (in priority order):**
1. `git diff --name-only <last_commit>..HEAD` — most accurate, requires git
2. mtime manifest comparison — fallback if git fails
3. Full scan — fallback if no previous state exists

---

## Output Structure

```
workspace/docs/
├── ARCHITECTURE.md          ← High-level overview: structure, components, data flow
├── FILE_INDEX.md            ← Quick-reference table linking every file to its doc
├── files/
│   ├── src/
│   │   ├── module.py.md     ← Per-file docs (mirrors source tree, .md appended)
│   │   └── other.cs.md
│   └── ...
├── .last_run.log            ← stdout from the most recent generator run
├── .scan_log.txt            ← One-line entry per completed scan
├── .last_scan_commit        ← Git commit hash stored after each run
└── .file_manifest.json      ← Mtime manifest for non-git fallback
```

### Per-file doc format

Each `*.md` file contains:

```markdown
# `src/module.py`
**Language:** Python | **Lines:** 312 | **Last Scanned:** 2025-01-15

## Purpose
What this file does and its role in the project.

## Dependencies
- `bpy` — Blender Python API for scene manipulation
- ...

## Classes

### `RenderPassManager`
Manages the two-pass render pipeline. Inherits: `object`

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(scene_path: Path, output_dir: Path)` | Initialize with paths |
| `render_ambient` | `() -> Path` | Ambient-only render pass |

## Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `setup_render_settings` | `(scene, mode: str) -> None` | Configure Blender render |

## Notes
- Blender 5.1.2 removed compositor node API — two-pass workaround required
```

---

## Using the Docs with an LLM

For targeted feature work, give the LLM the relevant file docs instead of raw source:

```
Here is the architecture overview:
[paste ARCHITECTURE.md]

Here are the relevant file docs:
[paste docs/files/src/relevant_module.py.md]
[paste docs/files/src/other_module.cs.md]

Task: Add X feature to Y module...
```

This typically uses 3–10x less context than pasting raw source while retaining all the structural information the LLM needs.

---

## Configuration Reference

All env vars in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DOC_MODEL` | `Qwen/Qwen3-VL-32B-Instruct` | Model ID for doc generation (must match vLLM) |
| `DOC_MAX_FILE_CHARS` | `12000` | Truncate source files larger than this before sending to LLM |
| `DOC_MAX_TOKENS` | `2048` | Max response tokens per file doc |
| `DOC_VLLM_TIMEOUT` | `300` | Seconds to wait per vLLM call |
| `CODEBASE_PATH` | `<workspace>/codebase` | Path inside container where source is mounted |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM API base URL |

---

## Running the Generator Directly

You can invoke `doc_generator.py` directly without the agent, which is useful for testing or scripting:

```bash
# From inside the container
docker compose exec docs-agent python3 /root/.openclaw/workspace/tools/doc_generator.py --full

# Or specific files
docker compose exec docs-agent python3 /root/.openclaw/workspace/tools/doc_generator.py \
    --files src/pipeline.py src/shadow_detector.py
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| No `[skills]` log line at startup | Verify `workspace/skills/docs-generator/SKILL.md` exists and `cron.enabled: true` in openclaw.json |
| Generator runs but produces no output | Check `workspace/docs/.last_run.log` for vLLM error messages |
| `<think>` blocks in docs | Ensure `chat_template_kwargs.enable_thinking: false` is set in both openclaw.json and your vLLM launch flags |
| Incremental scan re-docs everything | No `.last_scan_commit` or `.file_manifest.json` found — expected on first run; subsequent runs will be faster |
| Very large files produce poor docs | Lower `DOC_MAX_FILE_CHARS` to force more truncation, or split large files in your source |
| vLLM timeout errors | Increase `DOC_VLLM_TIMEOUT` and `timeoutSeconds` in openclaw.json |
| Container can't reach `localhost:8000` | Verify `network_mode: "host"` is set in docker-compose.yml |
