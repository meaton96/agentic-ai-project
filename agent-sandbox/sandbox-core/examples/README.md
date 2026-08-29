# Examples

1. Create a credential store and add the key `file-writer.yaml` references (`sandbox-model-api-key`): `mkdir -p ~/.sandbox && printf 'sandbox-model-api-key: sk-...\n' > ~/.sandbox/credentials.yaml` (or point `SANDBOX_CREDENTIALS_PATH` at a different file).
2. Set the two env vars the spec reads at load time: `export SANDBOX_MODEL_BASE_URL=https://your-openai-compatible-endpoint/v1` and `export SANDBOX_MODEL_NAME=your-model-name`.
3. From `agent-sandbox/sandbox-core/`, run: `sandbox run examples/agents/file-writer.yaml --task "Write a short haiku about agents to output.txt, then read it back to confirm."` (requires `npx`/Node available on PATH — it launches `@modelcontextprotocol/server-filesystem` on first run).
4. Output lands in `examples/workspace/output.txt`; the full event log lands in `./runs/<run_id>/events.jsonl`.
