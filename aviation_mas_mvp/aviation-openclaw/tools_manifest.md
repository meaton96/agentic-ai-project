# Step 2 — Single orchestrator in OpenClaw

This is the exact contract to wire into OpenClaw. It deliberately stops at the
contract (command in, JSON out) rather than guessing your OpenClaw config schema —
plug these four `exec` tools into your existing single-agent setup and point the
system prompt at `agent_task_brief.md` (already written).

## Model
Per your RIT fleet report: **orchestrator = `qwen3-coder:30b`** via the local vLLM
endpoint (fastest 7/7, zero anomalies, strongest tool discipline; comfortable on 96 GB).
RIT API stays as overflow/fallback.

## Build-order reminder (do not skip)
Run the loop **non-headless locally first**. Your own SPEC notes the OpenClaw gotcha:
in headless Docker gateway mode `exec` needs a paired companion node and file ops want
`read`/`write` only. Get the four `exec` calls passing non-headless, *then* port to the
gateway. And before trusting the agent at all: diff its trace against the deterministic
oracle (`orchestrator_sim.run_pipeline`) — same tool order, same key numbers.

## The four tools (exec; each prints one JSON object to stdout)

### 1. inspect
```
python -m aviation_mas_mvp.tools inspect --flight-dir <DIR> --metadata <META.csv>
```
→ `{tool, n_flights, label_balance:{0,1}, n_planes, detected_channels[], n_channels, example_flight_steps, notes}`

### 2. featurize
```
python -m aviation_mas_mvp.tools featurize --flight-dir <DIR> --metadata <META.csv> --out <feats.csv>
```
→ `{tool, rows, n_features, out}`   (writes the feature table to `--out`)

### 3. classify
```
python -m aviation_mas_mvp.tools classify --feats <feats.csv> --model <model.joblib> --out <preds.json>
```
→ `{tool, n_scored, n_high, out}`   (writes ranked predictions to `--out`)

### 4. recommend  (write-only queue; call at most once per run)
```
python -m aviation_mas_mvp.tools recommend --preds <preds.json> --queue <maintenance_queue.jsonl> --top-k 10
```
→ `{tool, n_queued, queue, items[]}`   (appends top-k to the JSONL queue)

## Orchestrator rules (already in agent_task_brief.md)
- One tool at a time; read the JSON before the next step.
- Fixed order: inspect → featurize → classify → recommend. Do not skip.
- On a tool error: report it and stop; never invent results.
- Write the queue at most once per run.
- After `recommend`: STOP and emit the HITL summary. No action beyond queuing.

## HITL summary the agent must produce
Match the structure of `orchestrator_sim.build_hitl_summary`: run scope, findings
(HIGH/MED/LOW counts), top recommendations, a grounded confidence sentence, the
evidence trail (tool → key numbers), and the closing line
"Awaiting technician approval before committing recommendations."

## Step 3 preview — approval gate
Add a second `approve` exec that only writes `approved_queue.jsonl` when handed an
approval token. The agent stops before it; a human supplies the token. That is the
auditability story for the vision paper.

## Step 3 — approval gate (implemented)

A fifth tool, `approve`, exists in `tools.py` / the CLI, but is **deliberately not
included in the agent's `TOOLS_SCHEMA`** (see `agent_harness.py`). The model never
has the authority to commit its own recommendations -- run this yourself, from a
shell, after reading the agent's HITL summary:

```bash
export MAINTENANCE_APPROVAL_TOKEN="whatever-secret-you-choose"   # operator-only, once
python -m aviation_mas_mvp.tools approve --queue maintenance_queue.jsonl \
    --approved approved_queue.jsonl --token "whatever-secret-you-choose"
```

The secret lives ONLY in that environment variable -- never in a tool's JSON
result, never in the HITL summary, never in the model's context. Wrong or
missing token -> refuses, writes nothing. Correct token -> appends only the
lines added to the queue since the last approval (tracked in a small
`<approved>.cursor` sidecar), so re-running is idempotent and a queue that
grows across multiple agent runs gets approved incrementally, not as one
all-or-nothing blob.

## Step 5 preview — Conv-MHSA swap
Replace the body of `classify_flights` with a call to the trained Conv-MHSA model.
The CLI signature and JSON output stay identical, so nothing above this line changes —
same agent, same contract, paper-grade accuracy. That hot-swap is the concrete
"architecture optimization / what-if" demonstration for the AIOS sandbox.