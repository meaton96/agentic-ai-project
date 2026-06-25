# TOOLS.md — Local Notes (Aviation Predictive-Maintenance MVP)

Exact CLI contract for the four exec tools, plus the gotchas already hit
once during setup — read these before assuming something is broken.

## The four tools (exec; each prints ONE JSON object to stdout)

### 1. inspect

```
python -m aviation_mas_mvp.tools inspect --flight-dir <DIR> --metadata <META.csv>
```

→ `{tool, n_flights, label_balance:{0,1}, n_planes, detected_channels[], n_channels, example_flight_steps, notes}`

### 2. featurize

```
python -m aviation_mas_mvp.tools featurize --flight-dir <DIR> --metadata <META.csv> --out <feats.csv>
```

→ `{tool, rows, n_features, out}` (writes the feature table to `--out`)

### 3. classify

```
python -m aviation_mas_mvp.tools classify --feats <feats.csv> --model <model.joblib> --out <preds.json>
```

→ `{tool, n_scored, n_high, out}` (writes ranked predictions to `--out`)

### 4. recommend (write-only queue; call at most once per run)

```
python -m aviation_mas_mvp.tools recommend --preds <preds.json> --queue <maintenance_queue.jsonl> --top-k 10
```

→ `{tool, n_queued, queue, items[]}` (appends top-k to the JSONL queue)

## Path notes

- All commands run from the workspace root (your default cwd). Paths above
  can be relative to that root — e.g. `data/c28_demo/flights`.
- The output directory for `--out`/`--queue` must already exist; create it
  with a plain `mkdir -p` exec call first if it doesn't (pandas does not
  auto-create parent directories).
- `data/` is mounted read-only. Never attempt to write there.

## Known gotchas (already debugged once — don't re-derive these)

- **A 5th tool, `approve`, exists but is deliberately NOT in your tool set.**
  The whole point of the gate is that you never have authority to commit
  your own recommendations. Don't look for it; don't ask for the token.
- **The model behind the `vllm` provider is the RIT API**, not a literal
  local vLLM — that's just the OpenClaw provider id with working auth
  auto-pickup. Irrelevant to you at the tool-calling level; mentioned here
  only so the naming doesn't cause confusion if you ever see it in logs.
- **No headless companion-node issue**: exec runs directly in this
  container (sandboxing is off), so plain shell calls just work — no device
  pairing needed for tool calls themselves (pairing was only an issue for
  `cron`/`devices` management, not for `exec`).

## Step 5 preview (not yet wired)

`classify`'s body will eventually call a trained Conv-MHSA model instead of
the baseline classifier. The CLI signature and JSON shape will stay
identical — nothing above this line will need to change when that happens.