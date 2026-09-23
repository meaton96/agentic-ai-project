# Build the PdM data server: Postgres + MCP tools + bulk export

You are setting up a new, standalone service on this machine. It holds
predictive-maintenance datasets in Postgres and serves them to an LLM agent
platform (agent-sandbox, running on a different RIT server) in two ways:

1. **An MCP server** agents use to *explore* data: small, read-only, logged.
2. **A bulk export HTTP endpoint** deterministic jobs use to *pull* whole
   datasets for model training.

Read this whole document before starting. Work in small verified steps, and
confirm with me before installing system packages, running anything with
`sudo`, or opening firewall ports.

---

## 1. Context you need

### The consumer: agent-sandbox

- Agents are LLM agents defined in YAML. Each one can bind MCP servers:

  ```yaml
  mcp_servers:
  - name: pdm-data
    transport: http                      # MCP Streamable HTTP
    connection: {url: "http://<this-host>:<port>/mcp"}
    credential_ref: pdm-data-planner     # resolved from the user's credential store
    allowed_tools: [list_datasets, describe_dataset, column_stats]
    logging_policy: full                 # sandbox logs every tool call + result
  ```

  The sandbox sends the resolved credential as `Authorization: Bearer <token>`
  on every MCP request. So **per-agent credentials = per-agent bearer tokens**,
  and this service decides what each token may see.
- The sandbox builds a fresh MCP client for every agent run. Use **stateless**
  Streamable HTTP; don't depend on sessions surviving between requests.
- Deterministic pipeline steps ("gates" and long-running "jobs") are plain
  Python in Docker containers. A job will call the export endpoint with its
  own bearer token (the sandbox side for passing credentials into jobs is
  being built separately; not your concern).

### Network constraints (important)

- Sandbox containers reach the outside world **only through an HTTP egress
  proxy** that supports CONNECT tunnels and plain HTTP forward-proxy
  requests on an allowlist of ports. **No raw Postgres protocol can reach
  this machine from the sandbox**. Everything must go over HTTP.
- **HTTPS is currently blocked** somewhere upstream (an IT ticket is open;
  port 80/443 appear filtered, so ACME certificate issuance fails). So:
  - Serve **plain HTTP on a configurable non-standard port**, default
    **8766**.
  - Write the service so TLS can be added later with config only: uvicorn
    `--ssl-keyfile/--ssl-certfile`, or a reverse proxy in front. No code
    changes.
  - Tokens therefore travel in cleartext on the internal network for now.
    Keep every token read-only and narrowly scoped, and plan to rotate all
    of them once TLS is on.
- Postgres itself must **not** be reachable from outside this machine (bind
  to localhost or a private Docker network). Only the service port is
  exposed.
- The machine has no host firewall active (`ufw` is off). Don't change that
  without asking. Tell me what you'd recommend instead.

### The first datasets: NGAFID-MC (aviation maintenance)

From Yang, LaBella & Desell, "Predictive Maintenance for General Aviation
Using Convolutional Transformers" (arXiv 2110.03757). Two long-format CSVs,
one row per second of flight, many flights concatenated:

| File | Size | Flights | Label counts |
|---|---|---|---|
| `C28.csv` | 4.2 GB | 5,089 | `before_after=0`: 2,814 / `=1`: 2,275 |
| `C37.csv` | 2.0 GB | 2,416 | `before_after=0`: 1,432 / `=1`: 984 |

C28 is ~28.8M rows (mean flight length ~5,650 s, range 1,803–21,926); C37 is
roughly half that. I'll tell you where the CSVs are on this machine. Ask if
you don't know.

Columns, in file order:

```
volt1, volt2, amp1, amp2, FQtyL, FQtyR, E1 FFlow, E1 OilT, E1 OilP, E1 RPM,
E1 CHT1, E1 CHT2, E1 CHT3, E1 CHT4, E1 EGT1, E1 EGT2, E1 EGT3, E1 EGT4,
OAT, IAS, VSpd, NormAc, AltMSL, id, plane_id, split, date_diff, before_after
```

- 23 sensor channels (floats, ~1% empty/NaN; keep them NULL, don't fill).
- `id`: flight (sequence) id. A flight's rows are contiguous and in time
  order. **There is no timestamp column; row order within a flight is the
  time axis**, so preserve it explicitly.
- `plane_id`: aircraft. Evaluation must be plane-disjoint.
- `split`: the paper's 5 cross-validation folds (0–4), plane-disjoint.
- `before_after`: the label. **0 = pre-maintenance (the positive class in the
  paper), 1 = post-maintenance.** Verified against `date_diff`: label 0
  flights have `date_diff` ∈ {1, 2}, label 1 flights have ∈ {−2, −1}.
- `date_diff`: **leaks the label** (its sign is the label). It must be
  flagged as leaky everywhere, and excluded from exports by default.

**Stay generic.** NGAFID is the first dataset, not the only one. Nothing in
the service code may hardcode these column names. Column roles come from a
per-dataset catalog entry written at load time. Later datasets may be
tabular (one row per example, no sequences) or long-format like this one.

---

## 2. What to build

One Python service (FastAPI + the official `mcp` Python SDK's FastMCP,
mounted into the same app), Postgres 16, a loader CLI, and a token CLI.
Prefer Docker Compose (Postgres + service) if Docker is available here;
otherwise systemd. Check what exists first and tell me.

Suggested layout (adjust if you have a better one, and say why):

```
pdm-data-server/
  pyproject.toml
  docker-compose.yml
  .env.example
  README.md                 # runbook: setup, load, tokens, backup, TLS later
  src/pdm_data/
    config.py               # env-driven settings
    db.py                   # pool, per-request role switching
    catalog.py              # dataset catalog read/write
    auth.py                 # token verify -> principal (name, role, scopes, datasets)
    audit.py                # audit log writer
    mcp_tools.py            # FastMCP tool definitions
    export.py               # streaming export
    app.py                  # FastAPI app: /healthz, /v1/..., /mcp mount
    cli.py                  # pdm-data load-csv | token create/list/revoke | datasets
  sql/                      # schema migrations (plain SQL, applied in order)
  tests/
```

### 2.1 Database layout

- Schema `catalog`: table `datasets`, one row per dataset:
  - `id` (e.g. `ngafid_c28`), description, source file, `loaded_at`
  - `content_hash` (hash of the loaded data; changes if reloaded)
  - `row_count`, `sequence_count`
  - `kind`: `long_format` | `tabular`
  - `column_roles` (JSON): `sequence_id`, `order` (the row-index column),
    `group`, `label`, `fold`, `channels` (list), and `excluded`, a list of
    `{column, reason}`, e.g. `{date_diff, "sign equals the label"}`
  - `label_info` (JSON): classes, positive class, meaning of each value
  - `column_names` (JSON): original CSV name → SQL name (e.g.
    `E1 CHT1` → `e1_cht1`)
- Schema `data`: one table per dataset. For long-format data, add
  `row_idx bigint` (the row's position within its sequence, 0-based) and
  index `(sequence_id, row_idx)`, plus the group and fold columns. Also
  build a small `data.<dataset>__sequences` table: one row per sequence
  with group, fold, label, length, and any excluded columns (so agents can
  ask cheap questions without scanning 28M rows).
- Schema `auth`: `tokens` (see 2.3).
- Schema `audit`: `events` (see 2.5).

### 2.2 Loader CLI

```
pdm-data load-csv --dataset ngafid_c28 --csv /path/C28.csv \
  --kind long_format --sequence-column id --group-column plane_id \
  --label-column before_after --positive-label 0 \
  --label-meaning "0=pre-maintenance,1=post-maintenance" \
  --fold-column split --exclude "date_diff:sign equals the label" \
  --description "NGAFID-MC C28 (intake gasket), 1 Hz, 23 channels"
```

- Stream with Postgres `COPY`; never read the whole file into memory. Add
  `row_idx` while streaming.
- Check that each sequence's rows are contiguous and that label, group and
  fold are constant within a sequence. Fail loudly with the offending id.
- Load into a staging table and swap it in at the end, so a failed load
  leaves the previous version intact.
- Build the sequences table, compute `content_hash`, and write the catalog
  row.
- Report timing. Expect minutes, not hours, for C28. If it's slower, find
  out why before moving on.
- Must also work for a tabular CSV (`--kind tabular`, no sequence column).

### 2.3 Auth: tokens → Postgres roles

- The service connects to Postgres as a login role with almost no rights of
  its own. For each request it runs `SET LOCAL ROLE <role>` inside a
  transaction, where `<role>` comes from the token. **Postgres grants are
  the enforcement**, not string checks in Python.
- Roles are `NOLOGIN` and `SELECT`-only on the dataset tables they're
  granted, with `default_transaction_read_only = on` and a
  `statement_timeout`.
- Token CLI:

  ```
  pdm-data token create --name sandbox-planner --role pdm_explore \
    --datasets ngafid_c28,ngafid_c37 --scopes mcp --expires 90d
  pdm-data token create --name sandbox-train-job --role pdm_export \
    --datasets ngafid_c28 --scopes export --expires 90d
  pdm-data token list
  pdm-data token revoke sandbox-planner
  ```

  - Print the token **once**, at creation. Store only a hash (e.g. SHA-256
    with a public prefix id for lookup).
  - `scopes` gate the surface: `mcp` for tool calls, `export` for the bulk
    endpoint.
  - `datasets` restricts which datasets the token can name (checked in the
    service *and* enforced by the role's grants).
  - Expired or revoked tokens get 401. A wrong scope or dataset gets 403.
    Error messages must never echo the token.

### 2.4 MCP tools (read-only, small results)

Mount FastMCP's Streamable HTTP app at `/mcp`, stateless. Authenticate every
request with middleware that reads the bearer token and puts the principal
in a context variable the tools read. Every tool:

- has a precise docstring and typed parameters (these become the tool
  descriptions agents see, so write them for an LLM reader);
- returns compact JSON, capped at ~50 KB (truncate with an explicit
  `"truncated": true` and a note);
- runs under the token's role with a short statement timeout (~30 s);
- turns failures into clear messages, never stack traces.

Tools:

| Tool | Returns |
|---|---|
| `list_datasets()` | datasets this token may use: id, description, kind, row/sequence counts |
| `describe_dataset(dataset)` | columns with SQL types and original names, column roles, **excluded columns with reasons**, label meaning and positive class, fold scheme |
| `sequence_summary(dataset)` | label balance overall and per fold, sequences per group, length stats per label |
| `column_stats(dataset, column, by="label"\|"fold"\|null)` | count, null fraction, mean, std, min, p10/p50/p90, max, optionally grouped |
| `sample_sequences(dataset, n, label?, fold?, seed)` | sequence ids with their group, fold, label and length |
| `get_sequence(dataset, sequence_id, columns?, max_points=512)` | one sequence's values, downsampled by averaging windows to at most `max_points` rows |
| `query(sql, max_rows=200)` | results of one read-only `SELECT`/`WITH` statement |

For `query`: parse with `sqlglot` and reject anything but a single
SELECT/WITH before sending it. Wrap it to enforce `max_rows`. Rely on the
read-only role as the real guarantee; the parser only gives better error
messages.

### 2.5 Bulk export (for jobs)

```
GET /v1/datasets                       -> datasets for this token (export scope)
GET /v1/datasets/{id}/manifest         -> catalog entry incl. content_hash
GET /v1/datasets/{id}/export?format=parquet|csv
        &columns=a,b,c                 (default: all except excluded)
        &include_excluded=false
        &folds=0,1,2&sequence_ids=...  (optional filters)
```

- Stream: server-side cursor, fetch in batches (~100k rows), write Parquet
  row groups with `pyarrow` (or CSV). Memory stays flat regardless of
  dataset size.
- Rows are ordered by `(sequence_id, row_idx)` for long-format data.
- Response headers carry `X-Dataset-Id`, `X-Content-Hash`, and the row count
  when it's cheap to know. Jobs cache exports under the content hash and
  only re-download when it changes.
- Must work through a plain HTTP forward proxy (standard HTTP/1.1, chunked
  transfer). Prove it with `curl -x http://<some-proxy> ...` or a local
  `tinyproxy`.
- Excluded (leaky) columns are left out unless `include_excluded=true`, and
  that request is recorded in the audit log.

### 2.6 Audit log

Every MCP tool call and every export records: timestamp, token name
(**never the token**), role, endpoint/tool, arguments (including full SQL
text), rows and bytes returned, duration, and error if any. Write to
`audit.events` and to a JSONL file. Add a CLI to tail and filter it
(`pdm-data audit --token sandbox-planner --since 1h`).

---

## 3. Testing

Automated (pytest, against a throwaway Postgres: a compose test profile or
`testcontainers`):

- a token for dataset A can't read dataset B through any tool or endpoint;
- writes and DDL fail even through `query` (`INSERT`, `DROP`, `SET ROLE`,
  multiple statements, `COPY`, function calls with side effects);
- expired and revoked tokens get 401, wrong scope 403, and no response or
  log line contains a token;
- row and size caps and truncation flags work, and the statement timeout
  fires;
- the loader rejects non-contiguous sequences and inconsistent labels, and
  a failed load leaves the previous table intact;
- export row counts and a sample of values match a small fixture CSV
  exactly, including NULLs and row order;
- the MCP tools are listed and callable through the real `mcp` client over
  Streamable HTTP with a bearer header;
- audit rows are written for success and failure.

Against the real data, after loading C28 and C37:

- sequence counts and label counts match the table in §1;
- no plane appears in two folds;
- a full C28 Parquet export streams through a proxy with flat memory. Report
  its time, size, and the service's peak RSS.

---

## 4. Deliverables and report back

When done, send me:

1. The host/IP and port the service listens on, and whether you could
   reach it from another machine on the network.
2. The loaded datasets with `content_hash`, row counts, sequence counts,
   label counts, and load times.
3. The exact commands to create two tokens: one explore-only MCP token and
   one export-only token.
4. A ready-to-paste sandbox agent binding (YAML like §1's) and a short
   Python example of a job downloading an export through `HTTP_PROXY` with
   a bearer token and caching it by content hash.
5. Anything you'd recommend for TLS and firewalling once IT unblocks ports.

## 5. Not in scope

- Any change to agent-sandbox. Note what it will need instead, e.g. adding
  this port to the egress proxy's `EGRESS_ALLOWED_PORTS` and this host to
  users' egress allowlists.
- Write access of any kind through the service.
- Training, feature engineering, or model code.
- Hardcoding NGAFID column names anywhere outside the load command.
