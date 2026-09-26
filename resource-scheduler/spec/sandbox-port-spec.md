# Porting resource-scheduler into agent-sandbox

**Status (2026-09-16): implemented.** `gate_adapters.py`, `mcp_facts/`,
and `PersistentMailbox` all exist in `resource-scheduler` (168 tests
passing, zero regressions); `agents/rs-propose-*.yaml` and
`pipelines/resource-scheduler-*.yaml` exist in `agent-sandbox`, validated
against the real `AgentSpec`/`PipelineSpec`/`EnvironmentSpec` schemas.
Not yet done: a live end-to-end run against a real model (the build order
in §7 is still the right sequence for that pass), and the continuous
optimization loop stays out of scope per §7 point 5. Two design details
were refined during implementation versus this doc's original draft —
both noted inline below (§1's `PersistentMailbox` signature, §4's
failure-recovery routing) — everything else matches as originally
specified.

Written so `resource-scheduler` can be built out pipeline-by-pipeline in
the sandbox UI the same way `agentic-ml-classification` was, by making
the same three changes that made that port possible:

1. A `gate_adapters.py`-style module exposing this package's stages as
   `GateFn`-shaped functions (`(outputs: dict[str, str]) -> str | tuple[str, str|None]`),
   importable directly by `PipelineSpec.steps[].gate`.
2. Removing internal LLM calls from that trust boundary — every judgment call
   becomes a real sandbox `AgentSpec` (a "propose" agent step, logged like any
   other agent run) sitting between a deterministic "prepare" gate and a
   deterministic "decide" gate.
3. An MCP server exposing this package's harness-computed facts as tools, so
   the propose agent step can fetch what it needs without ever touching
   Python objects, file paths, or the mailbox directly.

Reference implementation for all three: `agentic-ml-classification`'s
`src/agentic_ml/gate_adapters.py` and `src/agentic_ml/mcp_facts/`, now
proven end-to-end in `agent-sandbox` as the `titanic-static-new` pipeline
(`docs/future-work-roadmap.md`'s Phase 9 entry) — read that module's
docstring before starting. Also read `docs/persistent-environment-spec.md`
and Phase 12's entry in the same roadmap: the platform grew a genuine
persistent-session primitive (`EnvironmentSpec`/`GateStep.environment_id`,
Tier 1 shipped 2026-09-16) specifically while scoping this port, and §1
below now builds the mailbox on top of it instead of hand-rolling
persistence next to it.

## 0. Good news first: less work than agentic_ml's own migration needed

agentic_ml's pre-refactor `steps/*_step.py` functions had the LLM call and the
post-proposal validation logic tangled together in one function, so Phase 9
had to *extract* the pure decision logic before it could be called from a
gate. resource-scheduler doesn't have that problem — its post-proposal
decision logic already lives in `environment/*.py` as pure, LLM-free functions
that take `(facts, proposal)` and return a validation result:

- `environment.queue.validate_ranking_proposal` / `is_ranking_score_consistent`
- `environment.allocation.validate_allocation_structure` / `check_constraints` / `build_allocation_events` / `identify_risky_assignments`
- `environment.incidents.validate_recovery_proposal` / `reroutes_avoid_source_machine`
- `environment.policy_evidence.validate_policy_proposal`
- `environment.oversight.parse_oversight_verdict`

`gate_adapters.py`'s `*_decide` functions can call these **directly** — no
new extraction work in `environment/`, no touching `steps/*_step.py` or
`tools/*.py` at all. Those files, `agent_runtime.py`, `model_client.py`, and
every `scripts/run_*_agent.py` stay exactly as they are today, for standalone
CLI use outside the sandbox (same reason agentic_ml kept its own scripts
working unmodified).

## 1. The one real structural difference: the mailbox has to survive

agentic_ml's orchestration is entirely mediated by strings written into
`outputs: dict[str, str]` and JSON manifests on disk — nothing needs to
survive between a `prepare_*` gate call and its `*_decide` call except what's
explicitly persisted. resource-scheduler's `steps/*_step.py` additionally
depend on a **live, in-process `Mailbox` object shared across an entire
orchestrator run** (`a2a/mailbox.py`). In the sandbox, every `PipelineStep`
and `GateStep` executes as an independent call — potentially a separate
process or container entirely, once `SANDBOX_GATE_EXECUTOR=docker` /
`SANDBOX_RUN_EXECUTOR=docker` are in play. Nothing keeps a Python object
alive between them. The in-memory `Mailbox` cannot cross that boundary as-is.

**Fix: bind every mailbox-touching gate to a per-pipeline-run `EnvironmentSpec`
session, and back the mailbox with a small on-disk store scoped to that
session's own container.**

This is `docs/persistent-environment-spec.md`'s Tier 1, shipped 2026-09-16
specifically for "a stateful ... world model that several gate calls in
sequence read and mutate — all pure Python/deterministic-code-side state, no
agent needs to talk to it directly" — this mailbox is exactly that use case,
so use the platform primitive rather than reinventing an ad hoc one next to
it. Two things worth being precise about before designing around it:

- **It does not give in-memory Python continuity.** Per `EnvironmentSpec`'s
  own docstring, each call into a Tier-1 session still runs the gate
  entrypoint as a **fresh process** (`entrypoint.py execute module function
  inputs_b64`, via `container.exec_run()`) — what persists is the
  **container's own filesystem**, not any module-level Python object. A
  disk-backed mailbox is still required; what changes is *where* it lives.
- **What it actually buys**, versus the `RESOURCE_SCHEDULER_DATA_ROOT`-under-
  `/scratch` approach an earlier draft of this spec used: `/scratch` is a
  Docker volume shared by **every pipeline run a given owner ever starts**
  (`docs/future-work-roadmap.md` §1's own "accidental-shared-disk
  persistence" framing) — safe only as long as resource-scheduler code
  namespaces every path under `run_id` itself, and it never gets cleaned up
  automatically. A Tier-1 session container is created fresh per
  `(owner_id, pipeline_run_id, environment_id)` and torn down automatically
  once the run reaches `completed`/`errored` (`PipelineRunManager`'s
  lifecycle hook) — so a mailbox stored in that container's own private
  filesystem is scoped and garbage-collected by construction, with no
  cross-run collision possible and no manual cleanup code to write.

**Declare one `EnvironmentSpec` per pipeline** (each of the three pipelines
in §4 gets its own — sessions are scoped per `pipeline_run_id`, so they
can't be shared across pipelines):

```yaml
environments:
  - id: resource-scheduler-session
    name: resource scheduler mailbox session
    image: agent-sandbox-gate-runtime:latest   # same image throwaway gate
                                                # calls already use — Tier 1
                                                # sessions get the same
                                                # /pkgs (ro) / /scratch (rw)
                                                # mounts, nothing new to build
    kind: gate
    idle_timeout_s: 900
    max_lifetime_s: 14400
```

**Give every gate step that reads or writes the mailbox** an
`environment_id: resource-scheduler-session` field (added to `GateStep` by
this same feature) — concretely: `task_prioritization_decide`,
`prepare_resource_allocation`, `resource_allocation_decide`,
`prepare_decision_oversight`, `decision_oversight_decide`,
`failure_recovery_decide`, `reroute_validation_decide`, `optimization_decide`,
`prepare_oversight`, `oversight_decide`. Gates with no mailbox interaction
(`prepare_load_monitor`, `load_monitor_decide`, `prepare_task_prioritization`,
`prepare_failure_recovery`, `prepare_optimization`) skip it and keep using
the ordinary throwaway-container path — no reason to pay for a session on a
step that only reads df/harness facts. §4's pipeline YAML below is updated
with this field on the steps that need it.

Add one class to `a2a/mailbox.py` (not a new file — same message-type
contracts, same callers as today's in-process `Mailbox`):

```python
class PersistentMailbox:
    """Same send/inbox_for/peek contract as Mailbox, backed by JSON files
    under a small local root (default: RESOURCE_SCHEDULER_SESSION_STATE_ROOT,
    itself defaulting to /var/lib/resource-scheduler-session — deliberately
    NOT under RESOURCE_SCHEDULER_DATA_ROOT/`/scratch`, since it's meant to
    live in a Tier-1 EnvironmentSpec session container's own private,
    per-pipeline-run filesystem rather than the shared per-owner scratch
    volume; see spec §1), namespaced under run_id. One file per
    <recipient>__<message_type>; inbox_for's pop semantics are preserved by
    rewriting the file after consuming. Each send is also emitted as an
    event via on_event, same as Mailbox already does.

    run_id namespacing is defense-in-depth, not the primary isolation
    mechanism (implementation refinement over this doc's original draft,
    which proposed dropping run_id entirely): in the sandbox deployment,
    every gate step touching the mailbox shares one Tier-1 session scoped
    to exactly one pipeline run, so there's only ever one run_id's files
    under this root in practice — but keeping run_id in the path costs
    nothing and keeps this class correct standalone too (a non-Docker/
    in-process deployment, or a local test run, can share one process/
    filesystem across several pipeline runs with no container boundary
    between them at all)."""

    def __init__(self, run_id: str, on_event=None, root: Path | None = None): ...
    def send(self, sender, recipient, message_type, payload) -> Message: ...
    def inbox_for(self, recipient, message_type=None) -> list[Message]: ...
    def peek(self, recipient) -> list[Message]: ...
```

Nothing in `environment/`, `tools/`, or `steps/` needs to know which backend
it's talking to — `gate_adapters.py` is the only caller that ever constructs
a `PersistentMailbox()` instead of a plain `Mailbox()`, and it only does so
inside the mailbox-touching gates listed above. This, plus the
`environments:`/`environment_id:` YAML wiring, is the only new infrastructure
this port needs beyond a direct port of the agentic_ml pattern.

**Known limitation, inherited rather than introduced**: if a paused
pipeline's session gets idle-reaped before it's resumed, the mailbox state
inside it is gone — resume starts a fresh session (§4/§7 of
`docs/persistent-environment-spec.md`, explicitly scoped as the accepted v1
tradeoff, with `docker commit`-based checkpointing noted as a possible,
not-yet-built follow-up). Size each pipeline's `idle_timeout_s` /
`max_lifetime_s` generously enough that a normal end-to-end run (minutes,
not hours) never gets reaped mid-flight; a *paused* run surviving a long
human-review gap is the one case actually worth tuning for.

## 2. `src/resource_scheduler/mcp_facts/` — new package

Near-verbatim port of `agentic_ml.mcp_facts.fact_store` — `write_fact`,
`read_fact`, `read_fact_or_default` against
`runs/<run_id>/facts/<name>.json`, using `resource_scheduler.paths.run_dir`.
No resource-scheduler-specific logic needed here.

Unlike agentic_ml, **skip the `LocalToolProvider`/`McpToolProvider` dual
abstraction** (`provider.py`). agentic_ml needs it because it supports both
in-process and MCP tool-serving as a runtime choice; resource-scheduler's
sandbox port only needs the MCP path (the in-process `ToolCallingAgent` path
stays untouched for the standalone scripts, per §0) — so this would be
speculative generality with no second caller. Go straight from
`tools/*_tool.py`'s existing `build_*_fact` functions to `fact_store.write_fact`
inside each `prepare_*` gate.

`server.py` (FastMCP, run_id-scoped tools, same shape as agentic_ml's):

| MCP tool | fact name | built by (existing, unchanged) |
|---|---|---|
| `get_resource_snapshot` | `resource_snapshot` | `load_monitor_tool.build_resource_snapshot_fact` |
| `get_task_queue_profile` | `task_queue_profile` | `task_prioritization_tool.build_task_queue_fact` |
| `get_allocation_context` | `allocation_context` | `resource_allocation_tool.build_allocation_context_fact` |
| `get_incident_report` | `incident_report` | `failure_recovery_tool.build_incident_fact` |
| `get_policy_evidence` | `policy_evidence` | `optimization_tool.build_policy_evidence_fact` |
| `get_policy_review_bundle` | `policy_review_bundle` | `environment.oversight.build_oversight_review_bundle` |
| `get_decision_review_bundle` | `decision_review_bundle` | `environment.oversight.build_decision_review_bundle` |

Every tool takes `run_id: str` as its only argument, exactly like agentic_ml's
`mcp_facts/server.py`. Skip the bearer-auth/`build_http_app` machinery for
v1 — see §5 on why stdio transport makes it unnecessary here.

## 3. `src/resource_scheduler/gate_adapters.py` — the prepare/decide pairs

Same shape as agentic_ml's: `STEP_*` constants, one `prepare_<stage>` /
`<stage>_decide` pair per agent, `prepare_*` publishes facts and returns the
bare `run_id`, `*_decide` re-reads the manifest by `run_id`, judges the
`propose_*` agent step's output, and does whatever mailbox/event bookkeeping
the current `run_*_step` function does today.

One difference from agentic_ml worth using deliberately: because several of
resource-scheduler's stages are *conditional on a mailbox message existing*
(the current code's `stopped_reason="no_ranking_available"` /
`"no_incidents_detected"` / etc. short-circuits), a `prepare_*` gate here is
allowed to itself return a terminal decision (skipping the propose/decide
steps entirely) instead of always returning `"ready"` the way agentic_ml's
`prepare_*` gates do — there's no LLM call to skip *to* if the precondition
that would make one meaningful isn't met.

| Stage | `prepare_*` | `propose_*` (new agent step) | `*_decide` | Decisions |
|---|---|---|---|---|
| Load Monitor | `prepare_load_monitor` — `build_resource_snapshot_fact`, publish `resource_snapshot` | `propose_load_monitor` | `load_monitor_decide` — `_flags_match` soft check only | `reported` |
| Task Prioritization | `prepare_task_prioritization` — `build_task_queue_fact`, publish `task_queue_profile` | `propose_task_prioritization` | `task_prioritization_decide` — `validate_ranking_proposal` (hard), `is_ranking_score_consistent` (soft), `PersistentMailbox.send(task_ranking → resource_allocation)` on success | `ranked` / `invalid` |
| Resource Allocation | `prepare_resource_allocation` — reads `mailbox.inbox_for("resource_allocation", "task_ranking")`; if empty, short-circuits; else `build_allocation_context_fact`, publish `allocation_context` | `propose_resource_allocation` | `resource_allocation_decide` — `validate_allocation_structure`, `check_constraints`, `build_allocation_events`, `identify_risky_assignments`; `mailbox.send(risky_decision → human_oversight)` if any | `no_ranking` (prepare) / `accepted_clean` / `accepted_with_risk` / `invalid` |
| Reroute Validation | *(no propose — no LLM call today, stays a single deterministic gate)* | — | `reroute_validation_decide` — reads `mailbox.inbox_for("resource_allocation", "reroute_request")`, reuses `check_constraints`/`build_allocation_events`/`identify_risky_assignments`; `mailbox.send(risky_decision → human_oversight)` if any | `no_reroute` / `accepted_clean` / `accepted_with_risk` |
| Failure Recovery | `prepare_failure_recovery` — seed task is a **prior run's `run_id`**; reads that run's accepted-assignments fact, computes before/after snapshot, `build_incident_fact`; itself short-circuits with `no_incidents`/`no_affected_tasks` (no proposal to skip *to* if the precondition isn't met — mirrors `run_failure_recovery_step`'s current early returns) | `propose_failure_recovery` | `failure_recovery_decide` — `validate_recovery_proposal` (hard), `reroutes_avoid_source_machine` (soft), `mailbox.send(reroute_request → resource_allocation)` (this pipeline's own new `run_id`, not the source run's) | prepare: `ready` / `no_incidents` / `no_affected_tasks`; decide: `rerouted` / `invalid` |
| Optimization (one-shot) | `prepare_optimization` — `build_policy_evidence_fact(n_runs)`, publish `policy_evidence`; short-circuits if no run history | `propose_optimization` | `optimization_decide` — `validate_policy_proposal`, `mailbox.send(policy_update_proposal → human_oversight)` | `no_history` / `proposed` / `invalid` |
| Human Oversight (policy) | `prepare_oversight` — reads `mailbox.inbox_for("human_oversight", "policy_update_proposal")`; short-circuits if empty; `build_oversight_review_bundle`, publish `policy_review_bundle` | `propose_oversight` | `oversight_decide` — `parse_oversight_verdict` (degrade-to-`flagged` rule preserved verbatim) | `no_proposal` / `approved` / `rejected` / `flagged` |
| Human Oversight (decision) | `prepare_decision_oversight` — reads `mailbox.inbox_for("human_oversight", "risky_decision")`; short-circuits if empty; `build_decision_review_bundle`, publish `decision_review_bundle` | `propose_decision_oversight` | `decision_oversight_decide` — same `parse_oversight_verdict` | `no_decision` / `approved` / `rejected` / `flagged` |

**Found during implementation, not in the original draft**:
`environment.policy_evidence.collect_policy_evidence` (which
`prepare_optimization` relies on for cross-run evidence) reads three
specifically-named files — `task_prioritization_report.json`,
`resource_allocation_report.json`, `failure_recovery_report.json` — that
only `scripts/run_orchestrator.py`'s CLI path ever wrote. Left alone,
`prepare_optimization` would always see `no_history` for sandbox-driven
runs, since `gate_adapters.py`'s own manifests use a different naming
convention (`<step_id>_manifest.json`) entirely. Fix: `task_prioritization_decide`,
`resource_allocation_decide`, and `reroute_validation_decide` each
additionally write the small legacy-shaped file `collect_policy_evidence`
already knows how to read, purely additive alongside their real manifest
— no change to `environment/policy_evidence.py`'s file-reading contract
or the CLI path's own report shape.

`run_dir`/`run_id` creation: `prepare_load_monitor` calls
`resource_scheduler.cli_common.make_run_dir(None)` once, exactly like
agentic_ml's `prepare_intake` — every later stage in the same pipeline reads
it from the manifest chain. `prepare_failure_recovery` and `prepare_optimization`
each start a **new** `run_id` of their own (see §4), since they're separate
pipelines that reference an existing run only to read one fact out of it.

**Env var**: set `RESOURCE_SCHEDULER_DATA_ROOT` in the repo-root `.env`
(same mechanism as `AGENTIC_ML_DATA_ROOT` — `paths.py`'s `_resolve_root`
already supports this, confirmed present) so `runs/<run_id>` (facts,
reports, transcripts) lands under `resource-scheduler/`, not
`agent-sandbox/runs/`. This is unrelated to, and unaffected by, §1's
`PersistentMailbox` root — the mailbox deliberately does *not* live under
this shared root; see §1.

## 4. Three pipelines, not one

resource-scheduler's own README already treats these as three independently
triggerable flows (each has its own standalone script and synthetic-bypass
flag); the sandbox port should keep that shape rather than forcing everything
into one linear `PipelineSpec`, since Failure Recovery and Optimization both
key off state that isn't a same-run mailbox message (a past run's committed
assignments; N runs of history).

**`resource-scheduler-main`** — the live scheduling loop:

```yaml
id: resource-scheduler-main
name: resource scheduler — main loop
max_steps: 20
environments:
  - id: resource-scheduler-session
    name: resource scheduler mailbox session
    image: agent-sandbox-gate-runtime:latest
    kind: gate
    idle_timeout_s: 900
    max_lifetime_s: 14400
steps:
  - kind: gate
    step_id: prepare_load_monitor
    gate: "resource_scheduler.gate_adapters:prepare_load_monitor"
    on_result: {reported: propose_load_monitor}
  - kind: agent
    step_id: propose_load_monitor
    agent_id: rs-propose-load-monitor
    task_template: "{{steps.prepare_load_monitor.output}}"
  - kind: gate
    step_id: load_monitor_decide
    gate: "resource_scheduler.gate_adapters:load_monitor_decide"
    on_result: {reported: prepare_task_prioritization}
  - kind: gate
    step_id: prepare_task_prioritization
    gate: "resource_scheduler.gate_adapters:prepare_task_prioritization"
    on_result: {reported: propose_task_prioritization}
  - kind: agent
    step_id: propose_task_prioritization
    agent_id: rs-propose-task-prioritization
    task_template: "{{steps.prepare_task_prioritization.output}}"
  - kind: gate
    step_id: task_prioritization_decide
    gate: "resource_scheduler.gate_adapters:task_prioritization_decide"
    environment_id: resource-scheduler-session   # writes task_ranking
    on_result: {ranked: prepare_resource_allocation, invalid: __end__}
  - kind: gate
    step_id: prepare_resource_allocation
    gate: "resource_scheduler.gate_adapters:prepare_resource_allocation"
    environment_id: resource-scheduler-session   # reads task_ranking
    on_result: {ready: propose_resource_allocation, no_ranking: __end__}
  - kind: agent
    step_id: propose_resource_allocation
    agent_id: rs-propose-resource-allocation
    task_template: "{{steps.prepare_resource_allocation.output}}"
  - kind: gate
    step_id: resource_allocation_decide
    gate: "resource_scheduler.gate_adapters:resource_allocation_decide"
    environment_id: resource-scheduler-session   # writes risky_decision
    on_result:
      accepted_clean: prepare_optimization
      accepted_with_risk: prepare_decision_oversight
      invalid: __end__
  - kind: gate
    step_id: prepare_decision_oversight
    gate: "resource_scheduler.gate_adapters:prepare_decision_oversight"
    environment_id: resource-scheduler-session   # reads risky_decision
    on_result: {ready: propose_decision_oversight, no_decision: prepare_optimization}
  - kind: agent
    step_id: propose_decision_oversight
    agent_id: rs-propose-decision-oversight
    task_template: "{{steps.prepare_decision_oversight.output}}"
  - kind: gate
    step_id: decision_oversight_decide
    gate: "resource_scheduler.gate_adapters:decision_oversight_decide"
    environment_id: resource-scheduler-session
    on_result: {approved: prepare_optimization, rejected: prepare_optimization, flagged: prepare_optimization}
  - kind: gate
    step_id: prepare_optimization
    gate: "resource_scheduler.gate_adapters:prepare_optimization"
    on_result: {ready: propose_optimization, no_history: __end__}
  - kind: agent
    step_id: propose_optimization
    agent_id: rs-propose-optimization
    task_template: "{{steps.prepare_optimization.output}}"
  - kind: gate
    step_id: optimization_decide
    gate: "resource_scheduler.gate_adapters:optimization_decide"
    environment_id: resource-scheduler-session   # writes policy_update_proposal
    on_result: {proposed: prepare_oversight, invalid: __end__}
  - kind: gate
    step_id: prepare_oversight
    gate: "resource_scheduler.gate_adapters:prepare_oversight"
    environment_id: resource-scheduler-session   # reads policy_update_proposal
    on_result: {ready: propose_oversight, no_proposal: __end__}
  - kind: agent
    step_id: propose_oversight
    agent_id: rs-propose-oversight
    task_template: "{{steps.prepare_oversight.output}}"
  - kind: gate
    step_id: oversight_decide
    gate: "resource_scheduler.gate_adapters:oversight_decide"
    environment_id: resource-scheduler-session
    on_result: {approved: __end__, rejected: __end__, flagged: __end__}
```

Steps with no `environment_id` (`prepare_load_monitor`, `load_monitor_decide`,
`prepare_task_prioritization`, `prepare_optimization`) never touch the
mailbox and keep running as ordinary throwaway-container gate calls — the
session only starts (lazily, on first reference — §4 of
`docs/persistent-environment-spec.md`) once `task_prioritization_decide`
actually needs it.

Note the fan-in at `prepare_optimization`: both `accepted_clean` (no risky
decisions) and every `decision_oversight_decide` outcome route there — the
oversight review is advisory-only today (matches current behavior: nothing
downstream branches on its verdict), so the pipeline proceeds regardless.

**`resource-scheduler-failure-recovery`** — separate pipeline, seed task is
an existing main-run's `run_id`. Its own `environments:` block (a session is
scoped to *this* pipeline's `pipeline_run_id`, so it cannot reuse
`resource-scheduler-main`'s session even though the id/image are identical):

```
environments: [resource-scheduler-session]   # same shape as §4's main pipeline
prepare_failure_recovery                     # unmarked -- reads the SOURCE run's
  {ready: propose_failure_recovery,          # accepted_assignments fact by run_id,
   no_incidents: __end__,                    # not this pipeline's own mailbox
   no_affected_tasks: __end__}
propose_failure_recovery
failure_recovery_decide [env]                # writes reroute_request
  {rerouted: reroute_validation_decide, invalid: __end__}
reroute_validation_decide [env] (gate only, no propose step)
  {accepted_with_risk: prepare_decision_oversight, accepted_clean|no_reroute: __end__}
prepare_decision_oversight [env] → propose_decision_oversight → decision_oversight_decide [env] → __end__
```

(Implementation refinement over this doc's original draft, which had
`no_incidents`/`no_affected_tasks` as `failure_recovery_decide` outcomes —
they're actually `prepare_failure_recovery`'s own short-circuits, computed
before any proposal exists, matching `run_failure_recovery_step`'s current
early-return behavior exactly. `failure_recovery_decide` itself only ever
judges a proposal, so its outcomes are just `rerouted`/`invalid`.)

`[env]` marks the same `environment_id: resource-scheduler-session` field
as §4 — `failure_recovery_decide` writes `reroute_request`,
`reroute_validation_decide` reads it and may write `risky_decision`,
`prepare_decision_oversight`/`decision_oversight_decide` read/round out
that review. `prepare_failure_recovery` itself stays unmarked — it reads a
fact out of the *source* run (via `mcp_facts`/the fact store, not this
pipeline's mailbox), not this pipeline's own session state.

**`resource-scheduler-optimization`** — one-shot only, run manually/on a
schedule. Its own `environments:` block too:

```
environments: [resource-scheduler-session]
prepare_optimization → propose_optimization → optimization_decide [env]
  {proposed: prepare_oversight, no_history|invalid: __end__}
prepare_oversight [env] → propose_oversight → oversight_decide [env] → __end__
```

## 5. AgentSpec YAMLs — the `propose_*` steps

> **Superseded (2026-09-26): don't use the stdio binding below.** The
> sandbox can't spawn `python -m resource_scheduler.mcp_facts.server`, and
> that module no longer runs a server anyway. The agentic-ml-facts MCP
> server (the `agentic-mcp` deployment) serves these tools over HTTP next
> to agentic_ml's, so bind each propose agent to it instead:
>
> ```yaml
> mcp_servers:
>   - name: agentic-ml-facts
>     transport: http
>     connection: {url: "https://agentsandbox.gccis.rit.edu/agentic-ml-facts/mcp"}
>     credential_ref: <the credential holding AGENTIC_ML_MCP_AUTH_TOKEN>
>     allowed_tools: ["get_task_queue_profile"]
>     logging_policy: full
> ```
>
> Gates write facts to `GATE_SCRATCH_DIR/resource-scheduler/runs/`
> (see `paths.py`), which that server reads from the same scratch volume.

Eight new `agents/*.yaml` files, one per row in §3's table, each following
`file-writer.yaml`'s stdio-MCP pattern (the only working precedent in this
repo for an agent that actually uses `mcp_servers`) rather than
`mcp-gate-demo`'s pattern of a gate spawning its own client — that demo
wires MCP at the *gate* level because its demo agent has none; here the
facts genuinely belong to the *agent's* tool surface, so `AgentSpec.mcp_servers`
(resolved by `strands_adapter._mcp_client_for`, which already handles
spawning/tearing down a stdio subprocess per agent run) is the right,
already-built primitive — no custom subprocess-spawning code needed:

```yaml
id: rs-propose-task-prioritization
name: Task Prioritization (propose)
system_prompt: >
  <copied verbatim from resource-scheduler/prompts/task_prioritization.md,
  updated to call get_task_queue_profile(run_id) instead of a no-arg tool>
model:
  base_url: ${SANDBOX_MODEL_BASE_URL}
  model_name: ${SANDBOX_MODEL_NAME}
  api_key_ref: sandbox-model-api-key
  temperature: 0.0
mcp_servers:
  - name: resource_scheduler_facts
    transport: stdio
    connection:
      command: <path to sandbox-core's venv python, or sys.executable equivalent>
      args: ["-m", "resource_scheduler.mcp_facts.server"]
    allowed_tools: ["get_task_queue_profile"]
    logging_policy: full
max_turns: 4
```

`task_template` for each step is just `"{{steps.prepare_<stage>.output}}"` —
the bare `run_id` string, verbatim, per the same rule agentic_ml's
`propose_*` steps follow: an agent step substitutes a prior output into its
prompt as-is and can't read the sandbox filesystem, so it must never be
handed a path. The agent's system prompt tells it to call its one tool with
that run_id and respond with the same JSON contract it already produces
today (`ranked_task_ids`/`score_breakdown`/`reasoning`, etc. — the
`configs/schemas/*.json` files already document every one of these
contracts and don't need to change).

`allowed_tools` scopes each propose agent to exactly the one fact tool its
current prompt already relies on (mirrors today's one-tool-per-agent design)
— tighter than strictly necessary, but matches existing intent and costs
nothing to keep.

**Auth**: skip bearer-token/HTTP entirely for v1. `stdio` means Strands
spawns a fresh server subprocess per agent run and tears it down after —
there's no shared network listener to secure. Reach for
`build_http_app`/`RESOURCE_SCHEDULER_MCP_AUTH_TOKEN` only if a persistent,
concurrently-shared fact server becomes necessary later (e.g. if the
continuous optimization loop ever needs to feed a live sandbox pipeline
instead of running standalone). This also happens to be exactly the
still-open Phase 1 item in the sandbox roadmap ("a real second example agent
with a live MCP server (stdio, local only) — proves Strands' MCP wiring
end-to-end") — this port satisfies it as a side effect.

## 6. Package installation

Same as `agentic_ml`, per README.md's "Porting your own Python package in
for gates" section (already written with `resource-scheduler` as its named
example — that section still applies for making `resource_scheduler`
importable; what's new here is *how* the gate module looks once installed):

```bash
sandbox-core/.venv/bin/python -m pip install -e ../resource-scheduler
```

**No thin wrapper module needed in `agent-sandbox/gates/`.** That's the
old pattern (`gates/agentic_ml_static.py`, now fully commented-out/retired),
needed only because agentic_ml's pre-refactor step functions didn't match
`GateFn`'s shape. `gate_adapters.py` is written *as* `GateFn`-shaped
functions living inside the package itself, resolved directly by dotted
path — `"resource_scheduler.gate_adapters:prepare_task_prioritization"` —
the same way `agentic_ml.gate_adapters:prepare_intake` already is. Update
`docs/docker.md`'s "Bringing your own gates" walkthrough and the
`docker-compose.override.yml.example` comments once this lands — both still
describe the retired wrapper-module pattern.

## 7. Suggested build order

Mirrors the repo's own "Agent build order" (validate one agent against real
data before adding the next), plus the sandbox roadmap's own sequencing
logic (prove MCP wiring on something simple before the mailbox complicates
it):

1. **Load Monitor** — no mailbox, no branching, terminal. Proves
   `prepare_*` → real `AgentSpec` w/ stdio MCP → `*_decide` end-to-end
   before anything else is at stake.
2. **Task Prioritization → Resource Allocation** — the first real
   `PersistentMailbox`-over-`EnvironmentSpec` round trip (write in one gate
   call, read in a later, separate one, both `exec`'d into the same session
   container). Tier 1 itself is already verified against a real Docker
   daemon (`docs/future-work-roadmap.md`'s Phase 12 entry — session reuse,
   `/scratch`-equivalent persistence, and teardown all confirmed), but
   *this* round trip — a gate-owned, non-`/scratch` JSON store inside that
   session — has no existing precedent anywhere in `agent-sandbox`; get it
   solid before building on top of it.
3. **Failure Recovery + Reroute Validation** as their own pipeline — proves
   reading one run's fact from a *different* run's `prepare_*` gate, and
   reusing `check_constraints`/`build_allocation_events` unchanged in a
   pure-gate (no propose) step.
4. **Optimization (one-shot) + both Human Oversight branches** — proves the
   fan-in case (oversight reachable from three different producers).
5. **Continuous optimization loop: leave out of scope for now.** It's a
   cross-run hill-climbing search that itself re-invokes two other agents
   many times per iteration with its own persisted history file —
   representing that as a single `PipelineSpec` run would fight
   `max_steps`'s safety-cap intent rather than fit it. Keep running it as
   `scripts/run_optimization_loop.py`, standalone, same as today. Revisit
   only if there's a concrete need to drive it from inside the sandbox.

## 8. Testing

- `PersistentMailbox` gets the same round-trip tests `Mailbox` already
  implies (send → inbox_for pops → peek doesn't) plus one proving state
  survives a fresh instance pointed at the same root directory (the actual
  property being added — this alone doesn't need Docker or a real session,
  just two separate `PersistentMailbox` instances against the same temp
  dir). A separate, `-m docker`-gated integration test (mirroring
  `runner/tests/test_docker_integration.py`'s existing "Tier-1 persistent
  sessions" section) should confirm the same round trip actually holds
  across two real `exec_in_session` calls into one session container, not
  just two Python objects in a test process.
- `fact_store` round-trip tests, direct port of agentic_ml's
  `test_mcp_facts.py` shape — no provider-parity test needed here since
  there's only one provider (§2).
- `gate_adapters.py`'s `*_decide` functions: unit-test with canned proposal
  JSON strings, asserting the right decision string and the right mailbox
  side effects — no live model needed, matching resource-scheduler's
  existing test suite's own "no LLM/API key required" property.
- Full pipeline runs against a real model are a manual sandbox-UI smoke
  test per stage (per the build order above), not a CI test — same as how
  `titanic-static-copy` was proven out for agentic_ml.

## 9. Open decisions for you to make before/while building

- **Failure Recovery as a separate pipeline vs. a manual re-trigger of the
  main one** — §4 assumes separate; flag if you'd rather model it
  differently.
- **Whether to keep `run_optimization_loop.py` out of the sandbox
  permanently**, or revisit later — §7 point 5 recommends deferring, not
  ruling out.
- **`allowed_tools` scoping** — kept tight (one tool per agent) to match
  current prompts; loosen only if a propose step's prompt evolves to need
  more than one fact.
- **`PersistentMailbox`'s storage root inside the session container** — §1
  suggests a fixed local path (e.g. `/var/lib/resource-scheduler-session/mailbox/`)
  outside the `/pkgs`/`/scratch` mounts; the exact path is a small decision,
  not yet pinned down, and should land wherever `resource_scheduler`'s own
  config conventions (`paths.py`/env vars) most naturally take an override.
- **`idle_timeout_s`/`max_lifetime_s` tuning per pipeline** — §1's defaults
  (900s / 14400s, `EnvironmentSpec`'s own schema defaults) are a starting
  point, not measured against this pipeline's actual run times; revisit
  once real runs establish how long a full main-loop run or a paused
  human-review gap actually takes.
