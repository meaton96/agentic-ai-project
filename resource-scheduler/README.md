# Resource Scheduler (Use Case II)

Multi-agent resource scheduler for 6G-enabled factory floors — Use Case II of the Agentic Sandbox project. Companion to [../agentic-ml-classification](../agentic-ml-classification), which structural conventions this project mirrors (see "Relationship to agentic-ml-classification" below).

## Scenario

Hundreds of tasks (machining jobs, QA inspections, robotic arm operations, data uploads) compete for shared resources — compute nodes, network slices, bandwidth, machine time — in a 6G-connected smart factory. A network of specialized agents schedules these resources in real time, adapting to congestion, failures, and priority shifts, while remaining auditable and interruptible.

## Architecture: environment + direct agent-to-agent (A2A)

Unlike the ML classification pipeline (where everything routes through a deterministic harness, with no direct agent-to-agent communication), this project deliberately tests **direct A2A messaging** for the negotiation-shaped parts of the workflow — Task Prioritization → Resource Allocation, Failure Recovery → Resource Allocation — while keeping a **deterministic environment/state layer** (`src/resource_scheduler/environment/`) as the single source of truth for machine/slice state and constraint arithmetic. Agents never get to assert a number the environment didn't compute, and every allocation decision is logged as a traceable event — but agents talk to each other directly for negotiation, not only through a central orchestrator.

The A2A mechanism itself lives in `src/resource_scheduler/a2a/mailbox.py`: a minimal in-process, per-run message bus (no async, no network — deliberately dumb so the interesting question, whether direct agent channels change behavior versus routing through a deterministic gate, can be tested without also debugging message-bus infrastructure). Every send is still logged as an event; "direct" doesn't mean "unaudited". Task Prioritization sends its validated ranking directly to Resource Allocation's inbox — no orchestrator in between — and every proposed assignment produces a traceable event (`environment/allocation.py::build_allocation_events`) recording which agent fired, what signal triggered it (the task's upstream `final_score`), what was decided, and whether it passed the constraint gate.

## Agent build order

Built one agent at a time, validated against synthetic data before moving on — same approach used for the ML classification pipeline.

1. **Load Monitor** — read-only, flags congestion/SLA violations. ✅ implemented (`steps/load_monitor_step.py`)
2. **Task Prioritization** — re-ranks the pending queue by urgency/energy cost/availability. ✅ implemented (`steps/task_prioritization_step.py`) — first agent to send an A2A message (see below)
3. **Resource Allocation** — assigns machines/slices under hard constraints. ✅ implemented (`steps/resource_allocation_step.py`) — first agent to *read* from the A2A mailbox; `environment/allocation.py::check_constraints` is the real gate (no Maintenance-machine assignment, no slice over-capacity, checked cumulatively within a batch), independent of the agent's stated rationale. `scripts/run_resource_allocation_agent.py --synthetic-ranking` lets it be tested standalone without agent #2.
4. **Failure Recovery** — detects faults, reroutes via a second A2A hop into Resource Allocation. ✅ implemented (`steps/failure_recovery_step.py`) — detects a machine's fresh transition into a fault state (`environment/incidents.py::diff_snapshots`, comparing two replay snapshots), identifies which committed tasks sit on it, proposes reroutes, and sends them to Resource Allocation's mailbox as a `reroute_request`. The receiving side (`steps/resource_allocation_step.py::run_reroute_validation_step`) is deliberately **not** a second LLM call — it re-validates each reroute against the exact same `check_constraints` gate Resource Allocation already applies to its own proposals. Short-circuits with zero LLM calls when there's nothing to recover from (`no_incidents_detected` / `no_affected_tasks`). `scripts/run_failure_recovery_agent.py --synthetic-committed` allows isolated testing without agents #2/#3.
5. **Optimization** — background policy-tuning loop, same shape as the ML pipeline's Phase-9 drift/retrain-decision subsystem. ✅ implemented, **two modes**:
   - *One-shot* (`steps/optimization_step.py`, `scripts/run_optimization_agent.py`) — a single LLM-reasoned recommendation from existing run history. Takes no task-table input at all; aggregates outcomes by reading the report JSON files agents #2-4 already write to `runs/<run_id>/` (`environment/policy_evidence.py`). May only propose adjustments to parameters actually wired to a CLI flag (`queue_size`, `snapshot_window`, `slice_capacity`) — never an invented knob. Short-circuits with zero LLM calls when there's no run history yet.
   - *Continuous* (`environment/policy_search.py`, `scripts/run_optimization_loop.py`) — the actual "runs a background loop... to tune allocation policies over time" the spec describes, which the one-shot mode never was. Greedy hill-climbing (not full RL — three small-integer parameters don't warrant that machinery) over the same three parameters, scored against the acceptance/validity rates `policy_evidence.py` already computes. Each iteration genuinely re-runs Task Prioritization + Resource Allocation at a candidate setting; a persisted history file (`artifacts/policy_search_history.json`) lets the search improve *across* separate invocations, not just within one run. `--synthetic` exercises the whole loop — including the real `check_constraints` gate — with zero LLM calls, useful for validating the search mechanism itself without live tokens.

   Neither mode ever applies its own proposal — a genuine improvement found by the search loop is sent to Human Oversight via the exact same `policy_update_proposal` mailbox path the one-shot mode uses (reused, not duplicated) — the third A2A hop.
6. **Human Oversight** — approval gate on both policy changes and risky individual scheduling decisions, same shape as the ML pipeline's Verification agent. ✅ implemented (`steps/oversight_step.py`) — two review paths now: `run_oversight_step` reviews Optimization's policy proposal against the real underlying evidence (not just the proposer's own characterization of it); `run_decision_oversight_step` reviews a batch of risky_decision messages sent by Resource Allocation or Failure Recovery whenever an accepted assignment/reroute lands on an Overloaded machine (`environment/allocation.py::identify_risky_assignments`) — the fourth and fifth A2A hops, satisfying the spec's "surfaces scheduling decisions above a risk threshold for human approval" beyond just policy changes. Both paths are advisory: the decision under review is already committed by the time Human Oversight sees it (widening this into a real blocking approval gate would be a materially bigger workflow change — pause/resume semantics on a decision mid-flight — not attempted here). Same "degrade to flagged, never silently approve" rule as the ML pipeline's Verification agent, including when the LLM invents a verdict string outside the allowed enum. `scripts/run_human_oversight_agent.py --synthetic-proposal` allows isolated testing without agent #5.

All six agents are now scaffolded. What's not built: no orchestrator tying all six into one run (each is independently invocable via its own script and chains only where a real data dependency requires it — e.g. Resource Allocation needs Task Prioritization's ranking), and no actual "apply" mechanism for an approved policy change (Human Oversight's approval is currently just a verdict, not a config write) — both reasonable next steps once the individual agents have been validated against a live model.

## Data

`datasets/raw/industrial_scheduling_dataset.csv` — 1000 task records: `Task_ID, Machine_ID, Network_Slice_ID, Task_Type, Execution_Time, Machine_Status, Reallocation, Latency_ms, Sensor_Temp_C, URLLC_Score, Target`.

**Known data-quality issue:** `Execution_Time`, `Latency_ms`, `Sensor_Temp_C`, and `URLLC_Score` are constant across every row in this file — zero variance, zero signal for anything that needs to *detect* change (confirmed directly, see `tests/test_environment_state.py::test_real_dataset_has_known_constant_columns`). `environment/state.py::load_task_table` injects small seeded jitter into those columns by default (`inject_variance=True`) so early development has something real to threshold against; this is never silent — every snapshot's `synthetic_variance_injected` field says exactly which columns are fake. Pass `--no-inject-variance` once a livelier feed replaces this file.

There's also no timestamp column, so `environment/state.py::compute_snapshot` treats row order (by `Task_ID`) as a proxy for time and replays the most recent `window` rows as "current" state. This is a stand-in for a real live queue/stream — swap the data source in `load_task_table` without touching any agent or step code.

## Quickstart

```bash
pip install -e .
cp .env.example .env   # fill in RIT_API_KEY at minimum
set -a; source .env; set +a
python scripts/run_load_monitor_agent.py --data datasets/raw/industrial_scheduling_dataset.csv
```

Run tests (no LLM/API key required — these only exercise the deterministic environment layer):

```bash
pytest tests/
```

Run the full six-agent chain in one process:

```bash
python scripts/run_orchestrator.py --data datasets/raw/industrial_scheduling_dataset.csv
```

## Viewing results

Two tools, both read whatever's already in `runs/<run_id>/` — no live model call needed to use either:

- `python scripts/render_run_report.py [--run-id <id>]` — a single self-contained HTML file (`runs/<run_id>/report.html`), open it in any browser or attach it to an email.
- `python scripts/generate_showcase_notebook.py [--run-id <id>]` — `notebooks/agent_showcase.ipynb`, one section per agent (what it does, its gate logic, its actual transcript from that run) with real outputs pre-populated. Regenerate it against any newer run to refresh.

Both default to the most recent run under `runs/` if `--run-id` is omitted.

**A real infrastructure quirk worth knowing about**, found by actually running this live: the RIT GenAI endpoint returns intermittent `504 Gateway Time-out` errors under agentic tool-calling load, and it correlates with request size — Task Prioritization's larger tool payload (originally ranking 15 tasks) hit this far more often than Load Monitor's smaller one. `queue_size` defaults to 8 for exactly this reason (see `environment/queue.py`'s docstring); lower it further with `--queue-size` if timeouts persist, or switch to `--use-local` for more reliable iteration.

## Relationship to agentic-ml-classification

This project deliberately reuses that project's infrastructure shape rather than importing it directly, since the two are meant to stay independently deployable:

| agentic-ml-classification | resource-scheduler | Role |
|---|---|---|
| `src/agentic_ml/agent_runtime.py` | `src/resource_scheduler/agent_runtime.py` | Minimal tool-calling agent loop |
| `src/agentic_ml/model_client.py` | `src/resource_scheduler/model_client.py` | OpenAI-compatible client (RIT / gateway / local) |
| `src/agentic_ml/cli_common.py` | `src/resource_scheduler/cli_common.py` | Script plumbing: run dirs, tracing, transcripts |
| `src/agentic_ml/prompt_loader.py` | `src/resource_scheduler/prompt_loader.py` | Loads `prompts/<agent>.md` |
| `src/agentic_ml/harness/` | `src/resource_scheduler/environment/` | Deterministic ground truth; agents never compute facts themselves |
| `src/agentic_ml/tools/*_tool.py` | `src/resource_scheduler/tools/*_tool.py` | Binds environment facts to LLM-callable tools |
| `src/agentic_ml/steps/*_step.py` | `src/resource_scheduler/steps/*_step.py` | One file per agent: prompt → tool call → validate → result |
| `configs/schemas/*.json` | `configs/schemas/*.json` | Documentation-grade JSON Schemas for each agent's proposal shape |

Not yet ported here: an orchestrator (static or dynamic), `mcp_facts/` MCP integration, and `priors/` — these get added once more than one agent exists and there's an actual coordination/negotiation problem to solve. `runs/` is gitignored here exactly as in the ML pipeline.
