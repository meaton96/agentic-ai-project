"""
gate_adapters.py's prepare_*/*_decide gates: resource_scheduler's six
agents with every internal LLM call removed, driven here by hand-written
proposals standing in for an external agent -- mirrors the sibling
agentic-ml-classification project's tests/test_gate_adapters.py.

Two things worth proving, not just exercising:

1. Nothing here can call a model -- ModelClient.call is patched to raise
   in every test (a structural guard, even though gate_adapters.py
   imports ModelClient nowhere at all), and every gate's hard-validation
   logic (validate_ranking_proposal / validate_allocation_structure /
   check_constraints / validate_recovery_proposal / validate_policy_proposal
   / parse_oversight_verdict) is exercised via canned JSON strings only.
2. PersistentMailbox actually crosses a call boundary: a mailbox.send()
   inside one gate function call and the matching mailbox.inbox_for() in
   a LATER, wholly separate call are two independent PersistentMailbox
   instances (never sharing Python state), standing in for the sandbox's
   fresh-process-per-gate-call model -- the one thing an in-process
   Mailbox could never survive. Exercised across three real hops:
   task_prioritization_decide -> prepare_resource_allocation,
   resource_allocation_decide -> prepare_decision_oversight, and
   failure_recovery_decide -> reroute_validation_decide.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from resource_scheduler import gate_adapters as ga
from resource_scheduler.cli_common import make_run_dir
from resource_scheduler.mcp_facts.fact_store import read_fact
from resource_scheduler.model_client import ModelClient

# tid -> (machine_id, network_slice_id), chosen so total per-slice load
# (4 already "current" per environment/allocation.py's compute_slice_load
# over this 8-task batch, see dataset_csv below, plus 3 more of these)
# stays comfortably under DEFAULT_SLICE_CAPACITY=8 on both slices.
_RISKY_PLACEMENT = {
    "T003": ("M01", "NS1"), "T004": ("M01", "NS2"),
    "T005": ("M01", "NS1"), "T006": ("M01", "NS2"),
    "T007": ("M03", "NS1"),  # Overloaded -- not blocked, but risky
    "T008": ("M03", "NS2"),  # Overloaded -- not blocked, but risky
}
_CLEAN_PLACEMENT = {
    "T003": ("M01", "NS1"), "T004": ("M01", "NS2"),
    "T005": ("M01", "NS1"), "T006": ("M01", "NS2"),
    "T007": ("M01", "NS1"), "T008": ("M01", "NS2"),
}


@pytest.fixture(autouse=True)
def isolated_run_root(tmp_path, monkeypatch):
    for var in ("RESOURCE_SCHEDULER_DATA_ROOT", "RESOURCE_SCHEDULER_RUNS_DIR",
                "RESOURCE_SCHEDULER_ARTIFACTS_DIR", "RESOURCE_SCHEDULER_DATASETS_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    # PersistentMailbox's default root is an absolute, real-filesystem
    # constant (see a2a/mailbox.py) -- exactly what a Tier-1 EnvironmentSpec
    # session container would override via this same env var (see
    # spec/sandbox-port-spec.md §1); tests must not touch the real path.
    monkeypatch.setenv("RESOURCE_SCHEDULER_SESSION_STATE_ROOT", str(tmp_path / "session"))


@pytest.fixture(autouse=True)
def no_llm_calls(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("gate_adapters made an LLM call")
    monkeypatch.setattr(ModelClient, "call", refuse)


@pytest.fixture
def dataset_csv(tmp_path):
    # First two rows are padding (outside the queue_size=8 pending window);
    # T003..T010 are the pending queue. M01 Active, M02 Idle, M03
    # Overloaded, M04 Maintenance -- gives Load Monitor real flags to
    # narrate, Resource Allocation a blocked machine (M04) and a risky-but-
    # legal one (M03), and Failure Recovery a genuine incident once the
    # "before" snapshot (row 0 only) is compared against the "after" one
    # (the full table): M03/M04 have no prior status to have transitioned
    # FROM, so both count as fresh incidents.
    rows = [
        dict(Task_ID="T001", Machine_ID="M01", Network_Slice_ID="NS1", Task_Type="Welding", Execution_Time=5.0, Machine_Status="Active", Reallocation="No", Latency_ms=2.0, Sensor_Temp_C=60.0, URLLC_Score=0.95, Target=0),
        dict(Task_ID="T002", Machine_ID="M01", Network_Slice_ID="NS1", Task_Type="Welding", Execution_Time=5.5, Machine_Status="Active", Reallocation="No", Latency_ms=2.1, Sensor_Temp_C=60.5, URLLC_Score=0.95, Target=0),
        dict(Task_ID="T003", Machine_ID="M01", Network_Slice_ID="NS1", Task_Type="Welding", Execution_Time=6.0, Machine_Status="Active", Reallocation="No", Latency_ms=2.2, Sensor_Temp_C=61.0, URLLC_Score=0.94, Target=0),
        dict(Task_ID="T004", Machine_ID="M01", Network_Slice_ID="NS1", Task_Type="Cutting", Execution_Time=7.0, Machine_Status="Active", Reallocation="Yes", Latency_ms=2.3, Sensor_Temp_C=61.5, URLLC_Score=0.93, Target=1),
        dict(Task_ID="T005", Machine_ID="M02", Network_Slice_ID="NS2", Task_Type="Welding", Execution_Time=8.0, Machine_Status="Idle", Reallocation="No", Latency_ms=2.4, Sensor_Temp_C=62.0, URLLC_Score=0.92, Target=0),
        dict(Task_ID="T006", Machine_ID="M02", Network_Slice_ID="NS2", Task_Type="Cutting", Execution_Time=9.0, Machine_Status="Idle", Reallocation="No", Latency_ms=2.5, Sensor_Temp_C=62.5, URLLC_Score=0.91, Target=0),
        dict(Task_ID="T007", Machine_ID="M03", Network_Slice_ID="NS1", Task_Type="Welding", Execution_Time=10.0, Machine_Status="Overloaded", Reallocation="Yes", Latency_ms=9.0, Sensor_Temp_C=63.0, URLLC_Score=0.80, Target=1),
        dict(Task_ID="T008", Machine_ID="M03", Network_Slice_ID="NS1", Task_Type="Cutting", Execution_Time=11.0, Machine_Status="Overloaded", Reallocation="No", Latency_ms=9.5, Sensor_Temp_C=63.5, URLLC_Score=0.79, Target=1),
        dict(Task_ID="T009", Machine_ID="M04", Network_Slice_ID="NS2", Task_Type="Welding", Execution_Time=12.0, Machine_Status="Maintenance", Reallocation="No", Latency_ms=3.0, Sensor_Temp_C=64.0, URLLC_Score=0.90, Target=0),
        dict(Task_ID="T010", Machine_ID="M04", Network_Slice_ID="NS2", Task_Type="Cutting", Execution_Time=13.0, Machine_Status="Maintenance", Reallocation="No", Latency_ms=3.1, Sensor_Temp_C=64.5, URLLC_Score=0.89, Target=0),
    ]
    path = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


# -- helpers: drive the pipeline forward to a given stage, no assertions --


def _prepare_through_load_monitor_decide(dataset_csv):
    _, lm_run_id = ga.prepare_load_monitor({"__task__": dataset_csv})
    authoritative = read_fact(lm_run_id, "resource_snapshot")["flags"]
    proposal = json.dumps({"flags": authoritative, "narrative_summary": "ok"})
    _, manifest_path = ga.load_monitor_decide({
        ga.STEP_PREPARE_LOAD_MONITOR: lm_run_id, ga.STEP_PROPOSE_LOAD_MONITOR: proposal,
    })
    return manifest_path


def _prepare_through_task_prioritization(dataset_csv):
    lm_manifest = _prepare_through_load_monitor_decide(dataset_csv)
    _, run_id = ga.prepare_task_prioritization({ga.STEP_LOAD_MONITOR_DECIDE: lm_manifest})
    return run_id


def _task_prioritization_decide_ranked(dataset_csv):
    run_id = _prepare_through_task_prioritization(dataset_csv)
    task_ids = [t["task_id"] for t in read_fact(run_id, "task_queue_profile")["pending_tasks"]]
    proposal = json.dumps({
        "ranked_task_ids": task_ids,
        "score_breakdown": {tid: {"final_score": len(task_ids) - i} for i, tid in enumerate(task_ids)},
        "reasoning": "descending by position",
    })
    decision, manifest_path = ga.task_prioritization_decide({
        ga.STEP_PREPARE_TASK_PRIORITIZATION: run_id, ga.STEP_PROPOSE_TASK_PRIORITIZATION: proposal,
    })
    assert decision == "ranked", decision
    return run_id, manifest_path


def _prepare_through_resource_allocation(dataset_csv):
    run_id, tp_manifest = _task_prioritization_decide_ranked(dataset_csv)
    decision, ra_run_id = ga.prepare_resource_allocation({ga.STEP_TASK_PRIORITIZATION_DECIDE: tp_manifest})
    assert decision == "ready", decision
    assert ra_run_id == run_id  # same run throughout -- the PersistentMailbox round trip's proof point
    return run_id


def _resource_allocation_decide_with_placement(dataset_csv, placement, expected_decision):
    run_id = _prepare_through_resource_allocation(dataset_csv)
    fact = read_fact(run_id, "allocation_context")
    assignments, rejected = [], []
    for t in fact["ranked_tasks"]:
        tid = t["task_id"]
        if tid in placement:
            machine_id, slice_id = placement[tid]
            assignments.append({"task_id": tid, "machine_id": machine_id, "network_slice_id": slice_id, "rationale": "placed"})
        else:
            rejected.append({"task_id": tid, "reason": "source machine in maintenance"})
    proposal = json.dumps({"assignments": assignments, "rejected": rejected})
    decision, manifest_path = ga.resource_allocation_decide({
        ga.STEP_PREPARE_RESOURCE_ALLOCATION: run_id, ga.STEP_PROPOSE_RESOURCE_ALLOCATION: proposal,
    })
    assert decision == expected_decision, decision
    return run_id, manifest_path


def _prepare_optimization_with_history(dataset_csv):
    _task_prioritization_decide_ranked(dataset_csv)  # leaves a task_prioritization_report.json behind
    decision, opt_run_id = ga.prepare_optimization({})
    assert decision == "ready", decision
    return opt_run_id


def _optimization_decide_proposed(dataset_csv):
    opt_run_id = _prepare_optimization_with_history(dataset_csv)
    proposal = json.dumps({"policy_updates": {"queue_size": 6}, "evidence": "ranking looks solid", "recommend_apply": True})
    decision, manifest_path = ga.optimization_decide({
        ga.STEP_PREPARE_OPTIMIZATION: opt_run_id, ga.STEP_PROPOSE_OPTIMIZATION: proposal,
    })
    assert decision == "proposed", decision
    return opt_run_id, manifest_path


def _prepare_oversight_ready(dataset_csv):
    opt_run_id, opt_manifest = _optimization_decide_proposed(dataset_csv)
    decision, ov_run_id = ga.prepare_oversight({ga.STEP_OPTIMIZATION_DECIDE: opt_manifest})
    assert decision == "ready", decision
    return ov_run_id


def _prepare_decision_oversight_ready(dataset_csv):
    run_id, ra_manifest = _resource_allocation_decide_with_placement(dataset_csv, _RISKY_PLACEMENT, "accepted_with_risk")
    decision, ov_run_id = ga.prepare_decision_oversight({ga.STEP_RESOURCE_ALLOCATION_DECIDE: ra_manifest})
    assert decision == "ready", decision
    return ov_run_id


def _prepare_through_failure_recovery_ready(dataset_csv):
    source_run_id, _ = _resource_allocation_decide_with_placement(dataset_csv, _RISKY_PLACEMENT, "accepted_with_risk")
    decision, fr_run_id = ga.prepare_failure_recovery({"__task__": source_run_id})
    assert decision == "ready", decision
    return fr_run_id


def _failure_recovery_decide_rerouted(dataset_csv):
    fr_run_id = _prepare_through_failure_recovery_ready(dataset_csv)
    fact = read_fact(fr_run_id, "incident_report")
    affected_ids = [t["task_id"] for t in fact["affected_tasks"]]
    reroutes = [
        {"task_id": tid, "new_machine_id": "M01", "new_network_slice_id": "NS1", "reasoning": "reroute to a healthy machine"}
        for tid in affected_ids
    ]
    proposal = json.dumps({"incidents": fact["incidents"], "reroute_proposals": reroutes})
    decision, manifest_path = ga.failure_recovery_decide({
        ga.STEP_PREPARE_FAILURE_RECOVERY: fr_run_id, ga.STEP_PROPOSE_FAILURE_RECOVERY: proposal,
    })
    assert decision == "rerouted", decision
    return fr_run_id, manifest_path, affected_ids


# -- 1. Load Monitor -------------------------------------------------


def test_load_monitor_prepare_publishes_snapshot_and_returns_bare_run_id(dataset_csv):
    decision, run_id = ga.prepare_load_monitor({"__task__": dataset_csv})
    assert decision == "reported"
    fact = read_fact(run_id, "resource_snapshot")
    assert set(fact) == {"snapshot", "thresholds", "flags"}
    assert fact["flags"], "fixture must produce at least one flag (M01/M03 sit at 100% utilization)"


def test_load_monitor_decide_no_mismatch_when_flags_match(dataset_csv):
    manifest_path = _prepare_through_load_monitor_decide(dataset_csv)
    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest["flag_mismatch"] is False


def test_load_monitor_decide_flags_a_mismatch(dataset_csv):
    _, run_id = ga.prepare_load_monitor({"__task__": dataset_csv})
    decision, manifest_path = ga.load_monitor_decide({
        ga.STEP_PREPARE_LOAD_MONITOR: run_id,
        ga.STEP_PROPOSE_LOAD_MONITOR: json.dumps({"flags": [], "narrative_summary": "nothing to see"}),
    })
    assert decision == "reported"
    assert json.loads(Path(manifest_path).read_text())["flag_mismatch"] is True


def test_load_monitor_decide_malformed_json_never_crashes_counts_as_mismatch(dataset_csv):
    _, run_id = ga.prepare_load_monitor({"__task__": dataset_csv})
    decision, manifest_path = ga.load_monitor_decide({
        ga.STEP_PREPARE_LOAD_MONITOR: run_id, ga.STEP_PROPOSE_LOAD_MONITOR: "not json at all",
    })
    assert decision == "reported"
    assert json.loads(Path(manifest_path).read_text())["flag_mismatch"] is True


# -- 2. Task Prioritization -------------------------------------------


def test_task_prioritization_decide_accepts_a_valid_ranking(dataset_csv):
    run_id, manifest_path = _task_prioritization_decide_ranked(dataset_csv)
    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest["valid"] is True
    assert manifest["score_inconsistent"] is False
    legacy = json.loads((Path(manifest["run_dir"]) / "task_prioritization_report.json").read_text())
    assert legacy == {"valid": True, "score_inconsistent": False}


def test_task_prioritization_decide_rejects_an_invalid_permutation(dataset_csv):
    run_id = _prepare_through_task_prioritization(dataset_csv)
    proposal = json.dumps({"ranked_task_ids": ["not_a_real_task"], "score_breakdown": {}, "reasoning": "bad"})
    decision, _ = ga.task_prioritization_decide({
        ga.STEP_PREPARE_TASK_PRIORITIZATION: run_id, ga.STEP_PROPOSE_TASK_PRIORITIZATION: proposal,
    })
    assert decision == "invalid"


def test_task_prioritization_decide_malformed_json_is_invalid_not_a_crash(dataset_csv):
    run_id = _prepare_through_task_prioritization(dataset_csv)
    decision, _ = ga.task_prioritization_decide({
        ga.STEP_PREPARE_TASK_PRIORITIZATION: run_id, ga.STEP_PROPOSE_TASK_PRIORITIZATION: "garbage",
    })
    assert decision == "invalid"


def test_task_prioritization_decide_flags_score_inconsistency_without_invalidating(dataset_csv):
    run_id = _prepare_through_task_prioritization(dataset_csv)
    task_ids = [t["task_id"] for t in read_fact(run_id, "task_queue_profile")["pending_tasks"]]
    # ascending, not descending -- structurally valid, just inconsistent with its own scores
    proposal = json.dumps({
        "ranked_task_ids": task_ids,
        "score_breakdown": {tid: {"final_score": i} for i, tid in enumerate(task_ids)},
        "reasoning": "oops, ascending",
    })
    decision, manifest_path = ga.task_prioritization_decide({
        ga.STEP_PREPARE_TASK_PRIORITIZATION: run_id, ga.STEP_PROPOSE_TASK_PRIORITIZATION: proposal,
    })
    assert decision == "ranked"
    assert json.loads(Path(manifest_path).read_text())["score_inconsistent"] is True


# -- 3. Resource Allocation --------------------------------------------


def test_resource_allocation_short_circuits_with_no_mailbox_message(dataset_csv):
    run_id = _prepare_through_task_prioritization(dataset_csv)
    invalid_proposal = json.dumps({"ranked_task_ids": ["nope"], "score_breakdown": {}, "reasoning": "bad"})
    decision, tp_manifest = ga.task_prioritization_decide({
        ga.STEP_PREPARE_TASK_PRIORITIZATION: run_id, ga.STEP_PROPOSE_TASK_PRIORITIZATION: invalid_proposal,
    })
    assert decision == "invalid"  # never sent to the mailbox
    decision2, _ = ga.prepare_resource_allocation({ga.STEP_TASK_PRIORITIZATION_DECIDE: tp_manifest})
    assert decision2 == "no_ranking"


def test_prepare_resource_allocation_reads_the_mailbox_across_a_separate_call(dataset_csv):
    """The core PersistentMailbox proof for this hop: task_prioritization_decide's
    mailbox.send() and prepare_resource_allocation's mailbox.inbox_for()
    are separate PersistentMailbox() instances in separate function calls."""
    run_id = _prepare_through_resource_allocation(dataset_csv)
    fact = read_fact(run_id, "allocation_context")
    assert len(fact["ranked_tasks"]) == 8


def test_resource_allocation_decide_accepts_clean_when_nothing_is_risky(dataset_csv):
    run_id, manifest_path = _resource_allocation_decide_with_placement(dataset_csv, _CLEAN_PLACEMENT, "accepted_clean")
    manifest = json.loads(Path(manifest_path).read_text())
    accepted_ids = {a["task_id"] for a in manifest["accepted_assignments"]}
    assert accepted_ids == set(_CLEAN_PLACEMENT)
    assert manifest["environment_rejected"] == []
    persisted = read_fact(run_id, "accepted_assignments")
    assert persisted["accepted_assignments"] == manifest["accepted_assignments"]
    legacy = json.loads((Path(manifest["run_dir"]) / "resource_allocation_report.json").read_text())
    assert legacy["accepted_assignments"] == manifest["accepted_assignments"]


def test_resource_allocation_decide_environment_vetoes_maintenance_assignment_regardless_of_agent_claim(dataset_csv):
    run_id = _prepare_through_resource_allocation(dataset_csv)
    fact = read_fact(run_id, "allocation_context")
    assignments, rejected = [], []
    for t in fact["ranked_tasks"]:
        tid = t["task_id"]
        if tid == "T009":
            assignments.append({"task_id": tid, "machine_id": "M04", "network_slice_id": "NS2", "rationale": "agent (wrongly) thinks this is fine"})
        elif tid == "T010":
            rejected.append({"task_id": tid, "reason": "agent declines"})
        else:
            machine_id, slice_id = _CLEAN_PLACEMENT[tid]
            assignments.append({"task_id": tid, "machine_id": machine_id, "network_slice_id": slice_id, "rationale": "fits"})
    proposal = json.dumps({"assignments": assignments, "rejected": rejected})
    decision, manifest_path = ga.resource_allocation_decide({
        ga.STEP_PREPARE_RESOURCE_ALLOCATION: run_id, ga.STEP_PROPOSE_RESOURCE_ALLOCATION: proposal,
    })
    manifest = json.loads(Path(manifest_path).read_text())
    veto = {e["task_id"]: e for e in manifest["environment_rejected"]}
    assert "T009" in veto
    assert veto["T009"]["violation"]["violation"] == "machine_in_maintenance"
    assert "T009" not in {a["task_id"] for a in manifest["accepted_assignments"]}


def test_resource_allocation_decide_flags_risky_assignment_and_sends_to_human_oversight(dataset_csv):
    run_id, manifest_path = _resource_allocation_decide_with_placement(dataset_csv, _RISKY_PLACEMENT, "accepted_with_risk")
    decision, ov_run_id = ga.prepare_decision_oversight({ga.STEP_RESOURCE_ALLOCATION_DECIDE: manifest_path})
    assert decision == "ready"
    assert ov_run_id == run_id
    bundle = read_fact(run_id, "decision_review_bundle")
    assert bundle["source_agent"] == "resource_allocation"
    assert {"T007", "T008"} <= {d["task_id"] for d in bundle["risky_decisions"]}


def test_prepare_decision_oversight_short_circuits_with_no_risky_decision(dataset_csv):
    _, manifest_path = _resource_allocation_decide_with_placement(dataset_csv, _CLEAN_PLACEMENT, "accepted_clean")
    decision, _ = ga.prepare_decision_oversight({ga.STEP_RESOURCE_ALLOCATION_DECIDE: manifest_path})
    assert decision == "no_decision"


# -- 4. Failure Recovery + Reroute Validation ---------------------------


def test_failure_recovery_prepare_uses_a_fresh_run_id_and_detects_the_incident(dataset_csv):
    source_run_id, _ = _resource_allocation_decide_with_placement(dataset_csv, _RISKY_PLACEMENT, "accepted_with_risk")
    decision, fr_run_id = ga.prepare_failure_recovery({"__task__": source_run_id})
    assert decision == "ready"
    assert fr_run_id != source_run_id
    fact = read_fact(fr_run_id, "incident_report")
    assert any(i["machine_id"] == "M03" for i in fact["incidents"])
    assert {"T007", "T008"} <= {t["task_id"] for t in fact["affected_tasks"]}


def test_failure_recovery_decide_accepts_valid_reroute_and_sends_request(dataset_csv):
    _failure_recovery_decide_rerouted(dataset_csv)  # asserts "rerouted" internally


def test_failure_recovery_decide_rejects_invalid_reroute_structure(dataset_csv):
    fr_run_id = _prepare_through_failure_recovery_ready(dataset_csv)
    proposal = json.dumps({"incidents": [], "reroute_proposals": [{"task_id": "not_affected", "new_machine_id": "M01", "new_network_slice_id": "NS1", "reasoning": "x"}]})
    decision, _ = ga.failure_recovery_decide({
        ga.STEP_PREPARE_FAILURE_RECOVERY: fr_run_id, ga.STEP_PROPOSE_FAILURE_RECOVERY: proposal,
    })
    assert decision == "invalid"


def test_reroute_validation_decide_reads_the_reroute_request_across_a_separate_call(dataset_csv):
    """Second PersistentMailbox proof: failure_recovery_decide's send()
    and reroute_validation_decide's inbox_for() are separate calls."""
    fr_run_id, manifest_path, affected_ids = _failure_recovery_decide_rerouted(dataset_csv)
    decision, manifest_path2 = ga.reroute_validation_decide({ga.STEP_FAILURE_RECOVERY_DECIDE: manifest_path})
    assert decision in ("accepted_clean", "accepted_with_risk")
    manifest2 = json.loads(Path(manifest_path2).read_text())
    assert {a["task_id"] for a in manifest2["accepted_reroutes"]} == set(affected_ids)
    legacy = json.loads((Path(manifest2["run_dir"]) / "failure_recovery_report.json").read_text())
    assert legacy["reroute_validation"]["accepted_reroutes"] == manifest2["accepted_reroutes"]


def test_reroute_validation_decide_short_circuits_with_no_reroute_request(dataset_csv):
    run_id, run_dir = make_run_dir(None)
    manifest_path = str(run_dir / "fake_manifest.json")
    Path(manifest_path).write_text(json.dumps({"run_id": run_id, "run_dir": str(run_dir), "csv_path": dataset_csv}))
    decision, _ = ga.reroute_validation_decide({ga.STEP_FAILURE_RECOVERY_DECIDE: manifest_path})
    assert decision == "no_reroute"


# -- 5. Optimization (one-shot) -------------------------------------------


def test_optimization_prepare_short_circuits_with_no_run_history():
    decision, _ = ga.prepare_optimization({})
    assert decision == "no_history"


def test_optimization_prepare_finds_history_a_prior_pipeline_run_left_on_disk(dataset_csv):
    opt_run_id = _prepare_optimization_with_history(dataset_csv)
    evidence = read_fact(opt_run_id, "policy_evidence")
    assert evidence["n_runs_scanned"] >= 1
    assert evidence["ranking_valid_rate"] == 1.0


def test_optimization_decide_accepts_valid_policy_and_sends_to_oversight(dataset_csv):
    _optimization_decide_proposed(dataset_csv)  # asserts "proposed" internally


def test_optimization_decide_rejects_unknown_policy_key(dataset_csv):
    opt_run_id = _prepare_optimization_with_history(dataset_csv)
    proposal = json.dumps({"policy_updates": {"not_a_real_knob": 1}, "evidence": "x", "recommend_apply": False})
    decision, _ = ga.optimization_decide({
        ga.STEP_PREPARE_OPTIMIZATION: opt_run_id, ga.STEP_PROPOSE_OPTIMIZATION: proposal,
    })
    assert decision == "invalid"


# -- 6. Human Oversight: policy review -------------------------------------


def test_prepare_oversight_reads_the_policy_proposal_across_a_separate_call(dataset_csv):
    opt_run_id, opt_manifest = _optimization_decide_proposed(dataset_csv)
    decision, ov_run_id = ga.prepare_oversight({ga.STEP_OPTIMIZATION_DECIDE: opt_manifest})
    assert decision == "ready"
    assert ov_run_id == opt_run_id
    bundle = read_fact(ov_run_id, "policy_review_bundle")
    assert bundle["policy_updates"] == {"queue_size": 6}


def test_prepare_oversight_short_circuits_with_no_proposal(dataset_csv):
    opt_run_id = _prepare_optimization_with_history(dataset_csv)
    decision, manifest_path = ga.optimization_decide({
        ga.STEP_PREPARE_OPTIMIZATION: opt_run_id, ga.STEP_PROPOSE_OPTIMIZATION: "garbage",
    })
    assert decision == "invalid"
    decision2, _ = ga.prepare_oversight({ga.STEP_OPTIMIZATION_DECIDE: manifest_path})
    assert decision2 == "no_proposal"


def test_oversight_decide_approves(dataset_csv):
    ov_run_id = _prepare_oversight_ready(dataset_csv)
    proposal = json.dumps({"verdict": "approved", "concerns": [], "reasoning": "fine"})
    decision, _ = ga.oversight_decide({ga.STEP_PREPARE_OVERSIGHT: ov_run_id, ga.STEP_PROPOSE_OVERSIGHT: proposal})
    assert decision == "approved"


def test_oversight_decide_degrades_malformed_verdict_to_flagged_never_approved(dataset_csv):
    ov_run_id = _prepare_oversight_ready(dataset_csv)
    decision, manifest_path = ga.oversight_decide({
        ga.STEP_PREPARE_OVERSIGHT: ov_run_id, ga.STEP_PROPOSE_OVERSIGHT: "not json at all",
    })
    assert decision == "flagged"
    manifest = json.loads(Path(manifest_path).read_text())
    assert "never treat an unparseable or invalid response as approval" in manifest["reasoning"]


def test_oversight_decide_degrades_unrecognized_verdict_string_to_flagged(dataset_csv):
    ov_run_id = _prepare_oversight_ready(dataset_csv)
    proposal = json.dumps({"verdict": "sure_why_not", "concerns": [], "reasoning": "??"})
    decision, _ = ga.oversight_decide({ga.STEP_PREPARE_OVERSIGHT: ov_run_id, ga.STEP_PROPOSE_OVERSIGHT: proposal})
    assert decision == "flagged"


# -- 7. Human Oversight: risky-decision review ------------------------------


def test_decision_oversight_decide_approves(dataset_csv):
    ov_run_id = _prepare_decision_oversight_ready(dataset_csv)
    proposal = json.dumps({"verdict": "approved", "concerns": [], "reasoning": "it's just overloaded, not blocked"})
    decision, _ = ga.decision_oversight_decide({
        ga.STEP_PREPARE_DECISION_OVERSIGHT: ov_run_id, ga.STEP_PROPOSE_DECISION_OVERSIGHT: proposal,
    })
    assert decision == "approved"


def test_decision_oversight_decide_degrades_malformed_to_flagged(dataset_csv):
    ov_run_id = _prepare_decision_oversight_ready(dataset_csv)
    decision, _ = ga.decision_oversight_decide({
        ga.STEP_PREPARE_DECISION_OVERSIGHT: ov_run_id, ga.STEP_PROPOSE_DECISION_OVERSIGHT: "garbage",
    })
    assert decision == "flagged"


# -- structural invariant ---------------------------------------------------


def test_gate_adapters_never_imports_the_model_client_or_agent_loop():
    assert not hasattr(ga, "ModelClient")
    assert not hasattr(ga, "ToolCallingAgent")
    assert "resource_scheduler.model_client" not in [
        getattr(v, "__module__", None) for v in vars(ga).values()
    ]
