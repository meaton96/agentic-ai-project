"""The experiment-loop gates and training job, driven the way agent-sandbox
drives them: an `outputs` dict of every completed step's output (plus
"__task__"), each function returning (decision, output)."""
import json

import pytest

from agentic_pdm import pipeline
from agentic_pdm.ingest import IngestConfig, ingest_long_csv

TINY = {"name": "tiny", "architecture": {"id": "small_cnn", "params": {"width": 8, "depth": 3}},
        "input": {"max_len": 64, "pool": 1}, "training": {"epochs": 3, "batch_size": 16}}


@pytest.fixture
def env(long_csv, tmp_path, monkeypatch):
    datasets = tmp_path / "datasets"
    ingest_long_csv(long_csv, datasets / "toy", IngestConfig(
        id_column="seq", label_column="target", group_column="unit", fold_column="fold",
        exclude_columns=["leak"], max_len=64))
    (datasets / "toy" / "baselines.json").write_text(json.dumps({"dummy": {"mean": {"roc_auc": 0.5}}}))
    monkeypatch.setenv("PDM_DATASETS_DIR", str(datasets))
    monkeypatch.setenv("PDM_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.delenv("GATE_SCRATCH_DIR", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    return tmp_path


def contract(**overrides):
    base = {"dataset": "toy", "goal": "separate the classes",
            "target": {"metric": "roc_auc", "protocol": "best", "value": None},
            "budget": {"max_experiments": 3, "max_minutes": 60}}
    return json.dumps({**base, **overrides})


def start(task):
    outputs = {"__task__": task}
    decision, out = pipeline.init_run(outputs)
    outputs["init"] = out
    return decision, outputs


def propose(outputs, experiment):
    outputs["plan"] = "Here is my proposal:\n```json\n" + json.dumps(experiment) + "\n```"
    decision, out = pipeline.validate_proposal(outputs)
    outputs["validate"] = out
    return decision


def train_and_record(outputs):
    decision, out = pipeline.train_experiment(outputs)
    assert decision == "done"
    outputs["train"] = out
    decision, out = pipeline.record_results(outputs)
    outputs["record"] = out
    return decision, out


def test_contract_parsing():
    parsed, errors = pipeline.parse_contract(contract())
    assert not errors and parsed["budget"]["max_experiments"] == 3 and parsed["target"]["protocol"] == "best"
    for bad, fragment in (("not json", "JSON run contract"), (json.dumps({"goal": "x"}), "dataset is required"),
                          (contract(target={"metric": "f2"}), "target.metric"),
                          (contract(budget={"max_experiments": 0}), "max_experiments"),
                          (contract(surprise=1), "unknown contract key")):
        assert fragment in " ".join(pipeline.parse_contract(bad)[1])


def test_invalid_contract_and_missing_dataset(env):
    assert pipeline.init_run({"__task__": "{}"})[0] == "invalid_contract"
    decision, out = pipeline.init_run({"__task__": contract(dataset="absent")})
    assert decision == "invalid_contract" and "not found" in out


def test_full_loop_with_recording_calibration_and_report(env):
    decision, outputs = start(contract())
    assert decision == "ready"
    run_dir = json.loads(outputs["init"])["pdm_run_dir"]

    decision, brief = pipeline.compose_brief(outputs)
    assert decision == "ready"
    assert "No experiments yet" in brief and "Arguments for validate_experiment" in brief
    assert "dummy: roc_auc 0.500" in brief

    assert propose(outputs, TINY) == "valid"
    decision, record = train_and_record(outputs)
    assert decision == "continue" and "tiny: roc_auc best" in record

    state = json.loads(open(f"{run_dir}/state.json").read())
    assert state["experiments_submitted"] == 1 and state["pending"] is None
    assert set(state["calibration"]) == {"default", "small_cnn"} and len(state["time_ratios"]["small_cnn"]) == 1
    assert state["calibration"]["small_cnn"] == state["calibration"]["default"] != 1.0

    outputs["analyze"] = "Underfitting; try more epochs."
    decision, brief = pipeline.compose_brief(outputs)
    assert "tiny:" in brief and "Underfitting" in brief and "Curves:" in brief

    # Reusing a name is rejected; the rejection shows up in the next brief.
    assert propose(outputs, TINY) == "invalid"
    assert "already ran" in outputs["validate"]
    assert "previous proposal was rejected" in pipeline.compose_brief(outputs)[1]

    assert propose(outputs, {**TINY, "name": "tiny-2", "folds": [0]}) == "valid"
    decision, record = train_and_record(outputs)
    assert decision == "continue" and "(screen)" in record

    decision, report = pipeline.finalize(outputs)
    assert decision == "done" and "Best:** tiny (full CV)" in report and "| 2 | tiny-2 | screen |" in report
    assert json.loads(open(f"{run_dir}/report.json").read())["experiments"] == 2


def test_target_met_only_counts_full_cross_validation(env):
    _, outputs = start(contract(target={"metric": "roc_auc", "protocol": "best", "value": 0.01}))
    assert propose(outputs, {**TINY, "folds": [0]}) == "valid"
    assert train_and_record(outputs)[0] == "continue"  # a screen can't meet the target
    assert propose(outputs, {**TINY, "name": "full"}) == "valid"
    assert train_and_record(outputs)[0] == "target_met"


def test_budget_is_enforced(env):
    _, outputs = start(contract(budget={"max_experiments": 1, "max_minutes": 60}))
    assert propose(outputs, TINY) == "valid"
    assert train_and_record(outputs)[0] == "budget_exhausted"
    assert pipeline.compose_brief(outputs)[0] == "budget_exhausted"
    assert propose(outputs, {**TINY, "name": "again"}) == "budget_exhausted"


def test_an_experiment_over_the_time_budget_is_rejected(env, monkeypatch):
    monkeypatch.setenv("PDM_EFFECTIVE_GFLOPS", "0.0001")  # a very slow machine
    _, outputs = start(contract(budget={"max_experiments": 3, "max_minutes": 1}))
    assert propose(outputs, TINY) == "invalid"
    assert "exceeds the" in outputs["validate"]


def test_repeated_invalid_proposals_give_up(env):
    _, outputs = start(contract())
    assert propose(outputs, {"name": "x"}) == "invalid"
    outputs["plan"] = "I refuse to answer in JSON."
    assert pipeline.validate_proposal(outputs)[0] == "invalid"
    assert propose(outputs, {"name": "x", "architecture": {"id": "nope"}}) == "give_up"


def test_a_failed_job_is_recorded_and_the_loop_continues(env):
    _, outputs = start(contract())
    assert propose(outputs, TINY) == "valid"
    outputs["train"] = "job was killed for exceeding its memory limit (2048 MiB)"  # the job's __error__ output
    decision, out = pipeline.record_results(outputs)
    assert decision == "continue" and "FAILED" in out and "memory limit" in out
    brief = pipeline.compose_brief(outputs)[1]
    assert "FAILED tiny" in brief and "It failed" in brief
    report = pipeline.finalize(outputs)[1]
    assert "No experiment completed" in report and "FAILED" in report


def test_proposal_extraction_variants():
    assert pipeline._extract_json_object('prefix {"a": {"b": 1}} suffix') == {"a": {"b": 1}}
    assert pipeline._extract_json_object('```json\n{"a": 2}\n```') == {"a": 2}
    with pytest.raises(ValueError, match="no JSON object"):
        pipeline._extract_json_object("no braces here")
