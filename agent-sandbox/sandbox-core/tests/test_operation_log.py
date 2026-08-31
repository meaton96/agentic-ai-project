from sandbox_core.runtime.operation_log import OperationLog, read_operations
from sandbox_core.schemas.operations import AlterModelOperation, SwapAgentOperation
from sandbox_core.schemas.agent_spec import ModelConfig

TARGET_TYPE = "pipeline"
TARGET_ID = "pipe-1"

BEFORE = {"id": "pipe-1", "steps": [{"agent_id": "agent-a"}]}
AFTER = {"id": "pipe-1", "steps": [{"agent_id": "agent-c"}]}


def make_swap_op(**overrides):
    fields = dict(step_id="a", new_agent_id="agent-c")
    fields.update(overrides)
    return SwapAgentOperation(**fields)


def test_append_assigns_monotonic_seq(tmp_path):
    log = OperationLog(TARGET_TYPE, TARGET_ID, root=tmp_path)

    first = log.append(operation=make_swap_op(), actor="user", before=BEFORE, after=AFTER)
    second = log.append(operation=make_swap_op(), actor="user", before=BEFORE, after=AFTER)
    third = log.append(operation=make_swap_op(), actor="user", before=BEFORE, after=AFTER)

    assert [first.seq, second.seq, third.seq] == [1, 2, 3]


def test_append_writes_one_json_line_per_record(tmp_path):
    log = OperationLog(TARGET_TYPE, TARGET_ID, root=tmp_path)
    log.append(operation=make_swap_op(), actor="user", before=BEFORE, after=AFTER)
    log.append(operation=make_swap_op(), actor="user", before=BEFORE, after=AFTER)

    lines = log.path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_append_records_target_and_actor(tmp_path):
    log = OperationLog(TARGET_TYPE, TARGET_ID, root=tmp_path)
    record = log.append(operation=make_swap_op(), actor="alice", before=BEFORE, after=AFTER)

    assert record.target_type == TARGET_TYPE
    assert record.target_id == TARGET_ID
    assert record.actor == "alice"


def test_append_computes_unified_diff(tmp_path):
    log = OperationLog(TARGET_TYPE, TARGET_ID, root=tmp_path)
    record = log.append(operation=make_swap_op(), actor="user", before=BEFORE, after=AFTER)

    assert "agent-a" in record.diff
    assert "agent-c" in record.diff


def test_read_operations_roundtrips_typed_records_in_order(tmp_path):
    log = OperationLog(TARGET_TYPE, TARGET_ID, root=tmp_path)
    log.append(operation=make_swap_op(), actor="user", before=BEFORE, after=AFTER)
    model = ModelConfig(base_url="http://x", model_name="m", api_key_ref="k")
    log.append(operation=AlterModelOperation(model=model), actor="user", before=BEFORE, after=AFTER)

    records = read_operations(log.path)

    assert len(records) == 2
    assert isinstance(records[0].operation, SwapAgentOperation)
    assert isinstance(records[1].operation, AlterModelOperation)
    assert [r.seq for r in records] == [1, 2]


def test_read_operations_missing_file_returns_empty_list(tmp_path):
    assert read_operations(tmp_path / "nope.jsonl") == []


def test_reopening_operation_log_continues_seq_from_existing_file(tmp_path):
    first_log = OperationLog(TARGET_TYPE, TARGET_ID, root=tmp_path)
    first_log.append(operation=make_swap_op(), actor="user", before=BEFORE, after=AFTER)
    first_log.append(operation=make_swap_op(), actor="user", before=BEFORE, after=AFTER)

    reopened = OperationLog(TARGET_TYPE, TARGET_ID, root=tmp_path)
    third = reopened.append(operation=make_swap_op(), actor="user", before=BEFORE, after=AFTER)

    assert third.seq == 3
    assert len(read_operations(reopened.path)) == 3


def test_operation_log_creates_output_directory(tmp_path):
    root = tmp_path / "does" / "not" / "exist"
    log = OperationLog(TARGET_TYPE, TARGET_ID, root=root)
    assert log.path.parent.is_dir()
    assert log.path == root / TARGET_TYPE / f"{TARGET_ID}.jsonl"


def test_operation_logs_for_different_targets_are_independent_files(tmp_path):
    log_a = OperationLog(TARGET_TYPE, "pipe-a", root=tmp_path)
    log_b = OperationLog(TARGET_TYPE, "pipe-b", root=tmp_path)

    log_a.append(operation=make_swap_op(), actor="user", before=BEFORE, after=AFTER)

    assert len(read_operations(log_a.path)) == 1
    assert len(read_operations(log_b.path)) == 0
