"""Module-level gate functions for pipeline_runner tests. Must live in an
importable module rather than being defined inline in a test — GateStep.gate
resolves by "module.path:function_name" import path, exactly like a real
gate (e.g. agentic_ml.harness.verification:check) would."""

reject_then_approve_calls = 0


def always_approve(outputs: dict[str, str]) -> str:
    return "approved"


def reject_then_approve(outputs: dict[str, str]) -> str:
    """Rejects on its first call, approves on every call after — lets tests
    exercise a real reject/retry loop that eventually exits."""
    global reject_then_approve_calls
    reject_then_approve_calls += 1
    return "approved" if reject_then_approve_calls > 1 else "rejected"


def always_loop(outputs: dict[str, str]) -> str:
    return "loop"


async def async_always_approve(outputs: dict[str, str]) -> str:
    return "approved"


def approve_with_artifact_path(outputs: dict[str, str]) -> tuple[str, str]:
    """Stands in for a deterministic pipeline stage (e.g. a DataFrame
    transform): "does work", then hands forward a reference to wherever it
    wrote the real result — never the payload itself."""
    return "approved", "artifacts/fake_features.parquet"


def echo_seed_task(outputs: dict[str, str]) -> tuple[str, str]:
    """A gate as the very first step in a pipeline, reading the seed task via
    the reserved "__task__" key instead of a {{task}} template placeholder
    (which gates don't have) — the shape a real "intake"-style stage needs."""
    return "approved", outputs["__task__"]


not_callable = "not a function"
