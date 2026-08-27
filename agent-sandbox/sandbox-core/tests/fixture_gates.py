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


not_callable = "not a function"
