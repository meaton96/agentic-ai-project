"""Demo gate functions for manually testing Phase 3. Not real
agentic_ml.harness logic — just enough branching to exercise routing."""

_modeling_attempts = 0


def approve(outputs: dict[str, str]) -> str:
    """Always approves — for the simplest happy-path gate test."""
    return "approved"


def reject_once_then_approve(outputs: dict[str, str]) -> str:
    """Rejects the first time it's called, approves every time after —
    for testing a real reject/retry loop that eventually exits."""
    global _modeling_attempts
    _modeling_attempts += 1
    return "approved" if _modeling_attempts > 1 else "rejected"


def always_loop(outputs: dict[str, str]) -> str:
    """Never approves — for testing that max_steps actually trips."""
    return "loop"
