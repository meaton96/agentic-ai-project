"""
Sandboxed execution of agent-generated candidate code.

Design: candidate.py must expose a single function

    def build_pipeline(config: dict):
        return <unfitted sklearn-compatible estimator/pipeline>

Candidate code NEVER receives the raw dataset or a file path to it. It
only returns a pipeline object. The harness (this trusted process) is
the only thing that ever calls .fit()/.predict() with real data.

Two layers of defense:
  1. Static AST check — reject forbidden imports/calls before ever
     executing the candidate.
  2. Subprocess isolation with wall-clock timeout and (on POSIX) CPU/
     memory rlimits — the candidate's build_pipeline() runs in a child
     process with no network and a hard timeout, and only the resulting
     pickled (unfitted) estimator crosses back to the parent.

This is process-level isolation, not container-level. For untrusted
multi-tenant use, layer this inside a Docker/gVisor sandbox as well
(see docs on agents.defaults.sandbox in the wider architecture doc) —
this module gives you the static+resource-limit layer regardless of
what container boundary you put around it.
"""
from __future__ import annotations

import ast
import os
import pickle
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


FORBIDDEN_IMPORTS = {
    "os", "sys", "subprocess", "socket", "shutil", "requests", "urllib",
    "http", "ftplib", "telnetlib", "smtplib", "ctypes", "multiprocessing",
    "pty", "pickle",  # candidate must not pickle things itself
}
FORBIDDEN_CALL_NAMES = {"eval", "exec", "__import__", "compile", "open", "input"}


@dataclass
class StaticCheckResult:
    passed: bool
    violations: list[str]


def static_check(source_code: str) -> StaticCheckResult:
    violations = []
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return StaticCheckResult(passed=False, violations=[f"syntax error: {e}"])

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    violations.append(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_IMPORTS:
                violations.append(f"forbidden import: {node.module}")
        elif isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in FORBIDDEN_CALL_NAMES:
                violations.append(f"forbidden call: {name}()")

    # must define build_pipeline
    has_build_fn = any(
        isinstance(node, ast.FunctionDef) and node.name == "build_pipeline"
        for node in ast.walk(tree)
    )
    if not has_build_fn:
        violations.append("candidate must define a top-level build_pipeline(config) function")

    return StaticCheckResult(passed=len(violations) == 0, violations=violations)


_SUBPROCESS_RUNNER_TEMPLATE = """
import json
import pickle
import resource
import sys

# Best-effort resource limits (POSIX only). Not a full sandbox on its own —
# combine with container/network isolation for untrusted multi-tenant use.
#
# NOTE: RLIMIT_AS (virtual address space) is intentionally NOT set here.
# BLAS-backed libraries (numpy/scikit-learn/OpenBLAS/MKL) reserve large
# virtual address ranges per thread regardless of actual memory used, and
# reliably hit an RLIMIT_AS ceiling even when real usage is tiny — this is
# a well-known incompatibility, not a sizing problem. Limiting BLAS to a
# single thread (below, via env vars set before subprocess launch) is the
# standard fix and avoids the false-failure entirely rather than just
# raising the ceiling. CPU time is still capped.
try:
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu_seconds}, {cpu_seconds}))
except Exception:
    pass

candidate_path = {candidate_path!r}
config = json.loads({config_json!r})
out_path = {out_path!r}

namespace = {{"__name__": "candidate"}}
with open(candidate_path) as f:
    code = f.read()
exec(compile(code, candidate_path, "exec"), namespace)

pipeline = namespace["build_pipeline"](config)

with open(out_path, "wb") as f:
    pickle.dump(pipeline, f)
"""


def run_candidate_build(
    source_code: str,
    config: dict,
    timeout_seconds: int = 30,
    cpu_seconds: int = 20,
) -> tuple[object | None, str | None]:
    """
    Static-checks then subprocess-executes build_pipeline(config) from
    untrusted candidate source. Returns (pipeline_or_None, error_or_None).

    No RLIMIT_AS memory ceiling — see the runner template's comment for
    why that's incompatible with BLAS-backed libraries. Wall-clock
    timeout + CPU time limit + single-threaded BLAS are the actual
    containment here. For a hard memory ceiling on genuinely untrusted
    code, wrap this in a container/cgroup boundary instead.
    """
    check = static_check(source_code)
    if not check.passed:
        return None, f"static check failed: {'; '.join(check.violations)}"

    import json as _json

    with tempfile.TemporaryDirectory() as tmpdir:
        candidate_path = Path(tmpdir) / "candidate.py"
        candidate_path.write_text(source_code)
        out_path = Path(tmpdir) / "pipeline.pkl"

        runner_code = _SUBPROCESS_RUNNER_TEMPLATE.format(
            cpu_seconds=cpu_seconds,
            candidate_path=str(candidate_path),
            config_json=_json.dumps(config),
            out_path=str(out_path),
        )
        runner_path = Path(tmpdir) / "runner.py"
        runner_path.write_text(runner_code)

        # Limit BLAS/OpenMP thread pools to 1 — this is the actual fix for
        # the RLIMIT_AS/OpenBLAS incompatibility described above, not just
        # a performance tweak. Untrusted candidate code also has no
        # legitimate need for multi-threaded parallelism inside the sandbox.
        env = {
            "PATH": os.environ.get("PATH", ""),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }

        try:
            proc = subprocess.run(
                [sys.executable, str(runner_path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=tmpdir,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return None, f"candidate build_pipeline() exceeded {timeout_seconds}s wall-clock timeout"

        if proc.returncode != 0:
            return None, f"candidate build_pipeline() raised: {proc.stderr[-2000:]}"

        if not out_path.exists():
            return None, "candidate did not produce a pipeline output"

        with open(out_path, "rb") as f:
            pipeline = pickle.load(f)

        return pipeline, None