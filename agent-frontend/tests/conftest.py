"""
Shared fixtures: an isolated AGENTIC_ML_DATA_ROOT per test (so runs/
artifacts/datasets never touch this repo's real directories), a small
synthetic churn dataset, and the same stubbed-ModelClient pattern the
pipeline repo's own tests/test_events.py uses (dispatch by
system-prompt identity + message count) — never a real model endpoint.
"""
from __future__ import annotations

import functools
import json
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agentic_ml.model_client import ModelResponse
from agentic_ml.paths import datasets_root

from server.app import create_app
from server.run_manager import default_launch_run


@pytest.fixture(autouse=True)
def isolated_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_ML_DATA_ROOT", str(tmp_path / "data_root"))
    # dummy creds for resolve_model_endpoint's default ("rit") branch —
    # never actually dialed out to, since ModelClient.call is stubbed.
    monkeypatch.setenv("RIT_BASE_URL", "http://example.invalid/v1")
    monkeypatch.setenv("RIT_API_KEY", "dummy")


@pytest.fixture
def dataset_csv():
    """Writes datasets_root()/churn.csv and returns its filename relative
    to the root (what the API's `dataset` field expects)."""
    root = datasets_root()
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "age": rng.randint(18, 80, size=n),
        "income": rng.exponential(50000, size=n),
        "plan_type": rng.choice(["basic", "premium", "pro"], size=n),
        "region": rng.choice(["north", "south", "east", "west"], size=n),
        "churned": rng.binomial(1, 0.35, size=n),
    })
    (root / "churn.csv").write_text(df.to_csv(index=False))
    return "churn.csv"


def _resp(text=None, tool_calls=None):
    return ModelResponse(
        text=text, tool_calls=tool_calls or [], raw=None, latency_seconds=0.01,
        model="fake-model", input_tokens=1, output_tokens=1,
    )


PROFILER_NARRATIVE = json.dumps({
    "summary": "Synthetic churn dataset with mixed numeric/categorical features.",
    "recommended_split_strategy": "stratified",
    "key_risks": [],
    "recommended_next_steps": ["try a mixed baseline"],
})

CANDIDATE_A = json.dumps({
    "candidate_id": "candidate_a",
    "template_id": "sklearn_mixed_pipeline",
    "config": {
        "numeric_cols": ["age", "income"],
        "categorical_cols": ["plan_type", "region"],
        "classifier": "logistic_regression",
    },
    "explanation": "Mixed baseline; logistic regression as a cheap first pass.",
})

VERIFICATION_APPROVED = json.dumps({"verdict": "approved", "concerns": [], "reasoning": "looks fine"})


def static_fake_call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
    system_content = messages[0]["content"]
    n = len(messages)

    if "Profiler agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "t2", "name": "get_dataset_profile", "arguments": "{}"}])
        return _resp(text=PROFILER_NARRATIVE)

    if "Modeling agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "t3", "name": "get_dataset_profile", "arguments": "{}"}])
        if n == 4:
            return _resp(tool_calls=[{"id": "t4", "name": "list_templates", "arguments": "{}"}])
        return _resp(text=CANDIDATE_A)

    if "Verification agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "v1", "name": "get_candidate_review_bundle", "arguments": "{}"}])
        return _resp(text=VERIFICATION_APPROVED)

    # Analyst-style final summary: a plain text call with no tools.
    return _resp(text="This is a plain-language summary of the modeling run.")


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def stubbed_app():
    """A RunManager wired to the real orchestrator entry point, with
    ModelClient.call replaced by static_fake_call *inside the launched
    child process*. RunManager launches via "forkserver", which — unlike
    plain fork — does not give the child a copy of this (parent) test
    process's memory, so `monkeypatch.setattr(ModelClient, "call", ...)`
    here would be invisible to it; static_fake_call has to be handed to
    the child explicitly and applied there instead. See
    server/run_manager.py's module docstring for the full reasoning."""
    launch_fn = functools.partial(default_launch_run, model_client_call=static_fake_call)
    return create_app(launch_fn=launch_fn)


@pytest.fixture
def stubbed_client(stubbed_app):
    return TestClient(stubbed_app)


def wait_for_status(client, run_id, terminal=("completed", "failed"), timeout=30.0, interval=0.2):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"/api/runs/{run_id}").json()
        if last["status"] in terminal:
            return last
        time.sleep(interval)
    raise TimeoutError(f"run {run_id} did not reach a terminal status within {timeout}s (last={last})")
