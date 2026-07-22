"""
Prompt-override support: steps/*_step.py's previously inline
SYSTEM_PROMPT string constants are now extracted verbatim into
prompts/<agent>.md and loaded via agentic_ml.prompt_loader. Three things
worth proving, not just exercising:

1. The loader's per-agent fallback: an override directory that only has
   a file for SOME agents still supplies the shipped default for any
   agent missing there — overrides are per-agent, not all-or-nothing
   (unit tests, no LLM involved).
2. A supplied override actually changes what's sent as the system
   prompt to the model for one agent, while every other agent in the
   same run still gets its shipped default — proven with the same
   stubbed-ModelClient pattern as tests/test_orchestrator.py, but
   dispatching on which TOOLS a call has access to (a stable signal
   that doesn't depend on the system prompt's exact wording, since the
   whole point of an override is that the wording can be arbitrary)
   rather than matching a substring of the default prompt's own text.
3. Each run's events.jsonl records a "prompt_loaded" event per agent
   recording which file (default or override) was actually used — this
   is what makes an edited-prompt run auditable later from a UI.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agentic_ml.model_client import ModelClient, ModelResponse
from agentic_ml.prompt_loader import (
    DEFAULT_PROMPTS_DIR,
    PROMPT_OVERRIDE_DIR_ENV_VAR,
    load_prompt,
    prompt_source,
    resolve_prompt_override_dir,
)

import run_orchestrator  # noqa: E402

ALL_AGENTS = [
    "intake", "feature_engineering", "profiler", "modeling", "verification",
    "deep_dive", "planner",
]


# --- 1. Loader unit tests: per-agent fallback, no LLM involved ---

def test_default_prompts_dir_has_every_agent_file():
    for agent in ALL_AGENTS:
        assert (DEFAULT_PROMPTS_DIR / f"{agent}.md").is_file()


def test_load_prompt_with_no_override_dir_returns_shipped_default():
    for agent in ALL_AGENTS:
        expected = (DEFAULT_PROMPTS_DIR / f"{agent}.md").read_text()
        assert load_prompt(agent) == expected
        assert load_prompt(agent, override_dir=None) == expected


def test_prompt_source_reports_default_when_no_override_dir_given():
    source, path = prompt_source("modeling")
    assert source == "default"
    assert path == DEFAULT_PROMPTS_DIR / "modeling.md"


def test_load_prompt_uses_override_file_when_present(tmp_path):
    override_text = "OVERRIDDEN modeling prompt — completely different wording."
    (tmp_path / "modeling.md").write_text(override_text)

    assert load_prompt("modeling", override_dir=str(tmp_path)) == override_text
    source, path = prompt_source("modeling", override_dir=str(tmp_path))
    assert source == "override"
    assert path == tmp_path / "modeling.md"


def test_load_prompt_falls_back_to_default_when_override_dir_missing_this_agent(tmp_path):
    """Overrides are per-agent, not all-or-nothing: an override dir that
    only has profiler.md must not affect modeling, intake, etc."""
    (tmp_path / "profiler.md").write_text("OVERRIDDEN profiler prompt.")

    default_modeling = (DEFAULT_PROMPTS_DIR / "modeling.md").read_text()
    assert load_prompt("modeling", override_dir=str(tmp_path)) == default_modeling
    source, path = prompt_source("modeling", override_dir=str(tmp_path))
    assert source == "default"
    assert path == DEFAULT_PROMPTS_DIR / "modeling.md"

    # meanwhile profiler DOES pick up the override
    assert load_prompt("profiler", override_dir=str(tmp_path)) == "OVERRIDDEN profiler prompt."
    source, path = prompt_source("profiler", override_dir=str(tmp_path))
    assert source == "override"


def test_resolve_prompt_override_dir_explicit_wins_over_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv(PROMPT_OVERRIDE_DIR_ENV_VAR, str(tmp_path / "from_env"))
    assert resolve_prompt_override_dir(str(tmp_path / "explicit")) == str(tmp_path / "explicit")


def test_resolve_prompt_override_dir_falls_back_to_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv(PROMPT_OVERRIDE_DIR_ENV_VAR, str(tmp_path / "from_env"))
    assert resolve_prompt_override_dir(None) == str(tmp_path / "from_env")


def test_resolve_prompt_override_dir_none_when_neither_given(monkeypatch):
    monkeypatch.delenv(PROMPT_OVERRIDE_DIR_ENV_VAR, raising=False)
    assert resolve_prompt_override_dir(None) is None


# --- 2 & 3. Integration: an override changes one agent's system prompt,
# others keep their defaults, and events.jsonl records which file each
# agent actually used ---

INTAKE_PROPOSAL = json.dumps({
    "target_column": "churned", "task": "binary_classification",
    "id_columns": ["customer_id"], "group_column": None, "time_column": None,
    "positive_label": "1", "reasoning": "churned looks like the binary outcome column.",
})
FEATURE_ENGINEERING_PROPOSAL = json.dumps({
    "drop_columns": [], "derived_features": [], "explanation": "No changes needed.",
})
PROFILER_NARRATIVE = json.dumps({
    "summary": "Synthetic churn dataset.", "recommended_split_strategy": "stratified",
    "key_risks": [], "recommended_next_steps": [],
})
CANDIDATE_A = json.dumps({
    "candidate_id": "candidate_a", "template_id": "sklearn_mixed_pipeline",
    "config": {"numeric_cols": ["age", "income"], "categorical_cols": ["plan_type", "region"],
              "classifier": "logistic_regression"},
    "explanation": "Mixed baseline.",
})
VERIFICATION_APPROVED = json.dumps({"verdict": "approved", "concerns": [], "reasoning": "looks fine"})


def _resp(text=None, tool_calls=None):
    return ModelResponse(
        text=text, tool_calls=tool_calls or [], raw=None, latency_seconds=0.01,
        model="fake-model", input_tokens=1, output_tokens=1,
    )


def _tool_names(tools):
    if not tools:
        return frozenset()
    return frozenset(t["function"]["name"] for t in tools)


_INTAKE_TOOLS = frozenset({"get_raw_schema"})
_FEATURE_ENGINEERING_TOOLS = frozenset({"get_dataset_profile", "list_feature_ops"})
_PROFILER_TOOLS = frozenset({"get_dataset_profile"})
_MODELING_TOOLS = frozenset({"get_dataset_profile", "list_templates"})
_VERIFICATION_TOOLS = frozenset({"get_candidate_review_bundle"})


class RecordingFakeClient:
    """Dispatches purely on which TOOLS a call has access to (stable
    regardless of system-prompt wording) and how many messages are in
    the conversation so far — mirrors tests/test_orchestrator.py's
    dispatch discipline, just keyed differently since this test's whole
    point is that the system prompt's exact text is NOT a safe dispatch
    key once overrides are in play. Records every system prompt seen,
    per tool-set, so the test can assert what was actually sent."""

    def __init__(self):
        self.system_prompts_by_tools: dict[frozenset, list[str]] = {}

    def call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
        n = len(messages)
        names = _tool_names(tools)
        if names:
            self.system_prompts_by_tools.setdefault(names, []).append(messages[0]["content"])

        if names == _INTAKE_TOOLS:
            if n == 2:
                return _resp(tool_calls=[{"id": "t1", "name": "get_raw_schema", "arguments": "{}"}])
            return _resp(text=INTAKE_PROPOSAL)

        if names == _FEATURE_ENGINEERING_TOOLS:
            if n == 2:
                return _resp(tool_calls=[{"id": "f1", "name": "get_dataset_profile", "arguments": "{}"}])
            if n == 4:
                return _resp(tool_calls=[{"id": "f2", "name": "list_feature_ops", "arguments": "{}"}])
            return _resp(text=FEATURE_ENGINEERING_PROPOSAL)

        if names == _PROFILER_TOOLS:
            if n == 2:
                return _resp(tool_calls=[{"id": "t2", "name": "get_dataset_profile", "arguments": "{}"}])
            return _resp(text=PROFILER_NARRATIVE)

        if names == _MODELING_TOOLS:
            if n == 2:
                return _resp(tool_calls=[{"id": "t3", "name": "get_dataset_profile", "arguments": "{}"}])
            if n == 4:
                return _resp(tool_calls=[{"id": "t4", "name": "list_templates", "arguments": "{}"}])
            return _resp(text=CANDIDATE_A)

        if names == _VERIFICATION_TOOLS:
            if n == 2:
                return _resp(tool_calls=[{"id": "v1", "name": "get_candidate_review_bundle", "arguments": "{}"}])
            return _resp(text=VERIFICATION_APPROVED)

        # Analyst-style final summary: a plain text call with no tools.
        return _resp(text="This is a plain-language summary of the modeling run.")


@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    monkeypatch.setenv("RIT_BASE_URL", "http://example.invalid/v1")
    monkeypatch.setenv("RIT_API_KEY", "dummy")
    monkeypatch.delenv(PROMPT_OVERRIDE_DIR_ENV_VAR, raising=False)
    yield


@pytest.fixture
def dataset_csv(tmp_path):
    rng = np.random.RandomState(0)
    n = 400
    df = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "age": rng.randint(18, 80, size=n),
        "income": rng.exponential(50000, size=n),
        "plan_type": rng.choice(["basic", "premium", "pro"], size=n),
        "region": rng.choice(["north", "south", "east", "west"], size=n),
        "churned": rng.binomial(1, 0.35, size=n),
    })
    path = tmp_path / "churn.csv"
    df.to_csv(path, index=False)
    return path


def _run_in(tmp_path, argv):
    old_cwd = Path.cwd()
    old_argv = sys.argv
    os.chdir(tmp_path)
    sys.argv = argv
    try:
        run_orchestrator.main()
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv


def _read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_override_changes_one_agents_prompt_while_others_keep_defaults(dataset_csv, tmp_path, monkeypatch):
    override_dir = tmp_path / "prompt_overrides"
    override_dir.mkdir()
    override_text = "OVERRIDDEN MODELING PROMPT — nothing like the shipped default's wording."
    (override_dir / "modeling.md").write_text(override_text)

    fake_client = RecordingFakeClient()
    monkeypatch.setattr(ModelClient, "call", lambda self, *a, **k: fake_client.call(*a, **k))

    _run_in(tmp_path, [
        "run_orchestrator.py",
        "--data", str(dataset_csv),
        "--target", "churned",
        "--id-columns", "customer_id",
        "--max-candidates", "1",
        "--prompt-override-dir", str(override_dir),
        "--run-id", "test_prompt_override",
    ])

    report = json.loads((tmp_path / "runs" / "test_prompt_override" / "orchestrator_report.json").read_text())
    assert report["status"] == "success"

    # the overridden agent (modeling) got the override text, verbatim
    modeling_prompts = fake_client.system_prompts_by_tools[_MODELING_TOOLS]
    assert all(p == override_text for p in modeling_prompts)

    # every other agent in the SAME run still got its shipped default
    default_fe = (DEFAULT_PROMPTS_DIR / "feature_engineering.md").read_text()
    default_profiler = (DEFAULT_PROMPTS_DIR / "profiler.md").read_text()
    default_verification = (DEFAULT_PROMPTS_DIR / "verification.md").read_text()

    assert all(p == default_fe for p in fake_client.system_prompts_by_tools[_FEATURE_ENGINEERING_TOOLS])
    assert all(p == default_profiler for p in fake_client.system_prompts_by_tools[_PROFILER_TOOLS])
    assert all(p == default_verification for p in fake_client.system_prompts_by_tools[_VERIFICATION_TOOLS])
    assert override_text not in default_fe
    assert override_text not in default_profiler

    # events.jsonl records which file each agent actually used
    events = _read_events(tmp_path / "runs" / "test_prompt_override" / "events.jsonl")
    prompt_events = {e["phase"]: e["payload"] for e in events if e["type"] == "prompt_loaded"}

    assert prompt_events["modeling"]["source"] == "override"
    assert prompt_events["modeling"]["path"] == str(override_dir / "modeling.md")

    assert prompt_events["feature_engineering"]["source"] == "default"
    assert prompt_events["feature_engineering"]["path"] == str(DEFAULT_PROMPTS_DIR / "feature_engineering.md")
    assert prompt_events["profiler"]["source"] == "default"
    assert prompt_events["profiler"]["path"] == str(DEFAULT_PROMPTS_DIR / "profiler.md")
    assert prompt_events["verification"]["source"] == "default"
    assert prompt_events["verification"]["path"] == str(DEFAULT_PROMPTS_DIR / "verification.md")


def test_no_override_dir_given_every_agent_prompt_loaded_event_is_default(dataset_csv, tmp_path, monkeypatch):
    fake_client = RecordingFakeClient()
    monkeypatch.setattr(ModelClient, "call", lambda self, *a, **k: fake_client.call(*a, **k))

    _run_in(tmp_path, [
        "run_orchestrator.py",
        "--data", str(dataset_csv),
        "--target", "churned",
        "--id-columns", "customer_id",
        "--max-candidates", "1",
        "--run-id", "test_prompt_default",
    ])

    events = _read_events(tmp_path / "runs" / "test_prompt_default" / "events.jsonl")
    prompt_events = [e for e in events if e["type"] == "prompt_loaded"]
    assert prompt_events  # at least one agent ran and emitted a prompt_loaded event
    assert all(e["payload"]["source"] == "default" for e in prompt_events)
    for e in prompt_events:
        assert e["payload"]["path"] == str(DEFAULT_PROMPTS_DIR / f"{e['phase']}.md")
