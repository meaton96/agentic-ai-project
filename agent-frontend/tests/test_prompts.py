"""
GET/PUT/DELETE /api/prompts — override CRUD, and the two things that
matter most about it: overrides never touch ../agentic-ml-classification,
and use_prompt_overrides=False guarantees default-prompt behavior even
when an override already exists on disk for that agent.
"""
from __future__ import annotations

from agentic_ml.prompt_loader import DEFAULT_PROMPTS_DIR

from server.prompts import resolve_override_dir
from tests.conftest import wait_for_status

PIPELINE_REPO_ROOT = DEFAULT_PROMPTS_DIR.parent


def test_list_prompts_returns_all_known_agents_with_no_override_initially(client):
    resp = client.get("/api/prompts")
    assert resp.status_code == 200
    agents = {p["agent"] for p in resp.json()}
    assert agents == {"intake", "feature_engineering", "profiler", "modeling", "verification", "deep_dive", "planner"}
    for prompt in resp.json():
        assert prompt["has_override"] is False
        assert prompt["override_content"] is None
        assert prompt["default_content"] == (DEFAULT_PROMPTS_DIR / f"{prompt['agent']}.md").read_text()


def test_put_then_get_reflects_the_override(client):
    resp = client.put("/api/prompts/modeling", json={"content": "OVERRIDDEN modeling prompt."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["has_override"] is True
    assert body["override_content"] == "OVERRIDDEN modeling prompt."

    listed = {p["agent"]: p for p in client.get("/api/prompts").json()}
    assert listed["modeling"]["has_override"] is True
    assert listed["modeling"]["override_content"] == "OVERRIDDEN modeling prompt."
    # every other agent is untouched
    assert listed["profiler"]["has_override"] is False


def test_delete_reverts_to_default(client):
    client.put("/api/prompts/verification", json={"content": "custom verification prompt"})
    assert client.get("/api/prompts").json()

    resp = client.delete("/api/prompts/verification")
    assert resp.status_code == 200
    assert resp.json()["has_override"] is False
    assert resp.json()["override_content"] is None

    listed = {p["agent"]: p for p in client.get("/api/prompts").json()}
    assert listed["verification"]["has_override"] is False


def test_delete_when_no_override_exists_is_a_no_op_not_an_error(client):
    resp = client.delete("/api/prompts/intake")
    assert resp.status_code == 200
    assert resp.json()["has_override"] is False


def test_put_unknown_agent_is_404(client):
    resp = client.put("/api/prompts/not_a_real_agent", json={"content": "x"})
    assert resp.status_code == 404


def test_delete_unknown_agent_is_404(client):
    resp = client.delete("/api/prompts/not_a_real_agent")
    assert resp.status_code == 404


def test_override_dir_never_resolves_inside_the_pipeline_repo():
    resolved = resolve_override_dir().resolve()
    assert not resolved.is_relative_to(PIPELINE_REPO_ROOT), (
        f"prompt override dir {resolved} must never be inside the pipeline repo {PIPELINE_REPO_ROOT}"
    )


def test_writing_an_override_leaves_the_pipelines_own_prompt_file_untouched(client):
    shipped_path = DEFAULT_PROMPTS_DIR / "profiler.md"
    original = shipped_path.read_text()
    original_mtime = shipped_path.stat().st_mtime

    client.put("/api/prompts/profiler", json={"content": "some override text — must not land here"})

    assert shipped_path.read_text() == original
    assert shipped_path.stat().st_mtime == original_mtime


# --- end-to-end: the override actually reaches (or is correctly kept
# from reaching) a launched run ---

# Keeps the stub's "Profiler agent" substring dispatch working (see
# tests/conftest.py's static_fake_call) while still being obviously
# different text from the shipped default — proves the override was
# both recorded AND actually used for this agent's real LLM call.
PROFILER_OVERRIDE = (
    "You are the Profiler agent in a CUSTOM-OVERRIDDEN pipeline configuration. "
    "You have exactly one tool: get_dataset_profile. Call it once, then respond "
    "with ONLY valid JSON matching: "
    '{"summary": "...", "recommended_split_strategy": "...", "key_risks": [], '
    '"recommended_next_steps": []}'
)


def _prompt_loaded_events(run_dir_events):
    return {e["phase"]: e["payload"] for e in run_dir_events if e["type"] == "prompt_loaded"}


def _read_events(events_path):
    import json
    return [json.loads(line) for line in events_path.read_text().splitlines()]


def test_use_prompt_overrides_true_actually_uses_the_override(stubbed_client, dataset_csv, tmp_path):
    client = stubbed_client
    client.put("/api/prompts/profiler", json={"content": PROFILER_OVERRIDE})

    resp = client.post("/api/runs", json={
        "dataset": dataset_csv, "orchestrator": "static", "target": "churned",
        "skip_feature_engineering": True, "max_candidates": 1, "use_prompt_overrides": True,
    })
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]
    final = wait_for_status(client, run_id)
    assert final["status"] == "completed", final

    from agentic_ml.paths import run_dir as resolve_run_dir
    events = _read_events(resolve_run_dir(run_id) / "events.jsonl")
    prompt_events = _prompt_loaded_events(events)

    assert prompt_events["profiler"]["source"] == "override"
    assert prompt_events["profiler"]["path"] == str(resolve_override_dir() / "profiler.md")
    # untouched agents in the same run still got their shipped defaults
    assert prompt_events["modeling"]["source"] == "default"
    assert prompt_events["verification"]["source"] == "default"


def test_use_prompt_overrides_false_ignores_an_existing_override(stubbed_client, dataset_csv):
    client = stubbed_client
    # an override exists on disk for profiler...
    client.put("/api/prompts/profiler", json={"content": PROFILER_OVERRIDE})

    # ...but this run doesn't opt in (use_prompt_overrides defaults False)
    resp = client.post("/api/runs", json={
        "dataset": dataset_csv, "orchestrator": "static", "target": "churned",
        "skip_feature_engineering": True, "max_candidates": 1,
    })
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]
    final = wait_for_status(client, run_id)
    assert final["status"] == "completed", final

    from agentic_ml.paths import run_dir as resolve_run_dir
    events = _read_events(resolve_run_dir(run_id) / "events.jsonl")
    prompt_events = _prompt_loaded_events(events)

    assert prompt_events["profiler"]["source"] == "default"
    assert prompt_events["profiler"]["path"] == str(DEFAULT_PROMPTS_DIR / "profiler.md")
