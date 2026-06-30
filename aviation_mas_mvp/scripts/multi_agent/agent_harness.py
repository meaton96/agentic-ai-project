"""
agent_harness.py
=================
A minimal, standalone agentic loop that talks DIRECTLY to an OpenAI-compatible
chat-completions endpoint (vLLM, etc.) using the `tools` function-calling API.
No OpenClaw, no framework -- just HTTP + a tool dispatcher, so every message
the model sees and every tool result it gets back is visible and inspectable.

This is an alternative path through SPEC.md Step 2. The tool schemas and the
dispatcher below call the exact same Python functions tools.py already exposes
-- nothing about the tools changes, only who is calling them (a real LLM
instead of orchestrator_sim's fixed plan).

Two LLMClient implementations:
  - VLLMClient    : real HTTP calls to your local OpenAI-compatible endpoint.
                    Edit base_url/model to match your running vLLM server.
  - ScriptedClient: a fake client that plays back a fixed, well-behaved script
                    of tool calls shaped exactly like real tool-calling
                    responses. Used to validate the harness mechanics (message
                    threading, dispatch, stop condition, guardrails) WITHOUT a
                    live model. It does NOT reason about tool results when
                    deciding what to call next -- it's a script, not a model.
                    Swapping in VLLMClient is what closes that gap.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field

from scripts import tools as T

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI "tools" function-calling format). Names match the CLI
# subcommands in tools_manifest.md / agent_task_brief.md.
# ---------------------------------------------------------------------------
TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "inspect",
        "description": "Read-only sensor inventory, class balance, and channel "
                        "count for a flight directory + metadata file. Call "
                        "this FIRST, before any other tool.",
        "parameters": {"type": "object", "properties": {
            "flight_dir": {"type": "string", "description": "Directory of per-flight CSVs"},
            "metadata":   {"type": "string", "description": "Path to metadata.csv"},
        }, "required": ["flight_dir", "metadata"]}}},
    {"type": "function", "function": {
        "name": "featurize",
        "description": "Build the tabular feature table from the flight "
                        "directory. Call after inspect.",
        "parameters": {"type": "object", "properties": {
            "flight_dir": {"type": "string"},
            "metadata":   {"type": "string"},
            "out":        {"type": "string", "description": "Output path for feats.csv"},
        }, "required": ["flight_dir", "metadata", "out"]}}},
    {"type": "function", "function": {
        "name": "classify",
        "description": "Score flights with the trained model: maintenance "
                        "probability + urgency (LOW/MEDIUM/HIGH), ranked. "
                        "Call after featurize.",
        "parameters": {"type": "object", "properties": {
            "feats": {"type": "string"},
            "model": {"type": "string", "description": "Path to model.joblib"},
            "out":   {"type": "string", "description": "Output path for preds.json"},
        }, "required": ["feats", "model", "out"]}}},
    {"type": "function", "function": {
        "name": "recommend",
        "description": "Append the top-k highest-priority flights to the "
                        "write-only maintenance queue. Call LAST, exactly once "
                        "per run, after classify.",
        "parameters": {"type": "object", "properties": {
            "preds": {"type": "string"},
            "queue": {"type": "string"},
            "top_k": {"type": "integer", "default": 10},
        }, "required": ["preds", "queue"]}}},
]

EXPECTED_ORDER = ["inspect", "featurize", "classify", "recommend"]


@dataclass
class ToolCallRecord:
    name: str
    args: dict
    result: dict
    ok: bool


@dataclass
class AgentRun:
    messages: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    final_summary: str | None = None
    stopped_reason: str | None = None


def _dispatch(name: str, args: dict, called_so_far: list[str]) -> tuple[dict, bool]:
    """
    Execute one tool call for real, against the SAME functions tools.py exposes.
    Enforces the brief's hard rule at the tool layer (defense in depth -- don't
    rely on the prompt alone): `recommend` may run at most once per session.
    This guard lives HERE rather than in tools.recommend_maintenance itself,
    because the tool is intentionally stateless (append-only across runs is the
    point); "have I already called this in THIS session" is orchestrator state,
    not tool state.
    """
    if name == "recommend" and "recommend" in called_so_far:
        return {"error": "recommend already called once this run; refusing a "
                          "second write to the maintenance queue."}, False
    try:
        if name == "inspect":
            r = T.inspect_data(args["flight_dir"], args["metadata"])
        elif name == "featurize":
            r = T.featurize_flights(args["flight_dir"], args["metadata"], args["out"])
        elif name == "classify":
            r = T.classify_flights(args["feats"], args["model"], args["out"])
        elif name == "recommend":
            r = T.recommend_maintenance(args["preds"], args["queue"], top_k=int(args.get("top_k", 10)))
        else:
            return {"error": f"unknown tool {name!r}"}, False
        return r, True
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}, False


def run_agent(client, system_prompt: str, user_goal: str,
             max_turns: int = 12, verbose: bool = True,
             tools_schema: list | None = None, dispatch: dict | None = None) -> AgentRun:
    """
    Drive the function-calling loop:
      1. send messages + tools to client.chat(...)
      2. tool call requested -> dispatch for real -> append result as a
         tool-role message -> back to 1
      3. plain content, no tool call -> that's the HITL summary -> stop
    `client` needs one method: chat(messages, tools) -> an OpenAI-shaped
    `message` dict (content and/or tool_calls).

    `tools_schema`/`dispatch`: override the default single-orchestrator tool
    set (inspect/featurize/classify/recommend). Pass a custom schema + a
    {name: callable(**kwargs) -> dict} dispatch table to run a DIFFERENT
    agent with its own tool subset (see mas_coordinator.py) -- the default
    path (both None) is unchanged from before, so existing callers are
    unaffected.
    """
    schema = tools_schema if tools_schema is not None else TOOLS_SCHEMA
    run = AgentRun(messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_goal},
    ])
    called_so_far: list[str] = []

    for turn in range(1, max_turns + 1):
        msg = client.chat(run.messages, schema)
        run.messages.append(msg)

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            run.final_summary = msg.get("content", "")
            run.stopped_reason = "model returned final content (no tool call)"
            if verbose:
                print(f"[turn {turn}] model stopped with final summary.")
            break

        if len(tool_calls) > 1 and verbose:
            print(f"[turn {turn}] WARNING: model requested {len(tool_calls)} tool "
                  f"calls at once; brief says one at a time. Executing only the first.")
        call = tool_calls[0]
        name = call["function"]["name"]
        try:
            args = json.loads(call["function"]["arguments"])
        except json.JSONDecodeError as e:
            args, result, ok = {}, {"error": f"could not parse arguments JSON: {e}"}, False
        else:
            if dispatch is not None:
                fn = dispatch.get(name)
                if fn is None:
                    result, ok = {"error": f"unknown tool {name!r}"}, False
                else:
                    try:
                        result, ok = fn(**args), True
                    except Exception as e:
                        result, ok = {"error": f"{type(e).__name__}: {e}"}, False
            else:
                result, ok = _dispatch(name, args, called_so_far)

        if ok:
            called_so_far.append(name)
        run.trace.append(ToolCallRecord(name=name, args=args, result=result, ok=ok))
        if verbose:
            tag = "OK" if ok else "ERROR"
            shown = {k: v for k, v in result.items() if k != "items"}
            print(f"[turn {turn}] {name}({args}) -> {tag}: {shown}")

        run.messages.append({
            "role": "tool", "tool_call_id": call.get("id", f"call_{turn}"),
            "name": name, "content": json.dumps(result),
        })

        if not ok:
            run.stopped_reason = f"tool {name!r} errored; stopping per brief rules"
            if verbose:
                print(f"[turn {turn}] stopping: {run.stopped_reason}")
            break
    else:
        run.stopped_reason = f"hit max_turns={max_turns} without a final summary"

    return run


def order_diff(run: AgentRun) -> dict:
    """Compare the agent's actual tool-call order against the prescribed order."""
    actual = [t.name for t in run.trace if t.ok]
    return {
        "expected_order": EXPECTED_ORDER,
        "actual_order": actual,
        "matches_expected": actual == EXPECTED_ORDER,
        "recommend_call_attempts": sum(1 for t in run.trace if t.name == "recommend"),
        "recommend_successful_writes": sum(1 for t in run.trace if t.name == "recommend" and t.ok),
    }


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
class VLLMClient:
    """
    Real client: HTTP POST to an OpenAI-compatible /chat/completions endpoint.
    EDIT base_url and model to match your running vLLM server, e.g.:
        VLLMClient(base_url="http://localhost:8000/v1", model="qwen3-coder:30b")
    Run this from a machine that can actually reach that URL (not this sandbox).
    """
    def __init__(self, base_url: str, model: str, api_key: str = "not-needed",
                temperature: float = 0.1, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        import requests
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages, "tools": tools,
                  "tool_choice": "auto", "temperature": self.temperature},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]


class ScriptedClient:
    """
    Fake client: plays back a fixed, well-behaved script of tool calls (or a
    deliberately bad one, e.g. a double `recommend`) shaped exactly like real
    tool-calling responses. Proves the harness's plumbing is correct in
    isolation from whether a real model reasons well. Swap to VLLMClient to
    test that part.
    """
    def __init__(self, flight_dir: str, metadata: str, feats: str, model: str,
                preds: str, queue: str, top_k: int = 10,
                double_call_recommend: bool = False):
        self._script = [
            ("inspect",   {"flight_dir": flight_dir, "metadata": metadata}),
            ("featurize", {"flight_dir": flight_dir, "metadata": metadata, "out": feats}),
            ("classify",  {"feats": feats, "model": model, "out": preds}),
            ("recommend", {"preds": preds, "queue": queue, "top_k": top_k}),
        ]
        if double_call_recommend:
            self._script.append(("recommend", {"preds": preds, "queue": queue, "top_k": top_k}))
        self._i = 0

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        if self._i < len(self._script):
            name, args = self._script[self._i]
            self._i += 1
            return {"role": "assistant", "content": None,
                    "tool_calls": [{"id": f"call_{self._i}", "type": "function",
                                   "function": {"name": name, "arguments": json.dumps(args)}}]}
        last_tool_msgs = [m for m in messages if m.get("role") == "tool"]
        seen = {m["name"]: json.loads(m["content"]) for m in last_tool_msgs}
        insp, cla, rec = seen.get("inspect", {}), seen.get("classify", {}), seen.get("recommend", {})
        bal = {int(k): int(v) for k, v in insp.get("label_balance", {}).items()}
        text = (
            "HUMAN-IN-THE-LOOP MAINTENANCE SUMMARY\n"
            f"Run scope : {insp.get('n_flights')} flights | balance {bal} | "
            f"{insp.get('n_channels')} channels\n"
            f"Findings  : {cla.get('n_high')} HIGH of {cla.get('n_scored')} scored\n"
            f"Queued    : {rec.get('n_queued')} -> {rec.get('queue')}\n"
            "Confidence: MODERATE -- balance and channel detection look usable.\n"
            "Awaiting technician approval before committing recommendations."
        )
        return {"role": "assistant", "content": text, "tool_calls": []}
