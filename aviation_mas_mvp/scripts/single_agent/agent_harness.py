"""
agent_harness.py
=================
A standalone agentic loop that interfaces with an OpenAI-compatible
chat-completions endpoint using the function-calling API.

It handles message threading, tool dispatching to local functions,
and stop conditions. Includes both a real HTTP client (VLLMClient)
and a scripted mock client (ScriptedClient) for testing.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field

from scripts import tools as T

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI tools function-calling format).
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
    Execute a requested tool call against local functions.
    Enforces a strict limit of one `recommend` call per session.
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
             max_turns: int = 12, verbose: bool = True) -> AgentRun:
    """
    Executes the agent loop until a final text summary is produced or max turns are reached.
    """
    run = AgentRun(messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_goal},
    ])
    called_so_far: list[str] = []

    for turn in range(1, max_turns + 1):
        # 1. Fetch response from the client
        msg = client.chat(run.messages, TOOLS_SCHEMA)
        run.messages.append(msg)

        tool_calls = msg.get("tool_calls") or []
        
        # 2. Check for stop condition: model returns standard content instead of a tool
        if not tool_calls:
            run.final_summary = msg.get("content", "")
            run.stopped_reason = "model returned final content (no tool call)"
            if verbose:
                print(f"[turn {turn}] model stopped with final summary.")
            break

        # 3. Handle edge case of multiple parallel tool calls
        if len(tool_calls) > 1 and verbose:
            print(f"[turn {turn}] WARNING: model requested {len(tool_calls)} tool "
                  f"calls at once. Executing only the first.")
            
        call = tool_calls[0]
        name = call["function"]["name"]
        
        # 4. Parse arguments and dispatch
        try:
            args = json.loads(call["function"]["arguments"])
            result, ok = _dispatch(name, args, called_so_far)
        except json.JSONDecodeError as e:
            args, result, ok = {}, {"error": f"could not parse arguments JSON: {e}"}, False

        if ok:
            called_so_far.append(name)
            
        # Record trace history
        run.trace.append(ToolCallRecord(name=name, args=args, result=result, ok=ok))
        
        if verbose:
            tag = "OK" if ok else "ERROR"
            shown = {k: v for k, v in result.items() if k != "items"}
            print(f"[turn {turn}] {name}({args}) -> {tag}: {shown}")

        # 5. Append the executed tool's result to the message thread
        run.messages.append({
            "role": "tool", "tool_call_id": call.get("id", f"call_{turn}"),
            "name": name, "content": json.dumps(result),
        })

        # 6. Abort loop if a tool fails
        if not ok:
            run.stopped_reason = f"tool {name!r} errored; stopping per brief rules"
            if verbose:
                print(f"[turn {turn}] stopping: {run.stopped_reason}")
            break
    else:
        run.stopped_reason = f"hit max_turns={max_turns} without a final summary"

    return run


def order_diff(run: AgentRun) -> dict:
    """Compare the actual execution order against the expected sequence."""
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
    HTTP client for an OpenAI-compatible /chat/completions endpoint.
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


class OllamaClient:
    def __init__(self, base_url: str, model: str, api_key: str = "not-needed",
                temperature: float = 0.1, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self._warmed_up = False

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        import requests
        
        # Simple warm-up to force Ollama to load the model into VRAM
        if not self._warmed_up:
            print(f"[*] Warming up {self.model} on the server (this may take a minute)...")
            try:
                requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={"model": self.model, "messages": [{"role": "user", "content": "hi"}]},
                    timeout=self.timeout
                )
            except requests.exceptions.ReadTimeout:
                # Gateway timed out, but Ollama is likely still loading it in the background!
                print("[*] Gateway timed out during warm-up. Retrying actual request now...")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 504:
                    print("[*] 504 Gateway Timeout during warm-up. Retrying actual request now...")
            self._warmed_up = True

        # Actual Agentic Call
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