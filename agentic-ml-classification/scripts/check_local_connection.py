#!/usr/bin/env python3
"""
Verify a local OpenAI-compatible model server (vLLM/Ollama/SGLang) is
reachable and, critically, that its tool-calling actually works — every
agent in this pipeline runs through agent_runtime.ToolCallingAgent,
which depends on the server returning real tool_calls, not just plain
text. Added alongside check_rit_connection.py/check_gateway_connection.py
because RIT's shared endpoint returns real, intermittent 504s under
agentic tool-calling load; a local server sidesteps that for development.

Run:
    set -a; source .env; set +a
    python scripts/check_local_connection.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_ml.agent_runtime import Tool, ToolCallingAgent
from agentic_ml.model_client import ModelClient


def main():
    base_url = os.environ.get("LOCAL_MODEL_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("LOCAL_MODEL_API_KEY", "not-needed")
    model = os.environ.get("LOCAL_DEFAULT_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")

    client = ModelClient(base_url=base_url, api_key=api_key, default_model=model)
    report = {
        "base_url": base_url, "model": model,
        "tested_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    print(f"Checking {base_url} (model={model})...")

    # 1. Plain structured-JSON completion, no tools.
    prompt = 'Respond with ONLY valid JSON: {"status": "ok", "via": "local"}'
    parsed, response = client.call_json(prompt, model=model)
    report["plain_json_ok"] = parsed is not None
    report["plain_json_latency_seconds"] = response.latency_seconds
    print(f"  [{'OK' if report['plain_json_ok'] else 'FAILED'}] plain JSON completion "
          f"({response.latency_seconds}s)")

    # 2. Tool-calling — this is what every agent in this pipeline actually
    # depends on (ToolCallingAgent), not just plain completions.
    calls = {"n": 0}

    def get_fact() -> dict:
        calls["n"] += 1
        return {"fact": "the sky is blue"}

    tool = Tool(
        name="get_fact", description="Returns one short factual statement. Call it once.",
        parameters={"type": "object", "properties": {}, "required": []}, handler=get_fact,
    )
    agent = ToolCallingAgent(
        model_client=client, tools=[tool],
        system_prompt="Call get_fact once, then respond with ONLY valid JSON: "
                      '{"fact": "<the fact you got>"}',
        model=model, max_turns=4,
    )
    result = agent.run("Get the fact and report it.")
    tool_called = calls["n"] > 0
    report["tool_calling_ok"] = tool_called and result.final_text is not None
    report["tool_call_count"] = calls["n"]
    report["stopped_reason"] = result.stopped_reason
    print(f"  [{'OK' if report['tool_calling_ok'] else 'FAILED'}] tool-calling "
          f"(tool called {calls['n']}x, stopped_reason={result.stopped_reason})")

    Path("reports").mkdir(exist_ok=True)
    out_path = Path("reports/local_connection_report.json")
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {out_path}")

    if not (report["plain_json_ok"] and report["tool_calling_ok"]):
        print("FAILED: local server did not pass both checks.")
        sys.exit(1)
    print("SUCCESS: local server confirmed for both plain completions and tool-calling.")


if __name__ == "__main__":
    main()
