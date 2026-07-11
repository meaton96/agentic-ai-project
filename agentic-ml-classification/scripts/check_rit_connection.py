#!/usr/bin/env python3
"""
Milestone 0 (replaces the OpenClaw bootstrap milestone):
    call RIT through a direct OpenAI-compatible client, parse a
    structured JSON response, write a report.

No sessions, no compaction, no CLI wrapper — just the ModelClient.
Run:
    set -a; source .env; set +a
    python scripts/check_rit_connection.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_ml.model_client import ModelClient


def main():
    base_url = os.environ["RIT_BASE_URL"]
    api_key = os.environ["RIT_API_KEY"]
    models = [m.strip() for m in os.environ.get(
        "RIT_TEST_MODELS", "qwen3-coder:30b"
    ).split(",") if m.strip()]

    client = ModelClient(base_url=base_url, api_key=api_key, default_model=models[0])

    report = {
        "base_url": base_url,
        "api_key_present": bool(api_key),
        "tested_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": [],
    }

    prompt = (
        'Respond with ONLY valid JSON, no prose, no markdown fences. '
        'Schema: {"status": "ok", "model_ack": "<name you were called with>", '
        '"one_fact": "<one true short fact>"}'
    )

    for model in models:
        parsed, response = client.call_json(prompt, model=model)
        entry = {
            "model": model,
            "ok": parsed is not None,
            "latency_seconds": response.latency_seconds,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "raw_text_preview": (response.text or "")[:300],
            "parsed": parsed,
        }
        status = "OK" if entry["ok"] else "FAILED TO PARSE"
        print(f"  [{status}] {model} ({entry['latency_seconds']}s, "
              f"{entry['input_tokens']}in/{entry['output_tokens']}out tokens)")
        report["models"].append(entry)

    Path("reports").mkdir(exist_ok=True)
    out_path = Path("reports/rit_connection_report.json")
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {out_path}")

    if not any(m["ok"] for m in report["models"]):
        print("FAILED: no model produced a parseable structured response.")
        sys.exit(1)
    print("SUCCESS: at least one model responded with valid structured JSON.")


if __name__ == "__main__":
    main()
