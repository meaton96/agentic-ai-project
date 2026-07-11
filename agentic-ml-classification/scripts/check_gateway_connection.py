#!/usr/bin/env python3
"""
Verify the LiteLLM gateway is routing correctly to RIT (and local
models, if configured). Run this after check_rit_connection.py passes
and after `docker compose up -d model-gateway`.
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
    base_url = os.environ.get("MODEL_GATEWAY_BASE_URL", "http://localhost:4000/v1")
    api_key = os.environ.get("LITELLM_MASTER_KEY", "change_me_to_something_random")

    # these are the model_name values from configs/litellm.yaml, not the
    # raw provider model IDs
    models = ["rit-qwen3-coder-30b", "rit-gemma4-26b", "rit-qwen3-8b"]

    client = ModelClient(base_url=base_url, api_key=api_key, default_model=models[0])

    report = {"gateway_base_url": base_url, "models": []}
    prompt = 'Respond with ONLY valid JSON: {"status": "ok", "via": "gateway"}'

    for model in models:
        try:
            parsed, response = client.call_json(prompt, model=model)
            ok = parsed is not None
            entry = {
                "model": model, "ok": ok,
                "latency_seconds": response.latency_seconds,
                "raw_text_preview": (response.text or "")[:200],
            }
        except Exception as e:
            entry = {"model": model, "ok": False, "error": f"{type(e).__name__}: {e}"}
        status = "OK" if entry["ok"] else "FAILED"
        print(f"  [{status}] {model}")
        report["models"].append(entry)

    Path("reports").mkdir(exist_ok=True)
    Path("reports/gateway_connection_report.json").write_text(json.dumps(report, indent=2))

    if not any(m["ok"] for m in report["models"]):
        print("FAILED: gateway did not route any model successfully.")
        sys.exit(1)
    print("SUCCESS: gateway routing confirmed.")


if __name__ == "__main__":
    main()
