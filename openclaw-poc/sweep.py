#!/usr/bin/env python3
"""
OpenClaw model sweep runner.

For each model:
  1. Edit config/openclaw.json -> agents.defaults.model.primary
     (OpenClaw auto-restarts its node backend on config change)
  2. Wait for restart to settle
  3. Clean staging folders, drop task_brief.md into workspace/00_task_input/
  4. Poll workspace/03_final_reports/PIPELINE_FINAL.md until it appears
     (or timeout)
  5. Copy the report to results/<model>_FINAL.md, run the 7-point
     verification, append a row to results/sweep_results.csv
  6. Clean up and move to the next model

Run from the openclaw-poc root (the folder containing config/ and workspace/):
    python3 sweep.py
    python3 sweep.py --models qwen3:8b qwen3-coder:30b    # subset
    python3 sweep.py --timeout 900                        # slower models
"""

import argparse
import csv
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

# ── Paths (host side, relative to openclaw-poc root) ───────────────────────
ROOT          = Path(__file__).resolve().parent
CONFIG_PATH   = ROOT / "config" / "openclaw.json"
WORKSPACE     = ROOT / "workspace"
TASK_INPUT    = WORKSPACE / "00_task_input"
PLANNER_OUT   = WORKSPACE / "01_planner_out"
EXECUTOR_OUT  = WORKSPACE / "02_executor_out"
FINAL_REPORTS = WORKSPACE / "03_final_reports"
TASK_BRIEF    = ROOT / "task_brief.md"
RESULTS_DIR   = ROOT / "results"

FINAL_NAME    = "PIPELINE_FINAL.md"
PROVIDER      = "rit"   # must match the provider key in openclaw.json

# Models that passed the notebook benchmark (15 candidates).
DEFAULT_MODELS = [
    "qwen3:8b",
    "qwen3-coder:30b",
    "qwen3:latest",
    "gemma4:latest",
    "qwen3.6:35b-a3b-nvfp4",
    "qwen3-coder-next:latest",
    "qwen3.5:latest",
    "devstral:latest",
    "gemma4:26b",
    "qwen3.5:27b",
    "gpt-oss:120b",
    "qwen3.5:35b-a3b-coding-nvfp4",
    "qwen3:1.7b",
    "llama3.1:70b",
    "cogito:70b",
]

PASSING_MODEL_NAMES = [
    "gemma4", "qwen3.6", "cogito", "qwen3.5", "llama3.1",
    "qwen3-coder", "devstral", "gpt-oss", "qwen3",
]


# ── Helpers ─────────────────────────────────────────────────────────────────

def safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def set_model(model: str) -> None:
    cfg = json.loads(CONFIG_PATH.read_text())
    cfg["agents"]["defaults"]["model"] = {"primary": f"{PROVIDER}/{model}"}
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n")


def clean_staging() -> None:
    for folder in (TASK_INPUT, PLANNER_OUT, EXECUTOR_OUT, FINAL_REPORTS):
        folder.mkdir(parents=True, exist_ok=True)
        for f in folder.iterdir():
            if f.is_file():
                f.unlink()


def drop_brief() -> None:
    shutil.copy(TASK_BRIEF, TASK_INPUT / "task_brief.md")


def wait_for_report(timeout_s: int, poll_s: int = 20) -> Path | None:
    target = FINAL_REPORTS / FINAL_NAME
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if target.exists() and target.stat().st_size > 0:
            # give the writing agent a moment to finish the file
            time.sleep(5)
            return target
        remaining = int(deadline - time.time())
        stage = pipeline_stage()
        print(f"    … waiting ({remaining:4d}s left)  stage: {stage}")
        time.sleep(poll_s)
    return None


def pipeline_stage() -> str:
    """Human-readable indicator of where the pipeline currently is."""
    if any(TASK_INPUT.glob("*.md")):
        return "awaiting planner"
    if any(PLANNER_OUT.glob("*.md")):
        return "awaiting executor"
    if any(EXECUTOR_OUT.glob("*.md")):
        return "awaiting synthesizer"
    return "in flight / idle"


# ── 7-point verification ────────────────────────────────────────────────────

def verify(report_text: str) -> dict:
    text_lower = report_text.lower()
    words = len(report_text.split())
    models_mentioned = sum(1 for m in PASSING_MODEL_NAMES if m in text_lower)

    checks = {
        "exists_nonempty":   len(report_text.strip()) > 0,
        "has_4_headers":     report_text.count("## ") >= 4,
        "mentions_3_models": models_mentioned >= 3,
        "says_recommend":    "recommend" in text_lower,
        "has_role_section":  any(k in text_lower for k in
                                 ("use case", "planner", "executor", "synthesizer")),
        "over_400_words":    words >= 400,
        "mentions_speed":    any(k in text_lower for k in
                                 ("time", "seconds", " fast", " slow", "speed")),
    }
    checks["score"] = sum(bool(v) for k, v in checks.items() if k != "score")
    checks["word_count"] = words
    return checks


# ── Main loop ────────────────────────────────────────────────────────────────

def run_sweep(models: list[str], timeout_s: int, settle_s: int) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "sweep_results.csv"
    fieldnames = [
        "model", "completed", "wall_time_s", "score",
        "exists_nonempty", "has_4_headers", "mentions_3_models",
        "says_recommend", "has_role_section", "over_400_words",
        "mentions_speed", "word_count", "timestamp",
    ]
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, model in enumerate(models, 1):
            print(f"\n[{i:2d}/{len(models)}] ══ {model} ══")

            print(f"    setting model -> {PROVIDER}/{model}")
            set_model(model)
            print(f"    waiting {settle_s}s for OpenClaw restart …")
            time.sleep(settle_s)

            clean_staging()
            drop_brief()
            print("    task brief dropped, polling for final report")

            t0 = time.time()
            report = wait_for_report(timeout_s)
            wall = round(time.time() - t0, 1)

            row = {"model": model, "wall_time_s": wall,
                   "timestamp": datetime.now().isoformat(timespec="seconds")}

            if report is None:
                print(f"    ❌ TIMEOUT after {wall}s (stage: {pipeline_stage()})")
                row.update({"completed": False, "score": 0, "word_count": 0})
                for k in fieldnames:
                    row.setdefault(k, False)
            else:
                text = report.read_text(errors="replace")
                dest = RESULTS_DIR / f"{safe_name(model)}_FINAL.md"
                shutil.copy(report, dest)
                checks = verify(text)
                print(f"    ✅ report in {wall}s — score {checks['score']}/7 "
                      f"({checks['word_count']} words) -> {dest.name}")
                row.update({"completed": True, **checks})

            writer.writerow(row)
            fh.flush()
            clean_staging()

    print(f"\nDone. Results: {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="OpenClaw model sweep")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS,
                    help="Subset of models to test (default: all 15)")
    ap.add_argument("--timeout", type=int, default=600,
                    help="Per-model timeout in seconds (default 600)")
    ap.add_argument("--settle", type=int, default=25,
                    help="Seconds to wait after config change for restart")
    args = ap.parse_args()

    print(f"Sweeping {len(args.models)} models, "
          f"timeout {args.timeout}s each, settle {args.settle}s")
    run_sweep(args.models, args.timeout, args.settle)
