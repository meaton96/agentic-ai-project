#!/usr/bin/env python3
"""
OpenClaw model sweep runner.

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
import subprocess
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

FINAL_NAME         = "PIPELINE_FINAL.md"
PROVIDER           = "vllm"
CONTAINER_NAME     = "openclaw-poc"

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

# ── Logging ──────────────────────────────────────────────────────────────────

def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def log(msg: str, indent: int = 4) -> None:
    print(f"[{ts()}] {'  ' * indent}{msg}")

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


def folder_summary() -> str:
    """One-line snapshot of all staging folders."""
    def ls(p: Path) -> str:
        files = list(p.glob("*.md"))
        if not files:
            return "empty"
        return ", ".join(
            f"{f.name}({f.stat().st_size}B)" for f in files
        )
    return (
        f"task_input:[{ls(TASK_INPUT)}] "
        f"planner_out:[{ls(PLANNER_OUT)}] "
        f"executor_out:[{ls(EXECUTOR_OUT)}] "
        f"final:[{ls(FINAL_REPORTS)}]"
    )


def pipeline_stage() -> str:
    if any(TASK_INPUT.glob("*.md")):
        return "awaiting planner"
    if any(PLANNER_OUT.glob("*.md")):
        return "awaiting executor"
    if any(EXECUTOR_OUT.glob("*.md")):
        return "awaiting synthesizer"
    if any(FINAL_REPORTS.glob("*.md")):
        return "FINAL READY"
    return "idle (all folders empty)"


def tail_docker_logs(lines: int = 8) -> None:
    """Print the last N lines of container logs."""
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(lines), CONTAINER_NAME],
            capture_output=True, text=True, timeout=5
        )
        output = (result.stdout + result.stderr).strip()
        if output:
            log("── docker logs (tail) ──────────────────────", indent=1)
            for line in output.splitlines():
                log(line, indent=2)
            log("────────────────────────────────────────────", indent=1)
    except Exception as e:
        log(f"(could not read docker logs: {e})", indent=2)


def next_cron_in(interval_min: int = 3) -> int:
    """Seconds until the next cron window (*/N schedule)."""
    now = datetime.now()
    elapsed_in_window = (now.minute % interval_min) * 60 + now.second
    return interval_min * 60 - elapsed_in_window


# ── Report wait loop ─────────────────────────────────────────────────────────

def wait_for_report(timeout_s: int, poll_s: int = 15) -> Path | None:
    target = FINAL_REPORTS / FINAL_NAME
    deadline = time.time() + timeout_s
    last_stage = None
    poll_count = 0

    log(f"next cron window in ~{next_cron_in()}s")

    while time.time() < deadline:
        if target.exists() and target.stat().st_size > 0:
            time.sleep(3)   # let the write finish
            return target

        remaining = int(deadline - time.time())
        stage = pipeline_stage()
        poll_count += 1

        # Always print stage + folder snapshot
        log(f"poll #{poll_count:03d}  {remaining:4d}s left  │  {stage}")
        log(folder_summary(), indent=5)

        # Print docker logs every 4 polls (~60s) or when stage changes
        if stage != last_stage or poll_count % 4 == 0:
            tail_docker_logs(lines=6)
            last_stage = stage

        time.sleep(poll_s)

    return None


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
    checks["score"]      = sum(bool(v) for k, v in checks.items() if k != "score")
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
        "mentions_speed", "word_count", "stalled_stage", "timestamp",
    ]
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, model in enumerate(models, 1):
            print(f"\n{'═'*60}")
            print(f"[{ts()}] [{i:2d}/{len(models)}] {model}")
            print(f"{'═'*60}")

            log(f"writing config -> {PROVIDER}/{model}")
            set_model(model)
            log(f"waiting {settle_s}s for OpenClaw restart …")
            time.sleep(settle_s)

            log("checking staging folders before drop:")
            log(folder_summary(), indent=5)
            clean_staging()
            drop_brief()
            log(f"task_brief.md dropped into 00_task_input/")
            log(f"next cron window in ~{next_cron_in()}s — then up to 3min per stage")

            t0 = time.time()
            report = wait_for_report(timeout_s)
            wall = round(time.time() - t0, 1)

            row = {"model": model, "wall_time_s": wall,
                   "timestamp": datetime.now().isoformat(timespec="seconds")}

            if report is None:
                stalled = pipeline_stage()
                log(f"❌ TIMEOUT after {wall}s — stalled at: {stalled}")
                tail_docker_logs(lines=15)
                row.update({"completed": False, "score": 0, "word_count": 0,
                            "stalled_stage": stalled})
                for k in fieldnames:
                    row.setdefault(k, "")
            else:
                text = report.read_text(errors="replace")
                dest = RESULTS_DIR / f"{safe_name(model)}_FINAL.md"
                shutil.copy(report, dest)
                checks = verify(text)
                log(f"✅ completed in {wall}s — score {checks['score']}/7 "
                    f"({checks['word_count']} words)")
                log(f"   saved -> {dest.name}")
                row.update({"completed": True, **checks})

            writer.writerow(row)
            fh.flush()
            clean_staging()

    print(f"\n[{ts()}] Done. Results: {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="OpenClaw model sweep")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS,
                    help="Subset of models to test (default: all 15)")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Models to skip (already completed)")
    ap.add_argument("--timeout",  type=int, default=600,
                    help="Per-model timeout in seconds (default 600)")
    ap.add_argument("--settle",   type=int, default=25,
                    help="Seconds to wait after config change for restart")
    args = ap.parse_args()

    models_to_run = [m for m in args.models if m not in args.skip]
    if args.skip:
        print(f"[{ts()}] Skipping {len(args.skip)} already-completed models")
    print(f"[{ts()}] Sweeping {len(models_to_run)} models | "
          f"timeout {args.timeout}s | settle {args.settle}s")
    run_sweep(models_to_run, args.timeout, args.settle)