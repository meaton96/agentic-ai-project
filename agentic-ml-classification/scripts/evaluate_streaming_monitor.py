#!/usr/bin/env python3
"""
Real-LLM evaluation of the streaming monitor (run_streaming_monitor.py)
against a real model endpoint — the streaming analog of
evaluate_dynamic_orchestrator.py's task-routing scenario. Not part of
pytest; run manually, and expect real cost/latency/flakiness from
whichever endpoint you point it at.

Cold-starts on a subset of planes from
datasets/processed/ngafid_c28_flights_smoke.csv, then replays several
batches with ONE batch's sensor values deliberately, synthetically
shifted to inject real drift. The property under test: does a real
planner (via monitor_drift's deterministic report and the
retrain_decision agent) actually choose "retrain" for the
drift-injected batch, and "infer_only"/"no_action" for the others —
not whether the resulting model is any good.

Usage:
    set -a; source .env; set +a
    python scripts/evaluate_streaming_monitor.py --use-local
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd

from agentic_ml.harness.streaming import simulate_batches

NGAFID_SENSORS_MEAN_COLS = [
    "volt1__mean", "volt2__mean", "amp1__mean", "amp2__mean", "FQtyL__mean", "FQtyR__mean",
    "E1 FFlow__mean", "E1 OilT__mean", "E1 OilP__mean", "E1 RPM__mean", "E1 CHT1__mean",
    "E1 CHT2__mean", "E1 CHT3__mean", "E1 CHT4__mean", "E1 EGT1__mean", "E1 EGT2__mean",
    "E1 EGT3__mean", "E1 EGT4__mean", "OAT__mean", "IAS__mean", "VSpd__mean", "NormAc__mean",
    "AltMSL__mean",
]


def _run(cmd: list[str], timeout: int = 1200) -> dict:
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)
        returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        returncode, stdout, stderr = -1, e.stdout or "", f"TIMEOUT after {timeout}s"
    return {
        "cmd": " ".join(cmd), "returncode": returncode, "elapsed_s": round(time.time() - start, 1),
        "stdout_tail": (stdout or "")[-2000:], "stderr_tail": (stderr or "")[-2000:],
    }


def build_drift_injected_dataset(
    data_path: Path, group_column: str, id_column: str,
    n_initial_groups: int, batch_size_groups: int, seed: int,
    drift_batch_index: int, out_path: Path,
) -> dict:
    """Two-pass construction, same technique as tests/test_streaming_monitor.py:
    a neutral pass discovers which plane_ids land in `drift_batch_index`
    (group membership doesn't depend on feature values), then those
    planes' sensor means get a large synthetic shift before writing the
    real input CSV the streaming monitor will actually replay.

    Subsets down to only the "__mean" sensor columns (plus id/group/
    target/passthrough columns) before doing any of this: the full smoke
    file has 282 columns, and feature_engineering's get_dataset_profile
    tool output for that many columns overflows a 32K-context local
    model's request budget once combined with a second tool call and the
    response — a pre-existing scalability limit of feature_engineering_
    step/profiler_step on very wide tables, not something this phase
    introduces (see evaluate_dynamic_orchestrator.py's task-routing
    scenario, which sidesteps the same issue by using
    --skip-feature-engineering / --existing-model). The same handful-of-
    representative-sensors subsetting notebooks/dynamic_orchestrator.
    ipynb's deep-dive demo already uses (NGAFID_SENSORS_DEMO)."""
    df = pd.read_csv(data_path)
    keep_cols = [c for c in df.columns if c.endswith("__mean")] + [
        group_column, id_column, "before_after", "date_diff", "split",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    _, batches = simulate_batches(
        df, group_column=group_column, id_column=id_column,
        n_initial_groups=n_initial_groups, batch_size_groups=batch_size_groups, seed=seed,
    )
    if drift_batch_index >= len(batches):
        raise ValueError(f"drift_batch_index={drift_batch_index} but only {len(batches)} batches exist")
    drift_plane_ids = set(batches[drift_batch_index][group_column])

    rng = np.random.RandomState(seed + 1)
    injected = df.copy()
    mask = injected[group_column].isin(drift_plane_ids)
    present_cols = [c for c in NGAFID_SENSORS_MEAN_COLS if c in injected.columns]
    for col in present_cols:
        col_std = injected[col].std() or 1.0
        injected.loc[mask, col] = injected.loc[mask, col] + rng.normal(8.0 * col_std, col_std, size=mask.sum())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    injected.to_csv(out_path, index=False)
    return {
        "n_batches": len(batches), "drift_batch_index": drift_batch_index,
        "drift_plane_ids": sorted(str(p) for p in drift_plane_ids),
        "n_shifted_columns": len(present_cols),
    }


def scenario_streaming_drift(endpoint_args: list[str]) -> dict:
    data_path = REPO_ROOT / "datasets/processed/ngafid_c28_flights_smoke.csv"
    if not data_path.exists():
        return {"scenario": "streaming_drift", "skipped": True,
                "reason": f"{data_path} not found — run scripts/featurize_ngafid_flights.py first"}

    n_initial_groups, batch_size_groups, seed, drift_batch_index = 25, 7, 42, 1
    injected_path = REPO_ROOT / "runs" / "_eval_streaming_drift_input.csv"
    injection_info = build_drift_injected_dataset(
        data_path, group_column="plane_id", id_column="id",
        n_initial_groups=n_initial_groups, batch_size_groups=batch_size_groups, seed=seed,
        drift_batch_index=drift_batch_index, out_path=injected_path,
    )

    run_id = f"eval_streaming_{uuid.uuid4().hex[:6]}"
    res = _run([sys.executable, "scripts/run_streaming_monitor.py",
                "--data", str(injected_path), "--target", "before_after",
                "--group-column", "plane_id", "--id-columns", "id,date_diff,split",
                "--n-initial-groups", str(n_initial_groups), "--batch-size-groups", str(batch_size_groups),
                "--seed", str(seed), "--strategy", "group",
                "--max-iterations-per-batch", "15",
                "--run-id", run_id] + endpoint_args, timeout=1800)

    log_path = REPO_ROOT / "runs" / run_id / "streaming_log.jsonl"
    log_entries = []
    if log_path.exists():
        log_entries = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]

    drift_batch_decision = next(
        (e["decision"] for e in log_entries if e["batch_index"] == drift_batch_index), None,
    )
    other_decisions = {e["batch_index"]: e["decision"] for e in log_entries if e["batch_index"] != drift_batch_index}
    correctly_retrained = drift_batch_decision == "retrain"
    correctly_left_others_alone = all(d != "retrain" for d in other_decisions.values())

    return {
        "scenario": "streaming_drift", "run_id": run_id, "ok": res["returncode"] == 0,
        "elapsed_s": res["elapsed_s"], "injection_info": injection_info,
        "drift_batch_index": drift_batch_index, "drift_batch_decision": drift_batch_decision,
        "other_batch_decisions": other_decisions,
        "correctly_retrained_on_drift": correctly_retrained,
        "correctly_left_other_batches_alone": correctly_left_others_alone,
        "log_entries": log_entries,
        "stderr_tail": res["stderr_tail"] if res["returncode"] else "",
    }


def write_report(results: list[dict], out_path: Path) -> None:
    lines = [
        "# Streaming Monitor Evaluation Report", "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
        "| Scenario | Result |", "|---|---|",
    ]
    for r in results:
        if r.get("skipped"):
            lines.append(f"| {r['scenario']} | SKIPPED — {r.get('reason', '')} |")
        else:
            lines.append(
                f"| {r['scenario']} | drift_batch={r.get('drift_batch_index')} -> "
                f"decision={r.get('drift_batch_decision')}, "
                f"correctly_retrained={r.get('correctly_retrained_on_drift')}, "
                f"other_batches_left_alone={r.get('correctly_left_other_batches_alone')} |"
            )
    lines += ["", "## Full details", ""]
    for r in results:
        lines.append(f"### {r.get('scenario')}\n")
        lines.append(f"```json\n{json.dumps(r, indent=2, default=str)}\n```\n")
    out_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-gateway", action="store_true")
    parser.add_argument("--use-local", action="store_true")
    parser.add_argument("--out", default="runs/streaming_eval_report.md")
    args = parser.parse_args()

    endpoint_args = ["--use-local"] if args.use_local else (["--use-gateway"] if args.use_gateway else [])

    print("Scenario: streaming drift-triggered retrain...")
    results = [scenario_streaming_drift(endpoint_args)]

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(results, out_path)
    print(f"\nReport written to {out_path}")
    for r in results:
        print(f"  {r['scenario']}: " + ("SKIPPED" if r.get("skipped") else json.dumps({
            "correctly_retrained_on_drift": r.get("correctly_retrained_on_drift"),
            "correctly_left_other_batches_alone": r.get("correctly_left_other_batches_alone"),
        })))


if __name__ == "__main__":
    main()
