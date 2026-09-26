"""
agent-sandbox gates and the training job for the experiment loop
(sandbox/pipelines/pdm-experiment-loop.yaml):

    init (gate) ──ready──▶ brief (gate) ──▶ plan (agent, catalog tools)
                                              │
    ┌──── invalid ◀── validate (gate) ◀───────┘
    │                    │ valid
    ▼                    ▼
  brief            train (job) ──done / __error__──▶ record (gate)
                                                      │ continue
                   analyze (agent) ◀──────────────────┘
                      │
                      └──▶ brief ──▶ plan ...

    budget_exhausted / target_met / give_up ──▶ finalize (gate) ──▶ report (agent)

Agents propose; these functions decide. Every decision with consequences —
is this experiment valid, is it within budget, did it meet the target, is
the run over — is made here, deterministically, from files in the run
directory. The run directory is the single source of truth:

    <work>/pdm-runs/<run_id>/
        contract.json        the seed task, parsed
        dataset.json         dataset facts and profile (no raw data)
        state.json           counters, budget spent, pending experiment
        leaderboard.jsonl    one line per finished (or failed) experiment
        experiments/NN-name/ config.json, results.json (from the job)
        report.json/.md      written by finalize

Gates find the run directory through the init gate's output (a JSON object
with "pdm_run_dir"), wherever it sits in `outputs`. Three step ids are
conventions the pipeline YAML must keep: "plan" (the planner, whose reply
validate_proposal reads), "train" (the job, whose error text is recorded
if it fails) and "analyze" (the analyst, whose notes go into the next
brief).

Where things live:
    work dir:     $GATE_SCRATCH_DIR, else $PDM_WORK_DIR, else ./pdm-work
    datasets dir: inputs["__datasets_dir__"], else $DATASETS_DIR, else
                  $SANDBOX_DATASETS_DIR, else $PDM_DATASETS_DIR; dataset
                  <id> is a tensor store (see ingest.py) at <dir>/<id>
"""
from __future__ import annotations

import json
import math
import os
import re
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from agentic_pdm.catalog import DatasetFacts, build_experiment, validate_experiment
from agentic_pdm.harness.trainer import run_cv
from agentic_pdm.store import load_store

HIGHER_IS_BETTER = {"roc_auc": True, "pr_auc": True, "accuracy": True, "log_loss": False}
MAX_CONSECUTIVE_INVALID = 3
PLAN_STEP_ID = "plan"
TRAIN_STEP_ID = "train"
ANALYST_STEP_ID = "analyze"


# -- plumbing ----------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _work_dir() -> Path:
    base = os.environ.get("GATE_SCRATCH_DIR") or os.environ.get("PDM_WORK_DIR") or "pdm-work"
    return Path(base)


def _datasets_dir(outputs: dict) -> Path:
    for value in (outputs.get("__datasets_dir__"), os.environ.get("DATASETS_DIR"),
                  os.environ.get("SANDBOX_DATASETS_DIR"), os.environ.get("PDM_DATASETS_DIR")):
        if value:
            return Path(value)
    raise RuntimeError("no datasets directory: list the dataset under this step's `datasets:` "
                       "(or set PDM_DATASETS_DIR when running outside the sandbox)")


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


def _run_dir(outputs: dict) -> Path:
    for value in outputs.values():
        if isinstance(value, str) and '"pdm_run_dir"' in value:
            try:
                return Path(json.loads(value)["pdm_run_dir"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    raise RuntimeError("no pdm run directory in the pipeline's outputs — the init gate must run first")


def _state(run_dir: Path) -> dict:
    return _read(run_dir / "state.json")


def _leaderboard(run_dir: Path) -> list[dict]:
    path = run_dir / "leaderboard.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _budget_left(contract: dict, state: dict) -> tuple[int, float]:
    budget = contract["budget"]
    return (budget["max_experiments"] - state["experiments_submitted"],
            budget["max_minutes"] - state["minutes_spent"])


def _score(entry: dict, contract: dict) -> Optional[float]:
    target = contract["target"]
    value = (entry.get("summary") or {}).get(target["protocol"], {}).get(target["metric"])
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


def _ranked(leaderboard: list[dict], contract: dict, full_cv: bool) -> list[dict]:
    rows = [e for e in leaderboard if e["status"] == "completed" and e["full_cv"] == full_cv
            and _score(e, contract) is not None]
    return sorted(rows, key=lambda e: _score(e, contract), reverse=HIGHER_IS_BETTER[contract["target"]["metric"]])


def _fmt(value, digits=4):
    return "—" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


def _extract_json_object(text: str) -> Any:
    """The planner's proposal: a fenced ```json block if there is one, else
    the first balanced {...} in the text."""
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = fenced or []
    if not candidates:
        start = text.find("{")
        depth = 0
        for i in range(start, len(text)) if start >= 0 else []:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                candidates.append(text[start:i + 1])
                break
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON object found — reply with the experiment as one ```json block")


# -- contract ----------------------------------------------------------------------

DEFAULT_TARGET = {"metric": "roc_auc", "protocol": "last", "value": None}
DEFAULT_BUDGET = {"max_experiments": 6, "max_minutes": 120.0}


def parse_contract(task: str) -> tuple[Optional[dict], list[str]]:
    """The seed task is a JSON object:
        {"dataset": "c28", "goal": "...",
         "target": {"metric": "roc_auc", "protocol": "last", "value": 0.80},
         "budget": {"max_experiments": 6, "max_minutes": 120},
         "reference": {"source": "...", "roc_auc": 0.826}}
    Only "dataset" is required. `protocol` "last" scores final-epoch
    metrics (honest); "best" scores the best epoch per fold (what the NGAFID
    paper reports; optimistic)."""
    errors = []
    try:
        raw = json.loads(task)
    except json.JSONDecodeError as exc:
        return None, [f"the seed task must be a JSON run contract ({exc})"]
    if not isinstance(raw, dict):
        return None, ["the seed task must be a JSON object"]
    dataset = raw.get("dataset")
    if not isinstance(dataset, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", dataset):
        errors.append("contract.dataset is required: a dataset id like \"c28\"")
    target = {**DEFAULT_TARGET, **(raw.get("target") or {})}
    if target["metric"] not in HIGHER_IS_BETTER:
        errors.append(f"target.metric must be one of {sorted(HIGHER_IS_BETTER)}")
    if target["protocol"] not in ("last", "best"):
        errors.append("target.protocol must be \"last\" or \"best\"")
    if target["value"] is not None and not isinstance(target["value"], (int, float)):
        errors.append("target.value must be a number or null (no target: run until the budget is spent)")
    budget = {**DEFAULT_BUDGET, **(raw.get("budget") or {})}
    if not isinstance(budget["max_experiments"], int) or not 1 <= budget["max_experiments"] <= 50:
        errors.append("budget.max_experiments must be a whole number from 1 to 50")
    if not isinstance(budget["max_minutes"], (int, float)) or not 1 <= budget["max_minutes"] <= 24 * 60:
        errors.append("budget.max_minutes must be between 1 and 1440")
    unknown = sorted(set(raw) - {"dataset", "goal", "target", "budget", "reference"})
    if unknown:
        errors.append(f"unknown contract key(s) {unknown}")
    if errors:
        return None, errors
    return {
        "dataset": dataset,
        "goal": str(raw.get("goal", "Find the best model for this dataset within the budget.")),
        "target": target,
        "budget": {"max_experiments": budget["max_experiments"], "max_minutes": float(budget["max_minutes"])},
        "reference": raw.get("reference") or {},
    }, []


def _dataset_profile(store) -> dict:
    meta = store.meta
    classes = store.manifest["classes"]
    per_fold = {}
    for fold, rows in meta.groupby("fold"):
        counts = rows["y"].value_counts().to_dict()
        per_fold[int(fold)] = {"sequences": int(len(rows)), "groups": int(rows["group"].nunique()),
                               **{f"class_{classes[k]}": int(v) for k, v in sorted(counts.items())}}
    lengths = meta["length"]
    return {
        "channels": store.manifest["channels"],
        "classes": classes,
        "positive_class": store.manifest.get("positive_label", classes[-1]),
        "label_counts": {str(classes[k]): int(v) for k, v in meta["y"].value_counts().sort_index().items()},
        "folds": per_fold,
        "fold_source": store.manifest.get("fold_source"),
        "length_stats": {"min": int(lengths.min()), "median": int(lengths.median()), "max": int(lengths.max())},
        "stored_length": int(store.X.shape[2]),
        "missing_fraction": round(float(meta["nan_frac"].mean()), 4),
        "excluded_columns": store.manifest["columns"].get("excluded", []),
    }


# -- gates -------------------------------------------------------------------------


def init_run(outputs: dict) -> tuple[str, str]:
    """Parses the seed task (the run contract), profiles the dataset, and
    creates the run directory. Decisions: "ready", or "invalid_contract"
    (output says why)."""
    contract, errors = parse_contract(outputs["__task__"])
    if errors:
        return "invalid_contract", "Invalid run contract:\n- " + "\n- ".join(errors)
    dataset_dir = _datasets_dir(outputs) / contract["dataset"]
    if not (dataset_dir / "manifest.json").exists():
        return "invalid_contract", f"dataset {contract['dataset']!r} not found at {dataset_dir}"
    store = load_store(dataset_dir)
    facts = DatasetFacts.from_store(store)

    run_id = datetime.now(timezone.utc).strftime("%y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    run_dir = _work_dir() / "pdm-runs" / run_id
    baselines_path = dataset_dir / "baselines.json"
    baselines = ({name: result["mean"] for name, result in _read(baselines_path).items()}
                 if baselines_path.exists() else {})
    _write(run_dir / "contract.json", contract)
    _write(run_dir / "dataset.json", {"facts": facts.to_dict(), "profile": _dataset_profile(store),
                                      "baselines": baselines})
    _write(run_dir / "state.json", {
        "created_at": _now(), "experiments_submitted": 0, "minutes_spent": 0.0,
        "consecutive_invalid": 0, "pending": None, "last_invalid": None,
        # Measured factor (actual / FLOP-estimated time) per architecture,
        # plus "default" for architectures not measured yet.
        "calibration": {"default": 1.0}, "time_ratios": {},
    })
    return "ready", json.dumps({"pdm_run_dir": str(run_dir), "dataset": contract["dataset"], "run_id": run_id})


def compose_brief(outputs: dict) -> tuple[str, str]:
    """Everything the planner needs for its next proposal, as one text:
    contract and budget, dataset facts, leaderboard, the last experiment's
    learning curves, the analyst's notes, and the last proposal's
    validation errors. Decisions: "ready", or "budget_exhausted"."""
    run_dir = _run_dir(outputs)
    contract, state, dataset = _read(run_dir / "contract.json"), _state(run_dir), _read(run_dir / "dataset.json")
    experiments_left, minutes_left = _budget_left(contract, state)
    if experiments_left <= 0 or minutes_left < 1:
        return "budget_exhausted", "budget exhausted before another experiment could be planned"

    target = contract["target"]
    lines = [
        "## Run contract",
        f"Goal: {contract['goal']}",
        f"Ranking metric: {target['metric']} ({target['protocol']}-epoch, mean over folds; "
        f"{'higher' if HIGHER_IS_BETTER[target['metric']] else 'lower'} is better). "
        + (f"Target: {target['value']} with full cross-validation." if target["value"] is not None
           else "No target: the run ends when the budget is spent."),
        f"Budget left: {experiments_left} experiment(s), ~{minutes_left:.0f} estimated CPU-minutes. "
        f"Cost estimates are corrected by factors measured on this run's finished experiments: "
        f"{json.dumps(state['calibration'])}.",
    ]
    if contract["reference"]:
        lines.append(f"Reference result to compare against: {json.dumps(contract['reference'])}")

    profile = dataset["profile"]
    lines += [
        "", "## Dataset",
        f"{dataset['facts']['n_sequences']} sequences x {dataset['facts']['n_channels']} channels, stored as the "
        f"last {profile['stored_length']} timesteps (raw lengths {profile['length_stats']}), "
        f"{profile['missing_fraction']:.1%} missing values.",
        f"Classes {profile['classes']} (positive: {profile['positive_class']}); counts {profile['label_counts']}.",
        f"Folds ({profile['fold_source']}): {json.dumps(profile['folds'])}",
        f"Channels: {', '.join(profile['channels'])}",
    ]
    if dataset["baselines"]:
        lines.append("Baselines (5-fold means): " + "; ".join(
            f"{name}: roc_auc {_fmt(m.get('roc_auc'), 3)}" for name, m in dataset["baselines"].items()))
    lines.append("Arguments for validate_experiment (pass both): "
                 f"dataset_facts={json.dumps(dataset['facts'])} calibration={json.dumps(state['calibration'])}")

    board = _leaderboard(run_dir)
    lines += ["", "## Leaderboard"]
    if not board:
        lines.append("No experiments yet.")
    for label, full in (("Full cross-validation", True), ("Screens (fold subsets, noisier)", False)):
        ranked = _ranked(board, contract, full)
        if ranked:
            lines.append(f"{label}:")
            for e in ranked[:5]:
                s = e["summary"]
                lines.append(f"- {e['name']}: {target['metric']} last {_fmt(s['last'].get(target['metric']))} / "
                             f"best {_fmt(s['best'].get(target['metric']))}, "
                             f"{e['minutes']:.1f} min — {e['digest']}")
    failed = [e for e in board if e["status"] != "completed"]
    for e in failed[-3:]:
        lines.append(f"- FAILED {e['name']}: {e['error'][:300]}")

    if board:
        last = board[-1]
        lines += ["", f"## Last experiment: {last['name']}", f"Config: {json.dumps(last['config_summary'])}"]
        if last["status"] == "completed":
            lines.append(f"Curves: {last['curves']}")
        else:
            lines.append(f"It failed: {last['error'][:600]}")

    notes = outputs.get(ANALYST_STEP_ID)
    if notes and board:
        lines += ["", "## Analyst notes on the last experiment", notes.strip()[:3000]]

    if state.get("last_invalid"):
        lines += ["", "## Your previous proposal was rejected", *[f"- {e}" for e in state["last_invalid"]],
                  "Fix these and propose again."]

    lines += ["", "## Your task",
              "Propose the single next experiment as one ```json block (name, rationale, architecture, "
              "augmentations, schedule, loss, input, training, folds). Check it with validate_experiment "
              "before answering."]
    return "ready", "\n".join(lines)


def validate_proposal(outputs: dict) -> tuple[str, str]:
    """The authoritative check on the planner's proposal: catalog schema and
    bounds, then budget. Decisions: "valid" (the experiment is registered
    for the training job), "invalid" (back to the planner, with reasons),
    "give_up" (too many invalid proposals in a row), "budget_exhausted"."""
    run_dir = _run_dir(outputs)
    contract, state, dataset = _read(run_dir / "contract.json"), _state(run_dir), _read(run_dir / "dataset.json")
    experiments_left, minutes_left = _budget_left(contract, state)
    if experiments_left <= 0 or minutes_left < 1:
        return "budget_exhausted", "no budget left"

    proposal_text = outputs.get(PLAN_STEP_ID) or ""
    errors: list[str] = []
    report = None
    try:
        proposal = _extract_json_object(proposal_text)
    except ValueError as exc:
        errors = [str(exc)]
    else:
        report = validate_experiment(proposal, DatasetFacts(**dataset["facts"]), calibration=state["calibration"])
        errors = list(report["errors"])
        if report["valid"]:
            estimate = report["estimate"]["total_minutes"]
            if estimate > minutes_left:
                errors.append(f"estimated {estimate:.1f} min exceeds the {minutes_left:.1f} min left; "
                              "reduce epochs, folds, ensemble, model size, or input length")
            names = {e["name"] for e in _leaderboard(run_dir)}
            if report["normalized"]["name"] in names:
                errors.append(f"an experiment named {report['normalized']['name']!r} already ran; pick a new name")

    if errors:
        state["consecutive_invalid"] += 1
        state["last_invalid"] = errors
        _write(run_dir / "state.json", state)
        if state["consecutive_invalid"] >= MAX_CONSECUTIVE_INVALID:
            return "give_up", f"{state['consecutive_invalid']} invalid proposals in a row; last errors: {errors}"
        return "invalid", "Proposal rejected:\n- " + "\n- ".join(errors)

    normalized = report["normalized"]
    number = state["experiments_submitted"] + 1
    exp_dir = run_dir / "experiments" / f"{number:02d}-{normalized['name']}"
    _write(exp_dir / "config.json", {"normalized": normalized, "estimate": report["estimate"],
                                     "warnings": report["warnings"], "submitted_at": _now()})
    state.update(consecutive_invalid=0, last_invalid=None, experiments_submitted=number,
                 minutes_spent=state["minutes_spent"] + report["estimate"]["total_minutes"],
                 pending={"dir": str(exp_dir), "name": normalized["name"],
                          "estimated_minutes": report["estimate"]["total_minutes"],
                          "uncalibrated_minutes": report["estimate"]["uncalibrated_minutes"],
                          "architecture": normalized["architecture"]["id"]})
    _write(run_dir / "state.json", state)
    return "valid", json.dumps({"experiment": normalized["name"],
                                "estimated_minutes": round(report["estimate"]["total_minutes"], 2),
                                "warnings": report["warnings"]})


def _curves_digest(results: dict, metric: str) -> str:
    """A compact description of how training went, for agents: per-fold
    last/best, and on the first fold the metric and losses at a few epochs."""
    folds = results["folds"]
    parts = []
    for fold, r in folds.items():
        parts.append(f"fold {fold}: last {_fmt(r['last'][metric], 3)} best {_fmt(r['best'][metric], 3)} "
                     f"@epoch {r['best']['epoch']}")
    first = next(iter(folds.values()))["history"]
    picks = sorted({0, len(first) // 4, len(first) // 2, 3 * len(first) // 4, len(first) - 1})
    trace = ", ".join(f"e{first[i]['epoch']}: train_loss {first[i]['train_loss']:.3f} "
                      f"val_loss {first[i]['log_loss']:.3f} {metric} {_fmt(first[i][metric], 3)}" for i in picks)
    spread = [r["last"][metric] for r in folds.values()]
    return "; ".join(parts) + f". Fold-to-fold std {np.std(spread):.3f}. First fold trace: {trace}"


def record_results(outputs: dict) -> tuple[str, str]:
    """Records the pending experiment's outcome (success or failure) on the
    leaderboard, recalibrates cost estimates from the time it really took,
    and decides: "target_met", "budget_exhausted", or "continue"."""
    run_dir = _run_dir(outputs)
    contract, state = _read(run_dir / "contract.json"), _state(run_dir)
    pending = state.get("pending")
    if not pending:
        return "continue", "nothing pending to record"
    exp_dir = Path(pending["dir"])
    config = _read(exp_dir / "config.json")
    normalized = config["normalized"]
    results_path = exp_dir / "results.json"
    results = _read(results_path) if results_path.exists() else None
    metric = contract["target"]["metric"]
    config_summary = {
        "architecture": normalized["architecture"],
        "augmentations": [a["id"] for a in normalized["augmentations"]],
        "schedule": normalized["schedule"], "loss": normalized["loss"],
        "input": normalized["input"], "training": normalized["training"], "folds": normalized["folds"],
    }
    entry = {"name": pending["name"], "dir": str(exp_dir), "recorded_at": _now(),
             "estimated_minutes": pending["estimated_minutes"], "config_summary": config_summary,
             "full_cv": len(normalized["folds"]) == len(_read(run_dir / "dataset.json")["facts"]["folds"])}

    if results and results.get("status") == "completed":
        minutes = sum(h["train_seconds"] for r in results["folds"].values() for h in r["history"]) / 60
        entry.update(status="completed", summary={k: results["summary"][k] for k in ("last", "best")},
                     minutes=round(minutes, 2), curves=_curves_digest(results, metric),
                     digest=f"{normalized['architecture']['id']}, "
                            f"{'+'.join(config_summary['augmentations']) or 'no aug'}, "
                            f"{normalized['schedule']['id']}, {normalized['training']['epochs']} ep")
        # Recalibrate: charge the real time, and learn how far off estimates run.
        state["minutes_spent"] += minutes - pending["estimated_minutes"]
        if pending["uncalibrated_minutes"] > 0:
            arch = pending["architecture"]
            ratios = state["time_ratios"]
            ratios.setdefault(arch, []).append(minutes / pending["uncalibrated_minutes"])
            state["calibration"] = {
                "default": round(statistics.median([r for rs in ratios.values() for r in rs]), 3),
                **{a: round(statistics.median(rs), 3) for a, rs in ratios.items()},
            }
    else:
        error = outputs.get(TRAIN_STEP_ID) or "the training job failed (no error text available)"
        if results is not None:
            error = f"training stopped with status {results.get('status')!r}: {error}"
        entry.update(status="failed", error=error[:2000], minutes=0.0)

    with open(run_dir / "leaderboard.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    state["pending"] = None
    _write(run_dir / "state.json", state)

    if entry["status"] == "completed":
        score = _score(entry, contract)
        headline = (f"{entry['name']}: {metric} {contract['target']['protocol']} {_fmt(score)} in "
                    f"{entry['minutes']:.1f} min ({'full CV' if entry['full_cv'] else 'screen'}).")
        body = f"{headline}\nConfig: {json.dumps(config_summary)}\nCurves: {entry['curves']}"
    else:
        body = f"{entry['name']} FAILED: {entry['error']}\nConfig: {json.dumps(config_summary)}"

    target = contract["target"]["value"]
    if entry["status"] == "completed" and entry["full_cv"] and target is not None:
        better = score >= target if HIGHER_IS_BETTER[metric] else score <= target
        if better:
            return "target_met", body + f"\nTarget {metric} {target} met."
    experiments_left, minutes_left = _budget_left(contract, state)
    if experiments_left <= 0 or minutes_left < 1:
        return "budget_exhausted", body + "\nBudget exhausted."
    return "continue", body + f"\nBudget left: {experiments_left} experiment(s), ~{minutes_left:.0f} min."


def finalize(outputs: dict) -> tuple[str, str]:
    """Writes report.json/report.md: the best full-CV experiment (else the
    best screen), the whole leaderboard, and budget used."""
    run_dir = _run_dir(outputs)
    contract, state = _read(run_dir / "contract.json"), _state(run_dir)
    board = _leaderboard(run_dir)
    full, screens = _ranked(board, contract, True), _ranked(board, contract, False)
    best = full[0] if full else (screens[0] if screens else None)
    metric, protocol = contract["target"]["metric"], contract["target"]["protocol"]
    report = {
        "contract": contract, "finished_at": _now(), "best": best,
        "experiments": len(board), "failed": sum(e["status"] != "completed" for e in board),
        "minutes_spent": round(state["minutes_spent"], 1), "leaderboard": board,
    }
    _write(run_dir / "report.json", report)
    lines = [f"# Experiment run {run_dir.name}", "", f"Goal: {contract['goal']}", ""]
    if best:
        lines += [f"**Best:** {best['name']} ({'full CV' if best['full_cv'] else 'screen only'}) — {metric} "
                  f"{protocol} {_fmt(_score(best, contract))}; last/best: "
                  f"{json.dumps(best['summary'])}", "", f"Config: {json.dumps(best['config_summary'])}", ""]
    else:
        lines += ["No experiment completed.", ""]
    if contract["reference"]:
        lines += [f"Reference: {json.dumps(contract['reference'])}", ""]
    lines += [f"{len(board)} experiment(s), {report['failed']} failed, ~{report['minutes_spent']} CPU-minutes.", "",
              "| # | experiment | CV | " + metric + " last | " + metric + " best | minutes |", "|---|---|---|---|---|---|"]
    for i, e in enumerate(board, 1):
        s = e.get("summary") or {}
        lines.append(f"| {i} | {e['name']} | {'full' if e['full_cv'] else 'screen'} | "
                     f"{_fmt((s.get('last') or {}).get(metric))} | {_fmt((s.get('best') or {}).get(metric))} | "
                     f"{e.get('minutes', 0):.1f}{' FAILED' if e['status'] != 'completed' else ''} |")
    text = "\n".join(lines)
    (run_dir / "report.md").write_text(text)
    return "done", text


# -- the job ------------------------------------------------------------------------


def train_experiment(inputs: dict) -> tuple[str, str]:
    """JobStep entry point: trains the experiment validate_proposal
    registered, with the harness's cross-validation trainer, writing
    results.json (rewritten every epoch) into the experiment directory.
    Its printed per-epoch lines are the job's visible progress."""
    run_dir = _run_dir(inputs)
    contract, state = _read(run_dir / "contract.json"), _state(run_dir)
    pending = state.get("pending")
    if not pending:
        raise RuntimeError("no experiment is pending — validate_proposal must accept one first")
    exp_dir = Path(pending["dir"])
    normalized = _read(exp_dir / "config.json")["normalized"]
    threads = int(os.environ.get("OMP_NUM_THREADS", "0")) or None
    plugin, cfg, folds = build_experiment(normalized, threads=threads)
    store = load_store(_datasets_dir(inputs) / contract["dataset"])
    started = time.perf_counter()
    print(f"experiment {normalized['name']}: folds {folds}, {cfg.epochs} epochs, ensemble {cfg.ensemble}, "
          f"threads {threads or 'default'}", flush=True)
    results = run_cv(store, plugin, cfg, exp_dir, folds=folds)
    summary = results["summary"]
    return "done", json.dumps({
        "pdm_experiment": normalized["name"],
        "minutes": round((time.perf_counter() - started) / 60, 2),
        "last": {k: round(v, 4) for k, v in summary["last"].items()},
    })
