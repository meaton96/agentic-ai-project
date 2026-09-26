"""An experiment: the one thing a planning agent proposes. It is plain JSON
naming catalog entries and their parameters — no code — which the harness
validates, prices, and turns into a trainable plugin.

    {
      "name": "cnn-cosine-cutmix",
      "rationale": "why this, given what we've seen so far",
      "architecture": {"id": "small_cnn", "params": {"width": 32}},
      "augmentations": [{"id": "cutout", "params": {}}, {"id": "cutmix", "params": {"p": 0.5}}],
      "schedule": {"id": "cosine", "params": {"lr": 0.001}},
      "loss": {"id": "cross_entropy", "params": {"label_smoothing": 0.05}},
      "input": {"max_len": 4096, "pool": 4},
      "training": {"epochs": 30, "batch_size": 64, "ensemble": 1, "seed": 0},
      "folds": "all"
    }

Everything but "name" and "architecture" has defaults. validate_experiment()
returns every problem at once (so an agent can fix them in one pass), the
fully normalized config, warnings, and an estimated cost in minutes.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils.flop_counter import FlopCounterMode

from agentic_pdm.catalog.architectures import ARCHITECTURES
from agentic_pdm.catalog.augmentations import AUGMENTATIONS, compose
from agentic_pdm.catalog.params import CatalogEntry, ParamSpec, validate_params
from agentic_pdm.catalog.training import LOSSES, SCHEDULES
from agentic_pdm.harness.plugin import Plugin
from agentic_pdm.harness.trainer import TrainConfig

INPUT_PARAMS = CatalogEntry("input", "input", "Which part of each sequence the model sees, and at what resolution.", {
    "max_len": ParamSpec("int", 4096, "keep only the last max_len timesteps (the paper used the last 4096)",
                         min=64, max=1_000_000),
    "pool": ParamSpec("choice", 4, "average-pool non-overlapping windows of this many steps (4096 / 4 = 1024 "
                      "model inputs); larger is faster but blurrier", choices=(1, 2, 4, 8, 16)),
})

TRAINING_PARAMS = CatalogEntry("training", "training", "Training loop settings.", {
    "epochs": ParamSpec("int", 20, "passes over the training folds", min=1, max=100),
    "batch_size": ParamSpec("choice", 64, "sequences per step", choices=(16, 32, 64, 128)),
    "ensemble": ParamSpec("int", 1, "models trained per fold (different seeds) and averaged; multiplies cost",
                          min=1, max=3),
    "seed": ParamSpec("int", 0, "random seed", min=0, max=100_000),
})

KINDS = {
    "architecture": ARCHITECTURES,
    "augmentation": AUGMENTATIONS,
    "schedule": SCHEDULES,
    "loss": LOSSES,
}

# Measured: the reference small CNN (29.4 MFLOPs forward per sequence at
# 23 x 1024) trains at 2.2 s/epoch on 3,997 + evaluates 1,092 sequences on 3
# threads of a Ryzen 9 7950X => ~175 effective GFLOP/s, augmentation and
# data movement included. Other hosts differ; set PDM_EFFECTIVE_GFLOPS, and
# the pipeline also rescales by what its own finished experiments measured.
DEFAULT_EFFECTIVE_GFLOPS = 175.0
MIN_MODEL_INPUT_LENGTH = 32


@dataclass
class DatasetFacts:
    """What validation needs to know about the data, without touching it."""
    n_sequences: int
    n_channels: int
    seq_len: int  # stored length (timesteps) per sequence
    n_classes: int
    folds: list[int]

    @classmethod
    def from_store(cls, store) -> "DatasetFacts":
        n, c, t = store.X.shape
        return cls(n_sequences=n, n_channels=c, seq_len=t, n_classes=store.n_classes, folds=store.folds)

    def to_dict(self) -> dict:
        return {"n_sequences": self.n_sequences, "n_channels": self.n_channels, "seq_len": self.seq_len,
                "n_classes": self.n_classes, "folds": list(self.folds)}


def catalog_summary(kind: Optional[str] = None) -> list[dict]:
    kinds = [kind] if kind else list(KINDS)
    unknown = [k for k in kinds if k not in KINDS]
    if unknown:
        raise ValueError(f"unknown kind {unknown[0]!r}; kinds: {sorted(KINDS)}")
    return [entry.summary() for k in kinds for entry in KINDS[k].values()]


def describe_entry(entry_id: str) -> dict:
    for kind, entries in KINDS.items():
        if entry_id in entries:
            return entries[entry_id].describe()
    if entry_id in ("input", "training"):
        return (INPUT_PARAMS if entry_id == "input" else TRAINING_PARAMS).describe()
    known = sorted([e for entries in KINDS.values() for e in entries] + ["input", "training"])
    raise ValueError(f"no catalog entry {entry_id!r}; known: {known}")


def _entry_ref(value: Any, kind: str, where: str, errors: list[str], default_id: Optional[str] = None):
    """Validates {"id": ..., "params": {...}} against KINDS[kind]."""
    entries = KINDS[kind]
    if value is None and default_id is not None:
        value = {"id": default_id}
    if isinstance(value, str):
        value = {"id": value}
    if not isinstance(value, dict) or "id" not in value:
        errors.append(f"{where} must be an object like {{\"id\": ..., \"params\": {{...}}}}")
        return None
    extra = sorted(set(value) - {"id", "params"})
    if extra:
        errors.append(f"{where}: unexpected key(s) {extra}; only 'id' and 'params'")
    entry = entries.get(value["id"])
    if entry is None:
        article = "an" if kind[0] in "aeiou" else "a"
        errors.append(f"{where}.id {value['id']!r} is not {article} {kind}; choose one of {sorted(entries)}")
        return None
    params, param_errors = validate_params(entry, value.get("params"), f"{where}.params")
    errors.extend(param_errors)
    return {"id": entry.id, "params": params}


def normalize_experiment(config: Any, facts: DatasetFacts) -> tuple[Optional[dict], list[str], list[str]]:
    """(normalized config or None, errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(config, dict):
        return None, ["the experiment must be a JSON object"], warnings

    allowed = {"name", "rationale", "architecture", "augmentations", "schedule", "loss", "input", "training", "folds"}
    extra = sorted(set(config) - allowed)
    if extra:
        errors.append(f"unexpected top-level key(s) {extra}; allowed: {sorted(allowed)}")

    name = config.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name or ""):
        errors.append("name is required: 1-64 chars of letters, digits, '.', '_', '-'")

    out: dict = {"name": name, "rationale": str(config.get("rationale", ""))[:2000]}
    if "architecture" not in config:
        errors.append(f"architecture is required; choose one of {sorted(ARCHITECTURES)}")
    else:
        out["architecture"] = _entry_ref(config["architecture"], "architecture", "architecture", errors)

    augs = config.get("augmentations", [])
    if not isinstance(augs, list):
        errors.append("augmentations must be a list (use [] for none)")
        augs = []
    out["augmentations"] = []
    for i, aug in enumerate(augs):
        ref = _entry_ref(aug, "augmentation", f"augmentations[{i}]", errors)
        if ref is not None:
            out["augmentations"].append(ref)
    ids = [a["id"] for a in out["augmentations"]]
    if len(ids) != len(set(ids)):
        errors.append(f"each augmentation may appear once, got {ids}")

    out["schedule"] = _entry_ref(config.get("schedule"), "schedule", "schedule", errors, default_id="cosine")
    out["loss"] = _entry_ref(config.get("loss"), "loss", "loss", errors, default_id="cross_entropy")
    out["input"], input_errors = validate_params(INPUT_PARAMS, config.get("input"), "input")
    errors.extend(input_errors)
    out["training"], training_errors = validate_params(TRAINING_PARAMS, config.get("training"), "training")
    errors.extend(training_errors)

    if not input_errors:
        max_len, pool = out["input"]["max_len"], out["input"]["pool"]
        if max_len > facts.seq_len:
            errors.append(f"input.max_len {max_len} exceeds the stored sequence length {facts.seq_len}")
        elif max_len % pool:
            errors.append(f"input.max_len {max_len} must be divisible by input.pool {pool}")
        elif max_len // pool < MIN_MODEL_INPUT_LENGTH:
            errors.append(f"max_len / pool = {max_len // pool} model inputs is too short "
                          f"(minimum {MIN_MODEL_INPUT_LENGTH})")

    folds = config.get("folds", "all")
    if folds == "all":
        out["folds"] = list(facts.folds)
    elif isinstance(folds, list) and folds and all(isinstance(f, int) and not isinstance(f, bool) for f in folds):
        bad = sorted(set(folds) - set(facts.folds))
        if bad:
            errors.append(f"folds {bad} don't exist; the dataset has folds {facts.folds}")
        if len(folds) != len(set(folds)):
            errors.append("folds must not repeat")
        out["folds"] = sorted(folds)
    else:
        errors.append(f"folds must be \"all\" or a non-empty list of fold numbers from {facts.folds}")

    if errors:
        return None, errors, warnings

    arch_id = out["architecture"]["id"]
    if arch_id == "conv_mhsa" and not out["augmentations"]:
        warnings.append("conv_mhsa without augmentation overfit badly in the NGAFID paper")
    if out["schedule"]["id"] == "constant":
        warnings.append("a constant learning rate made the reference model's last-epoch score swing widely; "
                        "final-epoch ('last') results are what the leaderboard ranks")
    if len(out["folds"]) < len(facts.folds):
        warnings.append(f"screening on folds {out['folds']} only: results are noisier than full cross-validation "
                        "and can't satisfy the run's target")
    return out, errors, warnings


def build_model_for(normalized: dict, facts: DatasetFacts) -> torch.nn.Module:
    arch = ARCHITECTURES[normalized["architecture"]["id"]]
    length = normalized["input"]["max_len"] // normalized["input"]["pool"]
    return arch.builder(facts.n_channels, length, facts.n_classes, normalized["architecture"]["params"])


def calibration_for(calibration: "float | dict | None", arch_id: str) -> float:
    """A measured time-correction factor: a plain number, or a map from
    architecture id to factor with an optional "default" (architectures
    differ in how well FLOPs predict their real time — memory-bound ones
    like inception_time run slower than their FLOPs suggest)."""
    if calibration is None:
        return 1.0
    if isinstance(calibration, dict):
        return float(calibration.get(arch_id, calibration.get("default", 1.0)))
    return float(calibration)


def estimate_cost(normalized: dict, facts: DatasetFacts, calibration: "float | dict | None" = 1.0) -> dict:
    """Minutes this experiment should take on this deployment: forward FLOPs
    of the actual model (counted, not guessed), x3 for training, times
    sequences, epochs, folds and ensemble members, at the effective rate."""
    model = build_model_for(normalized, facts).eval()
    length = normalized["input"]["max_len"] // normalized["input"]["pool"]
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(torch.zeros(1, facts.n_channels, length))
    forward = counter.get_total_flops()
    n_params = sum(p.numel() for p in model.parameters())
    # `or`: an empty value (e.g. docker compose's "${PDM_EFFECTIVE_GFLOPS:-}") means unset.
    gflops = float(os.environ.get("PDM_EFFECTIVE_GFLOPS") or DEFAULT_EFFECTIVE_GFLOPS)
    factor = calibration_for(calibration, normalized["architecture"]["id"])

    n_folds_total = len(facts.folds)
    per_fold_val = facts.n_sequences / n_folds_total
    per_fold_train = facts.n_sequences - per_fold_val
    raw_epoch_seconds = (3 * per_fold_train + per_fold_val) * forward / (gflops * 1e9)
    t = normalized["training"]
    raw_minutes = raw_epoch_seconds * t["epochs"] * t["ensemble"] * len(normalized["folds"]) / 60
    return {
        "n_params": int(n_params),
        "forward_mflops_per_sequence": round(forward / 1e6, 2),
        "model_input_shape": [facts.n_channels, length],
        # Unrounded: budgets and calibration do arithmetic on these.
        "seconds_per_epoch": raw_epoch_seconds * factor,
        "total_minutes": raw_minutes * factor,
        "uncalibrated_minutes": raw_minutes,
        "effective_gflops": gflops,
        "calibration": factor,
    }


def validate_experiment(config: Any, facts: DatasetFacts, calibration: "float | dict | None" = 1.0) -> dict:
    normalized, errors, warnings = normalize_experiment(config, facts)
    report = {"valid": not errors, "errors": errors, "warnings": warnings, "normalized": normalized}
    if normalized is not None:
        try:
            report["estimate"] = estimate_cost(normalized, facts, calibration)
        except Exception as exc:  # a parameter combination the model itself rejects
            report.update(valid=False, errors=[f"the model can't be built with these parameters: {exc}"])
    return report


def build_experiment(normalized: dict, threads: Optional[int] = None) -> tuple[Plugin, TrainConfig, list[int]]:
    """Turns a normalized (validated) experiment into the harness's inputs."""
    arch = ARCHITECTURES[normalized["architecture"]["id"]]
    arch_params = normalized["architecture"]["params"]
    augment = compose([AUGMENTATIONS[a["id"]].builder(a["params"]) for a in normalized["augmentations"]])
    configure_optimizer, configure_scheduler = SCHEDULES[normalized["schedule"]["id"]].builder(
        normalized["schedule"]["params"])
    plugin = Plugin(
        name=normalized["name"],
        path="catalog",
        build_model=lambda c, t, k, cfg: arch.builder(c, t, k, arch_params),
        augment=augment,
        configure_optimizer=configure_optimizer,
        configure_scheduler=configure_scheduler,
    )
    loss = normalized["loss"]
    t = normalized["training"]
    cfg = TrainConfig(
        epochs=t["epochs"],
        batch_size=t["batch_size"],
        pool=normalized["input"]["pool"],
        max_len=normalized["input"]["max_len"],
        seed=t["seed"],
        ensemble=t["ensemble"],
        threads=threads,
        loss=loss["id"],
        label_smoothing=loss["params"].get("label_smoothing", 0.0),
        focal_gamma=loss["params"].get("gamma", 2.0),
    )
    return plugin, cfg, list(normalized["folds"])
