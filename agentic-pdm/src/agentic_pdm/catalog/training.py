"""Optimization catalog: learning-rate schedules (always AdamW) and losses.
A schedule builder returns (configure_optimizer, configure_scheduler) plugin
functions; a loss entry maps onto the harness's own TrainConfig loss fields
(the harness computes the loss — see harness/trainer.py)."""
from __future__ import annotations

import math

import torch

from agentic_pdm.catalog.params import CatalogEntry, ParamSpec


def _lr(default=1e-3):
    return ParamSpec("float", default, "peak learning rate (AdamW)", min=1e-5, max=1e-2, log_scale=True)


def _wd():
    return ParamSpec("float", 1e-4, "AdamW weight decay", min=0.0, max=0.1, log_scale=True)


def _schedule(lr, weight_decay, factor_at):
    """factor_at(step, total_steps) -> multiplier on lr."""
    def configure_optimizer(model, config):
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def configure_scheduler(optimizer, config, total_steps):
        total = max(1, total_steps)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: factor_at(step, total))

    return configure_optimizer, configure_scheduler


def _cosine(warmup_frac, floor):
    def factor(step, total):
        warmup = int(total * warmup_frac)
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return factor


def _step(every_frac, gamma):
    def factor(step, total):
        return gamma ** int(step / max(1, int(total * every_frac)))
    return factor


def _one_cycle(pct_start):
    def factor(step, total):
        peak = max(1, int(total * pct_start))
        if step < peak:
            return 0.04 + 0.96 * step / peak
        progress = (step - peak) / max(1, total - peak)
        return 0.0004 + (1 - 0.0004) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return factor


SCHEDULES: dict[str, CatalogEntry] = {e.id: e for e in (
    CatalogEntry(
        "constant", "schedule",
        "Fixed learning rate. The reference baseline used lr 1e-3 constant, and its validation score swung "
        "widely epoch to epoch.",
        {"lr": _lr(), "weight_decay": _wd()},
        builder=lambda q: _schedule(q["lr"], q["weight_decay"], lambda s, t: 1.0),
    ),
    CatalogEntry(
        "cosine", "schedule",
        "Linear warmup, then cosine decay to floor * lr. Usually steadier final-epoch results than constant.",
        {
            "lr": _lr(), "weight_decay": _wd(),
            "warmup_frac": ParamSpec("float", 0.05, "fraction of training spent warming up", min=0.0, max=0.3),
            "floor": ParamSpec("float", 0.01, "final lr as a fraction of peak", min=0.0, max=0.5),
        },
        builder=lambda q: _schedule(q["lr"], q["weight_decay"], _cosine(q["warmup_frac"], q["floor"])),
    ),
    CatalogEntry(
        "step", "schedule",
        "Multiply lr by gamma every `every_frac` of training.",
        {
            "lr": _lr(), "weight_decay": _wd(),
            "every_frac": ParamSpec("float", 0.33, "interval between drops, as a fraction of training",
                                    min=0.05, max=1.0),
            "gamma": ParamSpec("float", 0.3, "multiplier at each drop", min=0.01, max=0.9),
        },
        builder=lambda q: _schedule(q["lr"], q["weight_decay"], _step(q["every_frac"], q["gamma"])),
    ),
    CatalogEntry(
        "one_cycle", "schedule",
        "One-cycle: ramp from lr/25 up to lr, then cosine down to ~0. Tolerates a higher peak lr.",
        {"lr": _lr(3e-3), "weight_decay": _wd(),
         "pct_start": ParamSpec("float", 0.3, "fraction of training spent ramping up", min=0.05, max=0.5)},
        builder=lambda q: _schedule(q["lr"], q["weight_decay"], _one_cycle(q["pct_start"])),
    ),
)}

LOSSES: dict[str, CatalogEntry] = {e.id: e for e in (
    CatalogEntry(
        "cross_entropy", "loss",
        "Cross-entropy on (possibly soft) targets.",
        {"label_smoothing": ParamSpec("float", 0.0, "mix targets toward uniform by this much", min=0.0, max=0.3)},
    ),
    CatalogEntry(
        "focal", "loss",
        "Focal loss: down-weights examples already predicted confidently, focusing on hard ones.",
        {
            "gamma": ParamSpec("float", 2.0, "focusing strength (0 = cross-entropy)", min=0.0, max=5.0),
            "label_smoothing": ParamSpec("float", 0.0, "mix targets toward uniform by this much", min=0.0, max=0.3),
        },
    ),
)}
