"""
The plugin contract: the only surface agent-written code implements.

A plugin is one Python file defining:

    build_model(n_channels: int, seq_len: int, n_classes: int, config: dict) -> torch.nn.Module
        Required. forward(x) takes a float tensor (B, n_channels, seq_len),
        already scaled and NaN-free, and returns logits (B, n_classes).

    augment(x: Tensor, y: Tensor, generator: torch.Generator, config: dict) -> (Tensor, Tensor)
        Optional; identity if absent. Called on training batches only.
        x is (B, C, T); y is one-hot float targets (B, K). Must return
        tensors of the same shapes. y may become soft (mixup-style label
        mixing) but each row must still sum to 1. All randomness must come
        from `generator`, so a seeded run is reproducible.

    configure_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer
        Optional; AdamW(lr=config["lr"], weight_decay=config["weight_decay"])
        if absent.

The harness owns everything else: which sequences are in which fold,
scaling, batching, the loss (soft-target cross-entropy), evaluation, and
metrics. A plugin never receives validation data or labels outside the
training batches it is handed.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

DEFAULT_OPTIMIZER_CONFIG = {"lr": 1e-3, "weight_decay": 1e-4}


def _identity_augment(x, y, generator, config):
    return x, y


def _default_optimizer(model, config):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", DEFAULT_OPTIMIZER_CONFIG["lr"]),
        weight_decay=config.get("weight_decay", DEFAULT_OPTIMIZER_CONFIG["weight_decay"]),
    )


@dataclass
class Plugin:
    name: str
    path: str
    build_model: Callable
    augment: Callable = _identity_augment
    configure_optimizer: Callable = _default_optimizer


def load_plugin(path: str | Path) -> Plugin:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"plugin file not found: {path}")
    module_name = f"agentic_pdm_plugin_{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not callable(getattr(module, "build_model", None)):
        raise TypeError(f"plugin {path.name} must define a callable build_model(n_channels, seq_len, n_classes, config)")
    plugin = Plugin(name=path.stem, path=str(path), build_model=module.build_model)
    for attr in ("augment", "configure_optimizer"):
        fn = getattr(module, attr, None)
        if fn is not None:
            if not callable(fn):
                raise TypeError(f"plugin {path.name}: {attr} must be callable")
            setattr(plugin, attr, fn)
    return plugin


def check_augment_output(x_in: torch.Tensor, y_in: torch.Tensor, x_out, y_out) -> None:
    """Enforced by the trainer on every training batch, so a contract
    violation fails loudly instead of silently training on garbage."""
    if not (isinstance(x_out, torch.Tensor) and isinstance(y_out, torch.Tensor)):
        raise TypeError("augment must return a (Tensor, Tensor) pair")
    if x_out.shape != x_in.shape or y_out.shape != y_in.shape:
        raise ValueError(
            f"augment changed shapes: x {tuple(x_in.shape)}->{tuple(x_out.shape)}, "
            f"y {tuple(y_in.shape)}->{tuple(y_out.shape)}")
    if not torch.isfinite(x_out).all():
        raise ValueError("augment produced non-finite values in x")
    if not torch.allclose(y_out.sum(dim=1), torch.ones(y_out.shape[0], dtype=y_out.dtype), atol=1e-4):
        raise ValueError("augment produced target rows that do not sum to 1")
