"""Augmentation catalog. Each entry's builder returns a batch function
(x, y, generator) -> (x, y) with x: (B, C, T) scaled inputs and y: (B, K)
soft targets. All randomness comes from `generator`, so a seeded run is
reproducible. Only label_mixup changes targets; the rest leave labels alone.

The first three are the temporal augmentations from the NGAFID-MC paper
(Yang, LaBella & Desell 2021), with the paper's settings as defaults.
Segment lengths are fractions of the sequence length, so they mean the
same thing at any pooling level (the paper's 64-512 of 4096 steps is
0.0156-0.125).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from agentic_pdm.catalog.params import CatalogEntry, ParamSpec


def _p(default=0.4):
    return ParamSpec("float", default, "probability of applying this to each sequence (label_mixup: to each batch)",
                     min=0.0, max=1.0)


def _uniform(generator) -> float:
    return torch.rand((), generator=generator).item()


def _randint(generator, lo, hi) -> int:  # inclusive
    return int(torch.randint(lo, hi + 1, (), generator=generator).item())


def _segment_bounds(t, lo_frac, hi_frac):
    lo = max(1, round(t * lo_frac))
    return lo, min(t, max(lo, round(t * hi_frac)))


def _selected(batch, p, generator):
    return (torch.rand(batch, generator=generator) < p).nonzero().flatten().tolist()


def _seg_frac_check(params):
    if params["seg_min_frac"] > params["seg_max_frac"]:
        return "seg_min_frac must be <= seg_max_frac"
    return None


def cutout(p, seg_min_frac, seg_max_frac, channel_p):
    def fn(x, y, g):
        b, c, t = x.shape
        lo, hi = _segment_bounds(t, seg_min_frac, seg_max_frac)
        for i in _selected(b, p, g):
            length = _randint(g, lo, hi)
            start = _randint(g, 0, t - length)
            channels = torch.rand(c, generator=g) < channel_p
            x[i, channels, start:start + length] = 0.0
        return x, y
    return fn


def cutmix(p, seg_min_frac, seg_max_frac, channel_p):
    def fn(x, y, g):
        b, c, t = x.shape
        lo, hi = _segment_bounds(t, seg_min_frac, seg_max_frac)
        source = x.clone()
        for i in _selected(b, p, g):
            other = _randint(g, 0, b - 1)
            length = _randint(g, lo, hi)
            dst, src = _randint(g, 0, t - length), _randint(g, 0, t - length)
            channels = torch.rand(c, generator=g) < channel_p
            x[i, channels, dst:dst + length] = source[other, channels, src:src + length]
        return x, y
    return fn


def channel_mixup(p, m_min, m_max, channel_p):
    def fn(x, y, g):
        b, c, _ = x.shape
        source = x.clone()
        for i in _selected(b, p, g):
            other = _randint(g, 0, b - 1)
            m = m_min + (m_max - m_min) * _uniform(g)
            channels = torch.rand(c, generator=g) < channel_p
            x[i, channels, :] = m * x[i, channels, :] + (1 - m) * source[other, channels, :]
        return x, y
    return fn


def jitter(p, sigma):
    def fn(x, y, g):
        rows = _selected(x.shape[0], p, g)
        if rows:
            x[rows] = x[rows] + sigma * torch.randn(x[rows].shape, generator=g)
        return x, y
    return fn


def scaling(p, sigma):
    def fn(x, y, g):
        rows = _selected(x.shape[0], p, g)
        if rows:
            factors = 1.0 + sigma * torch.randn((len(rows), x.shape[1], 1), generator=g)
            x[rows] = x[rows] * factors
        return x, y
    return fn


def _smooth_curves(n, c, t, knots, sigma, g):
    """(n, c, t) random curves around 1.0, linearly interpolated between knots."""
    points = 1.0 + sigma * torch.randn((n * c, 1, knots + 2), generator=g)
    return F.interpolate(points, size=t, mode="linear", align_corners=True).reshape(n, c, t)


def magnitude_warp(p, sigma, knots):
    def fn(x, y, g):
        rows = _selected(x.shape[0], p, g)
        if rows:
            x[rows] = x[rows] * _smooth_curves(len(rows), x.shape[1], x.shape[2], knots, sigma, g)
        return x, y
    return fn


def time_warp(p, sigma, knots):
    """Resamples the time axis along a random smooth monotone warp (the
    same warp for every channel of a sequence)."""
    def fn(x, y, g):
        b, c, t = x.shape
        for i in _selected(b, p, g):
            speed = _smooth_curves(1, 1, t, knots, sigma, g)[0, 0].clamp(min=0.1)
            position = torch.cumsum(speed, 0)
            position = (position - position[0]) / (position[-1] - position[0]) * (t - 1)
            left = position.floor().long().clamp(0, t - 1)
            right = (left + 1).clamp(max=t - 1)
            frac = (position - left.float()).unsqueeze(0)
            x[i] = x[i][:, left] * (1 - frac) + x[i][:, right] * frac
        return x, y
    return fn


def window_slice(p, ratio):
    """Crops a random window covering `ratio` of the sequence and stretches
    it back to full length."""
    def fn(x, y, g):
        b, c, t = x.shape
        length = max(2, round(t * ratio))
        for i in _selected(b, p, g):
            start = _randint(g, 0, t - length)
            x[i] = F.interpolate(x[i:i + 1, :, start:start + length], size=t, mode="linear", align_corners=True)[0]
        return x, y
    return fn


def channel_dropout(p, channel_p):
    def fn(x, y, g):
        b, c, _ = x.shape
        for i in _selected(b, p, g):
            x[i, torch.rand(c, generator=g) < channel_p, :] = 0.0
        return x, y
    return fn


def label_mixup(p, alpha):
    """Classic mixup (Zhang et al. 2017): with probability p, the whole
    batch is blended with a shuffled copy of itself, inputs AND targets,
    lambda ~ Beta(alpha, alpha)."""
    def fn(x, y, g):
        if _uniform(g) >= p:
            return x, y
        seed = int(torch.randint(0, 2**31 - 1, (), generator=g).item())
        lam = float(np.random.default_rng(seed).beta(alpha, alpha))
        perm = torch.randperm(x.shape[0], generator=g)
        return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]
    return fn


_SEG = dict(
    seg_min_frac=ParamSpec("float", 64 / 4096, "shortest segment, as a fraction of sequence length", min=0.001, max=0.5),
    seg_max_frac=ParamSpec("float", 512 / 4096, "longest segment, as a fraction of sequence length", min=0.001, max=0.8),
)

AUGMENTATIONS: dict[str, CatalogEntry] = {e.id: e for e in (
    CatalogEntry(
        "cutout", "augmentation",
        "NGAFID paper: zero a random time segment on a random subset of channels.",
        {"p": _p(), **_SEG, "channel_p": ParamSpec("float", 0.3, "chance each channel is affected", min=0.0, max=1.0)},
        constraint=_seg_frac_check,
        builder=lambda q: cutout(q["p"], q["seg_min_frac"], q["seg_max_frac"], q["channel_p"]),
    ),
    CatalogEntry(
        "cutmix", "augmentation",
        "NGAFID paper: replace a random segment on some channels with a segment from another sequence in the "
        "batch (any label; the label is not changed).",
        {"p": _p(), **_SEG, "channel_p": ParamSpec("float", 0.3, "chance each channel is affected", min=0.0, max=1.0)},
        constraint=_seg_frac_check,
        builder=lambda q: cutmix(q["p"], q["seg_min_frac"], q["seg_max_frac"], q["channel_p"]),
    ),
    CatalogEntry(
        "channel_mixup", "augmentation",
        "NGAFID paper's 'temporal mixup': blend some channels with another sequence's, m*x + (1-m)*x_other over "
        "the whole sequence. The label is not changed (unlike label_mixup).",
        {
            "p": _p(),
            "m_min": ParamSpec("float", 0.6, "smallest weight on the original", min=0.0, max=1.0),
            "m_max": ParamSpec("float", 0.9, "largest weight on the original", min=0.0, max=1.0),
            "channel_p": ParamSpec("float", 0.4, "chance each channel is blended", min=0.0, max=1.0),
        },
        constraint=lambda q: "m_min must be <= m_max" if q["m_min"] > q["m_max"] else None,
        builder=lambda q: channel_mixup(q["p"], q["m_min"], q["m_max"], q["channel_p"]),
    ),
    CatalogEntry(
        "jitter", "augmentation",
        "Add Gaussian noise (inputs are min-max scaled to ~[0, 1], so sigma is a fraction of each channel's range).",
        {"p": _p(0.5), "sigma": ParamSpec("float", 0.02, "noise standard deviation", min=0.0, max=0.2)},
        builder=lambda q: jitter(q["p"], q["sigma"]),
    ),
    CatalogEntry(
        "scaling", "augmentation",
        "Multiply each channel by a random factor ~ N(1, sigma).",
        {"p": _p(0.5), "sigma": ParamSpec("float", 0.1, "factor standard deviation", min=0.0, max=0.5)},
        builder=lambda q: scaling(q["p"], q["sigma"]),
    ),
    CatalogEntry(
        "magnitude_warp", "augmentation",
        "Multiply each channel by a smooth random curve around 1.",
        {
            "p": _p(0.3),
            "sigma": ParamSpec("float", 0.2, "curve standard deviation", min=0.0, max=0.5),
            "knots": ParamSpec("int", 4, "curve control points", min=1, max=16),
        },
        builder=lambda q: magnitude_warp(q["p"], q["sigma"], q["knots"]),
    ),
    CatalogEntry(
        "time_warp", "augmentation",
        "Locally speed up / slow down time along a smooth random warp (same warp for all channels). The paper "
        "argued warping may not suit this non-periodic data, so treat as an experiment.",
        {
            "p": _p(0.3),
            "sigma": ParamSpec("float", 0.2, "speed variation", min=0.0, max=0.5),
            "knots": ParamSpec("int", 4, "warp control points", min=1, max=16),
        },
        builder=lambda q: time_warp(q["p"], q["sigma"], q["knots"]),
    ),
    CatalogEntry(
        "window_slice", "augmentation",
        "Crop a random window and stretch it back to full length. The paper warned slicing can drop the few, "
        "distant segments that matter; keep ratio high.",
        {"p": _p(0.3), "ratio": ParamSpec("float", 0.9, "fraction of the sequence kept", min=0.5, max=1.0)},
        builder=lambda q: window_slice(q["p"], q["ratio"]),
    ),
    CatalogEntry(
        "channel_dropout", "augmentation",
        "Zero entire channels (simulates a dropped-out sensor).",
        {"p": _p(0.3), "channel_p": ParamSpec("float", 0.1, "chance each channel is zeroed", min=0.0, max=0.5)},
        builder=lambda q: channel_dropout(q["p"], q["channel_p"]),
    ),
    CatalogEntry(
        "label_mixup", "augmentation",
        "Classic mixup: blend whole sequences AND their labels, lambda ~ Beta(alpha, alpha). Produces soft labels.",
        {"p": _p(0.5), "alpha": ParamSpec("float", 0.2, "Beta distribution parameter", min=0.05, max=2.0)},
        builder=lambda q: label_mixup(q["p"], q["alpha"]),
    ),
)}


def compose(functions):
    """One augment(x, y, generator, config) plugin function applying each
    built augmentation in order."""
    def augment(x, y, generator, config):
        for fn in functions:
            x, y = fn(x, y, generator)
        return x, y
    return augment
