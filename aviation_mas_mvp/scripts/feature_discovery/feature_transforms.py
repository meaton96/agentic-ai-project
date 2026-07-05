"""
feature_transforms.py
=====================
A registry of parameterized feature transformations for time-series data.

Provides mathematical transformations that map a 1-D channel array 
to a dictionary of named scalar features. These transforms are dynamically 
selected and tuned by the discovery agent via a JSON specification.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable


# --------------------------------------------------------------------------
# registry plumbing
# --------------------------------------------------------------------------

@dataclass
class Transform:
    name: str
    fn: Callable[..., dict]
    defaults: dict          # default params
    schema: dict            # param_name -> short human description (for the agent)
    doc: str                # one-line description shown to the agent


_REGISTRY: dict[str, Transform] = {}


def register(name, defaults, schema, doc):
    def deco(fn):
        _REGISTRY[name] = Transform(name, fn, defaults, schema, doc)
        return fn
    return deco


def get(name: str) -> Transform:
    if name not in _REGISTRY:
        raise KeyError(f"unknown transform '{name}'. "
                       f"available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def menu() -> list[dict]:
    """Machine-readable menu handed to the discovery agent (the `list_transforms`
    tool output). Small enough to sit in an LLM context window."""
    return [
        {"transform": t.name, "doc": t.doc,
         "defaults": t.defaults, "params": t.schema}
        for t in sorted(_REGISTRY.values(), key=lambda t: t.name)
    ]


def _finite(d: dict) -> dict:
    """Clamp every value to a finite float."""
    out = {}
    for k, v in d.items():
        v = float(v)
        out[k] = v if np.isfinite(v) else 0.0
    return out


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------

@register(
    "summary_stats",
    defaults={},
    schema={},
    doc="The 12 bridge-1 stats (mean,std,min,max,range,p10,p50,p90,slope,"
        "mean_abs_diff,max_abs_diff,last). Reproduces the existing baseline.",
)
def summary_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return {}
    diff = np.diff(x) if n > 1 else np.array([0.0])
    # slope: least-squares fit over the integer index, matching featurize.py
    if n > 1:
        idx = np.arange(n, dtype=float)
        slope = float(np.polyfit(idx, x, 1)[0])
    else:
        slope = 0.0
    return _finite({
        "mean": np.mean(x),
        "std": np.std(x),
        "min": np.min(x),
        "max": np.max(x),
        "range": np.max(x) - np.min(x),
        "p10": np.percentile(x, 10),
        "p50": np.percentile(x, 50),
        "p90": np.percentile(x, 90),
        "slope": slope,
        "mean_abs_diff": np.mean(np.abs(diff)),
        "max_abs_diff": np.max(np.abs(diff)),
        "last": x[-1],
    })


@register(
    "fft_band_energy",
    defaults={"n_bands": 5, "detrend": True},
    schema={"n_bands": "int, number of equal-width frequency bands (2-16)",
            "detrend": "bool, subtract linear trend before FFT"},
    doc="Fraction of signal energy in each of N equal-width frequency bands. "
        "Good for periodic degradation (vibration, rpm harmonics).",
)
def fft_band_energy(x: np.ndarray, n_bands: int = 5, detrend: bool = True) -> dict:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return {f"band{i}": 0.0 for i in range(n_bands)}
    if detrend:
        idx = np.arange(n, dtype=float)
        x = x - np.polyval(np.polyfit(idx, x, 1), idx)
    # one-sided power spectrum (drop DC). scipy.signal.welch is the upgrade path.
    spec = np.abs(np.fft.rfft(x)) ** 2
    spec = spec[1:]
    total = spec.sum()
    if total <= 0:
        return {f"band{i}": 0.0 for i in range(n_bands)}
    edges = np.linspace(0, spec.size, n_bands + 1).astype(int)
    out = {}
    for i in range(n_bands):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        out[f"band{i}"] = spec[lo:hi].sum() / total
    return _finite(out)


@register(
    "spectral_shape",
    defaults={"detrend": True},
    schema={"detrend": "bool, subtract linear trend before FFT"},
    doc="Spectral centroid, bandwidth, flatness (Wiener entropy) and 85% "
        "rolloff. Albedo-of-the-frequency-domain: shape, not magnitude.",
)
def spectral_shape(x: np.ndarray, detrend: bool = True) -> dict:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 8:
        return {"centroid": 0.0, "bandwidth": 0.0, "flatness": 0.0, "rolloff85": 0.0}
    if detrend:
        idx = np.arange(n, dtype=float)
        x = x - np.polyval(np.polyfit(idx, x, 1), idx)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n)
    spec, freqs = spec[1:], freqs[1:]
    p = spec.sum()
    if p <= 0:
        return {"centroid": 0.0, "bandwidth": 0.0, "flatness": 0.0, "rolloff85": 0.0}
    w = spec / p
    centroid = float((freqs * w).sum())
    bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * w).sum()))
    gmean = np.exp(np.mean(np.log(spec + 1e-12)))
    flatness = float(gmean / (spec.mean() + 1e-12))
    cumulative = np.cumsum(spec) / p
    rolloff = float(freqs[np.searchsorted(cumulative, 0.85)]) if cumulative[-1] >= 0.85 else float(freqs[-1])
    return _finite({"centroid": centroid, "bandwidth": bandwidth,
                    "flatness": flatness, "rolloff85": rolloff})


@register(
    "rolling_slope",
    defaults={"windows": [50, 200, 600]},
    schema={"windows": "list[int], window sizes (in timesteps) for trend slopes"},
    doc="Mean and max local trend slope over rolling windows. Catches "
        "gradual drift that a single global slope smears out.",
)
def rolling_slope(x: np.ndarray, windows=(50, 200, 600)) -> dict:
    x = np.asarray(x, dtype=float)
    n = x.size
    out = {}
    for w in windows:
        w = int(w)
        if n < w or w < 2:
            out[f"w{w}_meanslope"] = 0.0
            out[f"w{w}_maxslope"] = 0.0
            continue
        idx = np.arange(w, dtype=float)
        # slope over each non-overlapping window (cheap; stride=w)
        slopes = []
        for start in range(0, n - w + 1, w):
            seg = x[start:start + w]
            slopes.append(np.polyfit(idx, seg, 1)[0])
        slopes = np.asarray(slopes)
        out[f"w{w}_meanslope"] = slopes.mean()
        out[f"w{w}_maxslope"] = slopes[np.argmax(np.abs(slopes))]
    return _finite(out)


@register(
    "rolling_stat",
    defaults={"window": 200, "stat": "std", "agg": "max"},
    schema={"window": "int, rolling window size in timesteps",
            "stat": "str, per-window statistic: 'std'|'mean'|'range'",
            "agg": "str, how to collapse windows: 'max'|'mean'|'p90'"},
    doc="Aggregate of a rolling per-window statistic. e.g. max rolling-std "
        "surfaces a localized burst a global std would dilute.",
)
def rolling_stat(x: np.ndarray, window=200, stat="std", agg="max") -> dict:
    x = np.asarray(x, dtype=float)
    n, w = x.size, int(window)
    if n < w or w < 2:
        return {f"{stat}_{agg}": 0.0}
    vals = []
    for start in range(0, n - w + 1, max(1, w // 2)):  # 50% overlap
        seg = x[start:start + w]
        if stat == "std":
            vals.append(np.std(seg))
        elif stat == "mean":
            vals.append(np.mean(seg))
        elif stat == "range":
            vals.append(np.max(seg) - np.min(seg))
        else:
            raise ValueError(f"bad stat {stat}")
    vals = np.asarray(vals)
    collapse = {"max": vals.max, "mean": vals.mean,
                "p90": lambda: np.percentile(vals, 90)}[agg]
    return _finite({f"{stat}_{agg}": collapse()})


@register(
    "autocorr_lags",
    defaults={"lags": [1, 5, 20, 100]},
    schema={"lags": "list[int], autocorrelation lags (in timesteps)"},
    doc="Normalized autocorrelation at chosen lags. Periodicity / "
        "persistence signature; lag-1 ~ smoothness, longer lags ~ cycles.",
)
def autocorr_lags(x: np.ndarray, lags=(1, 5, 20, 100)) -> dict:
    x = np.asarray(x, dtype=float)
    n = x.size
    x = x - x.mean()
    denom = np.dot(x, x)
    out = {}
    for L in lags:
        L = int(L)
        if n <= L or denom <= 0:
            out[f"lag{L}"] = 0.0
        else:
            out[f"lag{L}"] = float(np.dot(x[:-L], x[L:]) / denom)
    return _finite(out)


@register(
    "peak_features",
    defaults={"threshold_std": 3.0},
    schema={"threshold_std": "float, peak = excursion beyond k*std from mean"},
    doc="Count and mean prominence of excursions beyond k*std. Targets "
        "intermittent spikes/transients rather than sustained level.",
)
def peak_features(x: np.ndarray, threshold_std=3.0) -> dict:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"count": 0.0, "mean_prom": 0.0, "rate": 0.0}
    mu, sd = x.mean(), x.std()
    if sd <= 0:
        return {"count": 0.0, "mean_prom": 0.0, "rate": 0.0}
    excursion = np.abs(x - mu) / sd
    mask = excursion > threshold_std
    count = float(mask.sum())
    return _finite({
        "count": count,
        "mean_prom": float(excursion[mask].mean()) if count else 0.0,
        "rate": count / x.size,
    })


@register(
    "diff_stats",
    defaults={"order": 1},
    schema={"order": "int, difference order (1 = velocity, 2 = acceleration)"},
    doc="Mean/std/max-abs of the n-th order difference. Roughness / "
        "jerk signature, scale-aware to step-to-step dynamics.",
)
def diff_stats(x: np.ndarray, order=1) -> dict:
    x = np.asarray(x, dtype=float)
    d = x
    for _ in range(int(order)):
        d = np.diff(d) if d.size > 1 else np.array([0.0])
    if d.size == 0:
        d = np.array([0.0])
    return _finite({"mean": d.mean(), "std": d.std(), "max_abs": np.max(np.abs(d))})


@register(
    "hist_entropy",
    defaults={"bins": 16},
    schema={"bins": "int, histogram bins for Shannon entropy estimate"},
    doc="Shannon entropy of the value histogram. Distributional spread / "
        "multimodality, invariant to mean and scale of the channel.",
)
def hist_entropy(x: np.ndarray, bins=16) -> dict:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.ptp(x) == 0:
        return {"entropy": 0.0}
    h, _ = np.histogram(x, bins=int(bins))
    p = h / h.sum()
    p = p[p > 0]
    return _finite({"entropy": float(-(p * np.log2(p)).sum())})


@register(
    "crossing_rate",
    defaults={},
    schema={},
    doc="Mean-crossing rate: fraction of consecutive samples that cross the "
        "channel mean. Cheap oscillation/noisiness proxy.",
)
def crossing_rate(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return {"rate": 0.0}
    centered = x - x.mean()
    crossings = np.sum(np.diff(np.sign(centered)) != 0)
    return _finite({"rate": crossings / (x.size - 1)})


# --------------------------------------------------------------------------
# (v2 extension point) cross-channel transforms would register here with a
# different fn signature fn(x, y, **params) and the spec featurizer would route
# {"transform": ..., "channel_pairs": [["a","b"]]}. Kept out of v1 on purpose.
# --------------------------------------------------------------------------


def _sanitize_series(x: np.ndarray) -> np.ndarray:
    """Make a channel finite without changing its length. Real sensor channels
    have dropouts (NaN) and occasional inf; spectral/temporal transforms can't
    consume those (e.g. np.histogram raises on a non-finite range). Interpolate
    internal gaps and clamp the edges to the nearest finite value; an all-missing
    channel becomes zeros. Applied identically at train and runtime, so the model
    never sees a different missingness treatment between the two."""
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    if mask.all():
        return x
    if not mask.any():
        return np.zeros_like(x)
    idx = np.arange(x.size)
    out = x.copy()
    out[~mask] = np.interp(idx[~mask], idx[mask], x[mask])  # clamps at the ends
    return out


def apply(transform_name: str, x: np.ndarray, params: dict | None = None) -> dict:
    """Run one transform on one channel, returning {feature_name: float}."""
    t = get(transform_name)
    p = {**t.defaults, **(params or {})}
    return t.fn(_sanitize_series(x), **p)