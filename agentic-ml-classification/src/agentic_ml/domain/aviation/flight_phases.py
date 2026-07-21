"""
flight_phases.py
================
Aviation-specific, deterministic flight-phase segmentation.

Segments a single flight's raw time-series into standardized phases
(ground, climb, cruise, descent) using smoothed altitude derivatives
rather than raw vertical speed (VSpd), for robustness against noisy or
missing sensor data. Also identifies runway-contact events (takeoffs and
landings), correctly handling touch-and-go training flights.

Aviation-domain-specific (hardcodes AltMSL/IAS/VSpd column names) —
unlike harness/attribution.py, this isn't reusable for a non-aviation
dataset, which is why it lives under domain/aviation/ rather than
harness/. Used by the deep-dive agent (steps/deep_dive_step.py) to give
localize_anomaly a phase to contrast against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    """Moving-average smoothing via 1D convolution."""
    if win <= 1:
        return x
    k = np.ones(win) / win
    return np.convolve(x, k, mode="same")


def _sanitize(x: np.ndarray) -> np.ndarray:
    """Interpolates NaN/Inf values (sensor dropouts) without changing
    array length, so it stays aligned with other time-series columns."""
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    if m.all():
        return x
    if not m.any():
        return np.zeros_like(x)
    idx = np.arange(x.size)
    out = x.copy()
    out[~m] = np.interp(idx[~m], idx[m], x[m])
    return out


def _merge_short(labels: list[str], min_len: int) -> list[str]:
    """Absorbs phase-label runs shorter than min_len into the longer
    adjacent segment, so sensor jitter doesn't shatter a continuous
    phase into unrealistic micro-segments."""
    labels = list(labels)
    runs = []  # [label, start, end)
    s = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[s]:
            runs.append([labels[s], s, i]); s = i
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for j, (lab, a, b) in enumerate(runs):
            if b - a < min_len:
                if j == 0:
                    runs[j + 1][1] = a
                elif j == len(runs) - 1:
                    runs[j - 1][2] = b
                else:
                    if (runs[j - 1][2] - runs[j - 1][1]) >= (runs[j + 1][2] - runs[j + 1][1]):
                        runs[j - 1][2] = b
                    else:
                        runs[j + 1][1] = a
                runs.pop(j); changed = True
                break
    out = labels[:]
    for lab, a, b in runs:
        for i in range(a, b):
            out[i] = lab
    return out


def segment_flight(df: pd.DataFrame, alt_col="AltMSL", ias_col="IAS", vspd_col="VSpd",
                    sample_hz: float = 1.0, smooth_s: float = 15.0,
                    agl_ground_ft: float = 150.0, ias_ground_kt: float = 35.0,
                    touch_agl_ft: float = 100.0, climb_fpm: float = 300.0,
                    min_phase_s: float = 20.0) -> dict:
    """
    Segments a single flight into discrete phases and identifies runway
    events.

    Note: phase labeling (e.g. 'ground') requires both low AGL and low
    airspeed. Runway-contact logic (liftoffs/touchdowns) strictly
    monitors AGL crossing `touch_agl_ft`, so touch-and-go maneuvers are
    captured even if airspeed remains high throughout.
    """
    if alt_col not in df.columns:
        raise KeyError(f"{alt_col!r} not in flight columns: {list(df.columns)[:8]}...")
    n = len(df)
    win = max(1, int(round(smooth_s * sample_hz)))
    alt = _sanitize(np.asarray(df[alt_col].values, dtype=float))
    alt_s = _smooth(alt, win)
    field = float(np.percentile(alt_s, 2))        # ground proxy = low percentile of MSL
    agl = alt_s - field
    vrate = np.gradient(alt_s) * 60.0 * sample_hz  # fpm, from smoothed altitude
    ias = (_sanitize(np.asarray(df[ias_col].values, dtype=float)) if ias_col in df.columns
           else np.full(n, np.nan))

    # ---- phase labels (taxi needs low AND slow) ----
    on_ground = agl < agl_ground_ft
    if ias_col in df.columns:
        on_ground = on_ground & (np.nan_to_num(ias, nan=0.0) < ias_ground_kt)
    labels = []
    for i in range(n):
        if on_ground[i]:
            labels.append("ground")
        elif vrate[i] > climb_fpm:
            labels.append("climb")
        elif vrate[i] < -climb_fpm:
            labels.append("descent")
        else:
            labels.append("cruise")
    labels = _merge_short(labels, max(1, int(round(min_phase_s * sample_hz))))

    segments, s = [], 0
    for i in range(1, n + 1):
        if i == n or labels[i] != labels[s]:
            seg_alt = alt[s:i]
            segments.append({
                "phase": labels[s], "start_idx": s, "end_idx": i - 1,
                "start_s": round(s / sample_hz, 1),
                "duration_s": round((i - s) / sample_hz, 1),
                "alt_min_ft": round(float(seg_alt.min()), 1),
                "alt_max_ft": round(float(seg_alt.max()), 1),
                "agl_max_ft": round(float(seg_alt.max() - field), 1),
            })
            s = i

    # ---- runway-contact events (AGL-only, catches touch-and-go) ----
    airborne = agl > touch_agl_ft
    liftoffs, touchdowns = [], []
    for i in range(1, n):
        if airborne[i] and not airborne[i - 1]:
            liftoffs.append(round(i / sample_hz, 1))       # ground -> air = takeoff
        elif not airborne[i] and airborne[i - 1]:
            touchdowns.append(round(i / sample_hz, 1))     # air -> ground = landing

    airborne_s = round(float(airborne.sum()) / sample_hz, 1)
    phase_seconds = {p: round(sum(seg["duration_s"] for seg in segments
                                   if seg["phase"] == p), 1)
                      for p in ("ground", "climb", "cruise", "descent")}

    return {
        "tool": "segment_flight",
        "n_steps": n, "sample_hz": sample_hz,
        "field_elevation_ft": round(field, 1),
        "n_takeoffs": len(liftoffs), "n_landings": len(touchdowns),
        "n_airborne_episodes": len(liftoffs),
        "takeoff_times_s": liftoffs, "landing_times_s": touchdowns,
        "airborne_s": airborne_s,
        "phase_seconds": phase_seconds,
        "segments": segments,
    }
