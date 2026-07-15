"""
anomaly_localization.py
========================
Aviation-specific raw-signal anomaly localization.

Detects cross-cylinder imbalances in engine temperature sensors (EGT and
CHT). Compares each cylinder's phase-averaged temperature against the
median of its siblings, then isolates load-dependent faults by
contrasting a phase's deviation against the cylinder's own calmest-phase
deviation — this is what lets it reject benign, constant cross-cylinder
offsets (present equally in every phase) rather than flagging every
flight where one cylinder simply always reads a bit warm.

Aviation-domain-specific (hardcodes EGT/CHT cylinder-group column names)
— lives under domain/aviation/ alongside flight_phases.py, not
harness/, for the same reason: not reusable for a non-aviation dataset.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

DEFAULT_GROUPS = {
    "EGT": ["E1 EGT1", "E1 EGT2", "E1 EGT3", "E1 EGT4"],
    "CHT": ["E1 CHT1", "E1 CHT2", "E1 CHT3", "E1 CHT4"],
}
DEFAULT_THRESHOLDS = {"EGT": 30.0, "CHT": 20.0}   # degF from the sibling median


def _cyl_num(ch: str):
    m = re.search(r"(\d+)\s*$", ch)
    return int(m.group(1)) if m else None


def localize_anomaly(df: pd.DataFrame, segments=None, groups=None, thresholds=None,
                      sample_hz: float = 1.0, min_valid_frac: float = 0.4) -> dict:
    """
    Identifies specific cylinders exhibiting abnormal temperature
    deviations.

    Args:
        df: raw time-series data for a single flight.
        segments: flight phase segments (from
            domain.aviation.flight_phases.segment_flight); computed if
            not given.
        groups: sensor-group -> channel-list mapping. Defaults to
            DEFAULT_GROUPS.
        thresholds: per-group excess-deviation threshold. Defaults to
            DEFAULT_THRESHOLDS.
        sample_hz: sampling frequency.
        min_valid_frac: minimum required non-NaN fraction for a channel
            to be evaluated (real-world CHT sensors have real dropout).

    Returns:
        dict of detected anomalies: channel, cylinder, deviation, worst
        phase, and whether the finding is corroborated by the other
        temperature group (EGT vs CHT) at the same cylinder.
    """
    groups = groups or DEFAULT_GROUPS
    thresholds = thresholds or DEFAULT_THRESHOLDS
    if segments is None:
        from .flight_phases import segment_flight
        segments = segment_flight(df, sample_hz=sample_hz)["segments"]
    phases = ["climb", "cruise", "descent"]

    findings = []
    checked = []
    for gname, chans in groups.items():
        present = [c for c in chans if c in df.columns]
        if len(present) < 3:                       # need >=3 for a robust median
            continue
        checked.append(gname)
        thr = thresholds.get(gname, 30.0)
        arr = df[present].to_numpy(dtype=float)
        valid_frac = {ch: float(np.isfinite(arr[:, i]).mean()) for i, ch in enumerate(present)}

        # rows belonging to each phase
        phase_rows = {ph: np.concatenate(
                        [np.arange(s["start_idx"], s["end_idx"] + 1)
                         for s in segments if s["phase"] == ph] or [np.array([], int)])
                      for ph in phases}
        phase_start = {ph: next((s["start_s"] for s in segments if s["phase"] == ph), None)
                       for ph in phases}

        # per-cylinder mean per phase, then deviation from the median of the 4 means
        chan_phase_dev = {ch: {} for ch in present}
        for ph, rows in phase_rows.items():
            if rows.size == 0:
                continue
            sub = arr[rows]                        # (m, k)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                cyl_means = np.nanmean(sub, axis=0)          # (k,)
                center = float(np.nanmedian(cyl_means))
            for i, ch in enumerate(present):
                if np.isfinite(cyl_means[i]):
                    chan_phase_dev[ch][ph] = round(float(cyl_means[i] - center), 1)

        for ch in present:
            if valid_frac[ch] < min_valid_frac or not chan_phase_dev[ch]:
                continue
            pdv = chan_phase_dev[ch]
            # Isolate load-dependent faults by comparing the worst phase against
            # the cylinder's calmest phase (baseline). This rejects benign constant offsets.
            baseline = min(pdv.values(), key=abs)              # calmest-phase deviation
            excess = {ph: round(pdv[ph] - baseline, 1) for ph in pdv}
            worst_ph = max(excess, key=lambda p: abs(excess[p]))
            if abs(excess[worst_ph]) > thr:
                findings.append({
                    "channel": ch, "group": gname, "cylinder": _cyl_num(ch),
                    "direction": "hot" if excess[worst_ph] > 0 else "cold",
                    "excess": excess[worst_ph], "deviation": pdv[worst_ph],
                    "threshold": thr, "worst_phase": worst_ph,
                    "worst_segment_start_s": phase_start[worst_ph],
                    "phase_deviations": pdv, "valid_frac": round(valid_frac[ch], 3),
                })
    findings.sort(key=lambda f: abs(f["excess"]), reverse=True)

    flagged = {(f["group"], f["cylinder"]) for f in findings}
    for f in findings:
        other = "CHT" if f["group"] == "EGT" else "EGT"
        f["corroborated_by_other_group"] = (other, f["cylinder"]) in flagged

    return {
        "tool": "localize_anomaly",
        "groups_checked": checked,
        "n_flagged": len(findings),
        "findings": findings,
        "note": "deviation = degF above(+)/below(-) the sibling-cylinder median, per "
                "phase; past 'threshold' is abnormal given the ~40 degF (EGT)/~19 degF "
                "(CHT) normal spread. CHT tolerates ~10% dropout (valid_frac).",
    }
