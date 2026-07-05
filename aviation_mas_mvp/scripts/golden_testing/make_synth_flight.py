"""
make_synth_flight.py
====================
Synthetic flight data generator for the NGAFID-MC schema.

Generates realistic synthetic flight data matching the schema of a Cirrus SR22T (C28),
including appropriate units, sensor ranges, and dropout patterns. This module is 
primarily used as a test fixture to validate anomaly detection and localization 
algorithms by planting known, controlled single-cylinder faults.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

SENSORS = ["volt1", "volt2", "amp1", "amp2", "FQtyL", "FQtyR", "E1 FFlow",
           "E1 OilT", "E1 OilP", "E1 RPM", "E1 CHT1", "E1 CHT2", "E1 CHT3",
           "E1 CHT4", "E1 EGT1", "E1 EGT2", "E1 EGT3", "E1 EGT4", "OAT",
           "IAS", "VSpd", "NormAc", "AltMSL"]
NON_SENSOR = ["id", "plane_id", "split", "date_diff", "before_after"]

# real per-channel NaN rates (from the C28 sample); default ~0 for the rest
NAN_RATE = {"E1 CHT2": 0.107, "E1 CHT3": 0.107, "E1 CHT4": 0.107, "E1 CHT1": 0.031,
            "volt2": 0.107, "amp2": 0.107}


def _agl_profile(profile: str) -> np.ndarray:
    """
    Generates a synthetic Above Ground Level (AGL) altitude profile.
    
    Args:
        profile (str): The flight profile type ('touch_and_go' or 'cross_country').
            
    Returns:
        np.ndarray: A 1D array representing the AGL altitude over time.
    """
    segs = []
    def ramp(d, a0, a1): segs.append(np.linspace(a0, a1, int(d)))
    def hold(d, a): segs.append(np.full(int(d), a))
    if profile == "touch_and_go":
        hold(120, 0); ramp(170, 0, 1800); hold(60, 1800)
        ramp(150, 1800, 40); ramp(150, 40, 1800); hold(40, 1800)
        ramp(150, 1800, 40); ramp(150, 40, 1800); ramp(120, 1800, 4000)
        hold(300, 4000); ramp(260, 4000, 0); hold(90, 0)
    else:  # cross_country: one clean climb / cruise / descent
        hold(150, 0); ramp(300, 0, 5500); hold(900, 5500)
        ramp(360, 5500, 0); hold(120, 0)
    return np.concatenate(segs)


def _lag(x, win):
    """
    Applies a simple moving average to smooth time-series transitions.
    
    Args:
        x (np.ndarray): The input 1D array to smooth.
        win (int): The smoothing window size.
        
    Returns:
        np.ndarray: The smoothed array.
    """
    return x if win <= 1 else np.convolve(x, np.ones(win) / win, mode="same")


def make_flight(seed=0, label=0, plane_id=37, fold=0, flight_id=0,
                profile="cross_country", anomaly: dict | None = None,
                field_elev=820.0) -> pd.DataFrame:
    """
    Generates a complete synthetic flight DataFrame with simulated sensor data.
    
    Models basic aircraft states (power, RPM, fuel flow, temperatures) to 
    create realistic multivariate time-series data. Incorporates defined sensor 
    noise and dropout rates. Can selectively inject thermal anomalies into 
    specific engine cylinders for evaluation purposes.
    
    Args:
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        label (int, optional): Target label (e.g., 0 for normal, 1 for faulty). Defaults to 0.
        plane_id (int, optional): Identifier for the simulated aircraft. Defaults to 37.
        fold (int, optional): Data split identifier. Defaults to 0.
        flight_id (int, optional): Identifier for the generated flight. Defaults to 0.
        profile (str, optional): The flight profile to follow. Defaults to 'cross_country'.
        anomaly (dict, optional): Configuration for planting a known fault. 
            Format: {'cyl': int, 'phase': str, 'egt_delta': float, 'cht_delta': float}. Defaults to None.
        field_elev (float, optional): Base field elevation in feet MSL. Defaults to 820.0.
        
    Returns:
        pd.DataFrame: A synthetic flight dataset matching the NGAFID-MC schema.
    """
    rng = np.random.default_rng(seed)
    agl = _agl_profile(profile)
    n = agl.size
    alt_msl = field_elev + agl
    airborne = agl > 30

    vrate = np.gradient(_lag(alt_msl, 15)) * 60.0
    power = np.full(n, 0.10)                       # ground idle
    power[airborne] = 0.70                         # cruise
    power[airborne & (vrate > 300)] = 0.95         # climb
    power[airborne & (vrate < -300)] = 0.22        # descent (throttled back)
    power = _lag(power, 20)

    rpm = np.clip(_lag(600 + 2050 * power, 6) + rng.normal(0, 14, n), 560, 2724)
    fflow = np.clip(_lag(1.5 + 14.5 * power, 6) + rng.normal(0, 0.3, n), 0.6, 16.4)

    # oil temp: cold start ~75 degF warming toward 150 + power, slow
    warm = 1 - np.exp(-np.arange(n) / 200.0)
    oilt = np.clip(75 * (1 - warm) + (150 + 38 * power) * warm + rng.normal(0, 1.2, n), 52, 195)
    oilp = np.clip(82 - (oilt - 75) * 0.18 + 7 * power + rng.normal(0, 1.3, n), 38, 91)

    # four cylinders track power; small per-cyl offsets give the real ~40/19 degF spread
    cht, egt = {}, {}
    cht_off = rng.normal(0, 11.0, 4)              # -> CHT max-min ~19 degF
    egt_off = rng.normal(0, 16.0, 4)              # -> EGT max-min ~40 degF
    for c in range(4):
        cht[c] = _lag(270 + 150 * power, 40) + cht_off[c] + rng.normal(0, 3.0, n)
        egt[c] = _lag(1100 + 450 * power, 12) + egt_off[c] + rng.normal(0, 10, n)

    if anomaly:                                   # plant a single hot cylinder
        c = int(anomaly["cyl"]) - 1
        ph = anomaly.get("phase", "climb")
        mask = (airborne & (vrate > 300)) if ph == "climb" else \
               (airborne & (np.abs(vrate) <= 300)) if ph == "cruise" else airborne
        ramp = _lag(mask.astype(float), 15)
        egt[c] = egt[c] + anomaly.get("egt_delta", 80) * ramp
        cht[c] = cht[c] + anomaly.get("cht_delta", 18) * ramp

    burn = np.cumsum(fflow) / 3600.0
    cols = {
        "volt1": 28.1 + rng.normal(0, 0.6, n),
        "volt2": 28.1 + rng.normal(0, 0.5, n),
        "amp1": 0.5 + 1.5 * power + rng.normal(0, 2.8, n),
        "amp2": rng.normal(0, 0.12, n),
        "FQtyL": np.clip(20.3 - burn * 0.5, 0, None),
        "FQtyR": np.clip(22.1 - burn * 0.5, 0, None),
        "E1 FFlow": fflow, "E1 OilT": oilt, "E1 OilP": oilp, "E1 RPM": rpm,
        "E1 CHT1": cht[0], "E1 CHT2": cht[1], "E1 CHT3": cht[2], "E1 CHT4": cht[3],
        "E1 EGT1": egt[0], "E1 EGT2": egt[1], "E1 EGT3": egt[2], "E1 EGT4": egt[3],
        "OAT": 6.0 - (alt_msl - field_elev) / 1000.0 * 2.0 + rng.normal(0, 0.5, n),
        "IAS": np.clip(np.where(airborne, 100, 0) + rng.normal(0, 6, n), -1, 160),
        "VSpd": np.clip(np.gradient(alt_msl) * 60.0 + rng.normal(0, 150, n), -4850, 3120),
        "NormAc": rng.normal(0, 0.07, n),
        "AltMSL": alt_msl + rng.normal(0, 8, n),
    }
    df = pd.DataFrame({k: np.asarray(v, dtype="float32") for k, v in cols.items()})

    for ch in df.columns:                          # real, concentrated dropout
        rate = NAN_RATE.get(ch, 0.0002)
        df.loc[rng.random(n) < rate, ch] = np.nan

    df["id"] = flight_id; df["plane_id"] = plane_id; df["split"] = fold
    df["date_diff"] = 1; df["before_after"] = label
    return df[SENSORS + NON_SENSOR]


if __name__ == "__main__":
    normal = make_flight(seed=1, label=0, flight_id=0)
    faulty = make_flight(seed=2, label=1, flight_id=1,
                         anomaly={"cyl": 3, "egt_delta": 80, "cht_delta": 18, "phase": "climb"})
    normal.to_csv("synth_flight.csv", index=False)
    faulty.to_csv("synth_flight_faulty.csv", index=False)
    egt = [c for c in SENSORS if "EGT" in c]; cht = [c for c in SENSORS if "CHT" in c]
    print("EGT range:", round(normal[egt].min().min()), "-", round(normal[egt].max().max()), "degF")
    print("CHT range:", round(normal[cht].min().min()), "-", round(normal[cht].max().max()), "degF")
    print("OilT range:", round(normal['E1 OilT'].min()), "-", round(normal['E1 OilT'].max()), "degF")
    print("RPM range:", round(normal['E1 RPM'].min()), "-", round(normal['E1 RPM'].max()))
    print("normal EGT cross-cyl spread (max-min) avg:",
          round((normal[egt].max(axis=1) - normal[egt].min(axis=1)).mean(), 1), "(real ~39.5)")
    print("normal CHT cross-cyl spread (max-min) avg:",
          round((normal[cht].max(axis=1) - normal[cht].min(axis=1)).mean(), 1), "(real ~18.9)")
    print("CHT3 dropout %:", round(normal['E1 CHT3'].isna().mean()*100, 1), "(real ~10.7)")
    f = faulty
    climb = (f['IAS'] > 30) & (f['VSpd'].rolling(15, min_periods=1).mean() > 300)
    sib = f[['E1 EGT1', 'E1 EGT2', 'E1 EGT4']].mean(axis=1)
    print("planted fault: EGT3 - siblings over climb =",
          round((f.loc[climb, 'E1 EGT3'] - sib[climb]).mean()), "degF (vs ~40 normal spread)")
