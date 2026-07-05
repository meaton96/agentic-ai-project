# """
# flight_phases.py
# ================
# Deterministic flight-phase segmentation module.

# Segments flight time-series data into standardized phases (taxi, climb, 
# cruise, descent). Utilizes smoothed altitude derivatives rather than raw vertical
# speed (VSpd) to ensure robustness against noisy or missing sensor data. Also 
# identifies runway contact events (takeoffs and landings), successfully handling
# complex operations like touch-and-go training flights.
# """
# from __future__ import annotations
# import numpy as np
# import pandas as pd


# def _smooth(x: np.ndarray, win: int) -> np.ndarray:
#     """
#     Applies a moving average smoothing filter using 1D convolution.
    
#     Args:
#         x (np.ndarray): 1D array of numeric data.
#         win (int): Window size for the moving average.
        
#     Returns:
#         np.ndarray: Smoothed array of the same shape as input.
#     """
#     if win <= 1:
#         return x
#     k = np.ones(win) / win
#     return np.convolve(x, k, mode="same")


# def _sanitize(x: np.ndarray) -> np.ndarray:
#     """
#     Interpolates NaN and infinite values to handle sensor dropouts.
    
#     Ensures the array length remains unchanged so that it aligns with 
#     other time-series columns.
    
#     Args:
#         x (np.ndarray): 1D array of numeric data containing potential NaNs.
        
#     Returns:
#         np.ndarray: Sanitized array with interpolated values.
#     """
#     x = np.asarray(x, dtype=float)
#     m = np.isfinite(x)
#     if m.all():
#         return x
#     if not m.any():
#         return np.zeros_like(x)
#     idx = np.arange(x.size)
#     out = x.copy()
#     out[~m] = np.interp(idx[~m], idx[m], x[m])
#     return out


# def _merge_short(labels: list[str], min_len: int) -> list[str]:
#     """
#     Absorbs categorical label runs shorter than `min_len` into adjacent segments.
    
#     Prevents sensor jitter from shattering a continuous flight phase into 
#     unrealistically short micro-segments.
    
#     Args:
#         labels (list[str]): List of sequential categorical phase labels.
#         min_len (int): Minimum acceptable length for a continuous phase run.
        
#     Returns:
#         list[str]: Filtered list of labels with short runs merged.
#     """
#     labels = list(labels)
#     # collapse to runs
#     runs = []  # [label, start, end)
#     s = 0
#     for i in range(1, len(labels) + 1):
#         if i == len(labels) or labels[i] != labels[s]:
#             runs.append([labels[s], s, i]); s = i
#     changed = True
#     while changed and len(runs) > 1:
#         changed = False
#         for j, (lab, a, b) in enumerate(runs):
#             if b - a < min_len:
#                 # merge into the longer neighbour
#                 if j == 0:
#                     runs[j + 1][1] = a
#                 elif j == len(runs) - 1:
#                     runs[j - 1][2] = b
#                 else:
#                     if (runs[j - 1][2] - runs[j - 1][1]) >= (runs[j + 1][2] - runs[j + 1][1]):
#                         runs[j - 1][2] = b
#                     else:
#                         runs[j + 1][1] = a
#                 runs.pop(j); changed = True
#                 break
#     out = labels[:]
#     for lab, a, b in runs:
#         for i in range(a, b):
#             out[i] = lab
#     return out


# def segment_flight(df: pd.DataFrame, alt_col="AltMSL", ias_col="IAS", vspd_col="VSpd",
#                    sample_hz: float = 1.0, smooth_s: float = 15.0,
#                    agl_ground_ft: float = 150.0, ias_ground_kt: float = 35.0,
#                    touch_agl_ft: float = 100.0, climb_fpm: float = 300.0,
#                    min_phase_s: float = 20.0) -> dict:
#     """
#     Segments a single flight into discrete phases and identifies runway events.
    
#     Note: Phase labeling (e.g., 'ground') requires both low AGL and low airspeed. 
#     Runway contact logic (liftoffs/touchdowns) strictly monitors AGL crossing 
#     `touch_agl_ft` to ensure touch-and-go maneuvers are captured, even if airspeed 
#     remains high.
    
#     Args:
#         df (pd.DataFrame): Time-series flight data.
#         alt_col (str): Column name for MSL altitude.
#         ias_col (str): Column name for Indicated Airspeed.
#         vspd_col (str): Column name for Vertical Speed (currently unused in favor of alt derivative).
#         sample_hz (float): Sampling frequency of the data.
#         smooth_s (float): Window size in seconds for altitude smoothing.
#         agl_ground_ft (float): AGL threshold below which the aircraft is considered grounded.
#         ias_ground_kt (float): Airspeed threshold below which the aircraft is considered grounded.
#         touch_agl_ft (float): AGL threshold triggering a liftoff or touchdown event.
#         climb_fpm (float): Vertical rate threshold in feet per minute for climb/descent phases.
#         min_phase_s (float): Minimum duration in seconds for a segment to be kept.
        
#     Returns:
#         dict: A JSON-serializable dictionary containing segment statistics, runway 
#             event times, and the list of flight segments.
#     """
#     if alt_col not in df.columns:
#         raise KeyError(f"{alt_col!r} not in flight columns: {list(df.columns)[:8]}...")
#     n = len(df)
#     win = max(1, int(round(smooth_s * sample_hz)))
#     alt = _sanitize(np.asarray(df[alt_col].values, dtype=float))
#     alt_s = _smooth(alt, win)
#     field = float(np.percentile(alt_s, 2))        # ground proxy = low percentile of MSL
#     agl = alt_s - field
#     vrate = np.gradient(alt_s) * 60.0 * sample_hz  # fpm, from smoothed altitude
#     ias = (_sanitize(np.asarray(df[ias_col].values, dtype=float)) if ias_col in df.columns
#            else np.full(n, np.nan))

#     # ---- phase labels (taxi needs low AND slow) ----
#     on_ground = agl < agl_ground_ft
#     if ias_col in df.columns:
#         on_ground = on_ground & (np.nan_to_num(ias, nan=0.0) < ias_ground_kt)
#     labels = []
#     for i in range(n):
#         if on_ground[i]:
#             labels.append("ground")
#         elif vrate[i] > climb_fpm:
#             labels.append("climb")
#         elif vrate[i] < -climb_fpm:
#             labels.append("descent")
#         else:
#             labels.append("cruise")
#     labels = _merge_short(labels, max(1, int(round(min_phase_s * sample_hz))))

#     segments, s = [], 0
#     for i in range(1, n + 1):
#         if i == n or labels[i] != labels[s]:
#             seg_alt = alt[s:i]
#             segments.append({
#                 "phase": labels[s], "start_idx": s, "end_idx": i - 1,
#                 "start_s": round(s / sample_hz, 1),
#                 "duration_s": round((i - s) / sample_hz, 1),
#                 "alt_min_ft": round(float(seg_alt.min()), 1),
#                 "alt_max_ft": round(float(seg_alt.max()), 1),
#                 "agl_max_ft": round(float(seg_alt.max() - field), 1),
#             })
#             s = i

#     # ---- runway-contact events (AGL-only, catches touch-and-go) ----
#     airborne = agl > touch_agl_ft
#     liftoffs, touchdowns = [], []
#     for i in range(1, n):
#         if airborne[i] and not airborne[i - 1]:
#             liftoffs.append(round(i / sample_hz, 1))       # ground -> air = takeoff
#         elif not airborne[i] and airborne[i - 1]:
#             touchdowns.append(round(i / sample_hz, 1))     # air -> ground = landing

#     airborne_s = round(float(airborne.sum()) / sample_hz, 1)
#     phase_seconds = {p: round(sum(seg["duration_s"] for seg in segments
#                                   if seg["phase"] == p), 1)
#                      for p in ("ground", "climb", "cruise", "descent")}

#     return {
#         "tool": "segment_flight",
#         "n_steps": n, "sample_hz": sample_hz,
#         "field_elevation_ft": round(field, 1),
#         "n_takeoffs": len(liftoffs), "n_landings": len(touchdowns),
#         "n_airborne_episodes": len(liftoffs),
#         "takeoff_times_s": liftoffs, "landing_times_s": touchdowns,
#         "airborne_s": airborne_s,
#         "phase_seconds": phase_seconds,
#         "segments": segments,
#     }


# if __name__ == "__main__":
#     import sys
#     df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "synth_flight.csv")
#     r = segment_flight(df)
#     import json
#     print(json.dumps({k: r[k] for k in
#           ("n_steps", "field_elevation_ft", "n_takeoffs", "n_landings",
#            "takeoff_times_s", "landing_times_s", "airborne_s", "phase_seconds")}, indent=2))
#     print("\nsegments:")
#     for seg in r["segments"]:
#         print(f"  {seg['start_s']:6.0f}s +{seg['duration_s']:4.0f}s  {seg['phase']:8s} "
#               f"AGL_max={seg['agl_max_ft']:6.0f}ft")
