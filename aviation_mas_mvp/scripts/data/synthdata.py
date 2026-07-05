"""
synthdata.py
============
Generates synthetic NGAFID-MC formatted flight data for testing and validation.
Provides functions to create both easy-to-learn baseline datasets and more 
complex, realistic datasets that simulate plane-specific feature leakage.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# Default sensor channel set for synthetic generation
CHANNELS = ["engine_rpm", "oil_temp", "oil_pressure", "cht", "egt",
            "fuel_flow", "vibration", "airspeed", "altitude"]

# Subset of channels injected with the synthetic maintenance signal
SIGNAL_CHANNELS = ("vibration", "oil_temp", "egt")


def make_synthetic(root: str | Path,
                   n_planes: int = 12,
                   flights_per_plane: int = 8,
                   steps: int = 600,
                   pos_rate: float = 0.5,
                   seed: int = 0) -> tuple[Path, Path]:
    """
    Generates individual flight CSVs and a central metadata table.
    Positive labels (maintenance required) receive a distinct upward drift 
    and increased variance on the designated signal channels.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    
    flight_dir = root / "flights"
    flight_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    
    for pid in range(n_planes):
        # Assign cluster roughly matching real dataset distribution
        cluster = "C28" if pid % 3 != 0 else "C37"  
        
        for f in range(flights_per_plane):
            label = int(rng.random() < pos_rate)
            t = np.linspace(0, 1, steps)
            data = {}
            
            for ch in CHANNELS:
                base = rng.normal(50, 5)
                sig = base + 5 * np.sin(2 * np.pi * t * rng.integers(1, 4))
                noise = rng.normal(0, 1.5, steps)
                
                # Inject maintenance drift signal for positive labels
                if label == 1 and ch in SIGNAL_CHANNELS:
                    sig = sig + 12 * t + rng.normal(0, 3, steps)
                    
                data[ch] = sig + noise
                
            fn = f"plane{pid}_flight{f}.csv"
            pd.DataFrame(data).to_csv(flight_dir / fn, index=False)
            
            meta.append({"filename": fn, "label": label,
                         "plane_id": f"P{pid}", "cluster": cluster})
                         
    meta_path = root / "metadata.csv"
    pd.DataFrame(meta).to_csv(meta_path, index=False)
    return flight_dir, meta_path


def make_synthetic_realistic(root: str | Path,
                             n_planes: int = 24,
                             flights_per_plane: int = 12,
                             steps: int = 600,
                             signal_strength: float = 1.2,
                             noise: float = 9.0,
                             signature_scale: float = 10.0,
                             sick_rate: float = 0.82,
                             healthy_rate: float = 0.18,
                             seed: int = 0) -> tuple[Path, Path]:
    """
    Generates a harder synthetic dataset designed to test group-disjoint validation.
    
    Introduces:
    1. Plane-correlated labels (latent health propensity per airframe).
    2. Per-plane signatures (fixed random offsets) that can cause data leakage 
       if splits are not properly grouped by tail number.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    
    flight_dir = root / "flights"
    flight_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    
    for pid in range(n_planes):
        cluster = "C28" if pid % 3 != 0 else "C37"
        
        # Determine if this specific tail number leans healthy or problematic
        plane_health = sick_rate if rng.random() < 0.5 else healthy_rate
        
        # Apply a fixed channel offset for this specific plane (leakage risk)
        signature = {ch: rng.normal(0, signature_scale) for ch in CHANNELS}  
        
        for f in range(flights_per_plane):
            label = int(rng.random() < plane_health)
            t = np.linspace(0, 1, steps)
            data = {}
            
            for ch in CHANNELS:
                base = rng.normal(50, 5) + signature[ch]
                sig = base + 5 * np.sin(2 * np.pi * t * rng.integers(1, 4))
                eps = rng.normal(0, noise, steps)
                
                if label == 1 and ch in SIGNAL_CHANNELS:
                    sig = sig + signal_strength * t + rng.normal(0, signal_strength * 0.4, steps)
                    
                data[ch] = sig + eps
                
            fn = f"plane{pid}_flight{f}.csv"
            pd.DataFrame(data).to_csv(flight_dir / fn, index=False)
            
            meta.append({"filename": fn, "label": label,
                         "plane_id": f"P{pid}", "cluster": cluster})
                         
    meta_path = root / "metadata.csv"
    pd.DataFrame(meta).to_csv(meta_path, index=False)
    return flight_dir, meta_path


# Canonical schema for the real NGAFID-MC dataset
NGAFID_SENSORS = ['volt1', 'volt2', 'amp1', 'amp2', 'FQtyL', 'FQtyR', 'E1 FFlow',
                  'E1 OilT', 'E1 OilP', 'E1 RPM', 'E1 CHT1', 'E1 CHT2', 'E1 CHT3',
                  'E1 CHT4', 'E1 EGT1', 'E1 EGT2', 'E1 EGT3', 'E1 EGT4', 'OAT',
                  'IAS', 'VSpd', 'NormAc', 'AltMSL']
                  
NGAFID_REAL_SIGNAL = ('E1 OilT', 'E1 EGT1', 'NormAc')


def make_ngafid_mock_csv(out_csv: str | Path,
                         n_planes: int = 20,
                         flights_per_plane: int = 15,
                         min_rows: int = 250,
                         max_rows: int = 600,
                         sick_rate: float = 0.78,
                         healthy_rate: float = 0.22,
                         signal_strength: float = 6.0,
                         noise: float = 7.0,
                         n_fold: int = 5,
                         seed: int = 0) -> Path:
    """
    Writes a single, continuous CSV matching the exact real NGAFID-MC schema.
    
    Used primarily to test streaming mechanisms and boundary carry-over logic, 
    ensuring variable-length flights are properly demarcated by `id`.
    """
    out_csv = Path(out_csv)
    rng = np.random.default_rng(seed)
    
    cols = NGAFID_SENSORS + ["id", "plane_id", "split", "date_diff", "before_after"]
    fid = 0
    
    with out_csv.open("w") as fh:
        fh.write(",".join(f'"{c}"' if " " in c else c for c in cols) + "\n")
        
        for pid in range(n_planes):
            health = sick_rate if rng.random() < 0.5 else healthy_rate
            signature = {c: rng.normal(0, 8) for c in NGAFID_SENSORS}
            
            # Assign fold id per tail to mock real file logic
            fold = pid % n_fold   
            
            for _ in range(flights_per_plane):
                is_before = rng.random() < health           
                nrows = int(rng.integers(min_rows, max_rows))
                t = np.linspace(0, 1, nrows)
                block = {}
                
                for c in NGAFID_SENSORS:
                    base = rng.normal(50, 6) + signature[c]
                    sig = base + 4 * np.sin(2 * np.pi * t * rng.integers(1, 4))
                    sig = sig + rng.normal(0, noise, nrows)
                    
                    if is_before and c in NGAFID_REAL_SIGNAL:
                        sig = sig + signal_strength * t + rng.normal(0, 2, nrows)
                        
                    block[c] = np.round(sig, 3)
                    
                date_diff = int(rng.choice([-2, -1, 1, 2]))
                df = pd.DataFrame(block)
                
                df["id"] = fid
                df["plane_id"] = f"N{1000+pid}"
                df["split"] = fold
                df["date_diff"] = date_diff if is_before else abs(date_diff)
                df["before_after"] = "before" if is_before else "after"
                
                df.to_csv(fh, header=False, index=False)
                fid += 1
                
    return out_csv


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_data"
    fd, mp = make_synthetic(out)
    print("flight_dir:", fd)
    print("metadata  :", mp)