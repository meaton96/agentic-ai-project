"""
make_golden_dataset.py
======================
Deterministic synthetic dataset generator for regression testing.

Creates a fixed, reproducible dataset containing a balanced mix of normal 
and anomalous (single-cylinder fault) flights. Outputs the data in two formats 
required by the aviation pipeline:
  1. A single concatenated CSV file (`C28_golden.csv`) for bulk processing/training.
  2. Individual flight CSV files within a `demo/` subdirectory, accompanied by a 
     `metadata.csv` index, simulating runtime directory ingestion.
"""
from __future__ import annotations
import os
import pandas as pd

try:
    from .make_synth_flight import make_flight, SENSORS, NON_SENSOR
except ImportError:
    from make_synth_flight import make_flight, SENSORS, NON_SENSOR


def _roster(n_per_fold=8, folds=("0", "1", "2", "3", "4")):
    """
    Generates a deterministic schedule of flight configurations.
    
    Ensures a balanced mix of normal and anomalous flights distributed 
    across multiple aircraft and data folds.
    
    Args:
        n_per_fold (int, optional): Number of flights to generate per fold. Defaults to 8.
        folds (tuple[str, ...], optional): Tuple of fold identifiers. Defaults to ("0", "1", "2", "3", "4").
        
    Returns:
        list[tuple]: A list of flight configurations in the format:
            (flight_id, plane_id, fold, label, has_fault).
    """
    roster, fid = [], 0
    for fold in folds:
        for k in range(n_per_fold):
            plane = int(fold) * 3 + (k % 3)          # a few planes per fold (group key)
            fault = (k % 2 == 0)                      # balanced
            roster.append((fid, plane, fold, 1 if fault else 0, fault))
            fid += 1
    return roster


def build(out_dir: str, n_per_fold: int = 8) -> dict:
    """
    Builds the complete golden dataset and writes it to disk.
    
    Generates synthetic flight time-series data using `make_flight`, injecting 
    specific thermal anomalies for positive class labels. Saves both a unified 
    CSV dataset and a directory of individual flight files.
    
    Args:
        out_dir (str): Destination directory path for the generated dataset.
        n_per_fold (int, optional): Number of flights per data split/fold. Defaults to 8.
        
    Returns:
        dict: Summary statistics and file paths for the generated dataset, including
            row counts, faulty flight counts, and directory locations.
    """
    demo = os.path.join(out_dir, "demo")
    os.makedirs(demo, exist_ok=True)
    roster = _roster(n_per_fold)

    all_rows, meta = [], []
    for fid, plane, fold, label, fault in roster:
        anom = ({"cyl": 3, "egt_delta": 85.0, "cht_delta": 20.0, "phase": "climb"}
                if fault else None)
        df = make_flight(seed=10_000 + fid, label=label, plane_id=plane,
                         fold=fold, flight_id=fid, anomaly=anom)
        all_rows.append(df)
        fname = f"flight_{fid:04d}.csv"
        df.to_csv(os.path.join(demo, fname), index=False)
        meta.append({"filename": fname, "label": label, "plane_id": plane, "paper_split": fold})

    big = pd.concat(all_rows, ignore_index=True)
    big.to_csv(os.path.join(out_dir, "C28_golden.csv"), index=False)
    pd.DataFrame(meta).to_csv(os.path.join(demo, "metadata.csv"), index=False)

    return {"n_flights": len(roster), "n_rows": len(big),
            "csv": os.path.join(out_dir, "C28_golden.csv"),
            "demo_dir": demo, "metadata": os.path.join(demo, "metadata.csv"),
            "n_faulty": sum(1 for r in roster if r[4])}


if __name__ == "__main__":
    import sys
    print(build(sys.argv[1] if len(sys.argv) > 1 else "golden_data"))
