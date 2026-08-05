#!/usr/bin/env python3
"""
NGAFID-MC aviation dataset adapter: rolls up a raw, long-format flight
sensor CSV (one row per timestep, many timesteps per flight, flights
concatenated and identified by an `id` column) into one row per flight —
the shape the rest of this pipeline already knows how to classify.

NGAFID-specific vocabulary (the sensor column names, the `before_after`
label strings) lives in agentic_ml.domain.aviation.ngafid_config — this
script is just a thin CLI wrapper around it. The actual rollup logic is
dataset-agnostic and lives in agentic_ml.harness.timeseries_features.
(orchestrator/dynamic_loop.py's featurize_timeseries agent uses the same
config + rollup engine to do this automatically as part of the dynamic
orchestrator's planner loop — this script remains useful standalone, or
for run_orchestrator.py's static pipeline, which has no such agent.)

Zero LLM/agent involvement — pure deterministic pre-processing, same as
scripts/run_baseline_ladder.py. Run this once to produce a tabular CSV,
then feed that CSV into the existing pipeline unchanged:

Usage:
    python scripts/featurize_ngafid_flights.py \\
        --csv /path/to/C28.csv \\
        --out datasets/processed/ngafid_c28_flights.csv \\
        --max-flights 100   # smoke test a slice first; omit for the full file

    python scripts/run_orchestrator.py \\
        --data datasets/processed/ngafid_c28_flights.csv \\
        --target before_after \\
        --group-column plane_id \\
        --id-columns id,date_diff,split \\
        --strategy group \\
        --max-candidates 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_ml.domain.aviation.ngafid_config import (
    NGAFID_EXTRA_COLUMNS, NGAFID_GROUP_COLUMN, NGAFID_ID_COLUMN, NGAFID_LABEL_COLUMN,
    NGAFID_LABEL_MAP, NGAFID_SENSORS,
)
from agentic_ml.harness.timeseries_features import build_flight_feature_table_streaming
from agentic_ml.paths import datasets_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="path to the raw long-format NGAFID CSV")
    parser.add_argument("--out", default=str(datasets_root() / "processed" / "ngafid_flights.csv"))
    parser.add_argument("--id-column", default=NGAFID_ID_COLUMN)
    parser.add_argument("--group-column", default=NGAFID_GROUP_COLUMN)
    parser.add_argument("--label-column", default=NGAFID_LABEL_COLUMN)
    parser.add_argument("--extra-columns", default=",".join(NGAFID_EXTRA_COLUMNS),
                         help="comma-separated columns to carry through unchanged "
                              "(not used as features)")
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--max-flights", type=int, default=None,
                         help="stop after this many flights; omit for the full file")
    args = parser.parse_args()

    extra_columns = [c.strip() for c in args.extra_columns.split(",") if c.strip()]

    table = build_flight_feature_table_streaming(
        args.csv,
        sensor_columns=NGAFID_SENSORS,
        id_column=args.id_column,
        group_column=args.group_column,
        label_column=args.label_column,
        label_map=NGAFID_LABEL_MAP,
        extra_columns=extra_columns,
        chunksize=args.chunksize,
        max_flights=args.max_flights,
    )

    n_unique_labels = table[args.label_column].nunique()
    if n_unique_labels != 2:
        raise SystemExit(
            f"expected exactly 2 distinct resolved label values in "
            f"'{args.label_column}', found {n_unique_labels}: "
            f"{sorted(table[args.label_column].unique())}. Passing --target "
            f"directly to run_orchestrator.py skips intake's own class-count "
            f"check, so this script checks it instead."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)

    n_feature_cols = len([c for c in table.columns if "__" in c])
    balance = table[args.label_column].value_counts().to_dict()
    print(
        f"[featurize_ngafid_flights] {len(table)} flights | "
        f"{table[args.group_column].nunique()} planes | "
        f"label balance {balance} | {n_feature_cols} feature columns -> {out_path}"
    )


if __name__ == "__main__":
    main()
