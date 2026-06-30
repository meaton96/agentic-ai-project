"""
train_real_baseline.py
=======================
A unified script to stream-featurize large NGAFID-MC cluster CSVs 
and train a classification model.

This automates data preprocessing and training into a single execution, 
supporting standard cross-validation, strict holdout-fold training for 
leakage-free agent evaluation, and custom spec-driven featurization 
discovered during the search phase.
"""
from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path

# --- bootstrap: make `scripts` importable without pip install -e -----------
_here = Path(__file__).resolve().parent
for _candidate in [_here, *_here.parents]:
    if (_candidate / "scripts").is_dir():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise SystemExit("Could not find a `scripts/` directory above this file. "
                     "Place train_real_baseline.py as a sibling of scripts/.")
# -----------------------------------------------------------------------------

from scripts.realdata import build_feature_table_streaming, peek_ngafid, fit_excluding_fold
from scripts.train_baseline import train



def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="path to the big per-cluster CSV (C28.csv / C37.csv)")
    ap.add_argument("--out-model", required=True, help="where to save the trained model (.joblib)")
    ap.add_argument("--out-feats", default=None, help="optionally save the feature table (.csv)")
    ap.add_argument("--out-metrics", default=None, help="optionally save CV metrics (.json)")
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--max-flights", type=int, default=None,
                    help="debug: cap the number of flights for a fast run")
    ap.add_argument("--skip-peek", action="store_true",
                    help="skip the pre-flight sanity check on the file's first chunk")
    ap.add_argument("--holdout-fold", default=None,
                    help="if set, fit ONLY on rows where the paper's `split` fold != this "
                        "value, and evaluate on the held-out fold (a genuine generalisation "
                        "test). Use this -- not the default whole-data fit -- for the model "
                        "an agent demo will classify against, and export the agent's demo "
                        "batch with `realdata export --only-fold <same value>` so the demo "
                        "flights were never seen during fitting. Without this, the default "
                        "fit uses every row, and any demo batch pulled from the same file is "
                        "largely the model's own training data (see fit_excluding_fold docs).")
    ap.add_argument("--spec", default=None,
                help="discovered feature spec (best_spec.json). If set, featurization "
                     "uses the spec instead of the hardcoded 12 stats.")
    args = ap.parse_args()

    if not args.skip_peek:
        print(f"[0/2] peeking at {args.csv} (first 200k rows)...")
        pk = peek_ngafid(args.csv, nrows=200_000)
        print(json.dumps({k: pk[k] for k in
              ("n_sensor_cols", "unexpected_sensors", "label_values_in_sample",
               "id_contiguous_in_sample")}, indent=2))
        if not pk["id_contiguous_in_sample"]:
            raise SystemExit("id is NOT contiguous in the sampled rows -- sort the CSV by "
                             "`id` before streaming, or featurize will misgroup flights. "
                             "Re-run with --skip-peek once you've confirmed it's safe.")
        if pk["unexpected_sensors"]:
            print(f"WARNING: unexpected sensor columns found: {pk['unexpected_sensors']}")
    if args.spec:
        import json as _json
        from scripts.featurize_spec import build_training_table
        spec = _json.loads(Path(args.spec).read_text())
        print(f"\n[1/2] spec-driven featurize ({spec.get('spec_id','?')}): {args.csv}")
        table = build_training_table(args.csv, spec, max_flights=args.max_flights,
                                    chunksize=args.chunksize)
    else:
        print(f"\n[1/2] streaming featurize: {args.csv}  (chunksize={args.chunksize})")
        table = build_feature_table_streaming(args.csv, chunksize=args.chunksize,
                                            max_flights=args.max_flights)
    if args.out_feats:
        table.to_csv(args.out_feats, index=False)
        print(f"      feature table saved -> {args.out_feats}")

    print(f"\n[2/2] training -> {args.out_model}")
    if args.holdout_fold is not None:
        if "paper_split" not in table.columns:
            raise SystemExit("--holdout-fold requires the `split` column from the source "
                             "CSV; it wasn't found in the featurized table.")
        metrics = fit_excluding_fold(table, args.holdout_fold, args.out_model)
        print(json.dumps(metrics, indent=2))
        if metrics["train_holdout_plane_overlap"] > 0:
            print("\nWARNING: train/holdout planes overlap -- the held-out evaluation "
                 "is not leakage-free. Check the `split` column's fold assignment.")
        if args.out_metrics:
            Path(args.out_metrics).write_text(json.dumps(metrics, indent=2))
    else:
        metrics = train(table, args.out_model, args.out_metrics)
        print(json.dumps({k: metrics[k] for k in
              ("roc_auc_mean", "roc_auc_std", "pr_auc_mean", "accuracy_mean",
               "n_flights", "n_planes", "class_balance")}, indent=2))
        print("\nNOTE: this model is fit on EVERY row in the table. Fine for reporting the "
             "honest CV baseline AUC above. NOT what you want to back an agent demo that "
             "classifies flights pulled from the same source file -- use --holdout-fold for "
             "that (see --help).")
    print(f"\nmodel saved -> {args.out_model}")
    if args.out_metrics:
        print(f"metrics saved -> {args.out_metrics}")


if __name__ == "__main__":
    main()