"""
agentic-pdm command line.

    agentic-pdm ingest   --csv C28.csv --out data/c28 --id-column id --label-column before_after \\
                         --group-column plane_id --fold-column split --exclude date_diff \\
                         --positive-label 0   # NGAFID: 0 = pre-maintenance
    agentic-pdm baseline --store data/c28 --out runs/c28_baselines.json
    agentic-pdm train    --store data/c28 --out runs/c28_ref --folds 0 --epochs 20 --threads 3
                         [--plugin path/to/plugin.py] [--plugin-config '{"width": 32}']
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agentic_pdm.baselines import run_baselines
from agentic_pdm.harness.plugin import load_plugin
from agentic_pdm.harness.trainer import TrainConfig, run_cv
from agentic_pdm.ingest import IngestConfig, ingest_long_csv
from agentic_pdm.reference import SMALL_CNN_PATH
from agentic_pdm.store import load_store


def _int_list(value: str | None):
    return None if value is None else [int(v) for v in value.split(",") if v.strip()]


def cmd_ingest(args) -> None:
    t0 = time.perf_counter()
    manifest = ingest_long_csv(args.csv, args.out, IngestConfig(
        id_column=args.id_column,
        label_column=args.label_column,
        group_column=args.group_column,
        fold_column=args.fold_column,
        exclude_columns=[c for c in (args.exclude or "").split(",") if c],
        positive_label=args.positive_label,
        max_len=args.max_len,
        chunksize=args.chunksize,
        max_sequences=args.max_sequences,
        n_folds=args.n_folds,
    ))
    print(json.dumps({k: manifest[k] for k in ("shape", "classes", "fold_source", "truncated")}))
    print(f"ingested in {time.perf_counter() - t0:.1f}s -> {args.out}")


def cmd_baseline(args) -> None:
    store = load_store(args.store)
    results = run_baselines(store, folds=_int_list(args.folds))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


def cmd_train(args) -> None:
    import torch
    store = load_store(args.store)
    plugin = load_plugin(args.plugin or SMALL_CNN_PATH)
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        pool=args.pool,
        seed=args.seed,
        threads=args.threads,
        plugin_config=json.loads(args.plugin_config),
    )
    print(f"plugin {plugin.name} | torch {torch.__version__} | threads {args.threads or torch.get_num_threads()}")
    results = run_cv(store, plugin, cfg, args.out, folds=_int_list(args.folds))
    print(json.dumps(results["summary"], indent=2))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="agentic-pdm")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="long-format CSV -> tensor store")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--id-column", required=True)
    p.add_argument("--label-column", required=True)
    p.add_argument("--group-column")
    p.add_argument("--fold-column")
    p.add_argument("--exclude", help="comma-separated columns that are neither roles nor channels")
    p.add_argument("--positive-label", help="binary only: raw label value scored as the positive class")
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--chunksize", type=int, default=1_000_000)
    p.add_argument("--max-sequences", type=int)
    p.add_argument("--n-folds", type=int, default=5)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("baseline", help="dummy, length-only probe, summary-stat GBM")
    p.add_argument("--store", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--folds")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("train", help="cross-validate a plugin (default: reference small CNN)")
    p.add_argument("--store", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--plugin")
    p.add_argument("--plugin-config", default="{}")
    p.add_argument("--folds")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--pool", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int)
    p.set_defaults(func=cmd_train)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
