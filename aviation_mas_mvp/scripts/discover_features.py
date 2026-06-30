"""
discover_features.py
====================
CLI entry point for automated feature discovery.

Executes an iterative search loop over a balanced sample of flight data to find 
the optimal feature specification. Supports both deterministic (scripted) and 
LLM-driven proposers. Can optionally finalize the pipeline by training the 
operational model on the best discovered features and exporting a demo dataset.
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
    raise SystemExit("Could not find a `scripts/` directory above this file.")
# ---------------------------------------------------------------------------

from scripts.featurize_spec import (collect_balanced_sample, realdata_flight_iter,
                                    materialize, build_training_table)
from scripts.discovery_agent import (run_discovery, scripted_proposer, make_llm_proposer,
                                     StopConfig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="big per-cluster CSV (C28.csv / C37.csv)")
    ap.add_argument("--out-spec", default="best_spec.json")
    ap.add_argument("--log", default="discovery_log.jsonl")
    ap.add_argument("--holdout-fold", default="0",
                    help="TEST fold: the winning spec is reported on this fold, which is "
                         "NEVER used to choose the spec (kept clean of selection bias). "
                         "Also the demo-export fold.")
    ap.add_argument("--select-fold", default="1",
                    help="SELECT fold: discovery scores and picks the best spec on this "
                         "fold. Must differ from --holdout-fold so the reported number "
                         "isn't selected on the same data it's reported on.")
    ap.add_argument("--per-fold", type=int, default=60,
                    help="flights per fold in the balanced small batch (small-batch-first)")
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--folds", default="0,1,2,3,4",
                    help="comma-separated fold ids present in the `split` column")
    ap.add_argument("--full", action="store_true",
                    help="ignore --per-fold and search over ALL flights (slower; the "
                         "definitive number once the small batch looks promising)")
    ap.add_argument("--proposer", choices=["scripted", "llm"], default="scripted")
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min-improvement", type=float, default=0.005)
    ap.add_argument("--feature-budget", type=int, default=150,
                    help="how many features the spec may add ABOVE the summary_stats "
                         "baseline. Cap = baseline_features + this, resolved after the "
                         "baseline runs, so it's always coherent with the baseline size.")
    ap.add_argument("--max-features", type=int, default=None,
                    help="absolute feature cap; overrides --feature-budget if set.")
    ap.add_argument("--time-budget-s", type=float, default=None)
    # --- finalize: do train + export in the same command -------------------
    ap.add_argument("--finalize", action="store_true",
                    help="after discovery, train the operational model on the FULL data "
                         "with the winning spec and export the demo batch -- the whole "
                         "pipeline in one command. Discovery searches the small batch; "
                         "the final fit uses every flight.")
    ap.add_argument("--out-model", default="data/c28_model.joblib")
    ap.add_argument("--out-metrics", default="data/c28_metrics.json")
    ap.add_argument("--demo-dir", default="data/c28_demo")
    ap.add_argument("--demo-fold", default=None,
                    help="fold to export for the demo batch (defaults to --holdout-fold)")
    ap.add_argument("--demo-max-flights", type=int, default=200)
    args = ap.parse_args()

    if str(args.select_fold) == str(args.holdout_fold):
        raise SystemExit("--select-fold must differ from --holdout-fold, or the reported "
                         "test number is selected on the same data it's reported on "
                         "(the bias this whole separation exists to remove).")

    folds = tuple(args.folds.split(","))

    if args.full:
        print(f"[discovery] materializing ALL flights from {args.csv} (one pass)...")
        flight_iter = materialize(realdata_flight_iter(args.csv))
    else:
        print(f"[discovery] collecting fold-balanced sample "
              f"({args.per_fold}/fold) from {args.csv}...")
        sample = collect_balanced_sample(args.csv, per_fold=args.per_fold, folds=folds)
        flight_iter = lambda: sample

    if args.proposer == "scripted":
        proposer = scripted_proposer
    else:
        proposer = _llm_proposer()

    stop = StopConfig(max_iters=args.max_iters, patience=args.patience,
                      min_improvement=args.min_improvement, time_budget_s=args.time_budget_s,
                      max_features=args.max_features, feature_budget=args.feature_budget)

    print(f"[discovery] SELECT fold = {args.select_fold} (specs are chosen here); "
          f"TEST fold = {args.holdout_fold} (reported, never used to choose)")
    out = run_discovery(flight_iter, proposer=proposer, holdout_fold=args.select_fold,
                        stop=stop, log_path=args.log)

    if out["best_spec"] is None:
        raise SystemExit("no best spec found")
    Path(args.out_spec).write_text(json.dumps(out["best_spec"], indent=2))
    print(f"\n[discovery] wrote winning spec -> {args.out_spec}")
    s = out["summary"]
    print(f"[discovery] SELECT fold {args.select_fold}: baseline {s['baseline_auc']} -> "
          f"best {s['best_auc']} (this is the optimistic, search-time number)")

    if args.finalize:
        demo_fold = args.demo_fold if args.demo_fold is not None else args.holdout_fold
        _finalize(args.csv, out["best_spec"], args.holdout_fold, args.select_fold,
                  args.out_model, args.out_metrics, args.demo_dir, demo_fold,
                  args.demo_max_flights, args.chunksize)
    else:
        print("[discovery] run with --finalize to get the unbiased TEST-fold number "
              f"(fold {args.holdout_fold}) and the operational model.")


def _finalize(csv, spec, test_fold, select_fold, out_model, out_metrics, demo_dir,
              demo_fold, demo_max, chunksize):
    """Deterministic plumbing + the HONEST evaluation. Trains the operational
    model on the FULL data with the winning spec and reports it on the TEST fold
    -- which was never used to choose the spec -- alongside the summary_stats
    baseline on that same test fold. The lift here is unbiased; the search-time
    select-fold number is not."""
    import json as _json
    from scripts.realdata import (fit_excluding_fold, export_flights_to_dir,
                                  NGAFID_NON_SENSOR)
    import pandas as pd

    print(f"\n[finalize 1/3] winning spec ({spec.get('spec_id','?')}) full train, "
          f"report on TEST fold {test_fold} -> {out_model}")
    table_best = build_training_table(csv, spec, max_flights=None, chunksize=chunksize)
    m_best = fit_excluding_fold(table_best, test_fold, out_model)
    Path(out_metrics).write_text(_json.dumps(m_best, indent=2))

    # baseline (summary_stats on all channels) on the SAME untouched test fold
    print(f"[finalize 2/3] summary_stats baseline on TEST fold {test_fold} (for honest lift)")
    sensors = [c for c in pd.read_csv(csv, nrows=0).columns if c not in NGAFID_NON_SENSOR]
    base_spec = {"spec_id": "baseline_summary_stats",
                 "features": [{"transform": "summary_stats", "channels": sensors}]}
    table_base = build_training_table(csv, base_spec, max_flights=None, chunksize=chunksize)
    import tempfile, os
    m_base = fit_excluding_fold(table_base, test_fold,
                                os.path.join(tempfile.gettempdir(), "baseline_model.joblib"))

    honest_lift = m_best["holdout_roc_auc"] - m_base["holdout_roc_auc"]
    import math
    n_h = m_best.get("n_holdout", 0)
    se = math.sqrt(max(1e-9, m_best["holdout_roc_auc"] * (1 - m_best["holdout_roc_auc"]))
                   / max(1, n_h / 2))
    print("\n" + "=" * 60)
    print(f"  SELECT fold {select_fold}: spec chosen here (optimistic, biased)")
    print(f"  TEST   fold {test_fold}: never used to choose the spec (HONEST)")
    print(f"    baseline summary_stats test-AUC : {m_base['holdout_roc_auc']:.4f}  "
          f"({m_base['n_features']} feats)")
    print(f"    winning spec        test-AUC    : {m_best['holdout_roc_auc']:.4f}  "
          f"({m_best['n_features']} feats)")
    print(f"    HONEST LIFT (test fold)         : {honest_lift:+.4f}   "
          f"(test holdout {n_h} flights, AUC SE ~{se:.3f})")
    print("=" * 60)
    if honest_lift <= se:
        print(f"  -> The lift is within ~1 standard error (~{se:.3f}) of zero on a "
              f"{n_h}-flight test fold: no meaningful improvement over the baseline. ")
    else:
        print(f"  -> Lift exceeds ~1 SE (~{se:.3f}) on a {n_h}-flight test fold it was never "
              "tuned on. Confirm by rotating the select/test pair before quoting it.")

    print(f"\n[finalize 3/3] export fold {demo_fold} demo batch -> {demo_dir}")
    export_flights_to_dir(csv, demo_dir, max_flights=demo_max, only_fold=demo_fold,
                          chunksize=chunksize)
    print(f"\n[finalize] done. model={out_model} metrics={out_metrics} demo={demo_dir}")
    print(f"[finalize] set DISCOVERY_SPEC={Path('best_spec.json').resolve()} (or pass "
          f"--spec) so the agent's featurize step uses this featurizer at runtime.")


def _http_chat_fn_from_env():
    """OpenAI-compatible chat function from env vars (works for the RIT API and a
    local vLLM server alike -- both expose /v1/chat/completions). Set API_URL,
    API_MODEL_NAME, and (for the RIT API) RIT_API_KEY."""
    import os
    base = os.getenv("API_URL")
    model = os.getenv("API_MODEL_NAME")
    key = os.getenv("RIT_API_KEY", "not-needed")
    if not base or not model:
        raise SystemExit("--proposer llm needs API_URL and API_MODEL_NAME in the "
                         "environment (and RIT_API_KEY for the RIT API).")
    import requests

    def chat(system, user):
        r = requests.post(
            f"{base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "temperature": 0.4,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
            timeout=180)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    return chat


def _llm_proposer():
    """Real LLM proposer wired to the OpenAI-compatible endpoint in your env.
    The discovery loop calls this each iteration; the model proposes the next
    feature spec from the channels, the transform menu, the AUC history, and the
    live feature budget. (To reuse your agent_harness client instead of raw HTTP,
    pass your own chat_fn(system, user)->str to make_llm_proposer in a notebook.)"""
    return make_llm_proposer(_http_chat_fn_from_env(), max_retries=5)


if __name__ == "__main__":
    main()