# agentic-pdm

Predictive-maintenance pipeline for agent-sandbox (use case 1). Agents
**propose experiments** from a fixed catalog of models, augmentations,
schedules and losses, as JSON, never code. Deterministic **gates** decide
what runs and when to stop, and a deterministic **harness** owns everything
that has to stay honest: data loading, folds, normalization, the training
loop, the loss, and metrics.

Separate from `../agentic-ml-classification` on purpose: that package is the
stable tabular-classification pipeline and is not modified here.

First dataset: NGAFID-MC (`C28.csv`, `C37.csv`) from Yang, LaBella & Desell,
*Predictive Maintenance for General Aviation Using Convolutional
Transformers* (arXiv 2110.03757). Targets to beat (augmented Conv-MHSA,
mean of best per-fold validation metrics over 5 plane-disjoint folds):

| | ROC-AUC | PR-AUC | Accuracy |
|---|---|---|---|
| C28 | 0.826 | 0.802 | 0.744 |
| C37 | 0.775 | 0.711 | 0.723 |

## Contents

- `ingest.py` — streams any long-format multivariate time-series CSV into a
  tensor store (`X.f32` of shape N×C×T, last `max_len` steps, NaN-left-padded;
  `meta.csv`; `manifest.json`). Column roles are arguments; nothing is
  NGAFID-specific. Memory is bounded by one CSV chunk.
- `harness/` — per-fold min/max scaling fit on training data only, metrics,
  the plugin loader + contract checks, and the cross-validation trainer.
- `reference/small_cnn.py` — hand-written plugin: ~130K-parameter 1D CNN plus
  the paper's temporal cutout / cutmix / mixup.
- `baselines.py` — dummy prior, a length-only shortcut probe, and the old
  agentic_ml summary-statistics rollup into gradient boosting.
- `catalog/` — what agents may choose from, each with typed, bounded
  parameters: 5 architectures (`small_cnn`, `inception_time`, `tcn`,
  `resnet1d`, `conv_mhsa`), 10 augmentations (the paper's `cutout`, `cutmix`,
  `channel_mixup`, plus `jitter`, `scaling`, `magnitude_warp`, `time_warp`,
  `window_slice`, `channel_dropout`, `label_mixup`), 4 schedules and 2
  losses. `experiment.py` validates a proposal (all errors at once), fills in
  defaults, and estimates its cost from the model's counted FLOPs.
- `catalog/tools.py` — the planning agents' catalog tools (`list_catalog`,
  `describe_entry`, `validate_experiment`) as plain functions in
  `CATALOG_TOOLS`. This package runs no server and doesn't depend on `mcp`:
  the agentic-ml-facts MCP server (the `agentic-mcp` deployment) registers
  them. Advisory only; the validate gate re-checks every proposal.
- `pipeline.py` — the gates and training job for
  `sandbox/pipelines/pdm-experiment-loop.yaml` (see below).
- `sandbox/` — agent-sandbox specs: the pipeline and three agents (planner,
  analyst, reporter).

## The experiment loop

```
init ─▶ brief ─▶ plan (agent + catalog tools) ─▶ validate ─▶ train (job) ─▶ record
          ▲                                      │ invalid                    │ continue
          └──────────────────────────────────────┘      analyze (agent) ◀─────┘
                                                              │
          brief ◀─────────────────────────────────────────────┘
record / validate / brief: target_met | budget_exhausted | give_up ─▶ finalize ─▶ report (agent)
```

Gates sit only where a decision has consequences:
- **validate** accepts a proposal only if it fits the catalog and the
  remaining budget. After 3 invalid proposals in a row the run gives up.
- **record** logs the result, recalibrates cost estimates from the real
  training time, and decides whether to continue. Only a full
  cross-validation experiment can meet the target.

The analyst hands straight to the next brief, with no gate between them.
All run state lives in a run directory (`pdm-runs/<run_id>/`: contract,
leaderboard, per-experiment config and results, final report), written only
by gates and the job.

The seed task is a JSON run contract:

```json
{"dataset": "c28", "goal": "Beat the NGAFID-MC paper on C28",
 "target": {"metric": "roc_auc", "protocol": "last", "value": 0.83},
 "budget": {"max_experiments": 6, "max_minutes": 180},
 "reference": {"source": "Yang et al. 2021 Conv-MHSA (best-epoch)", "roc_auc": 0.826}}
```

To use it in agent-sandbox:
1. Import this package on the Packages page (repo `agentic-ai-project`,
   subdirectory `agentic-pdm`). Set `RUNNER_PIP_EXTRA_INDEX_URL` to the
   PyTorch CPU index on a GPU-less host.
2. Put the ingested store in the sandbox's datasets directory as `c28/`
   (`X.f32`, `meta.csv`, `manifest.json`, and optionally `baselines.json`).
3. Create the three agents from `sandbox/agents/*.yaml` and the pipeline
   from `sandbox/pipelines/pdm-experiment-loop.yaml`.

This needs agent-sandbox with job steps (branch `phase1-job-infra`).

The step ids `plan`, `train` and `analyze` are conventions the gates rely
on; keep them if you edit the pipeline.

## Plugin contract

A plugin is one `.py` file (see `harness/plugin.py` for the full spec):

```python
def build_model(n_channels, seq_len, n_classes, config) -> torch.nn.Module  # required; logits (B, K)
def augment(x, y, generator, config) -> (x, y)          # optional; training batches only
def configure_optimizer(model, config) -> Optimizer     # optional; default AdamW
def configure_scheduler(optimizer, config, total_steps) # optional; stepped every batch
```

Catalog experiments are plugins too (`catalog.build_experiment`), so a
hand-written or agent-written plugin can be compared against them under the
same harness. Plugins never see validation data, fold ids, or metrics; the
harness picks the loss (`TrainConfig.loss`, `label_smoothing`,
`focal_gamma`) and can train seed ensembles (`TrainConfig.ensemble`).

## Reported numbers

Every run writes both, per fold and averaged:

- `last` — metrics after the final epoch. No validation peeking.
- `best` — the best-ROC epoch on the validation fold. Matches the paper's
  protocol, and is optimistic for the same reason.

## Usage

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e '.[dev]'

RAW=../agentic-ml-classification/datasets/raw
.venv/bin/agentic-pdm ingest --csv $RAW/C28.csv --out data/c28 --id-column id \
    --label-column before_after --group-column plane_id --fold-column split --exclude date_diff \
    --positive-label 0
.venv/bin/agentic-pdm baseline --store data/c28 --out runs/c28_baselines.json
.venv/bin/agentic-pdm train --store data/c28 --out runs/c28_ref --folds 0 --epochs 20 --threads 3
```

`date_diff` must be excluded: its sign is the label. `split` is the paper's
own fold assignment and is used as the fold column, not as a feature.
`before_after = 0` is **pre**-maintenance (the paper's positive class,
checked against `date_diff`), hence `--positive-label 0`.

Cost estimates assume ~175 effective GFLOP/s, measured on 3 threads of a
Ryzen 9 7950X. Set `PDM_EFFECTIVE_GFLOPS` for other hosts. The experiment
loop also self-calibrates from its own finished experiments.

## Target deployment

The RIT server has 4 vCPUs (EPYC Zen 2), about 12 GiB of free RAM, and no GPU.
Runs there should cap training at 3 threads and keep one training job at a time.
