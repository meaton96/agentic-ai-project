# agentic-pdm

Predictive-maintenance pipeline for agent-sandbox (use case 1). Agents will
write **model and augmentation plugins**; a deterministic **harness** owns
everything that has to stay honest — data loading, folds, normalization,
the training loop, and metrics.

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

## Status: Phase 0 (harness + reference, no agents)

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

## Plugin contract

A plugin is one `.py` file (see `harness/plugin.py` for the full spec):

```python
def build_model(n_channels, seq_len, n_classes, config) -> torch.nn.Module  # required; logits (B, K)
def augment(x, y, generator, config) -> (x, y)          # optional; training batches only
def configure_optimizer(model, config) -> Optimizer     # optional; default AdamW
```

Plugins never see validation data, fold ids, or metrics.

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
    --label-column before_after --group-column plane_id --fold-column split --exclude date_diff
.venv/bin/agentic-pdm baseline --store data/c28 --out runs/c28_baselines.json
.venv/bin/agentic-pdm train --store data/c28 --out runs/c28_ref --folds 0 --epochs 20 --threads 3
```

`date_diff` must be excluded: its sign is the label. `split` is the paper's
own fold assignment and is used as the fold column, not as a feature.

## Target deployment

The RIT server has 4 vCPUs (EPYC Zen 2), about 12 GiB of free RAM, and no GPU.
Runs there should cap training at 3 threads and keep one training job at a time.
