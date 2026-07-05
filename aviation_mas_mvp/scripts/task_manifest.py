"""
task_manifest.py
================
Phase-1: the declarative task description. Everything the aviation path used to
hardcode -- target, group column, metric, split, feature-spec path, model,
artifacts, serving knobs -- becomes a validated manifest ("what, not how").

The manifest is JSON on disk (so the Phase-5 profiling agent can draft one) and a
typed dataclass in code. `validate_manifest` returns a list of human-readable
errors (empty = valid), matching the rest of the suite; `validate_against_columns`
adds the data-aware checks (target exists, group exists, target-vs-group leakage).

Binding the feature-spec path here is deliberate: it's the exact coupling whose
absence let featurize (277) drift from a spec-trained model (412).
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

TASK_TYPES = {"binary_classification", "multiclass_classification", "regression"}
METRICS_BY_TASK = {
    "binary_classification": {"roc_auc", "pr_auc", "accuracy", "f1"},
    "multiclass_classification": {"macro_f1", "weighted_f1", "accuracy"},
    "regression": {"rmse", "mae", "r2"},
}
SPLIT_TYPES = {"group_kfold", "stratified_kfold", "time_series_split"}
ADAPTERS = {"aviation_flight_dir", "tabular_csv"}     # grows with the Phase-2 registry


@dataclass
class TaskManifest:
    name: str
    # --- task ---
    task_type: str
    target: str
    metric: str
    # --- data / adapter ---
    adapter: str
    data_path: str
    flight_dir: str | None = None
    metadata_path: str | None = None
    file_column: str | None = None
    group_column: str | None = None
    time_column: str | None = None
    folds: list | None = None
    positive_label: object = None
    # --- split ---
    split_type: str = "group_kfold"
    # --- features / model / artifacts ---
    feature_spec_path: str | None = None
    model_type: str = "HistGradientBoostingClassifier"
    model_path: str | None = None
    artifact_root: str | None = None
    # --- serving ---
    top_k: int = 10
    thresholds: dict = field(default_factory=lambda: {"high": 0.66, "med": 0.33})

    @classmethod
    def from_dict(cls, d: dict) -> "TaskManifest":
        known = {f.name for f in fields(cls)}
        m = cls(**{k: v for k, v in d.items() if k in known})
        object.__setattr__(m, "_unknown", sorted(set(d) - known))
        return m

    def to_dict(self) -> dict:
        return asdict(self)


def load_manifest(path: str | Path) -> TaskManifest:
    return TaskManifest.from_dict(json.loads(Path(path).read_text()))


def save_manifest(m: TaskManifest, path: str | Path) -> None:
    Path(path).write_text(json.dumps(m.to_dict(), indent=2))


def validate_manifest(m: TaskManifest, require_paths: bool = False) -> list[str]:
    """Structural + cross-field validation. Empty list = valid.
    Data-aware checks (does the target column exist?) are in
    validate_against_columns, which needs the dataset's column list."""
    e: list[str] = []

    if not m.name:
        e.append("name is empty")
    if m.task_type not in TASK_TYPES:
        e.append(f"unknown task_type {m.task_type!r}; valid: {sorted(TASK_TYPES)}")
    if not m.target:
        e.append("target is empty")

    # metric must fit the task type (rejects e.g. roc_auc on multiclass)
    valid_metrics = METRICS_BY_TASK.get(m.task_type, set())
    if m.task_type in METRICS_BY_TASK and m.metric not in valid_metrics:
        e.append(f"metric {m.metric!r} invalid for {m.task_type}; valid: {sorted(valid_metrics)}")

    if m.split_type not in SPLIT_TYPES:
        e.append(f"unknown split_type {m.split_type!r}; valid: {sorted(SPLIT_TYPES)}")
    if m.split_type == "group_kfold" and not m.group_column:
        e.append("split_type 'group_kfold' requires group_column")
    if m.split_type == "time_series_split" and not m.time_column:
        e.append("split_type 'time_series_split' requires time_column")

    # target-vs-group leakage sanity (the group key must not be the label)
    if m.target and m.group_column and m.target == m.group_column:
        e.append(f"target and group_column are the same column ({m.target!r}) -- leakage")

    if m.adapter not in ADAPTERS:
        e.append(f"unknown adapter {m.adapter!r}; valid: {sorted(ADAPTERS)}")
    if not m.data_path:
        e.append("data_path is empty")
    if not m.model_path:
        e.append("model_path is empty (needed to classify/serve)")

    if require_paths:
        for label, p in (("data_path", m.data_path), ("metadata_path", m.metadata_path),
                         ("feature_spec_path", m.feature_spec_path), ("model_path", m.model_path)):
            if p and not Path(p).exists():
                e.append(f"{label} does not exist on disk: {p}")

    for k in getattr(m, "_unknown", []):
        e.append(f"unknown manifest field {k!r} (typo?)")
    return e


def validate_against_columns(m: TaskManifest, columns) -> list[str]:
    """Data-aware checks once the dataset's columns are known (Phase-2 adapter
    profiles them). Catches nonexistent target and missing group column."""
    cols = set(columns)
    e = []
    if m.target and m.target not in cols:
        e.append(f"target {m.target!r} not found in dataset columns")
    if m.group_column and m.group_column not in cols:
        e.append(f"group_column {m.group_column!r} not found in dataset columns")
    if m.time_column and m.time_column not in cols:
        e.append(f"time_column {m.time_column!r} not found in dataset columns")
    return e
