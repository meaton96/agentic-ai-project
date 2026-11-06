import json

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from agentic_pdm.baselines import summary_features
from agentic_pdm.catalog import DatasetFacts, validate_experiment
from agentic_pdm.harness.metrics import classification_metrics
from agentic_pdm.harness.tabular import run_tabular_cv, tabular_features
from agentic_pdm.ingest import IngestConfig, ingest_long_csv
from agentic_pdm.store import load_store


@pytest.fixture
def store(long_csv, tmp_path):
    ingest_long_csv(long_csv, tmp_path / "store", IngestConfig(
        id_column="seq", label_column="target", group_column="unit",
        fold_column="fold", exclude_columns=["leak"], max_len=64))
    return load_store(tmp_path / "store")


def normalized(store, features, max_iter=20):
    report = validate_experiment({
        "name": "g", "architecture": {"id": "gbm", "params": {"features": features, "max_iter": max_iter}},
        "input": {"max_len": 64, "pool": 1}}, DatasetFacts.from_store(store))
    assert report["valid"], report["errors"]
    return report["normalized"]


def test_feature_sets(store):
    last = tabular_features(store.X, "last_record")
    stats = tabular_features(store.X, "summary_stats")
    np.testing.assert_array_equal(last, store.X[:, :, -1])
    np.testing.assert_array_equal(stats, summary_features(store.X))
    np.testing.assert_array_equal(tabular_features(store.X, "both"), np.hstack([last, stats]))
    with pytest.raises(ValueError):
        tabular_features(store.X, "nope")


@pytest.mark.parametrize("features", ["last_record", "summary_stats", "both"])
def test_last_stage_equals_a_plain_fit_on_the_same_features_and_fold(store, tmp_path, features):
    results = run_tabular_cv(store, normalized(store, features), tmp_path / "run", log=lambda m: None)
    F = tabular_features(store.X, features)
    y, fold_of = store.meta["y"].to_numpy(), store.meta["fold"].to_numpy()
    for fold in store.folds:
        train, val = fold_of != fold, fold_of == fold
        plain = HistGradientBoostingClassifier(max_iter=20, random_state=0, early_stopping=False).fit(F[train], y[train])
        want = classification_metrics(y[val], plain.predict_proba(F[val]))
        got = results["folds"][fold]["last"]
        for k in want:
            assert got[k] == pytest.approx(want[k])


def test_results_have_run_cvs_shape_and_predictions_match(store, tmp_path):
    results = run_tabular_cv(store, normalized(store, "last_record", max_iter=10), tmp_path / "run", log=lambda m: None)
    disk = json.loads((tmp_path / "run" / "results.json").read_text())
    assert disk["status"] == results["status"] == "completed"
    assert disk["summary"]["folds"] == store.folds and set(disk["summary"]["last"]) >= {"roc_auc", "pr_auc"}
    fold = str(store.folds[0])
    assert [h["epoch"] for h in disk["folds"][fold]["history"]] == list(range(1, 11))
    assert all(h["train_seconds"] > 0 and h["train_loss"] > 0 for h in disk["folds"][fold]["history"])

    d = np.load(tmp_path / "run" / "preds.npz")
    assert d["p"].shape == (10, len(store.meta)) and not np.isnan(d["p"]).any()
    for f in store.folds:
        val = d["fold"] == f
        for e, want in enumerate(disk["folds"][str(f)]["history"]):
            got = classification_metrics(d["y"][val], np.column_stack([1 - d["p"][e, val], d["p"][e, val]]))
            assert got["roc_auc"] == pytest.approx(want["roc_auc"]) and got["pr_auc"] == pytest.approx(want["pr_auc"])


def test_a_screen_leaves_other_folds_without_predictions(store, tmp_path):
    n = normalized(store, "last_record", max_iter=10)
    n["folds"] = store.folds[:1]
    run_tabular_cv(store, n, tmp_path / "run", log=lambda m: None)
    d = np.load(tmp_path / "run" / "preds.npz")
    assert not np.isnan(d["p"][:, d["fold"] == store.folds[0]]).any()
    assert np.isnan(d["p"][:, d["fold"] != store.folds[0]]).all()
