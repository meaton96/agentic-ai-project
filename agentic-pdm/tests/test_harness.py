import json

import numpy as np
import pytest
import torch

from agentic_pdm.baselines import run_baselines, summary_features
from agentic_pdm.harness.metrics import classification_metrics
from agentic_pdm.harness.normalize import ChannelMinMax
from agentic_pdm.harness.plugin import check_augment_output, load_plugin
from agentic_pdm.harness.trainer import TrainConfig, run_cv
from agentic_pdm.ingest import IngestConfig, ingest_long_csv
from agentic_pdm.reference import SMALL_CNN_PATH
from agentic_pdm.store import load_store
from conftest import make_long_frame


@pytest.fixture
def store(long_csv, tmp_path):
    ingest_long_csv(long_csv, tmp_path / "store", IngestConfig(
        id_column="seq", label_column="target", group_column="unit",
        fold_column="fold", exclude_columns=["leak"], max_len=64))
    return load_store(tmp_path / "store")


def test_minmax_fit_ignores_data_it_was_not_given():
    train = np.array([[[0.0, 10.0, np.nan]]], dtype=np.float32)
    scaler = ChannelMinMax.fit(train)
    val = np.array([[[1000.0, 5.0, np.nan]]], dtype=np.float32)
    out = scaler.transform(val)
    assert scaler.minimum[0] == 0.0 and scaler.maximum[0] == 10.0
    np.testing.assert_allclose(out, [[[100.0, 0.5, 0.0]]])


def test_metrics_binary_and_multiclass():
    m = classification_metrics(np.array([0, 0, 1, 1]), np.array([[.9, .1], [.6, .4], [.3, .7], [.2, .8]]))
    assert m["roc_auc"] == 1.0 and m["accuracy"] == 1.0
    proba = np.eye(3)[[0, 1, 2, 0]] * 0.8 + 0.2 / 3
    m3 = classification_metrics(np.array([0, 1, 2, 0]), proba)
    assert m3["roc_auc"] == pytest.approx(1.0)


def test_reference_augment_is_seeded_and_keeps_contract():
    plugin = load_plugin(SMALL_CNN_PATH)
    x = torch.randn(16, 5, 128)
    y = torch.nn.functional.one_hot(torch.arange(16) % 2, 2).float()
    cfg = {"aug_p": 1.0}
    a = plugin.augment(x.clone(), y.clone(), torch.Generator().manual_seed(3), cfg)
    b = plugin.augment(x.clone(), y.clone(), torch.Generator().manual_seed(3), cfg)
    torch.testing.assert_close(a[0], b[0])
    check_augment_output(x, y, *a)
    assert not torch.equal(a[0], x)
    torch.testing.assert_close(a[1], y)  # paper augmentations never touch labels


def test_reference_augment_can_be_disabled():
    plugin = load_plugin(SMALL_CNN_PATH)
    x, y = torch.randn(4, 3, 32), torch.eye(2)[[0, 1, 0, 1]]
    out = plugin.augment(x.clone(), y.clone(), torch.Generator().manual_seed(0), {"augmentations": []})
    torch.testing.assert_close(out[0], x)


def test_check_augment_output_rejects_contract_violations():
    x, y = torch.zeros(2, 3, 8), torch.eye(2)
    with pytest.raises(ValueError, match="shapes"):
        check_augment_output(x, y, x[:, :, :4], y)
    with pytest.raises(ValueError, match="sum to 1"):
        check_augment_output(x, y, x, y * 2)
    with pytest.raises(ValueError, match="non-finite"):
        check_augment_output(x, y, x + float("nan"), y)


def test_load_plugin_requires_build_model(tmp_path):
    bad = tmp_path / "bad_plugin.py"
    bad.write_text("def augment(x, y, g, c):\n    return x, y\n")
    with pytest.raises(TypeError, match="build_model"):
        load_plugin(bad)


def test_run_cv_learns_synthetic_signal_and_writes_results(store, tmp_path):
    cfg = TrainConfig(epochs=8, batch_size=8, pool=2, seed=0, threads=1,
                      plugin_config={"width": 8, "lr": 3e-3})
    results = run_cv(store, load_plugin(SMALL_CNN_PATH), cfg, tmp_path / "run", folds=[0, 1], log=lambda m: None)

    on_disk = json.loads((tmp_path / "run" / "results.json").read_text())
    assert on_disk["status"] == "completed"
    assert set(on_disk["folds"]) == {"0", "1"}
    fold0 = results["folds"][0]
    assert len(fold0["history"]) == 8
    assert fold0["n_train"] + fold0["n_val"] == 40
    assert results["summary"]["best"]["roc_auc"] >= results["summary"]["last"]["roc_auc"] - 1e-9
    assert results["summary"]["best"]["roc_auc"] > 0.8  # channel a carries a strong label signal


def test_run_cv_is_reproducible(store, tmp_path):
    cfg = TrainConfig(epochs=2, batch_size=8, pool=2, seed=1, threads=1, plugin_config={"width": 8})
    plugin = load_plugin(SMALL_CNN_PATH)
    a = run_cv(store, plugin, cfg, tmp_path / "a", folds=[2], log=lambda m: None)
    b = run_cv(store, plugin, cfg, tmp_path / "b", folds=[2], log=lambda m: None)
    assert a["folds"][2]["last"]["log_loss"] == b["folds"][2]["last"]["log_loss"]


def test_run_cv_rejects_unknown_fold(store, tmp_path):
    with pytest.raises(ValueError, match="unknown folds"):
        run_cv(store, load_plugin(SMALL_CNN_PATH), TrainConfig(epochs=1), tmp_path / "r", folds=[9])


def test_summary_features_match_naive_computation(store):
    feats = summary_features(store.X).reshape(len(store.meta), 3, 12)
    x = np.asarray(store.X[5, 0], dtype=np.float64)
    obs = x[~np.isnan(x)]
    assert feats[5, 0, 0] == pytest.approx(obs.mean(), rel=1e-5)
    assert feats[5, 0, 4] == pytest.approx(obs.max() - obs.min(), rel=1e-5)
    assert feats[5, 0, 11] == pytest.approx(obs[-1], rel=1e-5)
    t = np.flatnonzero(~np.isnan(x))
    assert feats[5, 0, 8] == pytest.approx(np.polyfit(t, obs, 1)[0], rel=1e-4, abs=1e-6)


def test_baselines_run_and_summary_beats_dummy(tmp_path):
    # Gradient boosting's default min_samples_leaf=20 needs more rows than
    # the 40-sequence fixture leaves per training fold.
    make_long_frame(n_seq=200).to_csv(tmp_path / "big.csv", index=False)
    ingest_long_csv(tmp_path / "big.csv", tmp_path / "big", IngestConfig(
        id_column="seq", label_column="target", group_column="unit",
        fold_column="fold", exclude_columns=["leak"], max_len=64))
    results = run_baselines(load_store(tmp_path / "big"), log=lambda m: None)
    assert set(results) == {"dummy", "length_only", "summary_stats"}
    assert results["summary_stats"]["mean"]["roc_auc"] > 0.8
