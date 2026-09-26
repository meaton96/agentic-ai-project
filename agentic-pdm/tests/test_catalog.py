import numpy as np
import pytest
import torch

from agentic_pdm.catalog import DatasetFacts, build_experiment, catalog_summary, describe_entry, validate_experiment
from agentic_pdm.catalog.architectures import ARCHITECTURES
from agentic_pdm.catalog.augmentations import AUGMENTATIONS
from agentic_pdm.catalog.params import validate_params
from agentic_pdm.catalog.training import SCHEDULES
from agentic_pdm.harness.plugin import check_augment_output
from agentic_pdm.harness.trainer import run_cv
from agentic_pdm.ingest import IngestConfig, ingest_long_csv
from agentic_pdm.store import load_store

FACTS = DatasetFacts(n_sequences=200, n_channels=3, seq_len=512, n_classes=2, folds=[0, 1, 2, 3])


def exp(**fields):
    return {"name": "e1", "architecture": {"id": "small_cnn"}, "input": {"max_len": 512, "pool": 4}, **fields}


@pytest.mark.parametrize("arch_id", sorted(ARCHITECTURES))
def test_every_architecture_builds_with_defaults_and_maps_to_logits(arch_id):
    report = validate_experiment(exp(architecture={"id": arch_id}), FACTS)
    assert report["valid"], report["errors"]
    assert report["estimate"]["n_params"] > 0 and report["estimate"]["forward_mflops_per_sequence"] > 0
    model = ARCHITECTURES[arch_id].builder(3, 128, 2, report["normalized"]["architecture"]["params"]).eval()
    assert model(torch.randn(4, 3, 128)).shape == (4, 2)


@pytest.mark.parametrize("aug_id", sorted(AUGMENTATIONS))
def test_every_augmentation_keeps_the_contract_and_is_seeded(aug_id):
    params, errors = validate_params(AUGMENTATIONS[aug_id], {"p": 1.0}, aug_id)
    assert not errors
    fn = AUGMENTATIONS[aug_id].builder(params)
    x = torch.rand(8, 3, 64)
    y = torch.eye(2)[torch.arange(8) % 2]
    a = fn(x.clone(), y.clone(), torch.Generator().manual_seed(1))
    b = fn(x.clone(), y.clone(), torch.Generator().manual_seed(1))
    check_augment_output(x, y, *a)
    torch.testing.assert_close(a[0], b[0])
    assert not torch.equal(a[0], x)
    if aug_id != "label_mixup":
        torch.testing.assert_close(a[1], y)


@pytest.mark.parametrize("schedule_id", sorted(SCHEDULES))
def test_schedules_produce_positive_learning_rates(schedule_id):
    params, _ = validate_params(SCHEDULES[schedule_id], {}, schedule_id)
    configure_optimizer, configure_scheduler = SCHEDULES[schedule_id].builder(params)
    model = torch.nn.Linear(2, 2)
    opt = configure_optimizer(model, {})
    sched = configure_scheduler(opt, {}, 100)
    lrs = []
    for _ in range(100):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert all(lr > 0 for lr in lrs) and max(lrs) <= params["lr"] * 1.0001


def test_validation_reports_every_error_at_once():
    report = validate_experiment({
        "name": "bad name!",
        "architecture": {"id": "small_cnn", "params": {"width": 999, "typo": 1}},
        "augmentations": [{"id": "cutout"}, {"id": "cutout"}, {"id": "nope"}],
        "input": {"max_len": 5000, "pool": 3},
        "training": {"epochs": 0},
        "folds": [9],
        "extra": True,
    }, FACTS)
    text = "\n".join(report["errors"])
    assert not report["valid"] and report["normalized"] is None
    for fragment in ("name is required", "width must be <= 64", "unknown parameter(s) ['typo']", "appear once",
                     "'nope' is not an augmentation", "pool must be one of", "epochs must be >= 1",
                     "folds [9] don't exist", "unexpected top-level key(s) ['extra']"):
        assert fragment in text, fragment


def test_cross_parameter_rules_and_length_checks():
    mhsa = validate_experiment(exp(architecture={"id": "conv_mhsa", "params": {"d_model": 60, "heads": 8}}), FACTS)
    assert "divisible by heads" in mhsa["errors"][0]
    short = validate_experiment(exp(input={"max_len": 64, "pool": 4}), FACTS)
    assert "too short" in short["errors"][0]
    indivisible = validate_experiment(exp(input={"max_len": 500, "pool": 8}), FACTS)
    assert "divisible by input.pool" in indivisible["errors"][0]


def test_defaults_and_warnings():
    report = validate_experiment(exp(architecture={"id": "conv_mhsa"}, schedule={"id": "constant"}, folds=[0]), FACTS)
    n = report["normalized"]
    assert n["schedule"]["id"] == "constant" and n["loss"]["id"] == "cross_entropy" and n["folds"] == [0]
    warnings = " ".join(report["warnings"])
    assert "overfit" in warnings and "constant learning rate" in warnings and "screening" in warnings


def test_cost_scales_with_folds_epochs_ensemble_and_calibration():
    ngafid = DatasetFacts(n_sequences=5089, n_channels=23, seq_len=4096, n_classes=2, folds=[0, 1, 2, 3])
    full = {"max_len": 4096, "pool": 4}
    base = validate_experiment(exp(input=full, folds=[0, 1]), ngafid)["estimate"]["total_minutes"]
    more = validate_experiment(exp(input=full, training={"epochs": 40, "ensemble": 2}), ngafid)["estimate"]
    assert more["total_minutes"] == pytest.approx(base * 2 * 2 * 2, rel=0.01)  # 2x folds, epochs, ensemble
    slow = validate_experiment(exp(input=full, folds=[0, 1]), ngafid, calibration=3.0)["estimate"]["total_minutes"]
    assert slow == pytest.approx(base * 3, rel=0.01)
    # A per-architecture map applies the matching factor, else "default".
    by_arch = validate_experiment(exp(input=full, folds=[0, 1]), ngafid,
                                  calibration={"default": 5.0, "small_cnn": 2.0})["estimate"]
    assert by_arch["total_minutes"] == pytest.approx(base * 2, rel=0.01) and by_arch["calibration"] == 2.0
    other = validate_experiment(exp(input=full, folds=[0, 1], architecture={"id": "tcn"}), ngafid,
                                calibration={"default": 5.0, "small_cnn": 2.0})["estimate"]
    assert other["calibration"] == 5.0


def test_catalog_listing_and_description():
    kinds = {e["kind"] for e in catalog_summary()}
    assert kinds == {"architecture", "augmentation", "schedule", "loss"}
    assert describe_entry("training")["params"]["ensemble"]["max"] == 3
    with pytest.raises(ValueError, match="no catalog entry"):
        describe_entry("nope")


def test_a_catalog_experiment_trains_through_the_harness(long_csv, tmp_path):
    ingest_long_csv(long_csv, tmp_path / "store", IngestConfig(
        id_column="seq", label_column="target", group_column="unit", fold_column="fold",
        exclude_columns=["leak"], max_len=64))
    store = load_store(tmp_path / "store")
    facts = DatasetFacts.from_store(store)
    report = validate_experiment({
        "name": "tiny", "architecture": {"id": "small_cnn", "params": {"width": 8, "depth": 3}},
        "augmentations": [{"id": "jitter"}, {"id": "label_mixup"}],
        "schedule": {"id": "one_cycle", "params": {"lr": 0.01}},
        "loss": {"id": "focal", "params": {"gamma": 1.0, "label_smoothing": 0.05}},
        "input": {"max_len": 64, "pool": 1}, "training": {"epochs": 6, "batch_size": 16, "ensemble": 2},
        "folds": [0, 1],
    }, facts)
    assert report["valid"], report["errors"]
    plugin, cfg, folds = build_experiment(report["normalized"], threads=1)
    assert (cfg.loss, cfg.ensemble, cfg.max_len, folds) == ("focal", 2, 64, [0, 1])
    results = run_cv(store, plugin, cfg, tmp_path / "run", folds=folds, log=lambda m: None)
    assert results["folds"][0]["ensemble"] == 2 and len(results["folds"][0]["history"]) == 6
    assert np.isfinite(results["summary"]["last"]["log_loss"])
    assert results["summary"]["best"]["roc_auc"] > 0.7


def test_an_empty_gflops_setting_means_the_default(monkeypatch):
    monkeypatch.setenv("PDM_EFFECTIVE_GFLOPS", "")
    report = validate_experiment(exp(), FACTS)
    assert report["valid"], report["errors"]
    assert report["estimate"]["effective_gflops"] == 175.0
