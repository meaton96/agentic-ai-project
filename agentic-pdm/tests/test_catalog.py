import numpy as np
import pytest
import torch

from agentic_pdm.catalog import DatasetFacts, build_experiment, catalog_summary, describe_entry, validate_experiment
from agentic_pdm.catalog.architectures import ARCHITECTURES, TABULAR_ARCHITECTURES
from agentic_pdm.catalog.augmentations import AUGMENTATIONS
from agentic_pdm.catalog.experiment import COST_FLOOR_FLOPS_PER_INPUT_VALUE, NEEDS_UNPOOLED_INPUT
from agentic_pdm.catalog.params import validate_params
from agentic_pdm.catalog.training import SCHEDULES
from agentic_pdm.harness.plugin import check_augment_output
from agentic_pdm.harness.trainer import run_cv
from agentic_pdm.ingest import IngestConfig, ingest_long_csv
from agentic_pdm.store import load_store

FACTS = DatasetFacts(n_sequences=200, n_channels=3, seq_len=512, n_classes=2, folds=[0, 1, 2, 3])


def exp(**fields):
    return {"name": "e1", "architecture": {"id": "small_cnn"}, "input": {"max_len": 512, "pool": 4}, **fields}


@pytest.mark.parametrize("arch_id", sorted(set(ARCHITECTURES) - TABULAR_ARCHITECTURES))
def test_every_architecture_builds_with_defaults_and_maps_to_logits(arch_id):
    pool = 1 if arch_id in NEEDS_UNPOOLED_INPUT else 4
    report = validate_experiment(exp(architecture={"id": arch_id}, input={"max_len": 512, "pool": pool}), FACTS)
    assert report["valid"], report["errors"]
    assert report["estimate"]["n_params"] > 0 and report["estimate"]["forward_mflops_per_sequence"] > 0
    model = ARCHITECTURES[arch_id].builder(3, 128, 2, report["normalized"]["architecture"]["params"]).eval()
    assert model(torch.randn(4, 3, 128)).shape == (4, 2)


def test_last_record_mlp_reads_only_the_final_timestep_and_needs_pool_1():
    model = ARCHITECTURES["last_record_mlp"].builder(3, 64, 2, {"width": 16, "depth": 1, "dropout": 0.0}).eval()
    x = torch.rand(4, 3, 64)
    changed = x.clone()
    changed[:, :, :-1] = 0.5
    torch.testing.assert_close(model(x), model(changed))
    rejected = validate_experiment(exp(architecture={"id": "last_record_mlp"}), FACTS)
    assert not rejected["valid"] and "input.pool must be 1" in rejected["errors"][0]


def test_window_stats_ignore_padding():
    model = ARCHITECTURES["window_stats_mlp"].builder(2, 8, 2, {"width": 16, "depth": 1, "dropout": 0.0})
    captured = {}
    model.net.register_forward_pre_hook(lambda _, args: captured.update(features=args[0]))
    x = torch.zeros(1, 2, 8)  # 5 padded steps, then 3 observed ones
    x[0, 0, 5:] = torch.tensor([0.2, 0.4, 0.6])
    x[0, 1, 5:] = torch.tensor([0.9, 0.9, 0.9])
    model.eval()(x)
    # Per channel: last, mean, std, min, max, slope; then the observed fraction.
    last, mean, std, low, high, slope, fraction = (captured["features"][0, :2], captured["features"][0, 2:4],
                                                   captured["features"][0, 4:6], captured["features"][0, 6:8],
                                                   captured["features"][0, 8:10], captured["features"][0, 10:12],
                                                   captured["features"][0, 12])
    torch.testing.assert_close(last, torch.tensor([0.6, 0.9]))
    torch.testing.assert_close(mean, torch.tensor([0.4, 0.9]), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(low, torch.tensor([0.2, 0.9]))
    torch.testing.assert_close(high, torch.tensor([0.6, 0.9]))
    assert std[1] < 1e-3 and std[0] > 0.1  # padding zeros don't inflate the spread
    assert slope[0] > 0 and abs(slope[1]) < 1e-5
    assert fraction.item() == pytest.approx(3 / 8)


def test_window_stats_survive_an_all_zero_sequence():
    model = ARCHITECTURES["window_stats_mlp"].builder(2, 8, 2, {"width": 16, "depth": 1, "dropout": 0.0}).eval()
    assert torch.isfinite(model(torch.zeros(2, 2, 8))).all()


@pytest.mark.parametrize("arch_id", ["last_record_mlp", "window_stats_mlp"])
def test_mlps_train_with_a_final_batch_of_one(arch_id):
    model = ARCHITECTURES[arch_id].builder(3, 64, 2, {"width": 16, "depth": 2, "dropout": 0.1}).train()
    assert model(torch.rand(1, 3, 64)).shape == (1, 2)


@pytest.mark.parametrize("arch_id", ["last_record_mlp", "window_stats_mlp"])
def test_warns_when_an_augmentation_can_change_the_final_record(arch_id):
    config = exp(architecture={"id": arch_id}, input={"max_len": 512, "pool": 1}, augmentations=[{"id": "cutout"}])
    assert any("final timestep" in w for w in validate_experiment(config, FACTS)["warnings"])
    clean = exp(architecture={"id": arch_id}, input={"max_len": 512, "pool": 1}, augmentations=[{"id": "jitter"}])
    assert not any("final timestep" in w for w in validate_experiment(clean, FACTS)["warnings"])


def test_cost_floor_applies_only_to_the_summary_models():
    assert set(COST_FLOOR_FLOPS_PER_INPUT_VALUE) == {"last_record_mlp", "window_stats_mlp"}
    floor = COST_FLOOR_FLOPS_PER_INPUT_VALUE["last_record_mlp"]
    estimate = validate_experiment(exp(architecture={"id": "last_record_mlp"}, input={"max_len": 512, "pool": 1}),
                                   FACTS)["estimate"]
    assert estimate["forward_mflops_per_sequence"] == pytest.approx(floor * 3 * 512 / 1e6, rel=0.01)


def test_original_architectures_have_no_floor_or_pool_rule():
    original = {"small_cnn", "inception_time", "tcn", "resnet1d", "conv_mhsa"}
    assert not original & set(COST_FLOOR_FLOPS_PER_INPUT_VALUE) and not original & NEEDS_UNPOOLED_INPUT
    # Recorded from version 0.1.3: the earlier runs' budget checks relied on these estimates.
    ngafid = DatasetFacts(n_sequences=5089, n_channels=23, seq_len=4096, n_classes=2, folds=[0, 1, 2, 3, 4])
    recorded = {"small_cnn": 4.232564066986666, "tcn": 9.275274910719997, "resnet1d": 9.385811790506665}
    for arch_id, minutes in recorded.items():
        config = {"name": "x", "architecture": {"id": arch_id}, "input": {"max_len": 4096, "pool": 4}}
        assert validate_experiment(config, ngafid)["estimate"]["uncalibrated_minutes"] == pytest.approx(minutes)


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


def gbm(params=None, **fields):
    return exp(architecture={"id": "gbm", "params": params or {}}, input={"max_len": 512, "pool": 1}, **fields)


def test_gbm_defaults_are_the_library_defaults_and_its_stages_are_its_epochs():
    report = validate_experiment(gbm(), FACTS)
    assert report["valid"], report["errors"]
    params = report["normalized"]["architecture"]["params"]
    assert (params["max_iter"], params["learning_rate"], params["max_leaf_nodes"], params["min_samples_leaf"],
            params["l2_regularization"]) == (100, 0.1, 31, 20, 0.0)
    assert report["normalized"]["training"]["epochs"] == 100 and report["warnings"] == []


def test_gbm_rejects_what_it_cannot_use_all_at_once():
    report = validate_experiment(exp(architecture={"id": "gbm"}, augmentations=[{"id": "jitter"}],
                                     input={"max_len": 512, "pool": 4}, training={"ensemble": 2}), FACTS)
    assert not report["valid"]
    joined = " ".join(report["errors"])
    assert "no augmentations" in joined and "input.pool must be 1" in joined and "ensemble must be 1" in joined


def test_gbm_warns_about_settings_it_ignores():
    report = validate_experiment(gbm({"max_iter": 50}, training={"epochs": 20}, schedule={"id": "cosine"}), FACTS)
    assert report["valid"], report["errors"]
    assert report["normalized"]["training"]["epochs"] == 50
    joined = " ".join(report["warnings"])
    assert "training.epochs" in joined and "schedule" in joined


def test_gbm_normalized_config_validates_again_unchanged():
    first = validate_experiment(gbm({"max_iter": 50}), FACTS)["normalized"]
    second = validate_experiment(first, FACTS)
    assert second["valid"], second["errors"]
    assert second["normalized"] == first


def test_gbm_cost_needs_no_model_and_uses_only_its_own_calibration():
    plain = validate_experiment(gbm(), FACTS)["estimate"]
    assert plain["n_params"] == 0 and plain["total_minutes"] > 0
    assert validate_experiment(gbm(), FACTS, calibration={"default": 9.0})["estimate"]["calibration"] == 1.0
    scaled = validate_experiment(gbm(), FACTS, calibration={"default": 9.0, "gbm": 2.0})["estimate"]
    assert scaled["total_minutes"] == pytest.approx(2 * plain["total_minutes"])
    fewer = validate_experiment(gbm({"features": "last_record"}), FACTS)["estimate"]
    assert fewer["total_minutes"] < plain["total_minutes"]


def test_build_experiment_refuses_a_tabular_model():
    with pytest.raises(ValueError, match="run_tabular_cv"):
        build_experiment(validate_experiment(gbm(), FACTS)["normalized"])
