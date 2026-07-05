"""
Phase-1 manifest validation tests. Pure Python -- runs anywhere, no data/model.
Covers the valid aviation manifest plus each rejection the Phase-5 planner needs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import copy
import pytest
from manifest.task_manifest import (TaskManifest, load_manifest, validate_manifest,
                           validate_against_columns)

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "manifests"


@pytest.fixture
def aviation():
    return load_manifest(MANIFEST_DIR / "aviation_c28.json")


def test_aviation_manifest_valid(aviation):
    assert validate_manifest(aviation) == []


def _mutate(m, **kw):
    d = m.to_dict(); d.update(kw)
    return TaskManifest.from_dict(d)


def test_reject_missing_target(aviation):
    assert any("target" in e for e in validate_manifest(_mutate(aviation, target="")))

def test_reject_bad_multiclass_metric(aviation):
    m = _mutate(aviation, task_type="multiclass_classification", metric="roc_auc")
    assert any("invalid for multiclass" in e for e in validate_manifest(m))

def test_reject_group_kfold_without_group(aviation):
    m = _mutate(aviation, group_column=None)
    assert any("group_kfold" in e for e in validate_manifest(m))

def test_reject_target_equals_group(aviation):
    m = _mutate(aviation, group_column="label")     # target is 'label'
    assert any("leakage" in e for e in validate_manifest(m))

def test_reject_unknown_split(aviation):
    assert any("split_type" in e for e in validate_manifest(_mutate(aviation, split_type="magic_split")))

def test_reject_unknown_adapter(aviation):
    assert any("adapter" in e for e in validate_manifest(_mutate(aviation, adapter="from_the_cloud")))

def test_reject_time_series_without_time_col(aviation):
    m = _mutate(aviation, split_type="time_series_split", time_column=None)
    assert any("time_series_split" in e for e in validate_manifest(m))

def test_reject_unknown_field(aviation):
    m = TaskManifest.from_dict({**aviation.to_dict(), "targett": "oops"})
    assert any("unknown manifest field" in e for e in validate_manifest(m))

def test_data_aware_target_missing(aviation):
    errs = validate_against_columns(aviation, ["plane_id", "filename", "volt1"])
    assert any("target 'label' not found" in e for e in errs)

def test_data_aware_all_present(aviation):
    assert validate_against_columns(aviation, ["label", "plane_id", "filename"]) == []
