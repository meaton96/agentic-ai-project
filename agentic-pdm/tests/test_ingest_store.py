import numpy as np
import pandas as pd
import pytest

from agentic_pdm.ingest import IngestConfig, ingest_long_csv
from agentic_pdm.store import load_store, pool_time
from conftest import make_long_frame


def _config(**overrides):
    base = dict(id_column="seq", label_column="target", group_column="unit",
                fold_column="fold", exclude_columns=["leak"], max_len=64, chunksize=500)
    return IngestConfig(**{**base, **overrides})


def test_ingest_shapes_roles_and_truncation(long_csv, tmp_path):
    manifest = ingest_long_csv(long_csv, tmp_path / "store", _config())
    store = load_store(tmp_path / "store")
    raw = pd.read_csv(long_csv)

    assert manifest["channels"] == ["a", "b", "c"]  # roles and excluded column removed
    assert store.X.shape == (40, 3, 64)
    assert manifest["classes"] == ["bad", "ok"]
    assert store.meta["y"].tolist() == [1 if i % 2 == 0 else 0 for i in range(40)]

    for seq_id in (0, 7, 39):
        seq = raw[raw["seq"] == seq_id]
        row = store.meta.index[store.meta["id"] == seq_id][0]
        assert store.meta.loc[row, "length"] == len(seq)
        expected_tail = seq[["a", "b", "c"]].to_numpy(np.float32).T[:, -64:]
        got = store.X[row][:, 64 - expected_tail.shape[1]:]
        np.testing.assert_array_equal(np.isnan(got), np.isnan(expected_tail))
        np.testing.assert_allclose(np.nan_to_num(got), np.nan_to_num(expected_tail))
        if len(seq) < 64:
            assert np.isnan(store.X[row][:, :64 - len(seq)]).all()  # left padding


def test_chunk_size_does_not_change_result(long_csv, tmp_path):
    ingest_long_csv(long_csv, tmp_path / "s1", _config(chunksize=37))
    ingest_long_csv(long_csv, tmp_path / "s2", _config(chunksize=100_000))
    a, b = load_store(tmp_path / "s1"), load_store(tmp_path / "s2")
    np.testing.assert_array_equal(np.asarray(a.X), np.asarray(b.X))
    pd.testing.assert_frame_equal(a.meta, b.meta)


def test_non_contiguous_sequence_raises(tmp_path):
    df = make_long_frame(n_seq=4)
    df = pd.concat([df[df.seq == 0].iloc[:10], df[df.seq == 1], df[df.seq == 0].iloc[10:]])
    df.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="non-contiguous"):
        ingest_long_csv(tmp_path / "bad.csv", tmp_path / "out", _config())


def test_label_changing_within_sequence_raises(tmp_path):
    df = make_long_frame(n_seq=4)
    df.loc[df.index[df.seq == 2][0], "target"] = "ok" if df.loc[df.seq == 2, "target"].iloc[1] == "bad" else "bad"
    df.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="exactly one value"):
        ingest_long_csv(tmp_path / "bad.csv", tmp_path / "out", _config())


def test_max_sequences_stops_early(long_csv, tmp_path):
    manifest = ingest_long_csv(long_csv, tmp_path / "store", _config(max_sequences=5))
    assert manifest["shape"][0] == 5 and manifest["truncated"] is True
    assert load_store(tmp_path / "store").X.shape[0] == 5


def test_assigned_folds_keep_groups_disjoint(long_csv, tmp_path):
    ingest_long_csv(long_csv, tmp_path / "store", _config(fold_column=None, n_folds=4))
    meta = load_store(tmp_path / "store").meta
    assert meta.groupby("group")["fold"].nunique().max() == 1
    assert sorted(meta["fold"].unique()) == [0, 1, 2, 3]


def test_missing_role_column_raises(long_csv, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        ingest_long_csv(long_csv, tmp_path / "out", _config(group_column="nope"))


def test_pool_time_nan_aware_mean():
    X = np.arange(2 * 1 * 8, dtype=np.float32).reshape(2, 1, 8)
    X[0, 0, :4] = np.nan          # a fully padded window stays NaN
    X[1, 0, 5] = np.nan           # a partially missing window ignores the NaN
    pooled = pool_time(X, 4)
    assert pooled.shape == (2, 1, 2)
    assert np.isnan(pooled[0, 0, 0])
    assert pooled[0, 0, 1] == pytest.approx(np.mean([4, 5, 6, 7]))
    assert pooled[1, 0, 1] == pytest.approx(np.mean([12, 14, 15]))


def test_pool_time_rejects_indivisible_length():
    with pytest.raises(ValueError, match="divisible"):
        pool_time(np.zeros((1, 1, 10), dtype=np.float32), 4)


def test_positive_label_becomes_class_one(long_csv, tmp_path):
    # "bad" sorts first, so this forces a reorder away from the default.
    manifest = ingest_long_csv(long_csv, tmp_path / "store", _config(positive_label="bad"))
    meta = load_store(tmp_path / "store").meta
    assert manifest["classes"] == ["ok", "bad"]
    assert (meta.loc[meta["label"] == "bad", "y"] == 1).all()
    assert (meta.loc[meta["label"] == "ok", "y"] == 0).all()
    with pytest.raises(ValueError, match="positive_label"):
        ingest_long_csv(long_csv, tmp_path / "s2", _config(positive_label="missing"))
