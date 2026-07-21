"""
Long-format time-series rollup (harness/timeseries_features.py). Two
things worth proving, not just exercising:

1. channel_features computes the stats its docstring claims, including
   the edge cases (empty/all-NaN -> zeros, single value -> degenerate
   stats).
2. The streaming chunked reader is correct across chunk boundaries — an
   id's rows can land on either side of a `chunksize` cut, and the
   carry-over logic must neither drop nor double-count them. This is
   checked against a naive non-streaming groupby computation on the same
   data, with a deliberately tiny chunksize to force multiple ids to
   straddle boundaries.
"""
import io

import numpy as np
import pandas as pd
import pytest

from agentic_ml.harness.timeseries_features import (
    build_flight_feature_table_streaming,
    channel_features,
    resolve_label,
)


# --- channel_features correctness ---

def test_channel_features_basic_stats():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    feats = channel_features(x)
    assert feats["mean"] == 3.0
    assert feats["min"] == 1.0
    assert feats["max"] == 5.0
    assert feats["range"] == 4.0
    assert feats["p50"] == 3.0
    assert feats["last"] == 5.0
    assert feats["mean_abs_diff"] == 1.0
    assert feats["max_abs_diff"] == 1.0
    assert feats["slope"] == pytest.approx(1.0)


def test_channel_features_empty_is_zeros():
    feats = channel_features(np.array([]))
    assert all(v == 0.0 for v in feats.values())


def test_channel_features_all_nan_is_zeros():
    feats = channel_features(np.array([np.nan, np.nan, np.inf]))
    assert all(v == 0.0 for v in feats.values())


def test_channel_features_single_value_is_degenerate():
    feats = channel_features(np.array([7.0]))
    assert feats["mean"] == 7.0
    assert feats["std"] == 0.0
    assert feats["range"] == 0.0
    assert feats["slope"] == 0.0
    assert feats["last"] == 7.0


def test_channel_features_negative_slope():
    x = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    feats = channel_features(x)
    assert feats["slope"] == pytest.approx(-2.0)


def test_channel_features_drops_nan_before_computing():
    x = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
    feats = channel_features(x)
    # only [1, 3, 5] are finite
    assert feats["mean"] == 3.0
    assert feats["min"] == 1.0
    assert feats["max"] == 5.0


# --- label resolution ---

def test_resolve_label_known_variants():
    label_map = {"before": 1, "post": 0}
    assert resolve_label("before", label_map) == 1
    assert resolve_label("Before", label_map) == 1
    assert resolve_label(" POST ", label_map) == 0


def test_resolve_label_unknown_raises():
    with pytest.raises(ValueError, match="Unrecognised label value"):
        resolve_label("unknown_flag", {"before": 1, "post": 0})


# --- streaming rollup: naive-vs-streaming equivalence across chunk boundaries ---

def _make_long_csv(n_flights: int, steps_per_flight: int, seed: int = 0) -> tuple[str, pd.DataFrame]:
    """Builds a long-format CSV (as a string) with n_flights ids, each
    with steps_per_flight rows, plus a plane_id group and a before_after
    label alternating by flight. Returns (csv_text, the raw long dataframe)."""
    rng = np.random.RandomState(seed)
    rows = []
    for fid in range(n_flights):
        plane = f"plane_{fid % 3}"
        label = "before" if fid % 2 == 0 else "post"
        for step in range(steps_per_flight):
            rows.append({
                "id": fid, "plane_id": plane, "before_after": label,
                "date_diff": fid,
                "s1": rng.normal(loc=fid, scale=1.0),
                "s2": rng.normal(loc=-fid, scale=1.0),
            })
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue(), df


def _naive_feature_table(long_df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    rows = []
    for fid, g in long_df.groupby("id", sort=True):
        feats = {}
        for c in sensors:
            for k, v in channel_features(g[c].values).items():
                feats[f"{c}__{k}"] = v
        feats["__n_steps"] = int(len(g))
        feats["id"] = fid
        feats["plane_id"] = g["plane_id"].iloc[0]
        feats["before_after"] = 1 if g["before_after"].iloc[0] == "before" else 0
        feats["date_diff"] = g["date_diff"].iloc[0]
        rows.append(feats)
    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)


@pytest.fixture
def long_csv_path(tmp_path):
    csv_text, long_df = _make_long_csv(n_flights=9, steps_per_flight=7)
    path = tmp_path / "long.csv"
    path.write_text(csv_text)
    return path, long_df


def test_streaming_matches_naive_across_chunk_boundaries(long_csv_path):
    path, long_df = long_csv_path
    sensors = ["s1", "s2"]
    label_map = {"before": 1, "post": 0}

    # chunksize=10 is smaller than one flight's 7 rows worth several
    # times over relative to file size, forcing multiple flights to
    # straddle a chunk boundary.
    streamed = build_flight_feature_table_streaming(
        path, sensor_columns=sensors, id_column="id", group_column="plane_id",
        label_column="before_after", label_map=label_map,
        extra_columns=["date_diff"], chunksize=10, progress_every=0,
    ).sort_values("id").reset_index(drop=True)

    naive = _naive_feature_table(long_df, sensors)

    assert list(streamed["id"]) == list(naive["id"])
    assert len(streamed) == 9  # every flight present exactly once
    pd.testing.assert_frame_equal(
        streamed.sort_index(axis=1), naive.sort_index(axis=1), check_dtype=False,
    )


def test_streaming_respects_max_flights(long_csv_path):
    path, _ = long_csv_path
    streamed = build_flight_feature_table_streaming(
        path, sensor_columns=["s1", "s2"], id_column="id", group_column="plane_id",
        label_column="before_after", label_map={"before": 1, "post": 0},
        chunksize=10, max_flights=3, progress_every=0,
    )
    assert len(streamed) == 3


def test_streaming_non_contiguous_id_raises(tmp_path):
    df = pd.DataFrame({
        "id": [0, 0, 1, 0],  # id 0 reappears after id 1 -> non-contiguous
        "plane_id": ["p"] * 4,
        "s1": [1.0, 2.0, 3.0, 4.0],
    })
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="non-contiguous"):
        build_flight_feature_table_streaming(
            path, sensor_columns=["s1"], id_column="id", group_column="plane_id",
            chunksize=2, progress_every=0,
        )


def test_streaming_without_label_map_passes_label_through(long_csv_path):
    path, _ = long_csv_path
    streamed = build_flight_feature_table_streaming(
        path, sensor_columns=["s1", "s2"], id_column="id", group_column="plane_id",
        label_column="before_after", label_map=None, chunksize=10, progress_every=0,
    )
    assert set(streamed["before_after"].unique()) == {"before", "post"}
