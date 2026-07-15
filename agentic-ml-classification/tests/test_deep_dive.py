"""
Deep-dive agent: flight-phase segmentation, cross-cylinder anomaly
localization, occlusion attribution, and the agent step that
synthesizes a hypothesis from them. Four things worth proving, not just
exercising:

1. _merge_short actually absorbs short spurious runs into a neighbor
   (domain/aviation/flight_phases.py) and segment_flight produces a
   sane phase/takeoff/landing structure for a realistic climb-cruise-
   descent profile.
2. localize_anomaly detects a genuine load-dependent single-cylinder
   fault (elevated in one phase only) AND correctly rejects a benign
   constant cross-cylinder offset (elevated equally in every phase) —
   the specific false-positive-rejection claim the module's docstring
   makes.
3. attribute_prediction, generalized in this repo to occlude columns by
   name in a DataFrame and call an actual fitted sklearn Pipeline
   (rather than the original prototype's raw-numpy-array classifier),
   still recovers the planted fault channel as the top driver — this is
   the regression test proving that generalization didn't break the
   original's core empirical claim.
4. run_deep_dive_step's conservative default: an unparseable LLM
   response degrades to the deterministic template, mirroring
   test_verification.py's equivalent test.
"""
import json

import numpy as np
import pandas as pd
import pytest

from agentic_ml.domain.aviation.flight_phases import _merge_short, segment_flight
from agentic_ml.domain.aviation.anomaly_localization import localize_anomaly
from agentic_ml.harness.attribution import attribute_prediction, channel_of, compute_background
from agentic_ml.model_client import ModelResponse
from agentic_ml.steps.deep_dive_step import run_deep_dive_step
from agentic_ml.templates.sources.xgboost_mixed import build_pipeline


# --- flight_phases ---

def test_merge_short_absorbs_short_runs_into_longer_neighbor():
    labels = ["a"] * 10 + ["b"] * 2 + ["a"] * 10
    assert _merge_short(labels, min_len=5) == ["a"] * 22


def test_merge_short_leaves_long_runs_alone():
    labels = ["a"] * 10 + ["b"] * 10 + ["a"] * 10
    assert _merge_short(labels, min_len=5) == labels


def test_segment_flight_recovers_climb_cruise_descent_profile():
    # ground(20s) -> climb(60s, ~2900 fpm) -> cruise(60s) -> descent(40s) -> ground(40s)
    # (final ground run is longer than the others so smoothing lag at the
    # descent->ground boundary can't shrink it below min_phase_s and get it
    # merged away by _merge_short)
    ground1 = np.full(20, 100.0)
    climb = 100.0 + np.linspace(0, 2900, 60)
    cruise = np.full(60, 3000.0)
    descent = 3000.0 - np.linspace(0, 2800, 40)
    ground2 = np.full(40, 100.0)
    alt = np.concatenate([ground1, climb, cruise, descent, ground2])
    ias = np.concatenate([np.zeros(20), np.full(60, 90.0), np.full(60, 100.0),
                           np.full(40, 90.0), np.zeros(40)])
    df = pd.DataFrame({"AltMSL": alt, "IAS": ias})

    result = segment_flight(df, sample_hz=1.0)
    phases_present = {seg["phase"] for seg in result["segments"]}
    assert phases_present == {"ground", "climb", "cruise", "descent"}
    assert result["n_takeoffs"] == 1
    assert result["n_landings"] == 1
    # first and last segments should be on the ground
    assert result["segments"][0]["phase"] == "ground"
    assert result["segments"][-1]["phase"] == "ground"


def test_segment_flight_requires_alt_column():
    with pytest.raises(KeyError):
        segment_flight(pd.DataFrame({"IAS": [0.0, 1.0]}))


# --- localize_anomaly ---

def test_localize_anomaly_detects_fault_and_rejects_benign_constant_offset():
    rng = np.random.RandomState(0)
    n = 150
    base = 1200.0
    df = pd.DataFrame({f"E1 EGT{i}": rng.normal(base, 1.0, n) for i in range(1, 5)})

    df["E1 EGT2"] += 50.0  # constant offset in every phase -> benign, should NOT be flagged
    df.loc[df.index[:50], "E1 EGT3"] += 80.0  # only elevated during "climb" -> load-dependent fault

    segments = [
        {"phase": "climb", "start_idx": 0, "end_idx": 49, "start_s": 0.0},
        {"phase": "cruise", "start_idx": 50, "end_idx": 99, "start_s": 50.0},
        {"phase": "descent", "start_idx": 100, "end_idx": 149, "start_s": 100.0},
    ]
    result = localize_anomaly(df, segments=segments)
    flagged = {f["channel"] for f in result["findings"]}

    assert "E1 EGT3" in flagged
    assert "E1 EGT2" not in flagged  # benign constant offset correctly rejected

    f3 = next(f for f in result["findings"] if f["channel"] == "E1 EGT3")
    assert f3["cylinder"] == 3
    assert f3["worst_phase"] == "climb"
    assert f3["direction"] == "hot"


def test_localize_anomaly_skips_group_with_fewer_than_three_channels():
    df = pd.DataFrame({"E1 EGT1": [1.0, 2.0], "E1 EGT2": [1.0, 2.0]})
    segments = [{"phase": "cruise", "start_idx": 0, "end_idx": 1, "start_s": 0.0}]
    result = localize_anomaly(df, segments=segments)
    assert result["groups_checked"] == []
    assert result["findings"] == []


# --- attribute_prediction (generalized to occlude a named DataFrame column,
# call an actual fitted sklearn Pipeline) ---

def test_channel_of_groups_by_naming_convention():
    assert channel_of("E1 EGT3__mean") == "E1 EGT3"
    assert channel_of("__n_steps") == "__global__"
    assert channel_of("family_size") == "family_size"  # no "__" -> its own channel


def _make_cyl3_fault_feature_table(n_per_class=60, seed=0):
    """A small synthetic feature table where the binary label IS a
    planted cyl-3 EGT/CHT fault, mirroring
    aviation_mas_mvp/scripts/deep_dive/validate_attribute.py's
    controlled test — adapted to fit one of this repo's own template
    pipelines instead of a bare classifier."""
    rng = np.random.RandomState(seed)
    channels = [f"E1 EGT{i}" for i in range(1, 5)] + [f"E1 CHT{i}" for i in range(1, 5)]
    rows = []
    for i in range(n_per_class * 2):
        label = i % 2
        row = {f"{ch}__mean": rng.normal(1200.0, 5.0) for ch in channels}
        if label == 1:
            row["E1 EGT3__mean"] += 80.0
            row["E1 CHT3__mean"] += 20.0
        row["label"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def test_attribute_prediction_recovers_planted_fault_channel_on_real_pipeline():
    table = _make_cyl3_fault_feature_table()
    feature_cols = [c for c in table.columns if c != "label"]

    train = table.iloc[:100]
    test = table.iloc[100:]

    pipeline = build_pipeline({"numeric_cols": feature_cols, "categorical_cols": [], "seed": 0})
    pipeline.fit(train[feature_cols], train["label"])

    background = compute_background(train, feature_cols, normal_mask=(train["label"] == 0))

    faulty_row = test[test["label"] == 1].iloc[0]
    result = attribute_prediction(faulty_row[feature_cols], pipeline, feature_cols, background, top_k=6)

    assert result["p_maintenance"] > 0.5
    ranked_channels = [a["channel"] for a in result["channel_attribution"]]
    top2 = ranked_channels[:2]
    assert "E1 EGT3" in top2 or "E1 CHT3" in top2


# --- run_deep_dive_step: conservative default on unparseable LLM output ---

def _resp(text=None, tool_calls=None):
    return ModelResponse(
        text=text, tool_calls=tool_calls or [], raw=None, latency_seconds=0.01,
        model="fake-model", input_tokens=1, output_tokens=1,
    )


def _make_fake_client(final_text):
    call_count = {"n": 0}

    class FakeClient:
        def call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _resp(tool_calls=[{"id": "d1", "name": "get_flight_deep_dive_evidence", "arguments": "{}"}])
            return _resp(text=final_text)

    return FakeClient()


@pytest.fixture
def deep_dive_fixture():
    table = _make_cyl3_fault_feature_table()
    feature_cols = [c for c in table.columns if c != "label"]
    pipeline = build_pipeline({"numeric_cols": feature_cols, "categorical_cols": [], "seed": 0})
    pipeline.fit(table[feature_cols], table["label"])
    background = compute_background(table, feature_cols, normal_mask=(table["label"] == 0))
    feature_row = table[table["label"] == 1].iloc[0][feature_cols]

    # minimal flight_df: enough for segment_flight to run; no EGT/CHT columns,
    # so localize_anomaly finds nothing -> exercises the "no localization" template branch.
    flight_df = pd.DataFrame({"AltMSL": np.full(30, 500.0), "IAS": np.zeros(30)})

    return flight_df, feature_row, pipeline, feature_cols, background


def test_run_deep_dive_step_approved(deep_dive_fixture):
    flight_df, feature_row, pipeline, feature_cols, background = deep_dive_fixture
    client = _make_fake_client(json.dumps({
        "hypothesis": "E1 EGT3 drove the flag.", "agrees_with_localization": None, "confidence": "medium",
    }))
    result = run_deep_dive_step(flight_df, feature_row, pipeline, feature_cols, background, client)
    assert result.ok
    assert not result.used_template_fallback
    assert result.hypothesis == "E1 EGT3 drove the flag."
    assert result.evidence is not None


def test_run_deep_dive_step_unparseable_defaults_to_template(deep_dive_fixture):
    flight_df, feature_row, pipeline, feature_cols, background = deep_dive_fixture
    client = _make_fake_client("not valid json")
    result = run_deep_dive_step(flight_df, feature_row, pipeline, feature_cols, background, client)
    assert not result.ok
    assert result.used_template_fallback
    assert result.confidence == "low"
    assert result.hypothesis is not None  # template still produces something, not None
    assert result.evidence is not None    # deterministic evidence unaffected by the parse failure


def test_run_deep_dive_step_missing_confidence_defaults_to_template(deep_dive_fixture):
    flight_df, feature_row, pipeline, feature_cols, background = deep_dive_fixture
    client = _make_fake_client(json.dumps({"hypothesis": "some cause", "confidence": "extremely sure"}))
    result = run_deep_dive_step(flight_df, feature_row, pipeline, feature_cols, background, client)
    assert not result.ok
    assert result.used_template_fallback
