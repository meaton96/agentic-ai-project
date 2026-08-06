"""
harness/column_grouping.py: compacting a wide per-column fact list into
fewer entries when many columns share a `{prefix}__{suffix}` naming
pattern. Built after a real incident: a 282-column rolled-up NGAFID
table serialized to ~19-25K tokens via raw_schema_summary()/
profile_dataset() alone, which blew a 32K-context local model's request
limit calling the intake agent (BadRequestError, context length
exceeded). Three things worth proving, not just exercising:

1. An ordinary dataset (no repeating "__" pattern — every dataset this
   pipeline ran against before the aviation rollup) is returned
   unchanged — zero behavior change for Titanic/Iris-shaped data.
2. A wide, repeating-pattern dataset gets compacted, and
   expand_grouped_columns() correctly resolves every real column name
   back to its facts either way — the property that keeps downstream
   validators (steps/modeling_step.py, harness/feature_engineering.py)
   correct against a compacted payload.
3. The actual regression this was built to prevent: a modeling
   candidate proposing a column that only exists inside a compacted
   group must still validate successfully — this is exactly what
   failed (a real, caught-in-testing bug, not hypothetical) before the
   validators were fixed to expand groups instead of assuming a flat
   name-keyed dict.
"""
from __future__ import annotations

import pandas as pd
import pytest

from agentic_ml.harness.column_grouping import expand_grouped_columns, group_columns_by_pattern
from agentic_ml.harness.feature_engineering import validate_feature_proposal
from agentic_ml.harness.intake import raw_schema_summary
from agentic_ml.harness.profiler import profile_dataset
from agentic_ml.steps.modeling_step import _validate_candidate_columns


def _stat_columns(prefix: str, n_stats: int) -> list[dict]:
    return [
        {"name": f"{prefix}__stat{i}", "dtype": "float64", "missing_frac": round(i * 0.01, 2)}
        for i in range(n_stats)
    ]


# --- 1 & 2: group_columns_by_pattern / expand_grouped_columns ---

def test_no_grouping_when_no_repeating_pattern_exists():
    columns = [{"name": "Age", "dtype": "float64"}, {"name": "Fare", "dtype": "float64"},
               {"name": "Survived", "dtype": "int64"}]
    assert group_columns_by_pattern(columns) == columns


def test_small_shared_prefix_below_min_group_size_stays_ungrouped():
    columns = _stat_columns("volt1", 3)  # below the default min_group_size=4
    assert group_columns_by_pattern(columns) == columns


def test_wide_repeating_pattern_gets_grouped():
    columns = _stat_columns("volt1", 12) + _stat_columns("amp1", 12) + [{"name": "plane_id", "dtype": "object"}]
    grouped = group_columns_by_pattern(columns)
    assert len(grouped) == 3  # volt1 group, amp1 group, plane_id passthrough
    group_names = {g["column_group"] for g in grouped if "column_group" in g}
    assert group_names == {"volt1", "amp1"}
    volt1_group = next(g for g in grouped if g.get("column_group") == "volt1")
    assert volt1_group["n_columns"] == 12
    assert volt1_group["columns"] == [f"volt1__stat{i}" for i in range(12)]
    assert volt1_group["missing_frac_max"] == 0.11  # max of 0.00..0.11
    assert any(c.get("name") == "plane_id" for c in grouped)


def test_expand_grouped_columns_resolves_every_real_name():
    columns = _stat_columns("volt1", 12) + [{"name": "plane_id", "dtype": "object"}]
    grouped = group_columns_by_pattern(columns)
    resolved = expand_grouped_columns(grouped)
    assert set(resolved) == {c["name"] for c in columns}
    # a grouped member resolves to the group's representative facts
    assert resolved["volt1__stat5"]["dtype"] == "float64"
    # an ungrouped column resolves to its own facts unchanged
    assert resolved["plane_id"]["dtype"] == "object"


def test_raw_schema_summary_compacts_a_wide_rollup_shaped_dataframe():
    n = 20
    data = {f"volt1__stat{stat}": list(range(n)) for stat in range(12)}
    data["plane_id"] = [f"p{i % 3}" for i in range(n)]
    df = pd.DataFrame(data)
    summary = raw_schema_summary(df)
    assert summary["n_columns"] == 13  # the real column count is untouched
    assert len(summary["columns"]) == 2  # volt1 group + plane_id


def test_profile_dataset_compacts_columns_without_affecting_leakage_or_split_facts():
    n = 20
    data = {f"volt1__stat{i}": list(range(n)) for i in range(12)}
    data["plane_id"] = [f"p{i % 3}" for i in range(n)]
    data["target"] = [i % 2 for i in range(n)]
    df = pd.DataFrame(data)
    report = profile_dataset(df, target_column="target")
    d = report.to_dict()
    assert len(d["columns"]) == 3  # volt1 group + plane_id + target, not 14 individual entries
    # facts computed over the FULL per-column list are unaffected by compaction
    assert d["n_columns"] == 14
    assert d["recommended_split_strategy"] in ("group", "group_time")  # plane_id still detected


# --- 3: the actual regression — validators must resolve a grouped column name ---

def test_modeling_validator_accepts_a_column_that_only_exists_inside_a_group():
    profile_report = {"columns": group_columns_by_pattern(
        _stat_columns("volt1", 12) + [{"name": "plane_id", "dtype": "object", "missing_frac": 0.0,
                                        "is_likely_id": False},
                                       {"name": "target", "dtype": "int64", "missing_frac": 0.0,
                                        "is_likely_id": False}],
    )}
    for c in profile_report["columns"]:
        if "representative_column" in c:
            c["representative_column"]["is_likely_id"] = False
    errors = _validate_candidate_columns(
        profile_report, target_column="target", group_column="plane_id", time_column=None,
        config={"numeric_cols": ["volt1__stat3"], "categorical_cols": []},
    )
    assert errors == []


def test_modeling_validator_still_rejects_a_genuinely_unknown_column():
    profile_report = {"columns": group_columns_by_pattern(_stat_columns("volt1", 12))}
    errors = _validate_candidate_columns(
        profile_report, target_column="target", group_column=None, time_column=None,
        config={"numeric_cols": ["totally_made_up_column"], "categorical_cols": []},
    )
    assert any("unknown column" in e for e in errors)


def test_feature_engineering_validator_accepts_a_column_that_only_exists_inside_a_group():
    columns = _stat_columns("volt1", 12) + [
        {"name": "plane_id", "dtype": "object"}, {"name": "target", "dtype": "int64"},
    ]
    profile_report = {"columns": group_columns_by_pattern(columns)}
    errors = validate_feature_proposal(
        profile_report, target_column="target", group_column="plane_id", time_column=None,
        proposal={"drop_columns": ["volt1__stat7"], "derived_features": []},
    )
    assert errors == []
