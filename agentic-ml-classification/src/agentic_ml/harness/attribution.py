"""
Occlusion-based feature attribution: deterministic, model-agnostic
explanation of why one accepted candidate pipeline scored one example
the way it did.

For each named group of feature columns ("channel" — see channel_of()),
replace that group's values with a reference "background" value and
measure the resulting drop in the model's predicted probability for the
positive class. A larger drop means that channel mattered more to the
prediction. This is the same idea as SHAP/occlusion-based explainers,
implemented as plain repeated predict_proba() calls rather than a
dependency, since the pipelines here are arbitrary sklearn Pipelines
(different templates use different classifiers) and occlusion needs no
model-specific support.

Deliberately generic, not aviation-specific: channel_of() only groups
columns that share harness/timeseries_features.py's own
"{sensor}__{stat}" naming convention. Every other column (a plain
tabular feature, or a feature_engineering.py-derived column like
"family_size") falls back to being its own single-column channel, so
this module produces sensible output for any accepted candidate from
this pipeline, not just the aviation dataset it was first built for.

Binary classification only for now (predict_proba(...)[:, 1] is the
"positive class" probability being explained) — the current use case
(flagging a flight for maintenance) is binary; a multiclass variant
would need to pick which class's probability to attribute, which isn't
part of this scope yet.
"""
from __future__ import annotations

import pandas as pd


def channel_of(col: str) -> str:
    """Extracts the base channel name from a generated feature column
    name (e.g. 'E1 EGT3__mean' -> 'E1 EGT3'). Columns without that
    naming convention are their own channel."""
    if col.startswith("__"):
        return "__global__"
    return col.split("__", 1)[0]


def compute_background(feature_table: pd.DataFrame, cols: list[str],
                        normal_mask=None) -> pd.Series:
    """Reference values (medians) used to occlude a channel. Pass
    `normal_mask` to scope the reference to a "healthy"/negative-class
    cohort (e.g. the training rows where the target is 0) rather than
    the whole table, so occlusion means "replace with what this channel
    normally looks like," not just "replace with the overall median."""
    tbl = feature_table[normal_mask] if normal_mask is not None else feature_table
    present = [c for c in cols if c in feature_table.columns]
    med = tbl[present].median(numeric_only=True)
    return med.reindex(cols).fillna(0.0)


def attribute_prediction(feature_row, pipeline, feature_columns: list[str],
                          background: pd.Series, top_k: int = 6) -> dict:
    """
    Ranks feature channels by how much they drive one example's
    predicted probability, via occlusion.

    Args:
        feature_row: the example's feature values (dict or pd.Series).
        pipeline: a fitted sklearn Pipeline/estimator exposing
            predict_proba on a DataFrame with `feature_columns` as its
            columns (this repo's accepted candidates are exactly this
            shape — see harness/sandbox.py / templates/).
        feature_columns: the exact column order/set the pipeline was
            fit on.
        background: reference values from compute_background(), indexed
            by feature_columns.
        top_k: how many top-driving channels to report.

    Returns:
        dict with the base probability, per-channel attribution, and
        the top_k channels whose occlusion actually moved the
        probability (prob_drop > 0.001).
    """
    row = feature_row if isinstance(feature_row, pd.Series) else pd.Series(feature_row)
    row = row.reindex(feature_columns).fillna(0.0)
    bg = background.reindex(feature_columns).fillna(0.0)

    row_df = row.to_frame().T
    p0 = float(pipeline.predict_proba(row_df)[0, 1])

    groups: dict[str, list[str]] = {}
    for c in feature_columns:
        groups.setdefault(channel_of(c), []).append(c)

    attribution = []
    for ch, cols in groups.items():
        occ_row = row.copy()
        occ_row[cols] = bg[cols]
        occ_df = occ_row.to_frame().T

        p_occ = float(pipeline.predict_proba(occ_df)[0, 1])
        attribution.append({
            "channel": ch,
            "prob_drop": round(p0 - p_occ, 4),
            "n_features": len(cols),
        })

    attribution.sort(key=lambda r: r["prob_drop"], reverse=True)

    return {
        "tool": "attribute_prediction",
        "p_maintenance": round(p0, 4),
        "channel_attribution": attribution,
        "top_channels": [a["channel"] for a in attribution[:top_k] if a["prob_drop"] > 0.001],
        "note": "prob_drop measures the decrease in positive-class probability when the "
                "channel is reset to its background (healthy-cohort median) value.",
    }
