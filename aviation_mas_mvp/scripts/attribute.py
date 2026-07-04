"""
attribute.py
============
Feature attribution module for sensor channels.

Implements an occlusion-based attribution method to identify which sensor
channels contribute most significantly to a classifier's positive prediction.
This provides deterministic, model-agnostic feature importance grouped by channel.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def channel_of(col: str) -> str:
    """
    Extracts the base channel name from a generated feature column name.
    
    Args:
        col (str): The raw feature column name (e.g., 'E1 EGT3__mean' or 
            'E1 EGT3__fft_band_energy__band0').
            
    Returns:
        str: The extracted channel name, or '__global__' for step/global features.
    """
    if col.startswith("__"):
        return "__global__"
    return col.split("__", 1)[0]


def compute_background(feature_table: pd.DataFrame, cols: list[str],
                       normal_mask=None) -> pd.Series:
    """
    Computes reference background values (medians) for feature occlusion.
    
    Args:
        feature_table (pd.DataFrame): The dataset containing feature columns.
        cols (list[str]): List of feature column names to compute medians for.
        normal_mask (pd.Series | np.ndarray, optional): Boolean mask to filter
            the table to a specific cohort (e.g., a healthy baseline). Defaults to None.
            
    Returns:
        pd.Series: Median values for the specified columns, filled with 0.0 for missing data.
    """
    tbl = feature_table[feature_table[cols].notna().any(axis=1)] if len(feature_table) else feature_table
    if normal_mask is not None:
        tbl = feature_table[normal_mask]
        
    present = [c for c in cols if c in feature_table.columns]
    med = feature_table[present].median(numeric_only=True)
    return med.reindex(cols).fillna(0.0)


def attribute_prediction(feature_row, bundle: dict, background: pd.Series,
                         top_k: int = 6) -> dict:
    """
    Calculates the drop in prediction probability for each channel via occlusion.
    
    Replaces each channel's features with the background reference values and
    measures the resulting change in the model's positive class probability.
    
    Args:
        feature_row (dict | pd.Series): The feature row for a single instance.
        bundle (dict): Dictionary containing the trained 'model' and 'feature_columns'.
        background (pd.Series): Reference median values for occlusion.
        top_k (int, optional): Number of top contributing channels to return. Defaults to 6.
        
    Returns:
        dict: A structured summary containing the base probability, channel attributions,
            and the top `top_k` driving channels.
    """
    clf, cols = bundle["model"], bundle["feature_columns"]
    row = feature_row if isinstance(feature_row, pd.Series) else pd.Series(feature_row)
    
    x = row.reindex(cols).fillna(0.0).to_numpy(dtype=float)
    med = background.reindex(cols).fillna(0.0).to_numpy(dtype=float)

    # Base probability for the positive class
    p0 = float(clf.predict_proba(x.reshape(1, -1))[0, 1])

    # Map channels to their respective column indices
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(cols):
        groups.setdefault(channel_of(c), []).append(i)

    attribution = []
    for ch, idxs in groups.items():
        x_occ = x.copy()
        x_occ[idxs] = med[idxs]
        
        p_occ = float(clf.predict_proba(x_occ.reshape(1, -1))[0, 1])
        attribution.append({
            "channel": ch, 
            "prob_drop": round(p0 - p_occ, 4),
            "n_features": len(idxs)
        })
        
    attribution.sort(key=lambda r: r["prob_drop"], reverse=True)

    return {
        "tool": "attribute_prediction",
        "p_maintenance": round(p0, 4),
        "channel_attribution": attribution,
        "top_channels": [a["channel"] for a in attribution[:top_k] if a["prob_drop"] > 0.001],
        "note": "prob_drop measures the decrease in positive probability when the channel is reset to the median background value.",
    }