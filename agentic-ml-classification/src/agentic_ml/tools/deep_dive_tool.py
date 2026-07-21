"""
Tool exposing a flagged flight's full deep-dive evidence bundle to the
deep-dive agent. Bundled into one tool call, same reasoning as
verification_tool.py's single get_candidate_review_bundle: the agent has
no real choice about which measurement to run — segmentation,
attribution, and localization always run together on the same flight —
so exposing three separate tools would just be three mandatory
sequential calls with no judgment in between. The handler runs the
harness's deterministic measurements directly; the agent cannot alter
what they compute.
"""
from __future__ import annotations

import pandas as pd

from agentic_ml.agent_runtime import Tool
from agentic_ml.domain.aviation.flight_phases import segment_flight
from agentic_ml.domain.aviation.anomaly_localization import localize_anomaly
from agentic_ml.harness.attribution import attribute_prediction


def gather_deep_dive_evidence(flight_df: pd.DataFrame, feature_row, pipeline,
                               feature_columns: list[str], background: pd.Series,
                               sample_hz: float = 1.0, loc_thresholds: dict | None = None) -> dict:
    """Runs the three deterministic measurement tools for one flight.
    Deterministic; no LLM. Exposed separately from make_deep_dive_evidence_tool
    so scripts/run_deep_dive_agent.py can also save this evidence even if
    the LLM synthesis step fails."""
    seg = segment_flight(flight_df, sample_hz=sample_hz)
    attr = attribute_prediction(feature_row, pipeline, feature_columns, background, top_k=6)
    loc = localize_anomaly(flight_df, seg["segments"], thresholds=loc_thresholds, sample_hz=sample_hz)
    seg_summary = {k: seg[k] for k in
                   ("n_takeoffs", "n_landings", "airborne_s", "phase_seconds")}
    return {
        "p_maintenance": attr["p_maintenance"],
        "segmentation": seg_summary,
        "attribution_top": attr["channel_attribution"][:6],
        "localization": loc["findings"],
        "n_localized": loc["n_flagged"],
    }


def make_deep_dive_evidence_tool(flight_df: pd.DataFrame, feature_row, pipeline,
                                  feature_columns: list[str], background: pd.Series,
                                  sample_hz: float = 1.0, loc_thresholds: dict | None = None) -> Tool:
    def handler() -> dict:
        return gather_deep_dive_evidence(
            flight_df, feature_row, pipeline, feature_columns, background,
            sample_hz=sample_hz, loc_thresholds=loc_thresholds,
        )

    return Tool(
        name="get_flight_deep_dive_evidence",
        description=(
            "Get the full deep-dive evidence for one flagged flight: the model's "
            "predicted maintenance probability, which sensor CHANNELS drove that "
            "prediction (occlusion attribution), flight-phase segmentation, and "
            "independent RAW-SIGNAL cross-cylinder imbalance findings. Call this "
            "once — you cannot compute any of this yourself and must not invent it."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )
