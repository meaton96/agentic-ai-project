"""
deep_dive_agent.py
==================
Orchestrates the deep-dive analysis for flagged flights.

This module combines deterministic signal measurements (flight phase segmentation,
model attribution, and raw-signal localization) with an LLM-based synthesis step.
The resulting output is a human-readable hypothesis detailing why a specific 
flight was flagged for maintenance, reconciling model attribution with raw signals.
"""
from __future__ import annotations
import json

try:
    from .attribute import attribute_prediction
    from .localize_anomaly import localize_anomaly
except ImportError:
    from scripts.deep_dive.flight_phases import segment_flight
    from attribute import attribute_prediction
    from localize_anomaly import localize_anomaly


def gather_evidence(flight_df, feature_row, bundle, background,
                    sample_hz: float = 1.0, loc_thresholds: dict | None = None) -> dict:
    """Run the three measurement tools. Deterministic; no LLM."""
    seg = segment_flight(flight_df, sample_hz=sample_hz)
    attr = attribute_prediction(feature_row, bundle, background, top_k=6)
    loc = localize_anomaly(flight_df, seg["segments"], thresholds=loc_thresholds,
                           sample_hz=sample_hz)
    seg_summary = {k: seg[k] for k in
                   ("n_takeoffs", "n_landings", "airborne_s", "phase_seconds")}
    return {"p_maintenance": attr["p_maintenance"],
            "segmentation": seg_summary,
            "attribution_top": attr["channel_attribution"][:6],
            "localization": loc["findings"],
            "n_localized": loc["n_flagged"]}


SYNTH_SYSTEM = (
    "You are a maintenance analyst explaining why a predictive model flagged a "
    "flight for inspection. You are given: the model's probability, which sensor "
    "CHANNELS the model relied on (occlusion attribution), and independent RAW-SIGNAL "
    "findings localizing cross-cylinder imbalances to a flight phase. Cylinder "
    "channels are named E1 EGT<n>/E1 CHT<n> (n=1-4). Write 2-4 sentences for an "
    "engineer: state the most likely cause, cite the specific channel, cylinder, "
    "phase and magnitude, and note whether the model's attribution and the raw signal "
    "AGREE. Hedge honestly -- if the two disagree, or nothing localized, say so and "
    "avoid inventing a cause. Do not recommend specific parts; this is a hypothesis to "
    "guide inspection, not a diagnosis.")


def synthesize(evidence: dict, chat_fn) -> str:
    """
    Generates a natural language explanation of the flight anomaly using an LLM.
    
    Args:
        evidence (dict): The compiled evidence package from `gather_evidence`.
        chat_fn (callable): A function `chat_fn(system_prompt, user_prompt) -> str` 
            that interfaces with an LLM. If None, falls back to a deterministic template.
            
    Returns:
        str: A concise, hedged hypothesis intended for a maintenance engineer.
    """
    user = "EVIDENCE:\n" + json.dumps(evidence, indent=2) + \
           "\n\nWrite the explanation."
    if chat_fn is None:
        return _template_explanation(evidence)
    try:
        return chat_fn(SYNTH_SYSTEM, user).strip()
    except Exception as e:
        return _template_explanation(evidence) + f"  [LLM synthesis unavailable: {e}]"


def _template_explanation(ev: dict) -> str:
    """
    Provides a deterministic, rule-based text explanation if the LLM is unavailable.
    
    Args:
        ev (dict): The compiled evidence package.
        
    Returns:
        str: A templated summary of the probability, attribution, and localization.
    """
    p = ev["p_maintenance"]
    loc = ev["localization"]
    attr = [a["channel"] for a in ev["attribution_top"][:3] if a["prob_drop"] > 0.001]
    if not loc:
        return (f"Flagged (p={p:.2f}). Model leaned on {', '.join(attr) or 'no single channel'}; "
                f"no cross-cylinder imbalance localized in the raw signal.")
    f = loc[0]
    agree = f["channel"] in attr
    return (f"Flagged (p={p:.2f}). {f['channel']} (cylinder {f['cylinder']}) ran "
            f"{f['deviation']:+.0f}\u00b0 vs its siblings, worst during {f['worst_phase']} "
            f"(~{f['worst_segment_start_s']}s)"
            + (f", corroborated by the other temp group" if f.get("corroborated_by_other_group") else "")
            + (f"; model attribution agrees ({f['channel']} among top drivers)."
               if agree else
               f"; note the model's top drivers ({', '.join(attr) or 'none'}) do not center on "
               f"this cylinder, so treat as a lead, not a confirmed cause."))


def deep_dive(flight_df, feature_row, bundle, background, chat_fn=None,
              flight_id: str = "?", sample_hz: float = 1.0,
              loc_thresholds: dict | None = None) -> dict:
    """Full deep dive for one flagged flight -> record for the maintenance queue."""
    evidence = gather_evidence(flight_df, feature_row, bundle, background,
                               sample_hz, loc_thresholds)
    explanation = synthesize(evidence, chat_fn)
    return {"flight": str(flight_id),
            "p_maintenance": evidence["p_maintenance"],
            "explanation": explanation,
            "evidence": evidence}