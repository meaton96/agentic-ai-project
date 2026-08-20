"""
Deterministic review-bundle assembly and verdict normalization for the
Human Oversight agent (agent #6) -- the third A2A hop this project
tests: Optimization -> Human Oversight. Mirrors the sibling ML
pipeline's Verification agent exactly (steps/verification_step.py /
harness/verification.py there): this module never re-decides anything,
it only assembles facts into a bundle for the LLM to review, and
normalizes whatever the LLM says. Unlike every other gate in this
project, there is no deterministic "correct" verdict to check the LLM
against here -- a human-judgment call is the entire point of delegating
to this agent, same as the ML pipeline's verification step.
"""
from __future__ import annotations

from typing import Optional


def build_oversight_review_bundle(policy_update_proposal: dict, evidence: dict) -> dict:
    """Bundles the Optimization agent's proposal with the evidence it was
    based on -- Human Oversight must see both, the same way the ML
    pipeline's Verification agent gets a review bundle carrying both a
    modeling candidate's config AND its gate results, never the
    candidate alone."""
    return {
        "policy_updates": policy_update_proposal.get("policy_updates"),
        "proposal_evidence_summary": policy_update_proposal.get("evidence"),
        "recommend_apply": policy_update_proposal.get("recommend_apply"),
        "underlying_evidence": evidence,
    }


def parse_oversight_verdict(raw: Optional[dict]) -> dict:
    """Degrades to 'flagged' -- never silently to 'approved' -- on any
    parse/shape problem. Deliberately duplicates the exact rule the ML
    pipeline's steps/verification_step.py documents for its own
    verification agent: an agent's approval is only meaningful if
    silence or failure can never be mistaken for it."""
    if not isinstance(raw, dict) or raw.get("verdict") not in ("approved", "rejected", "flagged"):
        return {
            "verdict": "flagged",
            "concerns": ["oversight agent response was missing, malformed, or used an unrecognized verdict"],
            "reasoning": "defaulted to flagged: never treat an unparseable or invalid response as approval",
        }
    return {
        "verdict": raw["verdict"],
        "concerns": raw.get("concerns") if isinstance(raw.get("concerns"), list) else [],
        "reasoning": raw.get("reasoning") if isinstance(raw.get("reasoning"), str) else "",
    }
