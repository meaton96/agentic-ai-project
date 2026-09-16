"""
Deterministic review-bundle assembly and verdict normalization for the
Human Oversight agent (agent #6). Two review paths now feed it:
Optimization -> Human Oversight (policy_update_proposal, the original
A2A hop) and Resource Allocation / Failure Recovery -> Human Oversight
(risky_decision -- an already-accepted assignment or reroute landing on
an Overloaded machine, per environment/allocation.py's
identify_risky_assignments). Mirrors the sibling ML pipeline's
Verification agent exactly (steps/verification_step.py /
harness/verification.py there): this module never re-decides anything,
it only assembles facts into a bundle for the LLM to review, and
normalizes whatever the LLM says. Unlike every other gate in this
project, there is no deterministic "correct" verdict to check the LLM
against here -- a human-judgment call is the entire point of delegating
to this agent, same as the ML pipeline's verification step.

Both review paths are advisory, not blocking: by the time either
message reaches Human Oversight, the decision it describes has already
been accepted by the deterministic gate (check_constraints) and
committed. Nothing here pauses that decision or can undo it -- the
verdict is logged for the audit trail, same as Optimization's proposal
is never auto-applied regardless of verdict. Making this a real
blocking approval gate would be a materially bigger workflow change
(pause/resume semantics on a decision mid-flight) than widening what
gets reviewed after the fact.
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


def build_decision_review_bundle(risky_decision_message: dict) -> dict:
    """Bundles a batch of risky scheduling decisions (all from the same
    Resource Allocation or Failure Recovery call) for Human Oversight.
    No aggregated evidence to attach here, unlike the policy path --
    each decision already carries its own risk_reason (why it was
    flagged) and rationale (why the agent chose it); Human Oversight's
    job is to judge whether that's still a good call given the risk,
    not to re-validate it structurally (it already passed
    check_constraints)."""
    return {
        "source_agent": risky_decision_message.get("source"),
        "risky_decisions": risky_decision_message.get("risky_decisions", []),
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
