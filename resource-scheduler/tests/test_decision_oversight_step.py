"""
Test for run_decision_oversight_step's deterministic short-circuit --
same pattern as test_reroute_validation.py's
test_no_reroute_request_available_returns_ok_false. The LLM-calling
path isn't unit-tested directly here, consistent with every other
step in this project (only the deterministic layer gets direct tests;
the LLM path is validated by actually running it, per the live
orchestrator runs this session).
"""
from resource_scheduler.a2a.mailbox import Mailbox
from resource_scheduler.steps.oversight_step import run_decision_oversight_step


def test_no_risky_decision_available_returns_ok_false():
    mailbox = Mailbox()
    result = run_decision_oversight_step(client=None, mailbox=mailbox)
    assert result.ok is False
    assert result.stopped_reason == "no_risky_decision_available"
    assert result.n_decisions_reviewed == 0


def test_ignores_unrelated_message_types_in_the_same_inbox():
    """human_oversight's inbox can also carry policy_update_proposal
    messages from Optimization -- confirms the message_type filter
    (fixed for exactly this multi-sender-shared-inbox scenario) keeps
    a risky_decision consumer from mistaking one for the other, and
    leaves the unrelated message queued rather than discarding it."""
    mailbox = Mailbox()
    mailbox.send(sender="optimization", recipient="human_oversight",
                 message_type="policy_update_proposal", payload={"policy_updates": {}})
    result = run_decision_oversight_step(client=None, mailbox=mailbox)
    assert result.ok is False
    remaining = mailbox.peek("human_oversight")
    assert len(remaining) == 1
    assert remaining[0].message_type == "policy_update_proposal"
