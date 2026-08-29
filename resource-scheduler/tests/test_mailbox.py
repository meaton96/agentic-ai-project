"""
Tests for a2a/mailbox.py -- the in-process A2A message bus. No LLM
involved; this only needs to prove the mechanism itself is correct
before any agent's real output flows through it.
"""
from resource_scheduler.a2a.mailbox import Mailbox


def test_send_then_inbox_for_returns_the_message():
    mailbox = Mailbox()
    mailbox.send("task_prioritization", "resource_allocation", "task_ranking", {"ranked_task_ids": ["T1"]})
    inbox = mailbox.inbox_for("resource_allocation")
    assert len(inbox) == 1
    assert inbox[0].sender == "task_prioritization"
    assert inbox[0].message_type == "task_ranking"
    assert inbox[0].payload == {"ranked_task_ids": ["T1"]}


def test_inbox_for_consumes_the_queue():
    mailbox = Mailbox()
    mailbox.send("a", "b", "msg", {})
    mailbox.inbox_for("b")
    assert mailbox.inbox_for("b") == []


def test_peek_does_not_consume():
    mailbox = Mailbox()
    mailbox.send("a", "b", "msg", {})
    assert len(mailbox.peek("b")) == 1
    assert len(mailbox.peek("b")) == 1  # still there
    assert len(mailbox.inbox_for("b")) == 1  # and still consumable


def test_recipients_are_isolated():
    mailbox = Mailbox()
    mailbox.send("a", "b", "msg", {})
    mailbox.send("a", "c", "msg", {})
    assert len(mailbox.inbox_for("b")) == 1
    assert len(mailbox.inbox_for("c")) == 1


def test_inbox_for_unknown_recipient_is_empty():
    mailbox = Mailbox()
    assert mailbox.inbox_for("nobody") == []


def test_on_event_fires_on_send():
    events = []
    mailbox = Mailbox(on_event=events.append)
    mailbox.send("a", "b", "msg", {"x": 1, "y": 2})
    assert len(events) == 1
    assert events[0]["type"] == "message_sent"
    assert events[0]["payload"]["sender"] == "a"
    assert events[0]["payload"]["recipient"] == "b"
    assert events[0]["payload"]["payload_keys"] == ["x", "y"]
    assert "payload" not in events[0]["payload"]  # full values never logged, only keys


def test_inbox_for_filters_by_message_type_leaves_others_queued():
    mailbox = Mailbox()
    mailbox.send("task_prioritization", "resource_allocation", "task_ranking", {"ranked_task_ids": ["T1"]})
    mailbox.send("failure_recovery", "resource_allocation", "reroute_request", {"reroute_proposals": []})

    rankings = mailbox.inbox_for("resource_allocation", message_type="task_ranking")
    assert len(rankings) == 1
    assert rankings[0].message_type == "task_ranking"

    # the reroute_request must NOT have been discarded by the filtered pop above
    still_queued = mailbox.peek("resource_allocation")
    assert len(still_queued) == 1
    assert still_queued[0].message_type == "reroute_request"

    reroutes = mailbox.inbox_for("resource_allocation", message_type="reroute_request")
    assert len(reroutes) == 1
    assert mailbox.peek("resource_allocation") == []


def test_inbox_for_no_type_filter_still_pops_everything():
    mailbox = Mailbox()
    mailbox.send("a", "b", "type_x", {})
    mailbox.send("a", "b", "type_y", {})
    assert len(mailbox.inbox_for("b")) == 2
    assert mailbox.peek("b") == []


def test_messages_preserve_send_order():
    mailbox = Mailbox()
    mailbox.send("a", "b", "msg", {"n": 1})
    mailbox.send("a", "b", "msg", {"n": 2})
    inbox = mailbox.inbox_for("b")
    assert [m.payload["n"] for m in inbox] == [1, 2]
