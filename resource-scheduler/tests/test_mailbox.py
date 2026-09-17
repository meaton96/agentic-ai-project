"""
Tests for a2a/mailbox.py -- the in-process A2A message bus, and its
disk-backed PersistentMailbox counterpart for the sandbox port (see
gate_adapters.py's module docstring). No LLM involved; this only needs
to prove the mechanism itself is correct before any agent's real output
flows through it.
"""
from resource_scheduler.a2a.mailbox import Mailbox, PersistentMailbox


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


# -- PersistentMailbox: same battery, plus the property it actually adds --


def test_persistent_send_then_inbox_for_returns_the_message(tmp_path):
    mailbox = PersistentMailbox("run_1", root=tmp_path)
    mailbox.send("task_prioritization", "resource_allocation", "task_ranking", {"ranked_task_ids": ["T1"]})
    inbox = mailbox.inbox_for("resource_allocation", message_type="task_ranking")
    assert len(inbox) == 1
    assert inbox[0].sender == "task_prioritization"
    assert inbox[0].message_type == "task_ranking"
    assert inbox[0].payload == {"ranked_task_ids": ["T1"]}


def test_persistent_inbox_for_consumes_the_queue(tmp_path):
    mailbox = PersistentMailbox("run_1", root=tmp_path)
    mailbox.send("a", "b", "msg", {})
    mailbox.inbox_for("b", message_type="msg")
    assert mailbox.inbox_for("b", message_type="msg") == []


def test_persistent_peek_does_not_consume(tmp_path):
    mailbox = PersistentMailbox("run_1", root=tmp_path)
    mailbox.send("a", "b", "msg", {})
    assert len(mailbox.peek("b")) == 1
    assert len(mailbox.peek("b")) == 1  # still there
    assert len(mailbox.inbox_for("b", message_type="msg")) == 1  # and still consumable


def test_persistent_inbox_for_filters_by_message_type_leaves_others_queued(tmp_path):
    mailbox = PersistentMailbox("run_1", root=tmp_path)
    mailbox.send("task_prioritization", "resource_allocation", "task_ranking", {"ranked_task_ids": ["T1"]})
    mailbox.send("failure_recovery", "resource_allocation", "reroute_request", {"reroute_proposals": []})

    rankings = mailbox.inbox_for("resource_allocation", message_type="task_ranking")
    assert len(rankings) == 1
    assert rankings[0].message_type == "task_ranking"

    still_queued = mailbox.peek("resource_allocation")
    assert len(still_queued) == 1
    assert still_queued[0].message_type == "reroute_request"

    reroutes = mailbox.inbox_for("resource_allocation", message_type="reroute_request")
    assert len(reroutes) == 1
    assert mailbox.peek("resource_allocation") == []


def test_persistent_on_event_fires_on_send(tmp_path):
    events = []
    mailbox = PersistentMailbox("run_1", on_event=events.append, root=tmp_path)
    mailbox.send("a", "b", "msg", {"x": 1, "y": 2})
    assert len(events) == 1
    assert events[0]["type"] == "message_sent"
    assert events[0]["payload"]["sender"] == "a"
    assert events[0]["payload"]["payload_keys"] == ["x", "y"]
    assert "payload" not in events[0]["payload"]


def test_persistent_mailbox_survives_a_fresh_instance_same_root(tmp_path):
    """The actual property PersistentMailbox exists to add: state written
    by one instance (standing in for one gate call/process) is visible to
    a brand new instance constructed later against the same root
    (standing in for a LATER, separate gate call into the same Tier-1
    session container) -- Mailbox itself (in-process) could never do
    this."""
    writer = PersistentMailbox("run_1", root=tmp_path)
    writer.send("task_prioritization", "resource_allocation", "task_ranking", {"ranked_task_ids": ["T1", "T2"]})

    reader = PersistentMailbox("run_1", root=tmp_path)
    inbox = reader.inbox_for("resource_allocation", message_type="task_ranking")
    assert len(inbox) == 1
    assert inbox[0].payload == {"ranked_task_ids": ["T1", "T2"]}

    # and it's really consumed -- a third instance sees nothing left
    assert PersistentMailbox("run_1", root=tmp_path).inbox_for("resource_allocation", message_type="task_ranking") == []


def test_persistent_mailbox_isolates_different_run_ids(tmp_path):
    PersistentMailbox("run_a", root=tmp_path).send("x", "human_oversight", "policy_update_proposal", {"n": 1})
    PersistentMailbox("run_b", root=tmp_path).send("x", "human_oversight", "policy_update_proposal", {"n": 2})

    a = PersistentMailbox("run_a", root=tmp_path).inbox_for("human_oversight", message_type="policy_update_proposal")
    b = PersistentMailbox("run_b", root=tmp_path).inbox_for("human_oversight", message_type="policy_update_proposal")
    assert [m.payload["n"] for m in a] == [1]
    assert [m.payload["n"] for m in b] == [2]
