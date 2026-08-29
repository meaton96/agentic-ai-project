"""
Minimal in-process agent-to-agent message bus.

This is the piece explicitly called out as the thing worth actually
testing in Use Case II that the ML classification pipeline never
exercises: an agent handing a structured proposal directly to another
agent, not only back through a central orchestrator. Kept deliberately
dumb for the MVP -- no async, no network, no persistence beyond one run
-- so the interesting question (does giving agents a direct channel
change behavior or failure modes versus routing everything through a
deterministic gate) can be tested without also debugging message-bus
infrastructure.

"Direct" does not mean "unaudited": every send is still logged as an
event the same way an allocation decision is (see Message.to_event),
and on_event, if given, fires on every send.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Message:
    sender: str
    recipient: str
    message_type: str
    payload: dict
    ts: float = field(default_factory=time.time)

    def to_event(self) -> dict:
        """Deliberately excludes payload values, not just for size --
        logging exactly what was decided (payload_keys) without logging
        the full content mirrors how events.py's module docstring
        already restricts every other step's event payloads in this
        project. Full payloads still live in the transcript/report the
        caller writes, if it chooses to."""
        return {
            "sender": self.sender, "recipient": self.recipient,
            "message_type": self.message_type, "ts": self.ts,
            "payload_keys": sorted(self.payload.keys()),
        }


class Mailbox:
    """Per-run, in-process, shared by every agent step in that run.
    send() appends to the recipient's queue; inbox_for() reads AND
    clears it -- a recipient consumes its queue once, like a real
    message queue, rather than every reader replaying full history."""

    def __init__(self, on_event: Optional[Callable[[dict], None]] = None):
        self._queues: dict[str, list[Message]] = {}
        self._on_event = on_event

    def send(self, sender: str, recipient: str, message_type: str, payload: dict) -> Message:
        msg = Message(sender=sender, recipient=recipient, message_type=message_type, payload=payload)
        self._queues.setdefault(recipient, []).append(msg)
        if self._on_event:
            self._on_event({"phase": "a2a", "type": "message_sent", "payload": msg.to_event()})
        return msg

    def inbox_for(self, recipient: str, message_type: Optional[str] = None) -> list[Message]:
        """Reads AND clears the recipient's queue. If message_type is
        given, only messages of that type are removed and returned --
        messages of a different type addressed to the same recipient
        (e.g. Resource Allocation's inbox carries both "task_ranking"
        from Task Prioritization and "reroute_request" from Failure
        Recovery) are left in place for their own consumer. Without
        this, one message type's consumer popping the whole queue would
        silently discard a different, still-unread message type -- real
        risk once more than one sender shares a recipient's inbox."""
        queue = self._queues.get(recipient, [])
        if message_type is None:
            self._queues.pop(recipient, None)
            return queue
        matching = [m for m in queue if m.message_type == message_type]
        remaining = [m for m in queue if m.message_type != message_type]
        if remaining:
            self._queues[recipient] = remaining
        else:
            self._queues.pop(recipient, None)
        return matching

    def peek(self, recipient: str) -> list[Message]:
        """Non-consuming read, for tests/debugging -- does not clear the queue."""
        return list(self._queues.get(recipient, []))
