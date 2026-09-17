"""
Minimal in-process agent-to-agent message bus, plus a disk-backed
variant for the sandbox port (PersistentMailbox, below).

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

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
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


# Default local root when nothing overrides it -- deliberately NOT under
# RESOURCE_SCHEDULER_DATA_ROOT/runs (see resource_scheduler/paths.py):
# that root is a shared, per-owner Docker volume in the sandbox deployment
# (/scratch), kept across every pipeline run an owner ever starts. The
# mailbox's real isolation boundary is meant to be a Tier-1 EnvironmentSpec
# session container's own private filesystem instead (see
# gate_adapters.py's module docstring and
# agent-sandbox/docs/persistent-environment-spec.md) -- a path outside any
# mounted volume, torn down automatically with the session. This constant
# is where that container-local root lives absent an override; it also
# doubles as a perfectly fine default for local/non-Docker runs, where it
# just persists for the life of the local process/filesystem.
_SESSION_STATE_ROOT_ENV = "RESOURCE_SCHEDULER_SESSION_STATE_ROOT"
_DEFAULT_SESSION_STATE_ROOT = "/var/lib/resource-scheduler-session"


class PersistentMailbox:
    """Same send/inbox_for/peek contract as Mailbox, backed by JSON files
    on disk instead of an in-process list, so a message survives between
    separate gate executions -- each Tier-1 EnvironmentSpec session call
    (see gate_adapters.py) runs the gate entrypoint as a fresh process, so
    nothing kept only in a Python object would otherwise survive from one
    gate call to the next.

    One JSON file per (recipient, message_type), holding a list of
    Message-shaped dicts, under `root` (default: run_id-namespaced under
    RESOURCE_SCHEDULER_SESSION_STATE_ROOT, itself defaulting to a fixed
    local path outside any shared/mounted volume). inbox_for's pop
    semantics are preserved by rewriting (or removing) the file after
    consuming. Each send is also emitted as an event via on_event, same
    as Mailbox already does.

    run_id namespacing is defense-in-depth, not the primary isolation
    mechanism: in the sandbox deployment, every gate step that touches the
    mailbox shares one Tier-1 session container scoped to exactly one
    pipeline run, so in practice there is only ever one run_id's worth of
    files under this root at a time. Keeping run_id in the path anyway
    costs nothing and keeps this class correct standalone too -- e.g. in
    a non-Docker/in-process deployment (or a local test run), where
    several pipeline runs can share one process/filesystem with no
    container boundary between them at all."""

    def __init__(self, run_id: str, on_event: Optional[Callable[[dict], None]] = None, root: Optional[Path] = None):
        self.run_id = run_id
        self._on_event = on_event
        base = root or Path(os.environ.get(_SESSION_STATE_ROOT_ENV, _DEFAULT_SESSION_STATE_ROOT))
        self._root = base / run_id / "mailbox"

    def _queue_path(self, recipient: str, message_type: str) -> Path:
        return self._root / f"{recipient}__{message_type}.json"

    def _read_queue(self, path: Path) -> list[Message]:
        if not path.exists():
            return []
        raw = json.loads(path.read_text())
        return [Message(**entry) for entry in raw]

    def _write_queue(self, path: Path, messages: list[Message]) -> None:
        if not messages:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(m) for m in messages]))

    def send(self, sender: str, recipient: str, message_type: str, payload: dict) -> Message:
        msg = Message(sender=sender, recipient=recipient, message_type=message_type, payload=payload)
        path = self._queue_path(recipient, message_type)
        queue = self._read_queue(path)
        queue.append(msg)
        self._write_queue(path, queue)
        if self._on_event:
            self._on_event({"phase": "a2a", "type": "message_sent", "payload": msg.to_event()})
        return msg

    def inbox_for(self, recipient: str, message_type: Optional[str] = None) -> list[Message]:
        """Same filtered-pop semantics as Mailbox.inbox_for. With an
        explicit message_type (the only way this project's own callers
        ever use it -- resource_allocation's inbox carries both
        task_ranking and reroute_request, human_oversight's carries both
        policy_update_proposal and risky_decision), only that one file is
        touched; a different message type addressed to the same recipient
        lives in its own file and is untouched. Without a message_type,
        every file for this recipient is popped and merged, oldest-file-
        order not guaranteed -- callers in this project always pass one."""
        if message_type is not None:
            path = self._queue_path(recipient, message_type)
            queue = self._read_queue(path)
            self._write_queue(path, [])
            return queue

        if not self._root.is_dir():
            return []
        prefix = f"{recipient}__"
        all_messages: list[Message] = []
        for path in self._root.glob(f"{prefix}*.json"):
            all_messages.extend(self._read_queue(path))
            path.unlink(missing_ok=True)
        return all_messages

    def peek(self, recipient: str) -> list[Message]:
        if not self._root.is_dir():
            return []
        prefix = f"{recipient}__"
        all_messages: list[Message] = []
        for path in self._root.glob(f"{prefix}*.json"):
            all_messages.extend(self._read_queue(path))
        return all_messages
