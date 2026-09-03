"""Tamper-evident audit trail.

Every agent decision is appended to a JSONL log as a link in a SHA-256 hash chain:

    entry_hash = SHA256(seq | inspection_id | timestamp | agent | action | payload_hash | prev_hash)

Because each entry commits to the previous entry's hash, editing or deleting any
historical record invalidates every entry after it, which ``verify_chain`` detects.
This is the governance property enterprise QA audits ask for: you cannot silently
rewrite what the system decided or when.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from mavia.schemas import AgentName, AuditEvent, utc_now

GENESIS_HASH = "0" * 64


def _canonical_json(payload: Any) -> str:
    """Deterministic JSON so the same payload always hashes to the same digest."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_entry_hash(
    *,
    seq: int,
    inspection_id: str,
    timestamp: datetime,
    agent: str,
    action: str,
    payload_hash: str,
    prev_hash: str,
) -> str:
    material = "|".join(
        [
            str(seq),
            inspection_id,
            timestamp.isoformat(),
            agent,
            action,
            payload_hash,
            prev_hash,
        ]
    )
    return _sha256(material)


def read_events(log_path: Path) -> Iterator[AuditEvent]:
    """Stream every audit event from a JSONL log."""
    path = Path(log_path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield AuditEvent.model_validate_json(line)


class AuditLogger:
    """Append-only hash-chained log backed by a JSONL file."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._seq, self._head_hash = self._resume()

    def _resume(self) -> tuple[int, str]:
        last: AuditEvent | None = None
        for last in self.read_all():  # noqa: B007 - we only want the final element
            pass
        if last is None:
            return 0, GENESIS_HASH
        return last.seq + 1, last.entry_hash

    @property
    def head_hash(self) -> str:
        return self._head_hash

    def log(
        self,
        *,
        inspection_id: str,
        agent: AgentName | str,
        action: str,
        payload: dict[str, Any] | None = None,
    ) -> AuditEvent:
        payload = payload or {}
        agent_value = str(agent)
        timestamp = utc_now()
        payload_hash = _sha256(_canonical_json(payload))
        entry_hash = compute_entry_hash(
            seq=self._seq,
            inspection_id=inspection_id,
            timestamp=timestamp,
            agent=agent_value,
            action=action,
            payload_hash=payload_hash,
            prev_hash=self._head_hash,
        )
        event = AuditEvent(
            seq=self._seq,
            inspection_id=inspection_id,
            timestamp=timestamp,
            agent=agent_value,
            action=action,
            payload=payload,
            payload_hash=payload_hash,
            prev_hash=self._head_hash,
            entry_hash=entry_hash,
        )
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")
        self._seq += 1
        self._head_hash = entry_hash
        return event

    def read_all(self) -> Iterator[AuditEvent]:
        return read_events(self.log_path)

    def events_for(self, inspection_id: str) -> list[AuditEvent]:
        return [e for e in self.read_all() if e.inspection_id == inspection_id]


@dataclass(frozen=True)
class ChainVerification:
    """Result of a chain integrity check; truthy when the chain is intact."""

    valid: bool
    checked: int
    broken_at: int | None = None

    def __bool__(self) -> bool:
        return self.valid


def verify_chain(log_path: Path) -> ChainVerification:
    """Re-derive every hash and confirm the chain is unbroken.

    ``broken_at`` carries the seq of the first entry that fails, or None.
    """
    prev_hash = GENESIS_HASH
    checked = 0

    for expected_seq, event in enumerate(read_events(log_path)):
        payload_hash = _sha256(_canonical_json(event.payload))
        entry_hash = compute_entry_hash(
            seq=event.seq,
            inspection_id=event.inspection_id,
            timestamp=event.timestamp,
            agent=str(event.agent),
            action=event.action,
            payload_hash=payload_hash,
            prev_hash=prev_hash,
        )
        broken = (
            event.seq != expected_seq
            or event.prev_hash != prev_hash
            or event.payload_hash != payload_hash
            or event.entry_hash != entry_hash
        )
        if broken:
            return ChainVerification(valid=False, checked=checked, broken_at=event.seq)
        prev_hash = event.entry_hash
        checked += 1

    return ChainVerification(valid=True, checked=checked)
