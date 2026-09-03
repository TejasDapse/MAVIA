import json
from itertools import pairwise
from pathlib import Path

from mavia.audit import GENESIS_HASH, AuditLogger, verify_chain
from mavia.schemas import AgentName


def _seed(log_path: Path, n: int = 3) -> AuditLogger:
    logger = AuditLogger(log_path)
    for i in range(n):
        logger.log(
            inspection_id="insp_test",
            agent=AgentName.VISION,
            action=f"step_{i}",
            payload={"anomaly_score": 0.1 * i, "category": "bottle"},
        )
    return logger


def test_first_entry_links_to_genesis(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    event = logger.log(inspection_id="i1", agent=AgentName.VISION, action="detect")
    assert event.seq == 0
    assert event.prev_hash == GENESIS_HASH
    assert len(event.entry_hash) == 64


def test_entries_form_a_chain(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    _seed(log_path, n=4)
    events = list(AuditLogger(log_path).read_all())
    assert [e.seq for e in events] == [0, 1, 2, 3]
    for prev, current in pairwise(events):
        assert current.prev_hash == prev.entry_hash


def test_verify_chain_accepts_untampered_log(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    _seed(log_path, n=5)
    result = verify_chain(log_path)
    assert result
    assert result.checked == 5
    assert result.broken_at is None


def test_verify_chain_detects_payload_tampering(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    _seed(log_path, n=4)

    lines = log_path.read_text().splitlines()
    record = json.loads(lines[2])
    record["payload"]["anomaly_score"] = 0.0  # someone hides a defect after the fact
    lines[2] = json.dumps(record)
    log_path.write_text("\n".join(lines) + "\n")

    result = verify_chain(log_path)
    assert not result
    assert result.broken_at == 2


def test_verify_chain_detects_deleted_entry(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    _seed(log_path, n=4)

    lines = log_path.read_text().splitlines()
    del lines[1]
    log_path.write_text("\n".join(lines) + "\n")

    assert not verify_chain(log_path)


def test_logger_resumes_chain_across_restarts(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    first = _seed(log_path, n=2)
    head = first.head_hash

    reopened = AuditLogger(log_path)
    event = reopened.log(inspection_id="insp_test", agent=AgentName.REPORTER, action="report")

    assert event.seq == 2
    assert event.prev_hash == head
    assert verify_chain(log_path)


def test_events_for_filters_by_inspection(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    logger.log(inspection_id="a", agent=AgentName.VISION, action="detect")
    logger.log(inspection_id="b", agent=AgentName.VISION, action="detect")
    logger.log(inspection_id="a", agent=AgentName.REPORTER, action="report")

    assert [e.action for e in logger.events_for("a")] == ["detect", "report"]
