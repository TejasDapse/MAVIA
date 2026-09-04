"""Tests for the dashboard data layer.

The interesting claim being tested is that an inspection's whole story can be
reconstructed from the audit log alone. If it cannot, the audit trail is not
actually a complete record - so the dashboard reads nothing else, and these
tests would catch it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mavia.audit import AuditLogger
from mavia.config import Settings
from mavia.dashboard import service
from mavia.schemas import AgentName


def _settings(tmp_path: Path) -> Settings:
    return Settings(artifacts_dir=tmp_path)


def _log_full_inspection(
    logger: AuditLogger,
    inspection_id: str,
    *,
    category: str = "bottle",
    verdict: str = "FAIL",
    risk: str = "CRITICAL",
    approval: str = "APPROVED",
    approver: str = "qa.lead@plant",
    score: float = 0.81,
    latency: float = 5400.0,
) -> None:
    logger.log(
        inspection_id=inspection_id,
        agent="pipeline",
        action="inspection_started",
        payload={"image_path": f"/data/{category}/test/x/000.png", "category": category},
    )
    logger.log(
        inspection_id=inspection_id,
        agent=AgentName.VISION,
        action="completed",
        payload={"vision": {"anomaly_score": score, "verdict": verdict}},
    )
    logger.log(
        inspection_id=inspection_id,
        agent=AgentName.ANALYST,
        action="completed",
        payload={"analysis": {"risk_level": risk, "confidence": 0.8}},
    )
    logger.log(
        inspection_id=inspection_id,
        agent=AgentName.APPROVAL,
        action=f"approval_{approval.lower()}",
        payload={"approver": approver, "rationale": "confirmed"},
    )
    logger.log(
        inspection_id=inspection_id,
        agent="report_writer",
        action="report_written",
        payload={"report_id": "rpt_abc123", "format": "pdf"},
    )
    logger.log(
        inspection_id=inspection_id,
        agent="pipeline",
        action="finalised",
        payload={
            "verdict": verdict,
            "approval": approval,
            "total_latency_ms": latency,
            "errors": [],
        },
    )


# ------------------------------------------------------- reconstruction


def test_inspection_is_reconstructed_from_the_audit_log_alone(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path / "audit" / "audit_log.jsonl")
    _log_full_inspection(logger, "insp_a")

    summary = service.load_inspections(_settings(tmp_path))[0]

    assert summary.inspection_id == "insp_a"
    assert summary.category == "bottle"
    assert summary.verdict == "FAIL"
    assert summary.risk_level == "CRITICAL"
    assert summary.approval == "APPROVED"
    assert summary.approver == "qa.lead@plant"
    assert summary.anomaly_score == 0.81
    assert summary.total_latency_ms == 5400.0
    assert summary.report_id == "rpt_abc123"
    assert summary.is_complete


def test_incomplete_inspection_is_marked_awaiting_approval(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path / "audit" / "audit_log.jsonl")
    logger.log(
        inspection_id="insp_b",
        agent="pipeline",
        action="inspection_started",
        payload={"category": "tile"},
    )
    logger.log(
        inspection_id="insp_b",
        agent=AgentName.ANALYST,
        action="completed",
        payload={"analysis": {"risk_level": "HIGH"}},
    )

    summary = service.load_inspections(_settings(tmp_path))[0]

    assert not summary.is_complete
    assert summary.awaiting_approval
    assert summary.verdict is None


def test_node_failures_are_surfaced(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path / "audit" / "audit_log.jsonl")
    logger.log(
        inspection_id="insp_c",
        agent="pipeline",
        action="inspection_started",
        payload={"category": "wood"},
    )
    logger.log(
        inspection_id="insp_c",
        agent=AgentName.VISION,
        action="failed",
        payload={"error": "camera offline"},
    )

    summary = service.load_inspections(_settings(tmp_path))[0]
    assert "camera offline" in summary.errors


def test_inspections_are_newest_first(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path / "audit" / "audit_log.jsonl")
    for name in ("insp_1", "insp_2", "insp_3"):
        _log_full_inspection(logger, name)

    summaries = service.load_inspections(_settings(tmp_path))
    assert [s.inspection_id for s in summaries] == ["insp_3", "insp_2", "insp_1"]


def test_limit_is_respected(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path / "audit" / "audit_log.jsonl")
    for name in ("a", "b", "c", "d"):
        _log_full_inspection(logger, name)
    assert len(service.load_inspections(_settings(tmp_path), limit=2)) == 2


def test_empty_log_yields_no_inspections(tmp_path: Path) -> None:
    assert service.load_inspections(_settings(tmp_path)) == []


# ------------------------------------------------------------- metrics


def test_metrics_compute_defect_and_escalation_rates(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path / "audit" / "audit_log.jsonl")
    _log_full_inspection(logger, "f1", verdict="FAIL", approval="APPROVED")
    _log_full_inspection(logger, "f2", verdict="FAIL", approval="REJECTED")
    _log_full_inspection(
        logger, "p1", verdict="PASS", risk="LOW", approval="NOT_REQUIRED", latency=900.0
    )
    _log_full_inspection(
        logger, "p2", verdict="PASS", risk="LOW", approval="NOT_REQUIRED", latency=1100.0
    )

    metrics = service.fleet_metrics(service.load_inspections(_settings(tmp_path)))

    assert metrics["total"] == 4
    assert metrics["completed"] == 4
    assert metrics["failed"] == 2
    assert metrics["defect_rate"] == 0.5
    assert metrics["escalated"] == 2
    assert metrics["escalation_rate"] == 0.5
    assert metrics["risk_breakdown"]["CRITICAL"] == 2


def test_median_latency_handles_even_and_odd_counts(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path / "audit" / "audit_log.jsonl")
    for index, latency in enumerate([1000.0, 2000.0, 3000.0, 4000.0]):
        _log_full_inspection(logger, f"i{index}", latency=latency)

    metrics = service.fleet_metrics(service.load_inspections(_settings(tmp_path)))
    assert metrics["median_latency_ms"] == 2500.0


def test_metrics_on_an_empty_fleet_do_not_divide_by_zero() -> None:
    metrics = service.fleet_metrics([])
    assert metrics["total"] == 0
    assert metrics["defect_rate"] == 0.0
    assert metrics["median_latency_ms"] is None


# --------------------------------------------------------------- chain


def test_chain_status_reports_valid(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path / "audit" / "audit_log.jsonl")
    _log_full_inspection(logger, "insp_ok")

    status = service.chain_status(_settings(tmp_path))
    assert status["valid"]
    assert status["checked"] == 6
    assert status["broken_at"] is None


def test_chain_status_reports_tampering(tmp_path: Path) -> None:
    """The dashboard must show a broken chain, not quietly render stale data."""
    import json

    path = tmp_path / "audit" / "audit_log.jsonl"
    logger = AuditLogger(path)
    _log_full_inspection(logger, "insp_bad")

    lines = path.read_text().splitlines()
    record = json.loads(lines[3])
    record["payload"]["approver"] = "someone.else"
    lines[3] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n")

    status = service.chain_status(_settings(tmp_path))
    assert not status["valid"]
    assert status["broken_at"] == 3


# --------------------------------------------------------------- reports


def test_report_paths_finds_what_exists(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "insp_x.md").write_text("# report")
    (reports / "insp_x.pdf").write_bytes(b"%PDF-1.7")

    found = service.report_paths("insp_x", _settings(tmp_path))
    assert set(found) == {"md", "pdf"}
    assert found["md"].read_text() == "# report"


def test_report_paths_empty_when_none_written(tmp_path: Path) -> None:
    assert service.report_paths("insp_missing", _settings(tmp_path)) == {}


# --------------------------------------------------------------- picker


def test_describe_image_is_readable() -> None:
    path = Path("/data/mvtec_ad/bottle/test/broken_large/000.png")
    assert service.describe_image(path) == "bottle / broken_large / 000.png"


def test_available_images_is_empty_without_a_dataset(tmp_path: Path) -> None:
    settings = Settings(artifacts_dir=tmp_path, data_dir=tmp_path / "nothing")
    assert service.available_images(settings) == []


# ----------------------------------------------------------- app smoke test


@pytest.mark.slow
def test_every_view_renders_without_exceptions() -> None:
    """Streamlit surfaces script errors as page exceptions rather than crashing,
    so a view can be silently broken. This executes all four headlessly."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = Path(__file__).resolve().parents[1] / "src" / "mavia" / "dashboard" / "app.py"

    for view in ("Inspect", "Approval queue", "History", "Audit trail"):
        harness = AppTest.from_file(str(app), default_timeout=120)
        harness.run()
        harness.sidebar.radio[0].set_value(view).run()
        assert not harness.exception, f"{view}: {[str(e.value) for e in harness.exception]}"
        assert harness.title, f"{view} rendered no title"
