"""Tests for the Report Writer.

The report is the artefact an auditor reads, so these check that it actually
contains the decision record - verdict, who approved, why, and the hash chain -
rather than merely rendering without error.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from mavia.agents.report import ReportWriter, _esc, render_pdf, summarise_for_dashboard
from mavia.audit import AuditLogger, verify_chain
from mavia.config import Settings
from mavia.schemas import (
    AgentName,
    ApprovalDecision,
    ApprovalStatus,
    BoundingBox,
    HistoricalCase,
    InspectionState,
    RetrievalResult,
    RiskLevel,
    RootCauseAnalysis,
    Verdict,
    VisionResult,
    utc_now,
)


def _state(
    *,
    approval: ApprovalDecision | None = None,
    risk: RiskLevel = RiskLevel.CRITICAL,
    with_analysis: bool = True,
    with_history: bool = True,
    errors: list[str] | None = None,
) -> InspectionState:
    return InspectionState(
        inspection_id="insp_test123456",
        image_path="/data/bottle/test/broken_large/000.png",
        category="bottle",
        vision=VisionResult(
            category="bottle",
            verdict=Verdict.FAIL,
            anomaly_score=0.813,
            decision_threshold=0.5,
            regions=[
                BoundingBox(x=72, y=53, width=146, height=171, area_px=13179, peak_score=0.94)
            ],
            overlay_path="/artifacts/inspections/bottle/000_overlay.png",
            model_name="PatchCore/WideResNet50-2/bottle",
            latency_ms=54.0,
        ),
        retrieval=RetrievalResult(
            query_text="Visual inspection of a bottle unit...",
            cases=[
                HistoricalCase(
                    case_id="bottle-broken_large-000",
                    category="bottle",
                    defect_type="broken_large",
                    root_cause="Thermal shock in the annealing lehr",
                    action_taken="Re-profile the lehr temperature curve",
                    outcome="Verified effective",
                    occurred_at=datetime(2026, 3, 1, tzinfo=UTC),
                    similarity=0.97,
                )
            ]
            if with_history
            else [],
            top_k=3,
            latency_ms=9.0,
            embedding_model="all-MiniLM-L6-v2",
        ),
        analysis=RootCauseAnalysis(
            root_cause="Thermal shock during annealing",
            evidence=["Anomaly score 0.813", "13179 px affected"],
            recommended_action="Re-profile the annealing lehr temperature curve",
            risk_level=risk,
            confidence=0.88,
            affected_process_step="glass forming / annealing",
            cited_case_ids=["bottle-broken_large-000"],
            latency_ms=1800.0,
            model_name="claude-opus-5",
        )
        if with_analysis
        else None,
        approval=approval
        or ApprovalDecision(
            status=ApprovalStatus.APPROVED,
            required_because="risk_level=CRITICAL",
            approver="qa.lead@plant",
            rationale="Confirmed fracture; line halted",
            decided_at=utc_now(),
        ),
        errors=errors or [],
        total_latency_ms=54321.0,
    )


def _writer(tmp_path: Path) -> tuple[ReportWriter, AuditLogger]:
    audit = AuditLogger(tmp_path / "audit.jsonl")
    for action in ("inspection_started", "started", "completed"):
        audit.log(inspection_id="insp_test123456", agent=AgentName.VISION, action=action)
    settings = Settings(artifacts_dir=tmp_path)
    return ReportWriter(settings=settings, audit=audit), audit


# ------------------------------------------------------------------ markdown


def test_markdown_leads_with_the_verdict(tmp_path: Path) -> None:
    writer, audit = _writer(tmp_path)
    markdown = writer.build_markdown(_state(), audit.events_for("insp_test123456"))

    assert markdown.startswith("# Quality Inspection Report")
    assert "**FAIL**" in markdown
    assert "**CRITICAL**" in markdown
    assert "APPROVED" in markdown


def test_markdown_records_who_approved_and_why(tmp_path: Path) -> None:
    """An audited QA document is worthless without attribution."""
    writer, audit = _writer(tmp_path)
    markdown = writer.build_markdown(_state(), audit.events_for("insp_test123456"))

    assert "qa.lead@plant" in markdown
    assert "Confirmed fracture; line halted" in markdown
    assert "risk_level=CRITICAL" in markdown


def test_markdown_carries_the_detection_evidence(tmp_path: Path) -> None:
    writer, audit = _writer(tmp_path)
    markdown = writer.build_markdown(_state(), audit.events_for("insp_test123456"))

    assert "0.813" in markdown
    assert "13179" in markdown
    assert "146x171" in markdown
    assert "PatchCore/WideResNet50-2/bottle" in markdown


def test_markdown_cites_the_retrieved_history(tmp_path: Path) -> None:
    writer, audit = _writer(tmp_path)
    markdown = writer.build_markdown(_state(), audit.events_for("insp_test123456"))

    assert "bottle-broken_large-000" in markdown
    assert "Thermal shock in the annealing lehr" in markdown


def test_markdown_embeds_the_audit_chain(tmp_path: Path) -> None:
    writer, audit = _writer(tmp_path)
    events = audit.events_for("insp_test123456")
    markdown = writer.build_markdown(_state(), events)

    assert "Audit trail" in markdown
    for event in events:
        assert event.entry_hash[:16] in markdown
    assert events[-1].entry_hash in markdown, "the full chain head must be quoted"


def test_markdown_states_when_no_human_was_needed(tmp_path: Path) -> None:
    writer, audit = _writer(tmp_path)
    state = _state(
        approval=ApprovalDecision(status=ApprovalStatus.NOT_REQUIRED), risk=RiskLevel.LOW
    )
    markdown = writer.build_markdown(state, audit.events_for("insp_test123456"))

    assert "below the escalation threshold" in markdown
    assert "proceeded" in markdown


def test_markdown_handles_a_missing_analysis(tmp_path: Path) -> None:
    writer, audit = _writer(tmp_path)
    markdown = writer.build_markdown(
        _state(with_analysis=False, with_history=False), audit.events_for("insp_test123456")
    )

    assert "No analysis was produced" in markdown
    assert "No comparable historical cases" in markdown


def test_markdown_surfaces_recorded_errors(tmp_path: Path) -> None:
    writer, audit = _writer(tmp_path)
    markdown = writer.build_markdown(
        _state(errors=["vision_inspector: camera offline"]), audit.events_for("insp_test123456")
    )
    assert "Recorded errors" in markdown
    assert "camera offline" in markdown


def test_report_is_labelled_as_decision_support(tmp_path: Path) -> None:
    """The limitation must travel with the document, not just the README."""
    writer, audit = _writer(tmp_path)
    markdown = writer.build_markdown(_state(), audit.events_for("insp_test123456"))
    assert "decision support" in markdown
    assert "not a confirmed finding" in markdown


# ---------------------------------------------------------------------- html


def test_html_is_standalone_and_styled(tmp_path: Path) -> None:
    writer, audit = _writer(tmp_path)
    html = writer.build_html(_state(), audit.events_for("insp_test123456"))

    assert html.startswith("<!doctype html>")
    assert "<style>" in html
    assert "@page" in html, "PDF page setup must be present"
    assert html.rstrip().endswith("</html>")


def test_html_escapes_model_output(tmp_path: Path) -> None:
    """Analysis text comes from an LLM; it must not be able to inject markup."""
    writer, audit = _writer(tmp_path)
    state = _state()
    assert state.analysis is not None
    state.analysis.root_cause = "<script>alert('xss')</script> & \"quoted\""

    html = writer.build_html(state, audit.events_for("insp_test123456"))

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&amp;" in html


def test_escape_helper_handles_types() -> None:
    assert _esc("a<b>c") == "a&lt;b&gt;c"
    assert _esc(42) == "42"
    assert "2026" in _esc(datetime(2026, 1, 1, tzinfo=UTC))


# --------------------------------------------------------------------- write


def test_write_produces_markdown_and_html(tmp_path: Path) -> None:
    writer, _ = _writer(tmp_path)
    report = writer.write(_state())

    directory = tmp_path / "reports"
    assert (directory / "insp_test123456.md").exists()
    assert (directory / "insp_test123456.html").exists()
    assert report.markdown
    assert report.report_id.startswith("rpt_")


def test_write_audits_the_report_and_binds_it_to_the_chain(tmp_path: Path) -> None:
    writer, audit = _writer(tmp_path)
    writer.write(_state())

    written = [e for e in audit.events_for("insp_test123456") if e.action == "report_written"]
    assert written, "writing a report must itself be audited"

    payload = written[0].payload
    assert payload["audit_events_included"] >= 3
    assert payload["chain_head"], "the report must be bound to the chain state it describes"
    assert verify_chain(tmp_path / "audit.jsonl")


def test_summary_is_actionable(tmp_path: Path) -> None:
    writer, _ = _writer(tmp_path)
    report = writer.write(_state())

    assert "bottle" in report.summary
    assert "FAIL" in report.summary
    assert "Recommended" in report.summary


def test_rejected_inspection_reports_inconclusive(tmp_path: Path) -> None:
    writer, audit = _writer(tmp_path)
    state = _state(
        approval=ApprovalDecision(
            status=ApprovalStatus.REJECTED, approver="qa.lead", rationale="disagree"
        )
    )
    markdown = writer.build_markdown(state, audit.events_for("insp_test123456"))

    assert "INCONCLUSIVE" in markdown
    assert "REJECTED" in markdown


# ----------------------------------------------------------------------- pdf


def test_pdf_failure_degrades_to_html(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing system library must not fail the inspection."""
    import builtins

    real_import = builtins.__import__

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("weasyprint"):
            raise ImportError("libpango not found")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.delitem(__import__("sys").modules, "weasyprint", raising=False)

    assert render_pdf("<html><body>x</body></html>", tmp_path / "out.pdf") is None

    writer, _ = _writer(tmp_path)
    report = writer.write(_state())
    assert report.pdf_path is not None
    assert report.pdf_path.endswith(".html"), "should fall back to the HTML document"


@pytest.mark.slow
def test_pdf_renders_when_the_toolchain_is_available(tmp_path: Path) -> None:
    pytest.importorskip("weasyprint")
    writer, _ = _writer(tmp_path)
    report = writer.write(_state())

    if report.pdf_path and report.pdf_path.endswith(".pdf"):
        pdf = Path(report.pdf_path)
        assert pdf.exists()
        assert pdf.stat().st_size > 1000
        assert pdf.read_bytes().startswith(b"%PDF")


# ----------------------------------------------------------------- dashboard


def test_dashboard_summary_is_flat_and_complete() -> None:
    summary = summarise_for_dashboard(_state())
    assert summary["inspection_id"] == "insp_test123456"
    assert summary["verdict"] == "FAIL"
    assert summary["risk_level"] == "CRITICAL"
    assert summary["approval"] == "APPROVED"
    assert summary["anomaly_score"] == pytest.approx(0.813)
