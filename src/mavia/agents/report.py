"""Agent 4 - Report Writer.

Renders a completed inspection into the QA record a plant actually files:
Markdown for the repo and the dashboard, PDF for the audit binder.

**The report is templated, not generated.** It would be easy to hand the whole
inspection to an LLM and ask for prose, and it would be a mistake. An audited
quality document needs a stable structure - the same sections in the same order
on every part, so a reviewer can find the verdict without reading, and so two
reports are diffable. The model's contribution belongs in the analysis fields it
already produced in Phase 4, which this template quotes verbatim. Nothing here
invents content, which is also why the report cannot hallucinate.

**The audit trail is part of the report.** Every report embeds the SHA-256 chain
head at time of writing plus the full event sequence for that inspection. A
reader can re-derive the chain from the log and confirm the document describes
what the system actually did, rather than trusting the PDF.

PDF rendering uses WeasyPrint, which needs pango. When that is unavailable the
writer degrades to styled HTML and says so, rather than failing - the same
principle applied everywhere else in the pipeline.
"""

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from mavia.audit import AuditLogger
from mavia.config import Settings, get_settings
from mavia.logging_setup import get_logger
from mavia.schemas import (
    ApprovalStatus,
    AuditEvent,
    InspectionState,
    QAReport,
    RiskLevel,
    Verdict,
    utc_now,
)

logger = get_logger(__name__)

# Homebrew prefixes where pango lives on macOS. WeasyPrint resolves libraries at
# import time, so this must be set before it is first imported.
_MACOS_LIB_PATHS = ("/opt/homebrew/lib", "/usr/local/lib")

RISK_STYLE = {
    RiskLevel.LOW: ("#0a7c42", "#e8f6ee"),
    RiskLevel.MEDIUM: ("#8a6100", "#fdf3e0"),
    RiskLevel.HIGH: ("#a33a00", "#fdeee6"),
    RiskLevel.CRITICAL: ("#a4161a", "#fdeaea"),
}

STYLESHEET = """
@page {
  size: A4; margin: 18mm 16mm;
  @bottom-center {
    content: "Page " counter(page) " of " counter(pages);
    font-size: 8pt; color: #6b7280;
  }
}
body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 10pt; color: #111827; line-height: 1.45;
}
h1 { font-size: 17pt; margin: 0 0 2mm; }
h2 {
  font-size: 11.5pt; margin: 7mm 0 2mm;
  padding-bottom: 1mm; border-bottom: 1px solid #d1d5db;
}
.sub { color: #6b7280; font-size: 8.5pt; margin-bottom: 4mm; }
table { width: 100%; border-collapse: collapse; margin: 2mm 0; }
th, td {
  text-align: left; padding: 1.6mm 2mm;
  border-bottom: 1px solid #e5e7eb; vertical-align: top;
}
th { width: 34%; color: #374151; font-weight: 600; }
.badge {
  display: inline-block; padding: 0.8mm 2.5mm;
  border-radius: 2mm; font-weight: 700; font-size: 9pt;
}
.mono {
  font-family: "SF Mono", Menlo, monospace;
  font-size: 7.5pt; word-break: break-all;
}
ul { margin: 1mm 0 1mm 4mm; padding: 0; }
.audit td { font-size: 8pt; padding: 1mm 2mm; }
.audit th { width: auto; font-size: 8pt; }
.note {
  background: #f9fafb; border-left: 2mm solid #d1d5db;
  padding: 2mm 3mm; font-size: 8.5pt; color: #374151;
}
"""


class ReportWriter:
    """Render an inspection into Markdown and PDF."""

    def __init__(self, settings: Settings | None = None, audit: AuditLogger | None = None) -> None:
        self.settings = settings or get_settings()
        self.audit = audit or AuditLogger(self.settings.audit_log_path)

    # ---------------------------------------------------------------- markdown

    def build_markdown(self, state: InspectionState, events: list[AuditEvent]) -> str:
        vision, analysis = state.vision, state.analysis
        lines: list[str] = [
            "# Quality Inspection Report",
            "",
            f"**Inspection ID:** `{state.inspection_id}`  ",
            f"**Generated:** {utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
            f"**Product category:** {state.category or 'unknown'}",
            "",
            "## 1. Verdict",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Final verdict | **{state.final_verdict.value}** |",
            f"| Approval status | {state.approval.status.value} |",
        ]
        if analysis:
            lines.append(f"| Risk level | **{analysis.risk_level.value}** |")
            lines.append(f"| Analysis confidence | {analysis.confidence:.2f} |")
        if state.total_latency_ms:
            lines.append(f"| End-to-end latency | {state.total_latency_ms:.0f} ms |")

        # --- detection
        lines += ["", "## 2. Detection", ""]
        if vision:
            lines += [
                "| Field | Value |",
                "|---|---|",
                f"| Detector | {vision.model_name} |",
                f"| Anomaly score | {vision.anomaly_score:.3f} "
                f"(threshold {vision.decision_threshold:.2f}) |",
                f"| Verdict | {vision.verdict.value} |",
                f"| Regions localised | {len(vision.regions)} |",
                f"| Inference latency | {vision.latency_ms:.0f} ms |",
            ]
            if vision.zero_shot_label:
                lines.append(
                    f"| Zero-shot fallback | {vision.zero_shot_label} "
                    f"({vision.zero_shot_confidence or 0:.2f}) - reduced confidence |"
                )
            if vision.regions:
                lines += [
                    "",
                    "| # | Position | Size | Area (px) | Peak severity |",
                    "|---|---|---|---|---|",
                ]
                for i, box in enumerate(vision.regions, start=1):
                    lines.append(
                        f"| {i} | ({box.x}, {box.y}) | {box.width}x{box.height} | "
                        f"{box.area_px} | {box.peak_score:.2f} |"
                    )
            if vision.overlay_path:
                lines += ["", f"Heatmap overlay: `{vision.overlay_path}`"]
        else:
            lines.append("_No detection was recorded._")

        # --- history
        lines += ["", "## 3. Historical context", ""]
        if state.retrieval and state.retrieval.cases:
            lines += [
                f"Retrieved {len(state.retrieval.cases)} comparable case(s) "
                f"using `{state.retrieval.embedding_model}` "
                f"in {state.retrieval.latency_ms:.0f} ms.",
                "",
                "| Case ID | Defect type | Similarity | Recorded root cause | Action taken |",
                "|---|---|---|---|---|",
            ]
            for case in state.retrieval.cases:
                lines.append(
                    f"| `{case.case_id}` | {case.defect_type} | {case.similarity:.3f} | "
                    f"{case.root_cause} | {case.action_taken} |"
                )
        else:
            lines.append("_No comparable historical cases were retrieved._")

        # --- analysis
        lines += ["", "## 4. Root cause analysis", ""]
        if analysis:
            lines += [
                f"**Probable root cause.** {analysis.root_cause}",
                "",
                f"**Recommended action.** {analysis.recommended_action}",
                "",
            ]
            if analysis.affected_process_step:
                lines += [f"**Affected process step.** {analysis.affected_process_step}", ""]
            if analysis.evidence:
                lines.append("**Supporting evidence.**")
                lines += [f"- {item}" for item in analysis.evidence]
                lines.append("")
            lines += [
                f"**Cited historical cases.** "
                f"{', '.join(f'`{c}`' for c in analysis.cited_case_ids) or 'none'}",
                "",
                f"_Analysis produced by `{analysis.model_name}` in "
                f"{analysis.latency_ms:.0f} ms at confidence {analysis.confidence:.2f}. "
                f"Every cited case ID was verified against the retrieved set._",
            ]
        else:
            lines.append("_No analysis was produced._")

        # --- approval
        lines += ["", "## 5. Human review", ""]
        approval = state.approval
        if approval.status == ApprovalStatus.NOT_REQUIRED:
            lines.append(
                "Risk was below the escalation threshold. The inspection proceeded "
                "automatically and was logged in full."
            )
        else:
            lines += [
                "| Field | Value |",
                "|---|---|",
                f"| Decision | **{approval.status.value}** |",
                f"| Approver | {approval.approver or 'unknown'} |",
                f"| Escalated because | {approval.required_because or '-'} |",
                f"| Rationale | {approval.rationale or '-'} |",
                f"| Decided at | {_when(approval.decided_at)} |",
            ]

        if state.errors:
            lines += ["", "## 6. Recorded errors", ""] + [f"- {e}" for e in state.errors]

        # --- audit
        section = 7 if state.errors else 6
        lines += [
            "",
            f"## {section}. Audit trail",
            "",
            "Every step below is a link in an append-only SHA-256 hash chain. Each "
            "entry commits to its predecessor's digest, so altering or removing any "
            "record invalidates every entry after it. Verify with `mavia audit verify`.",
            "",
            "| Seq | Timestamp (UTC) | Agent | Action | Entry hash |",
            "|---|---|---|---|---|",
        ]
        for event in events:
            lines.append(
                f"| {event.seq} | {event.timestamp.strftime('%H:%M:%S')} | {event.agent} | "
                f"{event.action} | `{event.entry_hash[:16]}...` |"
            )
        if events:
            lines += ["", f"**Chain head at time of writing:** `{events[-1].entry_hash}`"]

        lines += [
            "",
            "---",
            "",
            "_Generated by MAVIA - Multi-Agent Visual Inspection & Audit System. "
            "This is decision support: the root cause is a plausibility judgement over "
            "retrieved precedent, not a confirmed finding._",
        ]
        return "\n".join(lines)

    # -------------------------------------------------------------------- html

    def build_html(self, state: InspectionState, events: list[AuditEvent]) -> str:
        """Standalone HTML, used for PDF rendering and as the PDF fallback."""
        vision, analysis = state.vision, state.analysis
        risk = analysis.risk_level if analysis else RiskLevel.LOW
        colour, background = RISK_STYLE[risk]

        def row(label: str, value: object) -> str:
            return f"<tr><th>{_esc(label)}</th><td>{_esc(value)}</td></tr>"

        parts = [
            "<!doctype html><html><head><meta charset='utf-8'>",
            f"<title>QA Report {_esc(state.inspection_id)}</title>",
            f"<style>{STYLESHEET}</style></head><body>",
            "<h1>Quality Inspection Report</h1>",
            f"<div class='sub'>Inspection <span class='mono'>{_esc(state.inspection_id)}</span> "
            f"&middot; {utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')} "
            f"&middot; category: {_esc(state.category or 'unknown')}</div>",
            "<h2>1. Verdict</h2><table>",
            row("Final verdict", state.final_verdict.value),
            f"<tr><th>Risk level</th><td><span class='badge' "
            f"style='color:{colour};background:{background}'>{risk.value}</span></td></tr>",
            row("Approval status", state.approval.status.value),
        ]
        if analysis:
            parts.append(row("Analysis confidence", f"{analysis.confidence:.2f}"))
        if state.total_latency_ms:
            parts.append(row("End-to-end latency", f"{state.total_latency_ms:.0f} ms"))
        parts.append("</table>")

        parts.append("<h2>2. Detection</h2><table>")
        if vision:
            parts += [
                row("Detector", vision.model_name),
                row(
                    "Anomaly score",
                    f"{vision.anomaly_score:.3f} (threshold {vision.decision_threshold:.2f})",
                ),
                row("Regions localised", len(vision.regions)),
                row("Inference latency", f"{vision.latency_ms:.0f} ms"),
            ]
            if vision.zero_shot_label:
                parts.append(
                    row("Zero-shot fallback", f"{vision.zero_shot_label} - reduced confidence")
                )
        else:
            parts.append(row("Detection", "not recorded"))
        parts.append("</table>")

        parts.append("<h2>3. Historical context</h2>")
        if state.retrieval and state.retrieval.cases:
            parts.append(
                "<table><tr>"
                "<th style='width:auto'>Case</th><th style='width:auto'>Defect</th>"
                "<th style='width:auto'>Sim.</th>"
                "<th style='width:auto'>Recorded root cause</th></tr>"
            )
            for case in state.retrieval.cases:
                parts.append(
                    f"<tr><td class='mono'>{_esc(case.case_id)}</td>"
                    f"<td>{_esc(case.defect_type)}</td>"
                    f"<td>{case.similarity:.3f}</td>"
                    f"<td>{_esc(case.root_cause)}</td></tr>"
                )
            parts.append("</table>")
        else:
            parts.append("<p><em>No comparable historical cases were retrieved.</em></p>")

        parts.append("<h2>4. Root cause analysis</h2>")
        if analysis:
            parts += [
                f"<p><strong>Probable root cause.</strong> {_esc(analysis.root_cause)}</p>",
                f"<p><strong>Recommended action.</strong> {_esc(analysis.recommended_action)}</p>",
            ]
            if analysis.evidence:
                parts.append("<p><strong>Supporting evidence.</strong></p><ul>")
                parts += [f"<li>{_esc(item)}</li>" for item in analysis.evidence]
                parts.append("</ul>")
            cited = ", ".join(analysis.cited_case_ids) or "none"
            parts.append(
                "<p class='note'>Produced by <span class='mono'>"
                f"{_esc(analysis.model_name)}</span> "
                f"at confidence {analysis.confidence:.2f}. Cited cases: "
                f"<span class='mono'>{_esc(cited)}</span> - each verified against the "
                f"retrieved set.</p>"
            )
        else:
            parts.append("<p><em>No analysis was produced.</em></p>")

        parts.append("<h2>5. Human review</h2>")
        approval = state.approval
        if approval.status == ApprovalStatus.NOT_REQUIRED:
            parts.append(
                "<p>Risk was below the escalation threshold. The inspection proceeded "
                "automatically and was logged in full.</p>"
            )
        else:
            parts += [
                "<table>",
                row("Decision", approval.status.value),
                row("Approver", approval.approver or "unknown"),
                row("Escalated because", approval.required_because or "-"),
                row("Rationale", approval.rationale or "-"),
                "</table>",
            ]

        parts += [
            "<h2>6. Audit trail</h2>",
            "<p>Every step below is a link in an append-only SHA-256 hash chain. Each entry "
            "commits to its predecessor's digest, so altering or removing any record "
            "invalidates every entry after it.</p>",
            "<table class='audit'><tr><th>Seq</th><th>Time</th><th>Agent</th>"
            "<th>Action</th><th>Entry hash</th></tr>",
        ]
        for event in events:
            parts.append(
                f"<tr><td>{event.seq}</td><td>{event.timestamp.strftime('%H:%M:%S')}</td>"
                f"<td>{_esc(event.agent)}</td><td>{_esc(event.action)}</td>"
                f"<td class='mono'>{event.entry_hash[:24]}...</td></tr>"
            )
        parts.append("</table>")
        if events:
            parts.append(
                f"<p class='note'>Chain head at time of writing: "
                f"<span class='mono'>{events[-1].entry_hash}</span></p>"
            )

        parts.append(
            "<p class='sub' style='margin-top:6mm'>Generated by MAVIA. This is decision "
            "support: the root cause is a plausibility judgement over retrieved precedent, "
            "not a confirmed finding.</p></body></html>"
        )
        return "".join(parts)

    # ------------------------------------------------------------------ write

    def write(self, state: InspectionState, output_dir: Path | None = None) -> QAReport:
        """Render and persist the report. Returns the typed contract."""
        events = self.audit.events_for(state.inspection_id)
        directory = Path(output_dir or Path(self.settings.artifacts_dir) / "reports")
        directory.mkdir(parents=True, exist_ok=True)

        markdown = self.build_markdown(state, events)
        (directory / f"{state.inspection_id}.md").write_text(markdown, encoding="utf-8")

        html = self.build_html(state, events)
        html_path = directory / f"{state.inspection_id}.html"
        html_path.write_text(html, encoding="utf-8")

        pdf_path = render_pdf(html, directory / f"{state.inspection_id}.pdf")

        report = QAReport(
            markdown=markdown,
            summary=self._summary(state),
            pdf_path=str(pdf_path) if pdf_path else str(html_path),
        )
        self.audit.log(
            inspection_id=state.inspection_id,
            agent="report_writer",
            action="report_written",
            payload={
                "report_id": report.report_id,
                "format": "pdf" if pdf_path else "html",
                "audit_events_included": len(events),
                # Bind the report to the chain state it describes.
                "chain_head": events[-1].entry_hash if events else None,
            },
        )
        return report

    @staticmethod
    def _summary(state: InspectionState) -> str:
        verdict = state.final_verdict.value
        category = state.category or "unit"
        if state.analysis is None:
            return f"{category}: {verdict}, no analysis recorded."
        return (
            f"{category}: {verdict} at risk {state.analysis.risk_level.value}. "
            f"{state.analysis.root_cause} Recommended: {state.analysis.recommended_action}"
        )


# ------------------------------------------------------------------ helpers


def _prepare_macos_libraries() -> None:
    """Point WeasyPrint at Homebrew's pango before it resolves its libraries.

    WeasyPrint loads pango/cairo at import time via dlopen, and on macOS the
    Homebrew prefix is not on the default search path - so an otherwise correct
    `brew install pango` still produces an ImportError. Setting this here means
    the PDF path works without the user exporting anything.
    """
    if platform.system() != "Darwin":
        return
    existing = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    paths = [p for p in _MACOS_LIB_PATHS if Path(p).is_dir()]
    if not paths:
        return
    merged = ":".join([*paths, existing]) if existing else ":".join(paths)
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = merged


def render_pdf(html: str, destination: Path) -> Path | None:
    """Render HTML to PDF, returning None if the toolchain is unavailable.

    A missing system library degrades the report to HTML rather than failing the
    inspection - the same principle as everywhere else in the pipeline.
    """
    if "weasyprint" not in sys.modules:
        _prepare_macos_libraries()
    try:
        from weasyprint import HTML
    except (ImportError, OSError) as error:
        logger.warning("pdf_unavailable", error=str(error)[:160])
        return None

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        HTML(string=html).write_pdf(str(destination))
        return destination
    except Exception as error:
        logger.warning("pdf_render_failed", error=str(error)[:160])
        return None


def _when(value: datetime | None) -> str:
    return value.isoformat() if value else "-"


def _esc(value: object) -> str:
    """Escape untrusted text for HTML. Report content includes model output."""
    text = value.isoformat() if isinstance(value, datetime) else str(value)
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def summarise_for_dashboard(state: InspectionState) -> dict[str, Any]:
    """Flat view of an inspection, for the Phase 7 dashboard."""
    return {
        "inspection_id": state.inspection_id,
        "category": state.category,
        "verdict": state.final_verdict.value,
        "approval": state.approval.status.value,
        "risk_level": state.analysis.risk_level.value if state.analysis else None,
        "confidence": state.analysis.confidence if state.analysis else None,
        "anomaly_score": state.vision.anomaly_score if state.vision else None,
        "overlay_path": state.vision.overlay_path if state.vision else None,
        "latency_ms": state.total_latency_ms,
        "errors": state.errors,
    }


__all__ = ["ReportWriter", "Verdict", "render_pdf", "summarise_for_dashboard"]
