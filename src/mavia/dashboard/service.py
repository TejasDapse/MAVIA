"""Data layer for the dashboard.

Deliberately separate from ``app.py``. Streamlit re-executes its whole script on
every widget interaction, which makes UI code a poor place for logic and an
impossible place to test. Everything here is a plain function over the audit log
and the report directory, so the dashboard's behaviour is covered by ordinary
unit tests and the UI file stays a rendering layer.

The audit log is the source of truth for what happened, and the checkpoint
database is the source of truth for what is still waiting. Neither is re-derived
from the other.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from mavia.audit import AuditEvent, read_events, verify_chain
from mavia.config import Settings, get_settings

TERMINAL_ACTION = "finalised"
START_ACTION = "inspection_started"


@dataclass
class InspectionSummary:
    """One inspection, reconstructed from its audit events."""

    inspection_id: str
    started_at: datetime
    category: str | None = None
    image_path: str | None = None
    verdict: str | None = None
    risk_level: str | None = None
    approval: str | None = None
    approver: str | None = None
    anomaly_score: float | None = None
    total_latency_ms: float | None = None
    n_events: int = 0
    errors: list[str] = field(default_factory=list)
    report_id: str | None = None

    @property
    def is_complete(self) -> bool:
        return self.verdict is not None

    @property
    def awaiting_approval(self) -> bool:
        """Escalated but no decision recorded yet."""
        return self.approval is None and not self.is_complete and self.risk_level is not None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarise_events(events: list[AuditEvent]) -> InspectionSummary:
    """Rebuild an inspection's story from its audit trail alone.

    Nothing else is consulted. If the audit log cannot reconstruct what happened,
    the audit log is inadequate - so the dashboard uses it as its only source and
    surfaces that weakness rather than hiding it behind a second store.
    """
    first = events[0]
    summary = InspectionSummary(
        inspection_id=first.inspection_id,
        started_at=first.timestamp,
        n_events=len(events),
    )

    for event in events:
        payload = event.payload
        summary.category = payload.get("category") or summary.category
        summary.image_path = payload.get("image_path") or summary.image_path

        vision = payload.get("vision")
        if isinstance(vision, dict):
            summary.anomaly_score = _as_float(vision.get("anomaly_score")) or summary.anomaly_score

        analysis = payload.get("analysis")
        if isinstance(analysis, dict):
            summary.risk_level = analysis.get("risk_level") or summary.risk_level

        approval = payload.get("approval")
        if isinstance(approval, dict):
            summary.approval = approval.get("status") or summary.approval
            summary.approver = approval.get("approver") or summary.approver

        if event.action.startswith("approval_"):
            summary.approver = payload.get("approver") or summary.approver

        if event.action == "failed" and payload.get("error"):
            summary.errors.append(str(payload["error"]))

        if event.action == "report_written":
            summary.report_id = payload.get("report_id")

        if event.action == TERMINAL_ACTION:
            summary.verdict = payload.get("verdict") or summary.verdict
            summary.approval = payload.get("approval") or summary.approval
            summary.total_latency_ms = _as_float(payload.get("total_latency_ms"))
            errors = payload.get("errors")
            if isinstance(errors, list):
                summary.errors.extend(str(e) for e in errors if str(e) not in summary.errors)

    return summary


def load_inspections(
    settings: Settings | None = None, limit: int | None = None
) -> list[InspectionSummary]:
    """All inspections in the audit log, newest first."""
    settings = settings or get_settings()
    grouped: dict[str, list[AuditEvent]] = {}
    for event in read_events(settings.audit_log_path):
        grouped.setdefault(event.inspection_id, []).append(event)

    summaries = [summarise_events(events) for events in grouped.values() if events]
    summaries.sort(key=lambda s: s.started_at, reverse=True)
    return summaries[:limit] if limit else summaries


def fleet_metrics(summaries: list[InspectionSummary]) -> dict[str, Any]:
    """Headline numbers for the overview panel."""
    complete = [s for s in summaries if s.is_complete]
    failed = [s for s in complete if s.verdict == "FAIL"]
    latencies = [s.total_latency_ms for s in complete if s.total_latency_ms]
    escalated = [s for s in summaries if s.approval in {"APPROVED", "REJECTED"}]

    return {
        "total": len(summaries),
        "completed": len(complete),
        "failed": len(failed),
        "defect_rate": len(failed) / len(complete) if complete else 0.0,
        "awaiting_approval": sum(1 for s in summaries if s.awaiting_approval),
        "escalated": len(escalated),
        "escalation_rate": len(escalated) / len(complete) if complete else 0.0,
        "with_errors": sum(1 for s in summaries if s.errors),
        "median_latency_ms": _median(latencies),
        "risk_breakdown": dict(
            Counter(s.risk_level for s in summaries if s.risk_level).most_common()
        ),
        "category_breakdown": dict(
            Counter(s.category for s in summaries if s.category).most_common()
        ),
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def chain_status(settings: Settings | None = None) -> dict[str, Any]:
    """Live integrity check, shown prominently rather than buried."""
    settings = settings or get_settings()
    result = verify_chain(settings.audit_log_path)
    return {"valid": bool(result), "checked": result.checked, "broken_at": result.broken_at}


def report_paths(inspection_id: str, settings: Settings | None = None) -> dict[str, Path]:
    """Report files that exist on disk for an inspection."""
    settings = settings or get_settings()
    directory = Path(settings.artifacts_dir) / "reports"
    found: dict[str, Path] = {}
    for suffix in ("md", "html", "pdf"):
        candidate = directory / f"{inspection_id}.{suffix}"
        if candidate.exists():
            found[suffix] = candidate
    return found


def available_images(settings: Settings | None = None, per_defect: int = 2) -> list[Path]:
    """A browsable sample of MVTec images, for the demo picker."""
    settings = settings or get_settings()
    root = Path(settings.mvtec_dir)
    if not root.is_dir():
        return []

    images: list[Path] = []
    for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        test_dir = category_dir / "test"
        if not test_dir.is_dir():
            continue
        for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            images.extend(sorted(defect_dir.glob("*.png"))[:per_defect])
    return images


def describe_image(path: Path) -> str:
    """`bottle / broken_large / 000.png` - the label shown in the picker."""
    parts = path.parts
    if len(parts) >= 3:
        return f"{parts[-4]} / {parts[-2]} / {path.name}" if len(parts) >= 4 else path.name
    return path.name
