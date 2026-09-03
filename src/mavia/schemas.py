"""Domain contracts shared by every agent, the orchestrator, and the dashboard."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class ApprovalStatus(StrEnum):
    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class AgentName(StrEnum):
    VISION = "vision_inspector"
    RETRIEVAL = "history_retriever"
    ANALYST = "root_cause_analyst"
    APPROVAL = "approval_gate"
    REPORTER = "report_writer"


class MaviaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class BoundingBox(MaviaModel):
    """Axis-aligned region of the anomaly map, in pixel coordinates."""

    x: int
    y: int
    width: int
    height: int
    area_px: int
    peak_score: float = Field(ge=0.0, le=1.0)


class VisionResult(MaviaModel):
    """Output of the Vision Inspector agent."""

    category: str = Field(description="MVTec AD object/texture category, e.g. 'bottle'")
    verdict: Verdict
    anomaly_score: float = Field(ge=0.0, le=1.0, description="Calibrated image-level score")
    decision_threshold: float = Field(ge=0.0, le=1.0)
    regions: list[BoundingBox] = Field(default_factory=list)
    anomaly_map_path: str | None = None
    overlay_path: str | None = None
    zero_shot_label: str | None = Field(
        default=None, description="CLIP zero-shot defect label when the category is unseen"
    )
    zero_shot_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    model_name: str
    latency_ms: float


class HistoricalCase(MaviaModel):
    """One retrieved record from the defect-history vector store."""

    case_id: str
    category: str
    defect_type: str
    root_cause: str
    action_taken: str
    outcome: str | None = None
    occurred_at: datetime
    similarity: float = Field(ge=0.0, le=1.0)


class RetrievalResult(MaviaModel):
    """Output of the History Retriever agent."""

    query_text: str
    cases: list[HistoricalCase] = Field(default_factory=list)
    top_k: int
    latency_ms: float
    embedding_model: str


class RootCauseAnalysis(MaviaModel):
    """Structured output of the Root Cause Analyst agent."""

    root_cause: str
    evidence: list[str] = Field(default_factory=list)
    recommended_action: str
    risk_level: RiskLevel
    confidence: float = Field(ge=0.0, le=1.0)
    affected_process_step: str | None = None
    cited_case_ids: list[str] = Field(default_factory=list)
    latency_ms: float
    model_name: str


class ApprovalDecision(MaviaModel):
    """Human-in-the-loop checkpoint outcome."""

    status: ApprovalStatus
    required_because: str | None = None
    approver: str | None = None
    rationale: str | None = None
    decided_at: datetime | None = None


class QAReport(MaviaModel):
    """Output of the Report Writer agent."""

    report_id: str = Field(default_factory=lambda: new_id("rpt"))
    markdown: str
    summary: str
    pdf_path: str | None = None
    generated_at: datetime = Field(default_factory=utc_now)


class AuditEvent(MaviaModel):
    """One tamper-evident link in the inspection's hash chain."""

    seq: int
    inspection_id: str
    timestamp: datetime
    agent: AgentName | str
    action: str
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_hash: str
    prev_hash: str
    entry_hash: str


class InspectionState(MaviaModel):
    """The LangGraph state object threaded through the whole pipeline."""

    inspection_id: str = Field(default_factory=lambda: new_id("insp"))
    image_path: str
    category: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    vision: VisionResult | None = None
    retrieval: RetrievalResult | None = None
    analysis: RootCauseAnalysis | None = None
    approval: ApprovalDecision = Field(
        default_factory=lambda: ApprovalDecision(status=ApprovalStatus.NOT_REQUIRED)
    )
    report: QAReport | None = None

    errors: list[str] = Field(default_factory=list)
    total_latency_ms: float | None = None

    @property
    def final_verdict(self) -> Verdict:
        if self.vision is None:
            return Verdict.INCONCLUSIVE
        if self.approval.status == ApprovalStatus.REJECTED:
            return Verdict.INCONCLUSIVE
        return self.vision.verdict
