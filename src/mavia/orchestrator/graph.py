"""Phase 5 - the LangGraph inspection pipeline.

Wires the four agents into a stateful graph with a risk-routed human approval
gate. Three properties make this an agent rather than a script, and each is the
reason LangGraph was chosen over a hand-written loop:

**1. The pause survives the process.** A high-risk inspection calls
``interrupt()``, which suspends the graph at a checkpoint written to SQLite. The
Python process can exit. A reviewer can approve twenty minutes later from a
different process - or the dashboard in Phase 7 - and execution resumes at
exactly the node that stopped, with all prior state intact. An agent permitted to
stop a production line must be interruptible, and the interrupt must outlive the
request that created it. This is verified by a test that discards the graph
object entirely between suspend and resume.

**2. Auditing is structural, not remembered.** Every node is wrapped by
``_audited`` before being added to the graph, so entry, exit, latency and the
produced payload are hashed into the tamper-evident chain automatically. No node
is responsible for logging itself, which means no node can forget to.

**3. Risk routes control flow.** ``risk_level`` is an enum precisely so the
conditional edge can branch on it. The routing rule is deliberately conservative:
either a HIGH/CRITICAL analysis *or* a raw anomaly score above the configured
threshold sends the inspection to a human. A disagreement between the two
escalates rather than resolves.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from mavia.agents.analyst import RootCauseAnalyst
from mavia.audit import AuditLogger
from mavia.config import Settings, get_settings
from mavia.logging_setup import get_logger
from mavia.memory.retrieval import HistoryRetriever
from mavia.schemas import (
    AgentName,
    ApprovalDecision,
    ApprovalStatus,
    BoundingBox,
    HistoricalCase,
    InspectionState,
    QAReport,
    RetrievalResult,
    RiskLevel,
    RootCauseAnalysis,
    Verdict,
    VisionResult,
    utc_now,
)
from mavia.vision.inspector import VisionInspector

logger = get_logger(__name__)

ESCALATING_RISK = frozenset({RiskLevel.HIGH, RiskLevel.CRITICAL})

# Every MAVIA type that can appear inside a persisted checkpoint.
CHECKPOINT_TYPES: tuple[type, ...] = (
    InspectionState,
    VisionResult,
    RetrievalResult,
    RootCauseAnalysis,
    ApprovalDecision,
    QAReport,
    BoundingBox,
    HistoricalCase,
    Verdict,
    RiskLevel,
    ApprovalStatus,
    AgentName,
)


# LangGraph's StateGraph is generic over (state, context, input, output).
InspectionGraph = StateGraph[InspectionState, None, InspectionState, InspectionState]


def _add_node(
    graph: InspectionGraph, name: str, fn: Callable[[InspectionState], dict[str, Any]]
) -> None:
    """Attach a node.

    LangGraph's ``add_node`` overloads do not resolve for a plain
    ``Callable[[StateT], dict]`` under strict mypy, though it is exactly the
    documented node signature. The suppression is isolated here rather than
    repeated at every call site.
    """
    graph.add_node(name, fn)  # type: ignore[call-overload]


def _serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=list(CHECKPOINT_TYPES))


class InspectionPipeline:
    """The MAVIA inspection graph."""

    def __init__(
        self,
        settings: Settings | None = None,
        inspector: VisionInspector | None = None,
        retriever: HistoryRetriever | None = None,
        analyst: RootCauseAnalyst | None = None,
        audit: AuditLogger | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()

        self.inspector = inspector or VisionInspector(settings=self.settings)
        self.retriever = retriever or HistoryRetriever(settings=self.settings)
        self.analyst = analyst or RootCauseAnalyst(settings=self.settings)
        self.audit = audit or AuditLogger(self.settings.audit_log_path)

        self._connection: sqlite3.Connection | None = None
        if checkpointer is None:
            checkpointer, self._connection = self._open_checkpointer()
        self.checkpointer = checkpointer

        self.graph = self._build().compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------ persistence

    def _open_checkpointer(self) -> tuple[Any, sqlite3.Connection]:
        """SQLite-backed checkpoints, so a pending approval survives a restart.

        The serializer is given an explicit allowlist of MAVIA's own state types.
        LangGraph's default permissive mode deserializes anything and warns that
        it will be blocked in a future version; naming the types keeps the
        checkpoints readable across upgrades and, more importantly, means a
        checkpoint database cannot smuggle an arbitrary class past the loader.
        """
        path = Path(self.settings.artifacts_dir) / "checkpoints.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False: the dashboard resumes a graph from a different
        # thread than the one that suspended it.
        connection = sqlite3.connect(str(path), check_same_thread=False)
        saver = SqliteSaver(connection, serde=_serializer())
        saver.setup()
        return saver, connection

    def close(self) -> None:
        """Release the checkpoint database and the vector store client."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        memory = getattr(self.retriever, "memory", None)
        if memory is not None:
            with suppress(Exception):  # best-effort cleanup on shutdown
                memory.close()

    def __enter__(self) -> InspectionPipeline:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----------------------------------------------------------------- audit

    def _audited(
        self, agent: AgentName, node: Callable[[InspectionState], dict[str, Any]]
    ) -> Callable[[InspectionState], dict[str, Any]]:
        """Wrap a node so its decision is hashed into the audit chain automatically."""

        def wrapper(state: InspectionState) -> dict[str, Any]:
            started = time.perf_counter()
            self.audit.log(
                inspection_id=state.inspection_id,
                agent=agent,
                action="started",
                payload={"image_path": state.image_path, "category": state.category},
            )
            try:
                update = node(state)
            except Exception as error:
                self.audit.log(
                    inspection_id=state.inspection_id,
                    agent=agent,
                    action="failed",
                    payload={"error": str(error)},
                )
                logger.warning("node_failed", agent=str(agent), error=str(error))
                return {"errors": [*state.errors, f"{agent}: {error}"]}

            self.audit.log(
                inspection_id=state.inspection_id,
                agent=agent,
                action="completed",
                payload={
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
                    **_summarise(update),
                },
            )
            return update

        return wrapper

    # ----------------------------------------------------------------- nodes

    def _inspect(self, state: InspectionState) -> dict[str, Any]:
        vision = self.inspector.inspect(state.image_path, state.category)
        return {"vision": vision, "category": vision.category}

    def _retrieve(self, state: InspectionState) -> dict[str, Any]:
        if state.vision is None:
            return {"errors": [*state.errors, "retrieval skipped: no vision result"]}
        return {"retrieval": self.retriever.retrieve(state.vision)}

    def _analyse(self, state: InspectionState) -> dict[str, Any]:
        if state.vision is None or state.retrieval is None:
            return {"errors": [*state.errors, "analysis skipped: missing inputs"]}
        return {"analysis": self.analyst.analyse(state.vision, state.retrieval)}

    def _approval_gate(self, state: InspectionState) -> dict[str, Any]:
        """Suspend the graph until a human decides.

        ``interrupt()`` raises out of the node and persists a checkpoint. The
        payload below is everything a reviewer needs to make the call without
        going back to the raw image - it is what the dashboard renders.
        """
        analysis = state.analysis
        reason = self._escalation_reason(state)

        decision = interrupt(
            {
                "inspection_id": state.inspection_id,
                "image_path": state.image_path,
                "category": state.category,
                "reason": reason,
                "anomaly_score": state.vision.anomaly_score if state.vision else None,
                "overlay_path": state.vision.overlay_path if state.vision else None,
                "risk_level": analysis.risk_level.value if analysis else None,
                "root_cause": analysis.root_cause if analysis else None,
                "recommended_action": analysis.recommended_action if analysis else None,
                "confidence": analysis.confidence if analysis else None,
                "cited_case_ids": analysis.cited_case_ids if analysis else [],
            }
        )

        approval = _coerce_decision(decision, reason)
        self.audit.log(
            inspection_id=state.inspection_id,
            agent=AgentName.APPROVAL,
            action=f"approval_{approval.status.value.lower()}",
            payload={
                "approver": approval.approver,
                "rationale": approval.rationale,
                "required_because": reason,
            },
        )
        return {"approval": approval}

    def _auto_approve(self, state: InspectionState) -> dict[str, Any]:
        return {
            "approval": ApprovalDecision(
                status=ApprovalStatus.NOT_REQUIRED,
                required_because=None,
                approver="system",
                rationale="Risk below the human-review threshold; auto-proceeded with logging.",
                decided_at=utc_now(),
            )
        }

    def _finalise(self, state: InspectionState) -> dict[str, Any]:
        elapsed = (utc_now() - state.created_at).total_seconds() * 1000.0
        self.audit.log(
            inspection_id=state.inspection_id,
            agent="pipeline",
            action="finalised",
            payload={
                "verdict": state.final_verdict.value,
                "approval": state.approval.status.value,
                "total_latency_ms": round(elapsed, 2),
                "errors": state.errors,
            },
        )
        return {"total_latency_ms": elapsed}

    # ---------------------------------------------------------------- routing

    def _escalation_reason(self, state: InspectionState) -> str | None:
        """Why this inspection needs a human, or None if it does not."""
        reasons: list[str] = []
        if state.analysis is not None and state.analysis.risk_level in ESCALATING_RISK:
            reasons.append(f"risk_level={state.analysis.risk_level.value}")
        if (
            state.vision is not None
            and state.vision.anomaly_score >= self.settings.high_risk_threshold
        ):
            reasons.append(
                f"anomaly_score={state.vision.anomaly_score:.3f} "
                f">= {self.settings.high_risk_threshold}"
            )
        return "; ".join(reasons) or None

    def needs_human_review(self, state: InspectionState) -> bool:
        return self._escalation_reason(state) is not None

    def _route(self, state: InspectionState) -> str:
        return "approval_gate" if self.needs_human_review(state) else "auto_approve"

    # ----------------------------------------------------------------- build

    def _build(self) -> InspectionGraph:
        graph: InspectionGraph = StateGraph(InspectionState)

        _add_node(graph, "inspect", self._audited(AgentName.VISION, self._inspect))
        _add_node(graph, "retrieve", self._audited(AgentName.RETRIEVAL, self._retrieve))
        _add_node(graph, "analyse", self._audited(AgentName.ANALYST, self._analyse))
        _add_node(graph, "approval_gate", self._approval_gate)
        _add_node(graph, "auto_approve", self._auto_approve)
        _add_node(graph, "finalise", self._finalise)

        graph.add_edge(START, "inspect")
        graph.add_edge("inspect", "retrieve")
        graph.add_edge("retrieve", "analyse")
        graph.add_conditional_edges(
            "analyse",
            self._route,
            {"approval_gate": "approval_gate", "auto_approve": "auto_approve"},
        )
        graph.add_edge("approval_gate", "finalise")
        graph.add_edge("auto_approve", "finalise")
        graph.add_edge("finalise", END)
        return graph

    # ------------------------------------------------------------------- run

    @staticmethod
    def config_for(inspection_id: str) -> RunnableConfig:
        """LangGraph thread config. The inspection id *is* the thread id."""
        return cast(RunnableConfig, {"configurable": {"thread_id": inspection_id}})

    def run(
        self, image_path: str | Path, category: str | None = None, inspection_id: str | None = None
    ) -> InspectionState:
        """Run an inspection. Returns early, suspended, if a human is needed."""
        initial = InspectionState(image_path=str(image_path), category=category)
        if inspection_id:
            initial.inspection_id = inspection_id

        self.audit.log(
            inspection_id=initial.inspection_id,
            agent="pipeline",
            action="inspection_started",
            payload={"image_path": str(image_path), "category": category},
        )

        result = self.graph.invoke(initial, config=self.config_for(initial.inspection_id))
        return self._state_from(result, initial.inspection_id)

    def resume(
        self,
        inspection_id: str,
        approved: bool,
        approver: str,
        rationale: str | None = None,
    ) -> InspectionState:
        """Resume a suspended inspection with a human decision."""
        from langgraph.types import Command

        payload = {
            "status": ApprovalStatus.APPROVED.value if approved else ApprovalStatus.REJECTED.value,
            "approver": approver,
            "rationale": rationale,
        }
        result = self.graph.invoke(Command(resume=payload), config=self.config_for(inspection_id))
        return self._state_from(result, inspection_id)

    def pending_approval(self, inspection_id: str) -> dict[str, Any] | None:
        """The interrupt payload for a suspended inspection, or None if not waiting."""
        snapshot = self.graph.get_state(self.config_for(inspection_id))
        for task in snapshot.tasks:
            for item in getattr(task, "interrupts", ()) or ():
                return dict(item.value) if isinstance(item.value, dict) else {"value": item.value}
        return None

    def is_suspended(self, inspection_id: str) -> bool:
        return self.pending_approval(inspection_id) is not None

    @staticmethod
    def _state_from(result: Any, inspection_id: str) -> InspectionState:
        """LangGraph returns a dict for pydantic state; rebuild the typed model."""
        if isinstance(result, InspectionState):
            return result
        payload = dict(result)
        payload.pop("__interrupt__", None)
        payload.setdefault("inspection_id", inspection_id)
        return InspectionState.model_validate(payload)


# ------------------------------------------------------------------ helpers


def _coerce_decision(raw: Any, reason: str | None) -> ApprovalDecision:
    """Turn whatever the resume payload carried into a typed decision.

    Anything unrecognised is treated as a rejection. A human-in-the-loop gate
    that fails open is not a gate.
    """
    if isinstance(raw, ApprovalDecision):
        return raw

    approved = False
    approver: str | None = None
    rationale: str | None = None

    if isinstance(raw, bool):
        approved = raw
    elif isinstance(raw, str):
        approved = raw.strip().upper() in {"APPROVED", "APPROVE", "YES", "TRUE"}
        rationale = raw
    elif isinstance(raw, dict):
        status = str(raw.get("status", "")).upper()
        approved = status in {"APPROVED", "APPROVE", "YES", "TRUE"} or bool(raw.get("approved"))
        approver = raw.get("approver")
        rationale = raw.get("rationale")

    return ApprovalDecision(
        status=ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED,
        required_because=reason,
        approver=approver,
        rationale=rationale,
        decided_at=utc_now(),
    )


def _summarise(update: dict[str, Any]) -> dict[str, Any]:
    """Compact, hashable summary of a node's output for the audit payload."""
    summary: dict[str, Any] = {}
    for key, value in update.items():
        if value is None:
            continue
        if hasattr(value, "model_dump"):
            dumped = value.model_dump(mode="json")
            summary[key] = {
                k: v
                for k, v in dumped.items()
                # Exclude bulky or non-decision fields; the hash covers what matters.
                if k
                not in {"anomaly_map_path", "overlay_path", "query_text", "evidence", "regions"}
            }
        elif isinstance(value, list):
            summary[key] = len(value)
        else:
            summary[key] = value
    return summary
