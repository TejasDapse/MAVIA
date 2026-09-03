"""Agent 3 - Root Cause Analyst.

Takes the detection and the retrieved history and produces the thing every prior
layer exists to support: a stated probable cause, the evidence behind it, a
recommended action, and a risk level that decides whether a human must approve
before the line stops.

Three guardrails make this more than "ask an LLM and hope".

**1. Structured output, not prose.** The response is parsed into a Pydantic model
via ``client.messages.parse``. ``risk_level`` drives control flow in the
orchestrator, so it must be an enum the graph can branch on - never free text
that a downstream regex has to guess at.

**2. Citations are verified, not trusted.** The model is required to cite the
``case_id`` of any historical case it relies on. After parsing, every citation is
checked against the cases actually retrieved; invented ids are stripped and
counted. That check is what turns "grounded in history" from a claim into a
measurement (see EVALUATION.md - citation grounding rate).

**3. It degrades instead of failing.** With no API key, or on an API error, the
agent falls back to a deterministic rule-based analysis derived from the nearest
retrieved case and the knowledge base, at explicitly reduced confidence and
labelled as such in ``model_name``. A QA line cannot stop because a vendor is
having an outage.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from mavia.config import Settings, get_settings
from mavia.logging_setup import get_logger
from mavia.memory.knowledge import get_knowledge
from mavia.schemas import (
    RetrievalResult,
    RiskLevel,
    RootCauseAnalysis,
    Verdict,
    VisionResult,
)

logger = get_logger(__name__)

FALLBACK_MODEL_NAME = "rule-based-fallback"

SYSTEM_PROMPT = """\
You are a senior manufacturing quality engineer performing root-cause analysis on \
an automated visual inspection result.

You receive two things: what the vision system measured on this unit, and \
comparable defect cases retrieved from the plant's historical record. Your job is \
to explain the most probable cause and state what should be done.

Rules you must follow:

1. Ground every claim. When you rely on a historical case, cite its case_id in \
cited_case_ids. Do not cite a case_id that was not provided to you. If the \
retrieved history does not support a conclusion, say so and lower your confidence \
rather than inventing a cause.

2. Evidence must be observations, not restatements of your conclusion. Reference \
the measured anomaly score, the affected area, the number and shape of regions, \
and the specific historical precedents.

3. Set risk_level by consequence, not by defect size:
   - CRITICAL: safety, contamination, wrong product, or a defect that would reach \
an end user with functional impact. Also use this when several defect modes \
co-occur, which indicates a process out of control.
   - HIGH: functional impairment, structural compromise, or a defect likely to \
recur across the batch.
   - MEDIUM: cosmetic or dimensional deviation outside specification with no \
immediate safety impact.
   - LOW: minor cosmetic variation within normal process spread.

4. Calibrate confidence honestly. High confidence requires strong agreement \
between the current observation and the retrieved precedent. If the detection \
came from the zero-shot fallback model, or the retrieved cases disagree with each \
other, confidence must be below 0.5.

5. recommended_action must be a specific, actionable process intervention that a \
line engineer can carry out - name the equipment, parameter, or procedure. \
"Investigate further" is not an acceptable action.

Be concise. This is a QA record, not an essay."""


class AnalystOutput(BaseModel):
    """Schema the model is constrained to produce."""

    root_cause: str = Field(description="The single most probable cause, one or two sentences.")
    evidence: list[str] = Field(
        description="Specific observations supporting the conclusion. 2-4 items."
    )
    recommended_action: str = Field(
        description="A specific corrective action naming equipment, parameter, or procedure."
    )
    risk_level: RiskLevel
    confidence: float = Field(ge=0.0, le=1.0)
    affected_process_step: str | None = Field(
        default=None, description="Manufacturing step where the defect most likely originated."
    )
    cited_case_ids: list[str] = Field(
        default_factory=list, description="case_id values from the provided history only."
    )


class RootCauseAnalyst:
    """Reason over a detection and its retrieved history."""

    def __init__(self, settings: Settings | None = None, client: object | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._client_ready = client is not None

    # ------------------------------------------------------------------ client

    @property
    def client(self) -> object | None:
        """Anthropic client, or None when no credential is configured."""
        if not self._client_ready:
            self._client_ready = True
            key = self.settings.anthropic_api_key
            if key is None:
                logger.warning("no_anthropic_key_using_fallback")
                self._client = None
            else:
                import anthropic

                self._client = anthropic.Anthropic(api_key=key.get_secret_value())
        return self._client

    # ------------------------------------------------------------------ prompt

    @staticmethod
    def build_user_prompt(vision: VisionResult, retrieval: RetrievalResult) -> str:
        """Render the evidence package handed to the model."""
        lines = [
            "## Current inspection",
            f"- Product category: {vision.category}",
            f"- Verdict: {vision.verdict.value}",
            f"- Anomaly score: {vision.anomaly_score:.3f} "
            f"(decision threshold {vision.decision_threshold:.2f})",
            f"- Detector: {vision.model_name}",
        ]

        if vision.zero_shot_label:
            lines.append(
                f"- NOTE: no memory bank existed for this category. Zero-shot fallback "
                f"suggested '{vision.zero_shot_label}' "
                f"(confidence {vision.zero_shot_confidence or 0:.2f}). Treat as weak evidence."
            )

        if vision.regions:
            total = sum(box.area_px for box in vision.regions)
            lines.append(f"- Affected regions: {len(vision.regions)}, {total} px total")
            for index, box in enumerate(vision.regions, start=1):
                shape = (
                    "elongated"
                    if max(box.width, box.height) > 2.5 * max(1, min(box.width, box.height))
                    else "compact"
                )
                lines.append(
                    f"  {index}. {box.width}x{box.height} px at ({box.x}, {box.y}), "
                    f"area {box.area_px} px, peak severity {box.peak_score:.2f}, {shape}"
                )
        else:
            lines.append("- No discrete defect regions were localised.")

        lines.append("")
        lines.append("## Comparable historical cases")
        if retrieval.cases:
            for case in retrieval.cases:
                lines.extend(
                    [
                        f"### {case.case_id} (similarity {case.similarity:.3f})",
                        f"- Defect type: {case.defect_type}",
                        f"- Recorded root cause: {case.root_cause}",
                        f"- Action taken: {case.action_taken}",
                        f"- Outcome: {case.outcome or 'not recorded'}",
                        f"- Occurred: {case.occurred_at.date().isoformat()}",
                    ]
                )
        else:
            lines.append(
                "No comparable cases were retrieved. State this explicitly and "
                "reduce your confidence accordingly."
            )

        lines.extend(
            [
                "",
                "Produce the root-cause analysis. Cite only the case_id values listed above.",
            ]
        )
        return "\n".join(lines)

    # ----------------------------------------------------------------- analyse

    def analyse(self, vision: VisionResult, retrieval: RetrievalResult) -> RootCauseAnalysis:
        started = time.perf_counter()
        client = self.client

        if client is None:
            return self._fallback(vision, retrieval, started, reason="no_api_key")

        try:
            response = client.messages.parse(  # type: ignore[attr-defined]
                model=self.settings.llm_model,
                max_tokens=self.settings.llm_max_tokens,
                # The system prompt is fixed across every inspection, so caching it
                # makes each additional unit cheaper on a line running thousands.
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": self.build_user_prompt(vision, retrieval)}],
                output_format=AnalystOutput,
                thinking={"type": "adaptive"},
            )
            parsed: AnalystOutput = response.parsed_output
        except Exception as error:
            logger.warning("analyst_call_failed", error=str(error))
            return self._fallback(vision, retrieval, started, reason=str(error))

        cited, dropped = self.verify_citations(parsed.cited_case_ids, retrieval)
        if dropped:
            logger.warning("dropped_hallucinated_citations", ids=dropped)

        confidence = parsed.confidence
        if dropped:
            # A model that invented a citation is less reliable on the rest of it.
            confidence = min(confidence, 0.5)

        return RootCauseAnalysis(
            root_cause=parsed.root_cause,
            evidence=parsed.evidence,
            recommended_action=parsed.recommended_action,
            risk_level=parsed.risk_level,
            confidence=confidence,
            affected_process_step=parsed.affected_process_step,
            cited_case_ids=cited,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            model_name=self.settings.llm_model,
        )

    @staticmethod
    def verify_citations(
        cited: list[str], retrieval: RetrievalResult
    ) -> tuple[list[str], list[str]]:
        """Split citations into those that exist in the retrieved set and inventions."""
        available = {case.case_id for case in retrieval.cases}
        kept = [case_id for case_id in cited if case_id in available]
        dropped = [case_id for case_id in cited if case_id not in available]
        return kept, dropped

    # ---------------------------------------------------------------- fallback

    def _fallback(
        self,
        vision: VisionResult,
        retrieval: RetrievalResult,
        started: float,
        reason: str,
    ) -> RootCauseAnalysis:
        """Deterministic analysis from the nearest precedent and the knowledge base.

        Deliberately conservative: it reports the closest historical match without
        claiming to have reasoned about it, and caps confidence well below what
        the LLM path can assert.
        """
        if retrieval.cases:
            nearest = retrieval.cases[0]
            knowledge = get_knowledge(nearest.category, nearest.defect_type)
            risk = knowledge.severity if knowledge else RiskLevel.MEDIUM
            # Anchor on similarity: a weak match should not produce a confident claim.
            confidence = min(0.45, 0.5 * nearest.similarity)

            return RootCauseAnalysis(
                root_cause=(
                    f"Closest historical precedent ({nearest.case_id}, similarity "
                    f"{nearest.similarity:.2f}) recorded: {nearest.root_cause}"
                ),
                evidence=[
                    f"Anomaly score {vision.anomaly_score:.3f} against threshold "
                    f"{vision.decision_threshold:.2f}",
                    f"{len(vision.regions)} region(s) localised, "
                    f"{sum(b.area_px for b in vision.regions)} px affected",
                    f"Nearest precedent {nearest.case_id} ({nearest.defect_type}), "
                    f"similarity {nearest.similarity:.3f}",
                ],
                recommended_action=nearest.action_taken,
                risk_level=risk if vision.verdict == Verdict.FAIL else RiskLevel.LOW,
                confidence=confidence,
                affected_process_step=knowledge.process_step if knowledge else None,
                cited_case_ids=[nearest.case_id],
                latency_ms=(time.perf_counter() - started) * 1000.0,
                model_name=f"{FALLBACK_MODEL_NAME} ({reason[:60]})",
            )

        return RootCauseAnalysis(
            root_cause=(
                "No historical precedent and no LLM analysis available. "
                "Manual engineering review required."
            ),
            evidence=[
                f"Anomaly score {vision.anomaly_score:.3f} against threshold "
                f"{vision.decision_threshold:.2f}",
                "No comparable historical cases retrieved",
            ],
            recommended_action=(
                "Route to a quality engineer for manual assessment and record the "
                "outcome so future inspections have a precedent."
            ),
            risk_level=RiskLevel.HIGH if vision.verdict == Verdict.FAIL else RiskLevel.LOW,
            confidence=0.1,
            affected_process_step=None,
            cited_case_ids=[],
            latency_ms=(time.perf_counter() - started) * 1000.0,
            model_name=f"{FALLBACK_MODEL_NAME} ({reason[:60]})",
        )
