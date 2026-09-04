"""MAVIA operations dashboard.

Run with:  uv run streamlit run src/mavia/dashboard/app.py

Four views, matching what a line supervisor actually needs:

* **Inspect** - run a unit and see the heatmap, history and analysis.
* **Approval queue** - the escalated inspections waiting on a human, each shown
  with the overlay so the reviewer can see the defect before deciding. Approving
  from here resumes the suspended LangGraph run; the process that started the
  inspection may have exited hours ago.
* **History** - every inspection, reconstructed from the audit log alone.
* **Audit** - live chain verification and the raw event trail.

This file is a rendering layer only. All logic lives in ``service.py``, because
Streamlit re-executes the whole script on every interaction and that is a poor
place to put anything that deserves a test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `streamlit run src/mavia/dashboard/app.py` without an editable install.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mavia.audit import read_events  # noqa: E402
from mavia.config import get_settings  # noqa: E402
from mavia.dashboard import service  # noqa: E402

# Streamlit colour-text names, so risk reads at a glance in the queue.
RISK_COLOURS = {
    "LOW": "green",
    "MEDIUM": "orange",
    "HIGH": "red",
    "CRITICAL": "red",
}

st.set_page_config(page_title="MAVIA - Inspection & Audit", page_icon="🏭", layout="wide")


@st.cache_resource(show_spinner="Loading models (first run only)...")
def get_pipeline():  # type: ignore[no-untyped-def]
    """One pipeline for the session. Models are expensive to load; Streamlit
    would otherwise rebuild them on every widget interaction."""
    from mavia.orchestrator.graph import InspectionPipeline

    return InspectionPipeline()


def risk_badge(risk: str | None) -> str:
    if not risk:
        return "—"
    return f":{RISK_COLOURS.get(risk, 'gray')}[**{risk}**]"


# ---------------------------------------------------------------- sidebar

settings = get_settings()
st.sidebar.title("🏭 MAVIA")
st.sidebar.caption("Multi-Agent Visual Inspection & Audit System")

status = service.chain_status(settings)
if status["valid"]:
    st.sidebar.success(f"Audit chain intact\n\n{status['checked']} entries verified")
else:
    st.sidebar.error(f"AUDIT CHAIN BROKEN at seq {status['broken_at']}")

summaries = service.load_inspections(settings)
metrics = service.fleet_metrics(summaries)
if metrics["awaiting_approval"]:
    st.sidebar.warning(f"{metrics['awaiting_approval']} awaiting approval")

view = st.sidebar.radio(
    "View", ["Inspect", "Approval queue", "History", "Audit trail"], label_visibility="collapsed"
)
st.sidebar.divider()
st.sidebar.caption(
    f"LLM: `{settings.llm_model}`"
    + ("" if settings.anthropic_api_key else "  \n⚠️ no API key — rule-based fallback")
)


# ----------------------------------------------------------------- inspect

if view == "Inspect":
    st.title("Run an inspection")

    source = st.radio("Image source", ["Sample from MVTec AD", "Upload"], horizontal=True)
    image_path: Path | None = None

    if source == "Sample from MVTec AD":
        images = service.available_images(settings)
        if not images:
            st.warning("No dataset found. Run `uv run python scripts/download_mvtec.py`.")
        else:
            labels = [service.describe_image(p) for p in images]
            choice = st.selectbox(
                "Sample", options=range(len(images)), format_func=lambda i: labels[i]
            )
            image_path = images[choice]
    else:
        uploaded = st.file_uploader("Product image", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            destination = Path(settings.artifacts_dir) / "uploads" / uploaded.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(uploaded.getbuffer())
            image_path = destination

    if image_path:
        left, right = st.columns(2)
        left.image(str(image_path), caption="Input", width="stretch")

        if st.button("Inspect", type="primary"):
            with st.spinner("Running vision → retrieval → analysis..."):
                state = get_pipeline().run(image_path)
            st.session_state["last_inspection"] = state.inspection_id

            if state.vision and state.vision.overlay_path:
                right.image(state.vision.overlay_path, caption="Anomaly overlay", width="stretch")

            verdict = state.final_verdict.value
            cols = st.columns(4)
            cols[0].metric("Verdict", verdict)
            if state.vision:
                cols[1].metric("Anomaly score", f"{state.vision.anomaly_score:.3f}")
            if state.analysis:
                cols[2].metric("Risk", state.analysis.risk_level.value)
                cols[3].metric("Confidence", f"{state.analysis.confidence:.2f}")

            pending = get_pipeline().pending_approval(state.inspection_id)
            if pending:
                st.warning(
                    f"**Human approval required** — {pending['reason']}\n\n"
                    "This inspection is suspended and persisted. Approve it in the "
                    "**Approval queue** tab; the decision can be made any time, even "
                    "after this process restarts."
                )
            else:
                st.success("Completed without escalation. Report filed.")

            if state.analysis:
                st.subheader("Root cause analysis")
                st.markdown(f"**Probable cause.** {state.analysis.root_cause}")
                st.markdown(f"**Recommended action.** {state.analysis.recommended_action}")
                if state.analysis.evidence:
                    st.markdown("**Evidence.**")
                    for item in state.analysis.evidence:
                        st.markdown(f"- {item}")
                st.caption(
                    f"Produced by `{state.analysis.model_name}` · cited: "
                    f"{', '.join(state.analysis.cited_case_ids) or 'none'} "
                    "(each verified against the retrieved set)"
                )

            if state.retrieval and state.retrieval.cases:
                st.subheader("Comparable historical cases")
                st.dataframe(
                    [
                        {
                            "case": c.case_id,
                            "defect": c.defect_type,
                            "similarity": round(c.similarity, 3),
                            "recorded root cause": c.root_cause,
                        }
                        for c in state.retrieval.cases
                    ],
                    width="stretch",
                    hide_index=True,
                )


# ---------------------------------------------------------- approval queue

elif view == "Approval queue":
    st.title("Awaiting human approval")
    st.caption(
        "High-risk inspections suspend here and persist to disk. The process that "
        "started them may have exited; approving resumes the graph where it stopped."
    )

    pipeline = get_pipeline()
    waiting = []
    for summary in summaries:
        payload = pipeline.pending_approval(summary.inspection_id)
        if payload:
            waiting.append((summary, payload))

    if not waiting:
        st.success("Nothing awaiting approval.")
    for summary, payload in waiting:
        with st.container(border=True):
            header, image_col = st.columns([2, 1])
            header.subheader(f"{payload.get('category', '?')} · {summary.inspection_id}")
            header.markdown(f"**Escalated because:** {payload.get('reason')}")
            header.markdown(f"**Risk:** {risk_badge(payload.get('risk_level'))}")
            header.markdown(f"**Probable cause.** {payload.get('root_cause')}")
            header.markdown(f"**Recommended action.** {payload.get('recommended_action')}")
            header.caption(
                f"Confidence {payload.get('confidence', 0):.2f} · cited: "
                f"{', '.join(payload.get('cited_case_ids') or []) or 'none'}"
            )

            overlay = payload.get("overlay_path")
            if overlay and Path(overlay).exists():
                image_col.image(overlay, caption="Anomaly overlay", width="stretch")
            elif payload.get("image_path") and Path(payload["image_path"]).exists():
                image_col.image(payload["image_path"], width="stretch")

            approver = st.text_input(
                "Your identity", key=f"who-{summary.inspection_id}", placeholder="you@plant"
            )
            rationale = st.text_area(
                "Rationale (recorded in the audit trail)", key=f"why-{summary.inspection_id}"
            )
            approve_col, reject_col = st.columns(2)

            if approve_col.button("Approve", key=f"ok-{summary.inspection_id}", type="primary"):
                if not approver:
                    st.error("Identity is required — approvals are attributed in the audit trail.")
                else:
                    pipeline.resume(
                        summary.inspection_id, True, approver=approver, rationale=rationale or None
                    )
                    st.success("Approved and recorded.")
                    st.rerun()

            if reject_col.button("Reject", key=f"no-{summary.inspection_id}"):
                if not approver:
                    st.error("Identity is required — rejections are attributed too.")
                else:
                    pipeline.resume(
                        summary.inspection_id, False, approver=approver, rationale=rationale or None
                    )
                    st.warning("Rejected and recorded.")
                    st.rerun()


# ----------------------------------------------------------------- history

elif view == "History":
    st.title("Inspection history")

    cols = st.columns(5)
    cols[0].metric("Inspections", metrics["total"])
    cols[1].metric("Defect rate", f"{metrics['defect_rate']:.0%}")
    cols[2].metric("Escalation rate", f"{metrics['escalation_rate']:.0%}")
    cols[3].metric(
        "Median latency",
        f"{metrics['median_latency_ms'] / 1000:.1f}s" if metrics["median_latency_ms"] else "—",
    )
    cols[4].metric("With errors", metrics["with_errors"])

    if metrics["risk_breakdown"]:
        st.bar_chart(metrics["risk_breakdown"], horizontal=True)

    if not summaries:
        st.info("No inspections yet. Run one from the **Inspect** tab.")
    else:
        st.dataframe(
            [
                {
                    "inspection": s.inspection_id,
                    "started": s.started_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "category": s.category,
                    "verdict": s.verdict or "in progress",
                    "risk": s.risk_level,
                    "approval": s.approval,
                    "approver": s.approver,
                    "latency (s)": round(s.total_latency_ms / 1000, 1)
                    if s.total_latency_ms
                    else None,
                    "errors": len(s.errors),
                }
                for s in summaries
            ],
            width="stretch",
            hide_index=True,
        )

        chosen = st.selectbox("Open report", options=[s.inspection_id for s in summaries], index=0)
        files = service.report_paths(chosen, settings)
        if not files:
            st.info("No report on disk — the inspection may still be awaiting approval.")
        else:
            if "md" in files:
                with st.expander("Report", expanded=True):
                    st.markdown(files["md"].read_text(encoding="utf-8"))
            if "pdf" in files:
                st.download_button(
                    "Download PDF",
                    files["pdf"].read_bytes(),
                    file_name=files["pdf"].name,
                    mime="application/pdf",
                )


# ------------------------------------------------------------- audit trail

else:
    st.title("Audit trail")
    st.caption(
        "Append-only SHA-256 hash chain. Each entry commits to its predecessor's "
        "digest, so altering or deleting any record invalidates every entry after it."
    )

    if status["valid"]:
        st.success(f"Chain intact — {status['checked']} entries verified.")
    else:
        st.error(
            f"Chain BROKEN at sequence {status['broken_at']}. "
            "Records at or after this point have been altered or removed."
        )

    events = list(read_events(settings.audit_log_path))
    if not events:
        st.info("No audit events yet.")
    else:
        ids = ["(all)", *sorted({e.inspection_id for e in events})]
        chosen = st.selectbox("Filter by inspection", ids)
        shown = events if chosen == "(all)" else [e for e in events if e.inspection_id == chosen]

        st.dataframe(
            [
                {
                    "seq": e.seq,
                    "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "inspection": e.inspection_id,
                    "agent": str(e.agent),
                    "action": e.action,
                    "entry hash": e.entry_hash[:24] + "…",
                }
                for e in reversed(shown)
            ],
            width="stretch",
            hide_index=True,
            height=520,
        )
        st.caption(f"Chain head: `{events[-1].entry_hash}`")
