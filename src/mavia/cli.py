"""MAVIA command line interface."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from mavia import __version__
from mavia.audit import read_events, verify_chain
from mavia.config import get_settings

app = typer.Typer(help="MAVIA - Multi-Agent Visual Inspection & Audit System", no_args_is_help=True)
audit_app = typer.Typer(help="Inspect and verify the tamper-evident audit trail.")
app.add_typer(audit_app, name="audit")
console = Console()


@app.command()
def version() -> None:
    """Print the MAVIA version."""
    console.print(f"MAVIA {__version__}")


@app.command()
def inspect(
    image: Path = typer.Argument(..., help="Product image to inspect."),
    category: str = typer.Option(None, help="Product category; inferred from the path if omitted."),
) -> None:
    """Run a full inspection. Pauses for approval if the risk is high."""
    from mavia.orchestrator.graph import InspectionPipeline

    with InspectionPipeline() as pipeline:
        state = pipeline.run(image, category)
        _render_inspection(state)

        pending = pipeline.pending_approval(state.inspection_id)
        if pending:
            console.print("\n[bold yellow]HUMAN APPROVAL REQUIRED[/]")
            console.print(f"  Escalated because: {pending['reason']}")
            console.print(f"  Root cause  : {pending.get('root_cause')}")
            console.print(f"  Action      : {pending.get('recommended_action')}")
            console.print(
                f"\n  Approve with: [bold]mavia approve {state.inspection_id} "
                f'--approver you@plant --rationale "..."[/]'
            )
            console.print(f"  Reject with : [bold]mavia approve {state.inspection_id} --reject[/]")


@app.command()
def approve(
    inspection_id: str = typer.Argument(..., help="Inspection awaiting approval."),
    approver: str = typer.Option("cli-user", help="Who is making the decision."),
    rationale: str = typer.Option(None, help="Why."),
    reject: bool = typer.Option(False, "--reject", help="Reject instead of approving."),
) -> None:
    """Resume a suspended inspection with a human decision."""
    from mavia.orchestrator.graph import InspectionPipeline

    with InspectionPipeline() as pipeline:
        if not pipeline.is_suspended(inspection_id):
            console.print(f"[yellow]{inspection_id} is not awaiting approval.[/]")
            raise typer.Exit(code=1)

        state = pipeline.resume(
            inspection_id, approved=not reject, approver=approver, rationale=rationale
        )
        verb = "rejected" if reject else "approved"
        console.print(f"[green]Inspection {verb}[/] by {approver}")
        _render_inspection(state)


@app.command()
def pending() -> None:
    """List inspections waiting on a human decision."""
    from mavia.audit import read_events
    from mavia.orchestrator.graph import InspectionPipeline

    settings = get_settings()
    seen = {event.inspection_id for event in read_events(settings.audit_log_path)}

    with InspectionPipeline() as pipeline:
        rows = [
            (inspection_id, pipeline.pending_approval(inspection_id))
            for inspection_id in sorted(seen)
        ]
        waiting = [(i, p) for i, p in rows if p]

        if not waiting:
            console.print("No inspections awaiting approval.")
            return

        table = Table(title="Awaiting human approval")
        for column in ("inspection", "category", "risk", "reason"):
            table.add_column(column)
        for inspection_id, payload in waiting:
            table.add_row(
                inspection_id,
                str(payload.get("category")),
                str(payload.get("risk_level")),
                str(payload.get("reason")),
            )
        console.print(table)


def _render_inspection(state: object) -> None:
    """Print the outcome of an inspection."""
    from mavia.schemas import InspectionState

    assert isinstance(state, InspectionState)
    table = Table(title=f"Inspection {state.inspection_id}", show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")

    table.add_row("Image", state.image_path)
    table.add_row("Category", state.category or "-")
    if state.vision:
        table.add_row("Verdict", state.vision.verdict.value)
        table.add_row(
            "Anomaly score",
            f"{state.vision.anomaly_score:.3f} (threshold {state.vision.decision_threshold:.2f})",
        )
        table.add_row("Regions", str(len(state.vision.regions)))
        if state.vision.overlay_path:
            table.add_row("Overlay", state.vision.overlay_path)
    if state.retrieval:
        table.add_row("History retrieved", f"{len(state.retrieval.cases)} case(s)")
    if state.analysis:
        table.add_row("Risk level", state.analysis.risk_level.value)
        table.add_row("Root cause", state.analysis.root_cause)
        table.add_row("Action", state.analysis.recommended_action)
        table.add_row("Confidence", f"{state.analysis.confidence:.2f}")
        table.add_row("Cited cases", ", ".join(state.analysis.cited_case_ids) or "-")
    table.add_row("Approval", state.approval.status.value)
    if state.total_latency_ms:
        table.add_row("Total latency", f"{state.total_latency_ms:.0f} ms")
    if state.errors:
        table.add_row("Errors", "; ".join(state.errors))

    console.print(table)


@app.command()
def doctor() -> None:
    """Report which parts of the stack are configured and reachable."""
    settings = get_settings()
    settings.ensure_dirs()

    table = Table(title="MAVIA environment", show_lines=False)
    table.add_column("Component")
    table.add_column("Status")
    table.add_column("Detail")

    def row(name: str, ok: bool, detail: str) -> None:
        table.add_row(name, "[green]ready[/]" if ok else "[yellow]not set[/]", detail)

    row("Anthropic API key", settings.anthropic_api_key is not None, settings.llm_model)
    row(
        "Vector store",
        True,
        f"server {settings.qdrant_url}"
        if settings.uses_qdrant_server
        else f"embedded {settings.qdrant_path}",
    )
    row("MVTec AD dataset", settings.mvtec_dir.is_dir(), str(settings.mvtec_dir))
    row("Models dir", settings.models_dir.is_dir(), str(settings.models_dir))
    row("Audit log", settings.audit_log_path.exists(), str(settings.audit_log_path))

    console.print(table)


@audit_app.command("verify")
def audit_verify(
    log_path: Path = typer.Option(None, help="Defaults to the configured audit log."),
) -> None:
    """Re-derive the hash chain and report whether the trail is intact."""
    path = log_path or get_settings().audit_log_path
    result = verify_chain(path)
    if result:
        console.print(f"[green]Chain intact[/] - {result.checked} entries verified in {path}")
    else:
        console.print(f"[red]Chain BROKEN[/] at seq {result.broken_at} in {path}")
        raise typer.Exit(code=1)


@audit_app.command("show")
def audit_show(
    inspection_id: str = typer.Argument(None, help="Filter to one inspection."),
    log_path: Path = typer.Option(None),
    limit: int = typer.Option(50),
) -> None:
    """Print recent audit events."""
    path = log_path or get_settings().audit_log_path
    events = [e for e in read_events(path) if not inspection_id or e.inspection_id == inspection_id]

    table = Table(title=f"Audit trail ({path})")
    for column in ("seq", "timestamp", "inspection", "agent", "action", "entry_hash"):
        table.add_column(column)
    for event in events[-limit:]:
        table.add_row(
            str(event.seq),
            event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            event.inspection_id,
            str(event.agent),
            event.action,
            event.entry_hash[:12] + "...",
        )
    console.print(table)


if __name__ == "__main__":
    app()
