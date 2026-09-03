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
