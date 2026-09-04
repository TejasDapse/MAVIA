#!/usr/bin/env python3
"""Render REPORT.md to a typeset PDF.

Reuses ``mavia.agents.report.render_pdf`` - the same WeasyPrint path the system
uses for its own QA records - so the project's final report is produced by the
project's own machinery rather than a separate toolchain.

Usage:
    uv run python scripts/build_report.py
    uv run python scripts/build_report.py --input REPORT.md --output REPORT.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mavia.agents.report import render_pdf

REPO_ROOT = Path(__file__).resolve().parents[1]

STYLESHEET = """
@page {
  size: A4; margin: 20mm 18mm 18mm;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-size: 8pt; color: #6b7280;
  }
}
@page :first { @bottom-center { content: ""; } }
body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 9.6pt; line-height: 1.5; color: #111827;
}
h1 {
  font-size: 19pt; margin: 0 0 3mm; line-height: 1.25;
  color: #0f172a;
}
h2 {
  font-size: 12.5pt; margin: 8mm 0 2.5mm; padding-bottom: 1.2mm;
  border-bottom: 1.5px solid #cbd5e1; color: #0f172a;
  page-break-after: avoid;
}
h3 {
  font-size: 10.5pt; margin: 5mm 0 1.5mm; color: #334155;
  page-break-after: avoid;
}
p { margin: 0 0 2.2mm; text-align: justify; }
table {
  width: 100%; border-collapse: collapse; margin: 2.5mm 0 4mm;
  font-size: 8.8pt; page-break-inside: avoid;
}
th, td {
  text-align: left; padding: 1.5mm 2mm;
  border-bottom: 1px solid #e2e8f0; vertical-align: top;
}
th { background: #f8fafc; font-weight: 600; color: #334155; }
code {
  font-family: "SF Mono", Menlo, monospace; font-size: 8.4pt;
  background: #f1f5f9; padding: 0.3mm 1mm; border-radius: 1mm;
}
pre {
  background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 1.5mm;
  padding: 3mm; font-size: 8pt; line-height: 1.35; overflow-x: auto;
  page-break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: 8pt; }
ul, ol { margin: 1mm 0 3mm 5mm; padding: 0; }
li { margin-bottom: 1.2mm; }
hr { border: none; border-top: 1px solid #e2e8f0; margin: 6mm 0; }
blockquote {
  margin: 2mm 0; padding: 2mm 4mm; border-left: 2.5mm solid #cbd5e1;
  color: #475569;
}
strong { color: #0f172a; }
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "REPORT.md")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "REPORT.pdf")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"No report at {args.input}", file=sys.stderr)
        return 1

    try:
        import markdown
    except ImportError:
        print(
            "The `markdown` package is required.\n  uv sync --extra report",
            file=sys.stderr,
        )
        return 1

    body = markdown.markdown(
        args.input.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>MAVIA — Final Report</title>"
        f"<style>{STYLESHEET}</style></head><body>{body}</body></html>"
    )

    html_path = args.output.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")

    pdf = render_pdf(html, args.output)
    if pdf is None:
        print(f"PDF toolchain unavailable — wrote {html_path} instead.")
        print("  macOS: brew install pango    Debian: apt install libpango-1.0-0 libcairo2")
        return 1

    print(f"Wrote {pdf} ({pdf.stat().st_size / 1024:.0f} KB)")
    print(f"Wrote {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
