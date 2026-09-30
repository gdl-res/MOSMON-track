#!/usr/bin/env python3
"""Export a tracking run's report.html to a CSS-faithful PDF.

Thin CLI wrapper around ``mosmon_tracking.reports.export_report_pdf``: renders
``<run-dir>/report.html`` to a PDF that matches the HTML report (styled tables, figure
gallery, images sized to fit the page) via WeasyPrint.

The pipeline already writes report.pdf on every run (``reports.make_pdf: true`` by
default); this script is for one-off/after-the-fact export, or to append an extra PDF.

Requires the optional ``pdf`` extra:  pip install '.[pdf]'   (weasyprint + pypdf)

Usage:
    python scripts/report_to_pdf.py --run-dir outputs/example_run
    # append another PDF as an appendix:
    python scripts/report_to_pdf.py --run-dir outputs/example_run --report extra.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mosmon_tracking.reports import export_report_pdf


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="Run output folder containing report.html")
    p.add_argument("--report", default="none",
                   help="Optional PDF to append as an appendix (default: none)")
    p.add_argument("--out", default=None, help="Output PDF path (default: <run-dir>/report.pdf)")
    a = p.parse_args()

    html = Path(a.run_dir) / "report.html"
    if not html.exists():
        sys.exit(f"error: {html} not found (run track-video with make_html first).")
    appendix = None if a.report.lower() == "none" else a.report
    if appendix and not Path(appendix).exists():
        sys.exit(f"error: appendix PDF not found: {appendix}")
    out_pdf = Path(a.out) if a.out else Path(a.run_dir) / "report.pdf"

    try:
        export_report_pdf(html, out_pdf, appendix=appendix)
    except RuntimeError as exc:
        sys.exit(f"error: {exc}")

    from pypdf import PdfReader

    n = len(PdfReader(str(out_pdf)).pages)
    tail = " + appendix" if appendix else ""
    print(f"✅ Wrote {out_pdf} ({n} pages: report{tail})")


if __name__ == "__main__":
    main()
