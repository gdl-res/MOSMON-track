"""Tests for the report -> PDF export (skipped if the optional [pdf] deps are absent)."""
import pytest

pytest.importorskip("weasyprint")
pytest.importorskip("pypdf")

from pypdf import PdfReader, PdfWriter  # noqa: E402

from mosmon_tracking.reports import export_report_pdf  # noqa: E402

_HTML = "<h1>Behaviour report</h1><table><tr><th>total_tracks</th><td>3</td></tr></table>"


def _write_html(tmp_path):
    p = tmp_path / "report.html"
    p.write_text(_HTML, encoding="utf-8")
    return p


def test_export_without_appendix(tmp_path):
    html = _write_html(tmp_path)
    out = export_report_pdf(html, tmp_path / "report.pdf", appendix=None)
    assert out.exists()
    assert len(PdfReader(str(out)).pages) >= 1


def test_export_with_appendix_adds_pages_and_bookmarks(tmp_path):
    html = _write_html(tmp_path)
    # a 2-page stand-in appendix
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    w.add_blank_page(width=200, height=200)
    appendix = tmp_path / "appendix.pdf"
    with appendix.open("wb") as fh:
        w.write(fh)

    out = export_report_pdf(html, tmp_path / "report.pdf", appendix=appendix)
    reader = PdfReader(str(out))
    assert len(reader.pages) >= 3  # >=1 body page + 2 appendix pages
    titles = [o.title for o in reader.outline if hasattr(o, "title")]
    assert titles == ["Behaviour report", "Appendix"]


def test_export_missing_appendix_falls_back_to_body_only(tmp_path):
    html = _write_html(tmp_path)
    out = export_report_pdf(html, tmp_path / "report.pdf", appendix=tmp_path / "nope.pdf")
    # missing appendix -> body only, no crash; the appendix bookmark is not added.
    assert out.exists()
    titles = [o.title for o in PdfReader(str(out)).outline if hasattr(o, "title")]
    assert "Appendix" not in titles
