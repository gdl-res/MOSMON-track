"""Per-run reports: video_summary.json, report.md, and optional report.html.

Reports embed the interpretation caveats (see README) so that every output
ships with the warnings a reader needs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

CAVEATS = [
    "Bounding-box tracking estimates body-center trajectories, not posture or head-tail orientation.",
    "Heading from center displacement is unreliable when larvae pause or jitter.",
    "Apparent motion can be affected by camera motion, water perturbation, turbidity, reflections, and optical distortion.",
    "Without calibration, speed and distance are pixel-based, not biological physical units.",
    "Track IDs may fragment under occlusion or dense aggregation; filter behaviour metrics by track quality.",
    "In dual-container videos, species comes from the spatial compartment "
    "(regions.dual_container_split_x_fraction, verified per video against a frame); "
    "the detector's per-box class is not a validated species label and is not used as one.",
    "Frame subsampling changes apparent speed and event timing; true FPS and frame stride are used for temporal features.",
]


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_video_summary_json(summary: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
    return path


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "n/a" if (v != v) else f"{v:.4g}"
    return str(v)


def _dict_table(d: dict) -> str:
    lines = ["| metric | value |", "| --- | --- |"]
    for k, v in d.items():
        if isinstance(v, (dict, list)):
            continue
        lines.append(f"| {k} | {_fmt(v)} |")
    return "\n".join(lines)


def build_markdown(
    video_info: dict,
    model_settings: dict,
    qc: dict,
    video_summary: dict,
    plot_paths: list[str],
    heatmap_paths: list[str],
) -> str:
    parts: list[str] = []
    parts.append(f"# Behaviour report: {video_info.get('filename', 'video')}\n")
    parts.append("## 1. Input video metadata\n")
    parts.append(_dict_table(video_info) + "\n")
    parts.append("## 2. Model and tracker settings\n")
    parts.append(_dict_table(model_settings) + "\n")
    parts.append("## 3. Detection / tracking QC\n")
    parts.append(_dict_table(qc) + "\n")
    if qc.get("warnings"):
        parts.append("**QC warnings:**\n")
        parts.extend(f"- {w}" for w in qc["warnings"])
        parts.append("")
    parts.append("## 4. Behaviour & population summary\n")
    parts.append(_dict_table(video_summary) + "\n")
    parts.append("## 5. Plots\n")
    parts.extend(f"![{Path(p).stem}]({p})" for p in plot_paths)
    parts.append("\n## 6. Heatmaps\n")
    parts.extend(f"![{Path(p).stem}]({p})" for p in heatmap_paths)
    parts.append("\n## 7. Interpretation caveats\n")
    parts.extend(f"- {c}" for c in CAVEATS)
    return "\n".join(parts)


def write_markdown(md_text: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md_text, encoding="utf-8")
    return path


_HTML_STYLE = """
:root { --fg:#1a1a1a; --muted:#666; --line:#ddd; --accent:#0b6; --warn:#b22; }
* { box-sizing: border-box; }
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
       color: var(--fg); max-width: 1100px; margin: 0 auto; padding: 1.5rem 1.25rem 4rem; line-height: 1.5; }
h1 { font-size: 1.6rem; border-bottom: 2px solid var(--accent); padding-bottom: .4rem; }
h2 { font-size: 1.2rem; margin-top: 2rem; border-bottom: 1px solid var(--line); padding-bottom: .25rem; }
table { border-collapse: collapse; margin: .5rem 0 1rem; font-size: .92rem; }
th, td { border: 1px solid var(--line); padding: 5px 10px; text-align: left; vertical-align: top; }
th { background: #f6f6f6; font-weight: 600; width: 320px; }
.warn { color: var(--warn); }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1rem; }
.gallery figure { margin: 0; border: 1px solid var(--line); border-radius: 6px; padding: .5rem; background: #fafafa; }
.gallery img { width: 100%; height: auto; border-radius: 4px; }
.gallery figcaption { font-size: .8rem; color: var(--muted); margin-top: .35rem; word-break: break-word; }
ul.caveats li { margin: .25rem 0; color: #333; }
.meta { color: var(--muted); font-size: .85rem; }
"""


def _esc(v) -> str:
    import html

    return html.escape("" if v is None else str(v))


def _html_table(d: dict) -> str:
    rows = []
    for k, v in d.items():
        if isinstance(v, (dict, list)):
            continue
        rows.append(f"<tr><th>{_esc(k)}</th><td>{_esc(_fmt(v))}</td></tr>")
    return "<table>" + "".join(rows) + "</table>" if rows else "<p class='meta'>(no data)</p>"


def _html_gallery(paths: list[str]) -> str:
    from urllib.parse import quote

    if not paths:
        return "<p class='meta'>(none)</p>"
    figs = []
    for p in paths:
        src = quote(str(p))  # encode spaces in e.g. "occupancy_aedes aegypti.png"
        figs.append(
            f"<figure><a href='{src}' target='_blank'><img src='{src}' loading='lazy' "
            f"alt='{_esc(Path(p).stem)}'></a><figcaption>{_esc(Path(p).stem)}</figcaption></figure>"
        )
    return "<div class='gallery'>" + "".join(figs) + "</div>"


def build_html(
    video_info: dict,
    model_settings: dict,
    qc: dict,
    video_summary: dict,
    plot_paths: list[str],
    heatmap_paths: list[str],
) -> str:
    """Build a standalone, properly-rendered HTML report (real tables + images)."""
    title = video_info.get("filename") or "video"
    warnings = qc.get("warnings") or []
    warn_html = ""
    if warnings:
        items = "".join(f"<li class='warn'>{_esc(w)}</li>" for w in warnings)
        warn_html = f"<p><strong class='warn'>QC warnings:</strong></p><ul>{items}</ul>"
    caveats = "".join(f"<li>{_esc(c)}</li>" for c in CAVEATS)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MOSMON behaviour report — {_esc(title)}</title>
<style>{_HTML_STYLE}</style></head>
<body>
<h1>Behaviour report</h1>
<p class="meta">{_esc(title)}</p>
<h2>1. Input video metadata</h2>{_html_table(video_info)}
<h2>2. Model and tracker settings</h2>{_html_table(model_settings)}
<h2>3. Detection / tracking QC</h2>{_html_table(qc)}{warn_html}
<h2>4. Behaviour &amp; population summary</h2>{_html_table(video_summary)}
<h2>5. Plots</h2>{_html_gallery(plot_paths)}
<h2>6. Heatmaps</h2>{_html_gallery(heatmap_paths)}
<h2>7. Interpretation caveats</h2><ul class="caveats">{caveats}</ul>
</body></html>"""


def write_html(html_text: str, path: str | Path) -> Path:
    """Write a prebuilt HTML report string (see :func:`build_html`) to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# PDF export (CSS-faithful, via WeasyPrint) with optional appendix
# --------------------------------------------------------------------------- #

# Print stylesheet layered on top of the report's own CSS so the on-screen HTML
# paginates cleanly onto A4: real page margins, tables that wrap instead of
# overflowing the page, a 2-up figure gallery, and figures kept whole across breaks.
_PDF_PRINT_CSS = """
@page { size: A4; margin: 14mm 12mm; }
body { max-width: none !important; padding: 0 !important; }
table { width: 100% !important; table-layout: fixed; }
th { width: 38% !important; overflow-wrap: anywhere; }
td { overflow-wrap: anywhere; }
.gallery { grid-template-columns: repeat(2, 1fr) !important; }
.gallery figure { break-inside: avoid; }
img { max-width: 100% !important; height: auto !important; }
h1, h2 { break-after: avoid; }
p, .meta { overflow-wrap: anywhere; }
"""


def export_report_pdf(
    html_path: str | Path,
    out_path: str | Path,
    appendix: str | Path | None = None,
) -> Path:
    """Render an HTML report to a CSS-faithful PDF (optionally appending a PDF appendix).

    Uses WeasyPrint so the PDF matches the HTML report (styled tables, figure gallery)
    with images sized to fit the page. If ``appendix`` is given and exists, its pages are
    appended and two section bookmarks are added ("Behaviour report" / "Appendix ...").
    Requires the optional ``pdf`` extra (``weasyprint``, ``pypdf``); raises RuntimeError
    with a clear message if those are unavailable so callers can degrade gracefully.
    """
    html_path = Path(html_path)
    out_path = Path(out_path)
    try:
        from weasyprint import CSS, HTML
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PDF export needs WeasyPrint (pip install 'mosmon-tracking[pdf]')"
        ) from exc

    # base_url = the report's folder so relative plot/heatmap <img src> resolve + embed.
    body_pdf = HTML(str(html_path), base_url=str(html_path.parent)).render(
        stylesheets=[CSS(string=_PDF_PRINT_CSS)]
    ).write_pdf()

    appendix_path = Path(appendix) if appendix else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if appendix_path is None or not appendix_path.exists():
        out_path.write_bytes(body_pdf)
        return out_path

    import io

    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for page in PdfReader(io.BytesIO(body_pdf)).pages:
        writer.add_page(page)
    appendix_start = len(writer.pages)
    for page in PdfReader(str(appendix_path)).pages:
        writer.add_page(page)
    writer.add_outline_item("Behaviour report", 0)
    writer.add_outline_item("Appendix", appendix_start)
    with out_path.open("wb") as fh:
        writer.write(fh)
    return out_path


# --------------------------------------------------------------------------- #
# Aggregate (cross-video) report
# --------------------------------------------------------------------------- #

def _df_html(df) -> str:
    """Render a whole DataFrame as an HTML table (escaped, numbers tidied)."""
    if df is None or getattr(df, "empty", True):
        return "<p class='meta'>(no data)</p>"
    return df.to_html(index=False, na_rep="n/a", border=0,
                      float_format=lambda x: _fmt(float(x)))


def _df_md(df) -> str:
    """Render a whole DataFrame as a Markdown table (no tabulate dependency)."""
    if df is None or getattr(df, "empty", True):
        return "_(no data)_"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def build_aggregate_markdown(title: str, sections: list[tuple[str, Any]],
                             plot_paths: list[str] | None = None) -> str:
    """Markdown roll-up across videos/models. ``sections`` is (heading, DataFrame)."""
    parts = [f"# Aggregate behaviour report: {title}\n"]
    for i, (heading, df) in enumerate(sections, 1):
        parts.append(f"## {i}. {heading}\n")
        parts.append(_df_md(df) + "\n")
    if plot_paths:
        parts.append(f"## {len(sections) + 1}. Plots\n")
        parts.extend(f"![{Path(p).stem}]({p})" for p in plot_paths)
        parts.append("")
    parts.append(f"## {len(sections) + (2 if plot_paths else 1)}. Interpretation caveats\n")
    parts.extend(f"- {c}" for c in CAVEATS)
    return "\n".join(parts)


def build_aggregate_html(title: str, sections: list[tuple[str, Any]],
                         plot_paths: list[str] | None = None) -> str:
    """Standalone HTML roll-up across videos/models. ``sections`` is (heading, DataFrame)."""
    body = [f"<h1>Aggregate behaviour report</h1><p class='meta'>{_esc(title)}</p>"]
    for i, (heading, df) in enumerate(sections, 1):
        body.append(f"<h2>{i}. {_esc(heading)}</h2>{_df_html(df)}")
    n = len(sections)
    if plot_paths:
        n += 1
        body.append(f"<h2>{n}. Plots</h2>{_html_gallery(plot_paths)}")
    caveats = "".join(f"<li>{_esc(c)}</li>" for c in CAVEATS)
    body.append(f"<h2>{n + 1}. Interpretation caveats</h2><ul class='caveats'>{caveats}</ul>")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MOSMON aggregate report — {_esc(title)}</title>
<style>{_HTML_STYLE}
table th {{ width: auto; }}
</style></head>
<body>
{''.join(body)}
</body></html>"""
