"""Generates a single-page, scannable HTML "report card" for one
interview's analysis -- a confidence dial, strengths vs. issues at a
glance, and a question-by-question breakdown with suggested rewrites.

Written alongside the existing markdown report (see report.py) whenever
analysis finishes, and opened with the OS's default browser via the
dashboard's History tab "View infographic" button. Self-contained (no
external fonts/scripts/CSS) so it opens correctly offline.
"""
from __future__ import annotations

import html
import pathlib
from typing import Optional

from .config_loader import Config
from .db import InterviewRecord
from .report import _stringify

# Muted, professional palette -- avoids the near-universal AI-generated-
# design defaults (warm cream + terracotta, or neon-on-near-black).
_INK = "#1c232b"
_INK_SOFT = "#4a5563"
_INK_FAINT = "#7b8494"
_GROUND = "#f4f6f7"
_PANEL = "#ffffff"
_LINE = "#dde2e6"
_ACCENT = "#0f6e77"
_ACCENT_INK = "#0a4d54"
_ACCENT_TINT = "#e2f0f1"
_GOOD = "#3d7a4a"
_GOOD_TINT = "#e7f2e9"
_WATCH = "#b5701f"
_WATCH_TINT = "#faf0df"


def _e(value: object) -> str:
    """Escapes arbitrary (possibly model-generated) text for safe HTML
    embedding -- this file is opened directly in a real browser, so
    unescaped analysis text would be a real script-injection risk, not
    just a cosmetic one."""
    return html.escape(_stringify(value) if not isinstance(value, str) else value, quote=True)


def infographic_path(record: InterviewRecord, cfg: Config) -> pathlib.Path:
    out_dir = cfg.resolve(cfg.output.get("output_dir", "output")) / cfg.output.get(
        "reports_subdir", "reports"
    )
    date_str = record.started_at.split("T")[0]
    return out_dir / f"{date_str}_{record.source_app or 'interview'}_{record.id}_infographic.html"


def write_interview_infographic(record: InterviewRecord, cfg: Config) -> Optional[pathlib.Path]:
    """Writes the infographic HTML file and returns its path -- or None
    (writes nothing) if there's no usable analysis to visualize, same
    gating as the History tab's feedback panel (parse_error/no_speech_detected/
    no analysis at all -- see dashboard.py's _on_history_select)."""
    analysis = record.analysis
    if not analysis or analysis.get("parse_error") or analysis.get("no_speech_detected"):
        return None

    out_path = infographic_path(record, cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(record, analysis), encoding="utf-8")
    return out_path


def _confidence_dial_svg(score: Optional[int]) -> str:
    if score is None:
        return f"""<svg width="88" height="88" viewBox="0 0 88 88" role="img" aria-label="Confidence: not available">
<circle cx="44" cy="44" r="36" fill="none" stroke="{_LINE}" stroke-width="8"/>
<text x="44" y="48" text-anchor="middle" font-family="-apple-system,sans-serif" font-size="12" fill="{_INK_FAINT}">N/A</text>
</svg>"""
    circumference = 2 * 3.14159265 * 36
    offset = circumference * (1 - max(0, min(100, score)) / 100)
    return f"""<svg width="88" height="88" viewBox="0 0 88 88" role="img" aria-label="Confidence score: {score} out of 100">
<circle cx="44" cy="44" r="36" fill="none" stroke="{_LINE}" stroke-width="8"/>
<circle cx="44" cy="44" r="36" fill="none" stroke="{_ACCENT}" stroke-width="8" stroke-linecap="round"
        stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}" transform="rotate(-90 44 44)"/>
<text x="44" y="40" text-anchor="middle" font-family="Cascadia Code,SF Mono,Consolas,monospace"
      font-size="22" font-weight="600" fill="{_INK}">{score}</text>
<text x="44" y="55" text-anchor="middle" font-family="-apple-system,sans-serif" font-size="9" fill="{_INK_FAINT}">/ 100</text>
</svg>"""


def _issue_chip(issue) -> str:
    if isinstance(issue, dict):
        category = issue.get("category", "")
        detail = issue.get("detail", "")
        label = f"{category} — {detail}" if category and detail else (category or detail)
    else:
        label = _stringify(issue)
    return f'<span class="chip">{_e(label)}</span>'


def _qa_card(index: int, qa: dict) -> str:
    question = qa.get("question", "(question)")
    answer = qa.get("answer_summary", "")
    issues = qa.get("issues", []) or []
    improvement = qa.get("suggested_improvement", "")

    chips_html = "".join(_issue_chip(i) for i in issues)
    chips_block = f'<div class="chips">{chips_html}</div>' if chips_html else ""
    improvement_block = (
        f'<div class="improvement"><span class="improvement-label">Suggested improvement</span>{_e(improvement)}</div>'
        if improvement else ""
    )

    return f"""<div class="qa-card">
<span class="qnum">Q{index}</span>
<p class="question">{_e(question)}</p>
<p class="answer-label">Answer summary</p>
<p class="answer">{_e(answer)}</p>
{chips_block}
{improvement_block}
</div>"""


def _render(record: InterviewRecord, analysis: dict) -> str:
    qa_pairs = analysis.get("qa_pairs", []) or []
    summary = analysis.get("session_summary", {}) or {}
    confidence_info = analysis.get("confidence_info")
    score = (confidence_info or {}).get("score")

    date_str = record.started_at.split("T")[0]
    app_name = record.source_app or "Unknown app"
    title = f"Interview Report — {app_name}, {date_str}"

    strengths = summary.get("top_strengths") or []
    issues = summary.get("top_issues") or []
    focus = summary.get("one_thing_to_practice_next") or ""

    strengths_html = "".join(f"<li>{_e(_stringify(s))}</li>" for s in strengths) or "<li>None flagged</li>"
    issues_html = "".join(f"<li>{_e(_stringify(i))}</li>" for i in issues) or "<li>None flagged</li>"
    qa_html = "\n".join(_qa_card(i, qa) for i, qa in enumerate(qa_pairs, 1)) or (
        '<p class="empty-note">No individual questions were extracted from this transcript.</p>'
    )
    focus_block = (
        f'<div class="practice-note"><p class="label">Focus for next practice</p><p>{_e(focus)}</p></div>'
        if focus else ""
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title>
<style>
:root {{
  --ink: {_INK}; --ink-soft: {_INK_SOFT}; --ink-faint: {_INK_FAINT};
  --ground: {_GROUND}; --panel: {_PANEL}; --line: {_LINE};
  --accent: {_ACCENT}; --accent-ink: {_ACCENT_INK}; --accent-tint: {_ACCENT_TINT};
  --good: {_GOOD}; --good-tint: {_GOOD_TINT}; --watch: {_WATCH}; --watch-tint: {_WATCH_TINT};
  --font-display: Iowan Old Style, Palatino Linotype, Palatino, Georgia, serif;
  --font-body: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --font-mono: "Cascadia Code", "SF Mono", Consolas, "Courier New", monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --ink: #e9edf0; --ink-soft: #b3bcc6; --ink-faint: #7d8794;
    --ground: #14181c; --panel: #1b2126; --line: #2c343b;
    --accent: #4fb3ba; --accent-ink: #bfe6e8; --accent-tint: #1c2f31;
    --good: #7fbf8c; --good-tint: #1c2b1f; --watch: #e0a655; --watch-tint: #2e2416;
  }}
}}
* {{ box-sizing: border-box; }}
body {{ background: var(--ground); margin: 0; }}
.sheet {{ max-width: 760px; margin: 0 auto; padding: 2.5rem 1.75rem 3.5rem; font-family: var(--font-body); color: var(--ink); }}
.masthead {{ border-bottom: 1px solid var(--line); padding-bottom: 1.25rem; margin-bottom: 1.75rem; }}
.masthead .eyebrow {{ font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-faint); margin: 0 0 .35rem; }}
.masthead h1 {{ font-family: var(--font-display); font-weight: 600; font-size: 26px; margin: 0; }}
.masthead .meta {{ font-size: 13px; color: var(--ink-soft); margin-top: .4rem; }}
.masthead .meta code {{ font-family: var(--font-mono); font-size: 12px; }}
.top-grid {{ display: grid; grid-template-columns: minmax(0,1fr) 168px; gap: 1.25rem; margin-bottom: 2rem; }}
.practice-note {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 1.1rem 1.25rem; display: flex; flex-direction: column; justify-content: center; }}
.practice-note .label {{ font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: var(--accent-ink); margin: 0 0 .4rem; }}
.practice-note p {{ margin: 0; font-size: 15px; line-height: 1.5; }}
.confidence-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 1rem; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; }}
.confidence-card .dial-label {{ font-size: 10.5px; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-faint); margin: .5rem 0 0; }}
.confidence-card .dial-value {{ font-family: var(--font-mono); font-size: 12px; color: var(--ink-soft); }}
.columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-bottom: 2rem; }}
.panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 1.1rem 1.25rem 1.25rem; }}
.panel h2 {{ font-family: var(--font-display); font-size: 15px; font-weight: 600; margin: 0 0 .75rem; display: flex; align-items: center; gap: .4rem; }}
.panel h2 .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
.panel.strengths h2 .dot {{ background: var(--good); }}
.panel.issues h2 .dot {{ background: var(--watch); }}
.panel ul {{ margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column; gap: .55rem; }}
.panel li {{ font-size: 13.5px; line-height: 1.45; padding-left: .9rem; position: relative; color: var(--ink-soft); }}
.panel li::before {{ content: ""; position: absolute; left: 0; top: .5em; width: 5px; height: 5px; border-radius: 50%; }}
.panel.strengths li::before {{ background: var(--good); }}
.panel.issues li::before {{ background: var(--watch); }}
.qa-heading {{ font-family: var(--font-display); font-size: 17px; font-weight: 600; margin: 0 0 1rem; }}
.qa-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 1.1rem 1.25rem 1.25rem; margin-bottom: 1rem; }}
.qa-card .qnum {{ font-family: var(--font-mono); font-size: 11px; color: var(--accent-ink); background: var(--accent-tint); display: inline-block; padding: .15rem .5rem; border-radius: 5px; margin-bottom: .6rem; }}
.qa-card .question {{ font-size: 14.5px; font-weight: 600; line-height: 1.4; margin: 0 0 .5rem; }}
.qa-card .answer-label {{ font-size: 11px; letter-spacing: .04em; text-transform: uppercase; color: var(--ink-faint); margin: 0 0 .2rem; }}
.qa-card .answer {{ font-size: 13.5px; color: var(--ink-soft); line-height: 1.5; margin: 0 0 .75rem; }}
.chips {{ display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: .75rem; }}
.chip {{ font-size: 11px; background: var(--watch-tint); color: var(--watch); padding: .2rem .55rem; border-radius: 999px; font-weight: 600; }}
.improvement {{ font-size: 13.5px; line-height: 1.5; border-left: 2px solid var(--accent); padding-left: .75rem; }}
.improvement-label {{ display: block; font-size: 11px; letter-spacing: .04em; text-transform: uppercase; color: var(--accent-ink); margin-bottom: .2rem; }}
.empty-note {{ font-size: 13.5px; color: var(--ink-faint); }}
.footnote {{ font-size: 12px; color: var(--ink-faint); border-top: 1px solid var(--line); padding-top: 1rem; margin-top: .5rem; }}
@media (max-width: 560px) {{ .top-grid, .columns {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="sheet">
  <div class="masthead">
    <p class="eyebrow">Interview Analyzer &middot; session report</p>
    <h1>{_e(title)}</h1>
    <p class="meta">{_e(app_name)} &middot; {_e(date_str)} &middot; <code>#{record.id}</code></p>
  </div>

  <div class="top-grid">
    {focus_block}
    <div class="confidence-card">
      {_confidence_dial_svg(score)}
      <p class="dial-label">Confidence</p>
    </div>
  </div>

  <div class="columns">
    <div class="panel strengths">
      <h2><span class="dot"></span>Top strengths</h2>
      <ul>{strengths_html}</ul>
    </div>
    <div class="panel issues">
      <h2><span class="dot"></span>Top issues</h2>
      <ul>{issues_html}</ul>
    </div>
  </div>

  <p class="qa-heading">Question-by-question breakdown</p>
  {qa_html}

  <p class="footnote">Generated by Interview Analyzer from interview #{record.id}.</p>
</div>
</body>
</html>
"""
