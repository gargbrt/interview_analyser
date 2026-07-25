"""Generates single-page, scannable HTML "report cards":

  - `write_interview_infographic`: one interview's analysis -- a
    confidence dial, strengths vs. issues at a glance, and a
    question-by-question breakdown with suggested rewrites. Written
    alongside the existing markdown report (see report.py) whenever
    analysis finishes, opened via the History tab's "View infographic"
    button.
  - `write_trends_infographic`: recurring issues/strengths across every
    analyzed interview, as horizontal bar charts -- the visual
    counterpart to write_trends_report's markdown version, sharing the
    same underlying counts (see report.py's aggregate_trends). Written
    whenever the markdown trends report is, opened via the Trends tab's
    "View infographic" button.

Both are self-contained (no external fonts/scripts/CSS) so they open
correctly offline.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import html
import math
import pathlib
import re
from typing import Optional

from .config_loader import Config
from .confidence import estimate_selection_probability, weighted_competency_total
from .db import InterviewRecord
from .profiles import GENERIC_PROFILE, SENIORITIES, AssessmentProfile, competency_emphasis_map
from .report import _best_and_worst_scores, _questions_for_competency, _stringify, aggregate_trends, trends_report_path

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
_BAD = "#c0392b"


def _score_to_color(score: float) -> str:
    """Red (worst) -> amber -> green (best) gradient for a 0-100 score --
    same math as report_view.py's in-app color-coding (kept independent
    rather than imported, since this module's _GOOD/_WATCH constants are
    already the palette's source of truth that report_view.py matches)."""
    score = max(0.0, min(100.0, score))
    if score <= 50:
        t = score / 50
        c1, c2 = _BAD, _WATCH
    else:
        t = (score - 50) / 50
        c1, c2 = _WATCH, _GOOD
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _slugify(text: str) -> str:
    """A stable #anchor id for a competency name -- shared convention with
    report_view.py's Tk-side heading anchors (kept independent/duplicated
    there, same reasoning as the color palette) so a name like "Culture &
    Values Fit" always resolves to the same id in both places."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "row"


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


def _confidence_dial_svg(score: Optional[int], aria_label: str = "Confidence") -> str:
    """Draws a ring dial for any 0-100 score -- shared by the confidence
    dial and the selection-probability dial (see _render), which just pass
    a different aria_label so each stays correctly described for
    accessibility despite drawing identically.

    Colors are set via the `style` attribute using CSS custom properties
    (var(--ink) etc.), NOT literal hex passed to the `fill`/`stroke`
    attributes -- this SVG is inlined directly into the page, so it
    inherits the same :root variables the rest of the page uses, including
    the dark-mode override. A literal hex here would freeze the dial to
    light-mode colors, making the score text unreadable (near-black on a
    near-black panel) once dark mode swaps the panel background."""
    if score is None:
        return f"""<svg width="88" height="88" viewBox="0 0 88 88" role="img" aria-label="{aria_label}: not available">
<circle cx="44" cy="44" r="36" fill="none" style="stroke:var(--line);" stroke-width="8"/>
<text x="44" y="48" text-anchor="middle" font-family="-apple-system,sans-serif" font-size="12" style="fill:var(--ink-faint);">N/A</text>
</svg>"""
    circumference = 2 * 3.14159265 * 36
    offset = circumference * (1 - max(0, min(100, score)) / 100)
    return f"""<svg width="88" height="88" viewBox="0 0 88 88" role="img" aria-label="{aria_label} score: {score} out of 100">
<circle cx="44" cy="44" r="36" fill="none" style="stroke:var(--line);" stroke-width="8"/>
<circle cx="44" cy="44" r="36" fill="none" style="stroke:var(--accent);" stroke-width="8" stroke-linecap="round"
        stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}" transform="rotate(-90 44 44)"/>
<text x="44" y="40" text-anchor="middle" font-family="Cascadia Code,SF Mono,Consolas,monospace"
      font-size="22" font-weight="600" style="fill:var(--ink);">{score}</text>
<text x="44" y="55" text-anchor="middle" font-family="-apple-system,sans-serif" font-size="9" style="fill:var(--ink-faint);">/ 100</text>
</svg>"""


# Fixed zone boundaries for the selection-probability speedometer below --
# deliberately centered on the same neutral 50% the underlying estimate
# itself pulls toward (see confidence.py's _NEUTRAL_PERCENT), so "Maybe"
# reads as "too close to call" rather than an arbitrary middle third.
_GAUGE_ZONES = [(0, 35, "var(--bad)", "Not Hire"), (35, 65, "var(--watch)", "Maybe"), (65, 100, "var(--good)", "Hire")]


def _seniority_comparison(
    profile: AssessmentProfile,
    hire_recommendation: dict,
    competency_scores: list,
    confidence_info: Optional[dict],
) -> list[tuple[str, int]]:
    """What the selection probability would be for this SAME performance
    (the model's own hire-scale call, the actual competency scores, the
    actual assessment confidence) if judged one seniority level below and
    one above the profile's actual seniority, holding role/industry/
    company_type fixed -- lets a reader see which seniority bar this
    performance actually clears, not just the one it happened to be scored
    against, per an explicit user request for exactly this comparison.

    Deliberately NOT a fresh LLM call per adjacent level: re-running
    analysis would cost real API usage and introduce fresh sampling
    noise for a question that's really "how does this app's OWN
    seniority-weighting formula react to context", not a new judgment
    call -- so this just re-runs estimate_selection_probability with the
    seniority swapped, reusing the exact same anchor/competency/confidence
    inputs the real number was built from.

    Empty list if the profile has no seniority set (nothing to compare
    against) or its seniority is already at either end of the scale (no
    "one below"/"one above" exists there)."""
    if not profile.seniority or profile.seniority not in SENIORITIES:
        return []
    idx = SENIORITIES.index(profile.seniority)
    comparisons = []
    for adjacent_idx in (idx - 1, idx + 1):
        if not 0 <= adjacent_idx < len(SENIORITIES):
            continue
        adjacent_seniority = SENIORITIES[adjacent_idx]
        adjacent_profile = dataclasses.replace(profile, seniority=adjacent_seniority, name=None)
        result = estimate_selection_probability(
            hire_recommendation, competency_scores, profile=adjacent_profile, confidence_info=confidence_info,
        )
        comparisons.append((adjacent_seniority, result["percent"]))
    return comparisons


def _gauge_point(cx: float, cy: float, r: float, percent: float) -> tuple[float, float]:
    """A point on the gauge's semicircle for a 0-100 percent -- 0% is the
    leftmost point, 100% the rightmost, 50% dead center at the top,
    matching a real speedometer's left-low/right-high sweep."""
    angle_deg = 180 * (1 - percent / 100)
    angle_rad = math.radians(angle_deg)
    return cx + r * math.cos(angle_rad), cy - r * math.sin(angle_rad)


def _gauge_arc_path(cx: float, cy: float, r: float, p_start: float, p_end: float) -> str:
    """SVG path for the arc between two percentages along the gauge's
    semicircle -- large-arc-flag is always 0 since the full 0-100 range is
    exactly 180 degrees, so no sub-range can ever need the major arc;
    sweep-flag 1 draws left-to-right along the top, not under the bottom."""
    x1, y1 = _gauge_point(cx, cy, r, p_start)
    x2, y2 = _gauge_point(cx, cy, r, p_end)
    return f"M {x1:.2f},{y1:.2f} A {r:.2f},{r:.2f} 0 0 1 {x2:.2f},{y2:.2f}"


def _selection_probability_gauge_svg(percent: Optional[int], comparisons: Optional[list] = None) -> str:
    """A semicircular speedometer -- three fixed zones (Not Hire/Maybe/Hire,
    see _GAUGE_ZONES) with a needle pointing at the actual estimate --
    replacing a plain ring dial with something that reads at a glance the
    way a real gauge does, per the explicit user request this was built
    for. Zone colors are CSS vars (var(--bad) etc.), not literal hex, for
    the same dark-mode-correctness reason as _confidence_dial_svg above.

    `comparisons` (see _seniority_comparison), if given, adds a small ring
    marker at each adjacent seniority's own percentage, sitting right on
    the band's centerline -- distinct enough from the needle (which is
    always closer to center) to read as "here's where this same
    performance would land at a different bar" rather than a second
    estimate for THIS assessment.

    Layout constants below are chosen so every element -- especially the
    "Maybe" label, which sits at the gauge's 90-degree (straight up) point
    and therefore needs the MOST headroom above the center of any of the
    three zone labels -- stays inside the viewBox with margin to spare;
    reproduced directly: an earlier version placed "Maybe" just past the
    top edge (clipped/invisible) and put the percentage readout close
    enough to the needle's pivot circle to visually overlap it."""
    cx, cy, r, band_width = 110, 112, 78, 15
    label_radius = r + 20
    side_label_y = cy + 10
    needle_radius = r - band_width / 2 - 10
    value_text_y = cy + 42
    svg_height = value_text_y + 20

    band_paths = "".join(
        f'<path d="{_gauge_arc_path(cx, cy, r, p1, p2)}" fill="none" style="stroke:{color};" '
        f'stroke-width="{band_width}"/>'
        for p1, p2, color, _ in _GAUGE_ZONES
    )
    # "Not Hire"/"Hire" sit BELOW the arc's horizontal baseline (y=cy) at
    # its left/right ends, not out along their zone's own curve -- the
    # band is entirely drawn ABOVE y=cy, so this placement can never
    # overlap it regardless of label width. Reproduced directly: following
    # the curve at the zone's angular midpoint (like "Maybe" does, which
    # is fine since it sits at the isolated top point) put these two close
    # enough to the band's own curve at that oblique angle to overlap it.
    # "Maybe" alone still follows the curve, straight up, since that's the
    # one point with no better alternative placement.
    label_positions = [(cx - r, side_label_y), (cx, cy - label_radius), (cx + r, side_label_y)]
    label_svg_parts = []
    for (lx, ly), (_p1, _p2, _color, label) in zip(label_positions, _GAUGE_ZONES):
        label_svg_parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" font-family="-apple-system,sans-serif" '
            f'font-size="10.5" font-weight="600" style="fill:var(--ink-faint);">{label}</text>'
        )
    labels_svg = "".join(label_svg_parts)

    comparison_markers_svg = "".join(
        f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="5" style="fill:var(--panel); stroke:var(--ink);" stroke-width="2"/>'
        for mx, my in (_gauge_point(cx, cy, r, comp_percent) for _seniority, comp_percent in (comparisons or []))
    )

    if percent is None:
        needle_svg = ""
        value_text = "N/A"
        aria_value = "not available"
    else:
        clamped = max(0, min(100, percent))
        needle_x, needle_y = _gauge_point(cx, cy, needle_radius, clamped)
        needle_svg = (
            f'<line x1="{cx}" y1="{cy}" x2="{needle_x:.1f}" y2="{needle_y:.1f}" '
            f'style="stroke:var(--ink);" stroke-width="3" stroke-linecap="round"/>'
            f'<circle cx="{cx}" cy="{cy}" r="6" style="fill:var(--ink);"/>'
        )
        value_text = f"{percent}%"
        aria_value = f"{percent} out of 100"
    return f"""<svg width="220" height="{svg_height:.0f}" viewBox="0 0 220 {svg_height:.0f}" role="img" aria-label="Selection probability gauge: {aria_value}">
{band_paths}
{comparison_markers_svg}
{needle_svg}
{labels_svg}
<text x="{cx}" y="{value_text_y:.0f}" text-anchor="middle" font-family="Cascadia Code,SF Mono,Consolas,monospace"
      font-size="22" font-weight="700" style="fill:var(--ink);">{value_text}</text>
</svg>"""


def _score_summary_table_html(
    competency_scores: list, emphasis_map: dict, overall_score: Optional[float],
) -> str:
    """The "upfront" table: one row per competency (name links down to its
    full detail section via #comp-{slug}, see _competency_row_html), each
    with its score and profile-context weightage, plus a bolded total row.
    The single highest-scoring row is flagged "Strongest" and the single
    lowest-scoring "Needs focus" (see _best_and_worst_scores) so a reader
    can spot what to look at without scanning every row. Empty string
    (renders nothing) if there are no usable competency scores -- same
    gating as the detail section below it."""
    best_score, worst_score = _best_and_worst_scores(competency_scores)
    rows = []
    for entry in competency_scores:
        if not isinstance(entry, dict):
            continue
        name = _stringify(entry.get("name", ""))
        score = entry.get("score")
        has_score = isinstance(score, (int, float)) and not isinstance(score, bool)
        score_text = f"{score}/100" if has_score else "N/A"
        score_style = f' style="color:{_score_to_color(score)};"' if has_score else ""
        weight = emphasis_map.get(name)
        weight_text = weight.title() if weight else "—"
        row_class = ""
        badge_html = ""
        if has_score and best_score is not None and score == best_score:
            row_class = ' class="summary-best"'
            badge_html = '<span class="summary-badge summary-badge-best">&#9733; Strongest</span>'
        elif has_score and worst_score is not None and score == worst_score:
            row_class = ' class="summary-worst"'
            badge_html = '<span class="summary-badge summary-badge-worst">&#9888; Needs focus</span>'
        rows.append(f"""<tr{row_class}>
<td><a class="summary-link" href="#comp-{_e(_slugify(name))}">{_e(name)}</a>{badge_html}</td>
<td{score_style}>{_e(score_text)}</td>
<td>{_e(weight_text)}</td>
</tr>""")
    if not rows:
        return ""
    total_row = ""
    if overall_score is not None:
        total_row = (
            f'<tr class="summary-total"><td>Overall competency score</td>'
            f'<td style="color:{_score_to_color(overall_score)};">{round(overall_score)}/100</td><td></td></tr>'
        )
    return f"""<table class="summary-table">
<thead><tr><th>Parameter</th><th>Score</th><th>Weightage</th></tr></thead>
<tbody>
{"".join(rows)}
{total_row}
</tbody>
</table>"""


def _decision_summary_html(
    hire_level: str, selection_percent: Optional[int], binary_recommendation: Optional[str],
) -> str:
    """A compact "along with the decision" line shown right under the Score
    Summary table -- the same three signals already shown elsewhere on the
    page (hire-scale badge, selection-probability dial, recommendation
    pill), repeated here so the upfront table is a genuinely standalone
    summary a reader doesn't have to scroll further to understand."""
    parts = []
    if hire_level:
        parts.append(_e(hire_level))
    if selection_percent is not None:
        parts.append(f"{selection_percent}% selection probability")
    if binary_recommendation:
        parts.append(_e(binary_recommendation))
    if not parts:
        return ""
    return f'<p class="summary-decision"><strong>Decision:</strong> {" &middot; ".join(parts)}</p>'


def _seniority_comparison_caption_html(comparisons: list) -> str:
    """The text counterpart to the gauge's ring markers (see
    _seniority_comparison/_selection_probability_gauge_svg) -- spells out
    which seniority each marker is, since the markers alone don't carry a
    label. Empty string (renders nothing) if there's nothing to compare."""
    if not comparisons:
        return ""
    parts = [f"{_e(seniority)}: {percent}%" for seniority, percent in comparisons]
    return f'<p class="gauge-comparison">Same performance at&nbsp;&mdash;&nbsp;{" &middot; ".join(parts)}</p>'


def _related_questions_html(qa_pairs: Optional[list], name: str) -> str:
    """A native <details>/<summary> (real, browser-built-in expand/collapse,
    no JS needed) listing every question whose issues[] were tagged under
    this competency, each linking to its card in the Q&A breakdown (see
    _qa_card's id="q{i}") -- empty string if there's nothing to link, same
    gating as report.py's markdown counterpart (_questions_for_competency)."""
    if not qa_pairs:
        return ""
    related = _questions_for_competency(qa_pairs, name)
    if not related:
        return ""
    items_html = "".join(f'<li><a href="#q{i}">Q{i}. {_e(question)}</a></li>' for i, question in related)
    count = len(related)
    label = f"{count} question{'s' if count != 1 else ''}"
    return f"""<details class="competency-questions">
<summary>View details ({label})</summary>
<ul>{items_html}</ul>
</details>"""


def _competency_row_html(
    entry: dict, weight: Optional[str] = None, anchor_id: Optional[str] = None, qa_pairs: Optional[list] = None,
) -> str:
    """`anchor_id`, when given, makes this row a jump target for the Score
    Summary table's per-parameter links (see _score_summary_table_html) and
    adds a "back to top" link so the reader can return to that table --
    omitted for write_trends_infographic's reuse of this same row markup,
    since trends has no summary table linking into it (nor does it pass
    qa_pairs, since a trends average isn't about one interview's Q&A)."""
    if not isinstance(entry, dict):
        return ""
    name = _stringify(entry.get("name", ""))
    score = entry.get("score")
    has_score = isinstance(score, (int, float)) and not isinstance(score, bool)
    score_text = f"{score}/100" if has_score else "N/A"
    width_pct = max(4, min(100, score)) if has_score else 0
    bar_color = _score_to_color(score) if has_score else _ACCENT
    remark = entry.get("remark", "")
    remark_html = f'<p class="competency-remark">{_e(remark)}</p>' if remark else ""
    weight_html = f'<span class="competency-weight">{_e(weight)} weight</span>' if weight else ""
    id_attr = f' id="{_e(anchor_id)}"' if anchor_id else ""
    related_questions_html = _related_questions_html(qa_pairs, name)
    back_to_top_html = '<a class="back-to-top" href="#top">&uarr; Back to top</a>' if anchor_id else ""
    return f"""<div class="competency-row"{id_attr}>
<div class="competency-head"><span class="competency-name">{_e(name)}{weight_html}</span><span class="competency-score">{_e(score_text)}</span></div>
<div class="bar-track"><div class="bar-fill" style="width:{width_pct}%; background:{bar_color};"></div></div>
{remark_html}
{related_questions_html}
{back_to_top_html}
</div>"""


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

    return f"""<div class="qa-card" id="q{index}">
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
    selection_probability = analysis.get("selection_probability") or {}
    selection_percent = selection_probability.get("percent")
    selection_label = selection_probability.get("label")
    selection_basis = selection_probability.get("basis")
    hire_recommendation = summary.get("hire_recommendation") or {}
    competency_scores = summary.get("competency_scores") or []
    profile = record.profile or GENERIC_PROFILE
    emphasis_map = competency_emphasis_map(profile)
    overall_score = weighted_competency_total(competency_scores, profile)
    seniority_comparisons = _seniority_comparison(profile, hire_recommendation, competency_scores, confidence_info)

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
    selection_value_html = f'<p class="dial-value">{_e(selection_label)}</p>' if selection_label else ""
    selection_basis_html = f'<p class="dial-basis">{_e(selection_basis)}</p>' if selection_basis else ""
    binary_recommendation = selection_probability.get("binary_recommendation")
    recommendation_pill_html = (
        f'<p class="recommendation-pill recommendation-{"good" if binary_recommendation == "Recommended" else "bad"}">'
        f'{_e(binary_recommendation)}</p>'
        if binary_recommendation else ""
    )

    hire_level = hire_recommendation.get("level") or ""
    hire_rationale = hire_recommendation.get("rationale") or ""
    hire_block = (
        f'<div class="hire-badge"><p class="label">Hire recommendation</p>'
        f'<p class="hire-level">{_e(hire_level)}</p>'
        f'<p class="hire-rationale">{_e(hire_rationale)}</p></div>'
        if hire_level else ""
    )

    competency_rows_html = "".join(
        _competency_row_html(
            c, emphasis_map.get(_stringify(c.get("name", ""))),
            anchor_id=f"comp-{_slugify(_stringify(c.get('name', '')))}",
            qa_pairs=qa_pairs,
        )
        for c in competency_scores if isinstance(c, dict)
    )
    overall_score_html = (
        f'<div class="overall-score"><span class="overall-score-label">Overall competency score</span>'
        f'<span class="overall-score-value" style="color:{_score_to_color(overall_score)};">{round(overall_score)}/100</span></div>'
        if overall_score is not None else ""
    )
    competency_block = (
        f"""<p class="qa-heading">Competency scores</p>
{overall_score_html}
<div class="competency-list">{competency_rows_html}</div>"""
        if competency_rows_html else ""
    )
    score_summary_table_html = _score_summary_table_html(competency_scores, emphasis_map, overall_score)
    decision_summary_html = _decision_summary_html(hire_level, selection_percent, binary_recommendation)
    score_summary_block = (
        f"""<p class="qa-heading">Score Summary</p>
{score_summary_table_html}
{decision_summary_html}"""
        if score_summary_table_html else ""
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
  --good: {_GOOD}; --good-tint: {_GOOD_TINT}; --watch: {_WATCH}; --watch-tint: {_WATCH_TINT}; --bad: {_BAD};
  --font-display: Iowan Old Style, Palatino Linotype, Palatino, Georgia, serif;
  --font-body: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --font-mono: "Cascadia Code", "SF Mono", Consolas, "Courier New", monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --ink: #e9edf0; --ink-soft: #b3bcc6; --ink-faint: #7d8794;
    --ground: #14181c; --panel: #1b2126; --line: #2c343b;
    --accent: #4fb3ba; --accent-ink: #bfe6e8; --accent-tint: #1c2f31;
    --good: #7fbf8c; --good-tint: #1c2b1f; --watch: #e0a655; --watch-tint: #2e2416; --bad: #d97066;
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
.top-grid {{ display: grid; grid-template-columns: minmax(0,1fr) 168px minmax(220px, 240px); gap: 1.25rem; margin-bottom: 1.25rem; }}
.practice-note {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 1.1rem 1.25rem; display: flex; flex-direction: column; justify-content: center; }}
.practice-note .label {{ font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: var(--accent-ink); margin: 0 0 .4rem; }}
.practice-note p {{ margin: 0; font-size: 15px; line-height: 1.5; }}
.confidence-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 1rem; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; }}
.confidence-card .dial-label {{ font-size: 10.5px; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-faint); margin: .5rem 0 0; }}
.confidence-card .dial-value {{ font-family: var(--font-mono); font-size: 12px; color: var(--ink-soft); }}
.confidence-card .dial-basis {{ font-size: 10px; color: var(--ink-faint); line-height: 1.35; margin: .3rem 0 0; }}
.recommendation-pill {{ font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; border-radius: 999px; padding: .15rem .6rem; margin: .35rem 0 0; }}
.recommendation-pill.recommendation-good {{ color: var(--good); background: var(--good-tint); }}
.recommendation-pill.recommendation-bad {{ color: var(--bad); background: var(--watch-tint); }}
.gauge-comparison {{ font-size: 10px; color: var(--ink-faint); line-height: 1.4; margin: .4rem 0 0; }}
.hire-badge {{ background: var(--accent-tint); border: 1px solid var(--line); border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1.25rem; }}
.hire-badge .label {{ font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: var(--accent-ink); margin: 0 0 .3rem; }}
.hire-badge .hire-level {{ font-family: var(--font-display); font-size: 18px; font-weight: 600; margin: 0 0 .3rem; color: var(--accent-ink); }}
.hire-badge .hire-rationale {{ font-size: 13.5px; color: var(--ink-soft); margin: 0; line-height: 1.5; }}
.competency-list {{ margin-bottom: 2rem; }}
.competency-row {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: .85rem 1.1rem; margin-bottom: .75rem; }}
.competency-head {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: .4rem; }}
.competency-name {{ font-size: 13.5px; font-weight: 600; }}
.competency-weight {{ font-size: 10.5px; font-weight: 500; text-transform: uppercase; letter-spacing: .04em; color: var(--ink-faint); margin-left: .5rem; }}
.competency-score {{ font-family: var(--font-mono); font-size: 12px; color: var(--ink-soft); }}
.competency-remark {{ font-size: 13px; color: var(--ink-soft); line-height: 1.5; margin: .5rem 0 0; }}
.overall-score {{ display: flex; align-items: baseline; justify-content: space-between; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: .85rem 1.1rem; margin-bottom: .9rem; }}
.overall-score-label {{ font-size: 12.5px; font-weight: 600; color: var(--ink-soft); }}
.overall-score-value {{ font-family: var(--font-mono); font-size: 17px; font-weight: 700; }}
.summary-table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; margin-bottom: .75rem; }}
.summary-table th, .summary-table td {{ text-align: left; padding: .55rem .9rem; font-size: 13px; border-bottom: 1px solid var(--line); }}
.summary-table th {{ font-size: 11px; letter-spacing: .04em; text-transform: uppercase; color: var(--ink-faint); font-weight: 600; }}
.summary-table tr:last-child td {{ border-bottom: none; }}
.summary-table td:nth-child(2) {{ font-family: var(--font-mono); font-weight: 600; }}
.summary-link {{ color: var(--accent-ink); text-decoration: none; font-weight: 600; }}
.summary-link:hover {{ text-decoration: underline; }}
.summary-total td {{ font-weight: 700; border-top: 2px solid var(--line); }}
.summary-decision {{ font-size: 13.5px; color: var(--ink-soft); margin: 0 0 1.75rem; }}
.summary-table tr.summary-best td {{ background: var(--good-tint); }}
.summary-table tr.summary-worst td {{ background: var(--watch-tint); }}
.summary-badge {{ display: inline-block; margin-left: .5rem; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .03em; }}
.summary-badge-best {{ color: var(--good); }}
.summary-badge-worst {{ color: var(--watch); }}
.back-to-top {{ display: inline-block; font-size: 11.5px; color: var(--accent-ink); text-decoration: none; margin-top: .6rem; }}
.back-to-top:hover {{ text-decoration: underline; }}
.competency-questions {{ margin-top: .6rem; font-size: 12.5px; }}
.competency-questions summary {{ cursor: pointer; color: var(--accent-ink); font-weight: 600; }}
.competency-questions summary:hover {{ text-decoration: underline; }}
.competency-questions ul {{ margin: .5rem 0 0; padding: 0; list-style: none; display: flex; flex-direction: column; gap: .4rem; }}
.competency-questions li a {{ color: var(--ink-soft); text-decoration: none; }}
.competency-questions li a:hover {{ color: var(--accent-ink); text-decoration: underline; }}
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
<div class="sheet" id="top">
  <div class="masthead">
    <p class="eyebrow">Interview Analyzer &middot; session report</p>
    <h1>{_e(title)}</h1>
    <p class="meta">{_e(app_name)} &middot; {_e(date_str)} &middot; <code>#{record.id}</code></p>
  </div>

  {score_summary_block}

  <div class="top-grid">
    {focus_block}
    <div class="confidence-card">
      {_confidence_dial_svg(score)}
      <p class="dial-label">Confidence</p>
    </div>
    <div class="confidence-card gauge-card">
      {_selection_probability_gauge_svg(selection_percent, seniority_comparisons)}
      <p class="dial-label">Selection probability</p>
      {selection_value_html}
      {recommendation_pill_html}
      {_seniority_comparison_caption_html(seniority_comparisons)}
      {selection_basis_html}
    </div>
  </div>

  {hire_block}
  {competency_block}

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


def trends_infographic_path(cfg: Config, user_id: Optional[int] = None) -> pathlib.Path:
    """Alongside the markdown trends report (trends_<user_id>_infographic.html
    next to trends_<user_id>.md) -- same per-user file-naming reasoning as
    report.trends_report_path (a single shared file used to mean one
    profile's refresh could silently show stale/wrong data to another)."""
    md_path = trends_report_path(cfg, user_id)
    return md_path.with_name(f"{md_path.stem}_infographic.html")


def _bar_rows_html(items: list[tuple[str, int]], color: str, tint: str) -> str:
    if not items:
        return '<p class="empty-note">None flagged yet.</p>'
    max_count = max(count for _, count in items) or 1
    rows = []
    for label, count in items:
        width_pct = max(6, round(100 * count / max_count))  # 6% floor so a count of 1 is still visible
        rows.append(f"""<div class="bar-row">
<span class="bar-label">{_e(label)}</span>
<div class="bar-track"><div class="bar-fill" style="width:{width_pct}%; background:{color};"></div></div>
<span class="bar-count" style="color:{color}; background:{tint};">{count}</span>
</div>""")
    return "\n".join(rows)


def write_trends_infographic(
    records: list[InterviewRecord], cfg: Config, user_id: Optional[int] = None
) -> pathlib.Path:
    """Writes the HTML trends infographic and returns its path. Unlike
    write_interview_infographic, this always writes something -- even zero
    analyzed interviews gets a real (if sparse) page, same as
    write_trends_report's markdown version does."""
    out_path = trends_infographic_path(cfg, user_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_trends(records), encoding="utf-8")
    return out_path


def _render_trends(records: list[InterviewRecord]) -> str:
    agg = aggregate_trends(records)
    issue_counter, strength_counter, analyzed_count = (
        agg["issue_counter"], agg["strength_counter"], agg["analyzed_count"],
    )
    competency_scores = agg["competency_scores"]
    updated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    title = "Interview trends"

    if analyzed_count == 0:
        body = '<p class="empty-note">No analyzed interviews yet -- trends will appear here once at least one interview has been analyzed.</p>'
    else:
        issues_html = _bar_rows_html(issue_counter.most_common(10), _WATCH, _WATCH_TINT)
        strengths_html = _bar_rows_html(strength_counter.most_common(10), _GOOD, _GOOD_TINT)
        interview_rows = "".join(
            f'<li><span class="interview-date">{_e(r.started_at.split("T")[0])}</span>'
            f'<span class="interview-app">{_e(r.source_app or "unknown")}</span>'
            f'<span class="interview-status">{"report generated" if r.report_path else "not yet generated"}</span></li>'
            for r in records
        )
        # weakest average first -- the recurring areas most worth practicing,
        # same reasoning and reuse as report.py's write_trends_report
        competency_entries = sorted(
            (
                {
                    "name": name, "score": round(sum(scores) / len(scores)),
                    "remark": f"Average across {len(scores)} interview(s).",
                }
                for name, scores in competency_scores.items()
            ),
            key=lambda e: e["score"],
        )
        competency_block = (
            f"""<p class="qa-heading">Competency averages</p>
  <div class="competency-list">{"".join(_competency_row_html(e) for e in competency_entries)}</div>"""
            if competency_entries else ""
        )
        body = f"""<div class="columns">
    <div class="panel issues">
      <h2><span class="dot"></span>Most frequent issues</h2>
      {issues_html}
    </div>
    <div class="panel strengths">
      <h2><span class="dot"></span>Most frequent strengths</h2>
      {strengths_html}
    </div>
  </div>

  {competency_block}

  <p class="qa-heading">All interviews</p>
  <ul class="interview-list">{interview_rows}</ul>"""

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
.columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-bottom: 2rem; }}
.panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 1.1rem 1.25rem 1.25rem; }}
.panel h2 {{ font-family: var(--font-display); font-size: 15px; font-weight: 600; margin: 0 0 .9rem; display: flex; align-items: center; gap: .4rem; }}
.panel h2 .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
.panel.strengths h2 .dot {{ background: var(--good); }}
.panel.issues h2 .dot {{ background: var(--watch); }}
.bar-row {{ display: grid; grid-template-columns: minmax(0,1fr) 90px 34px; align-items: center; gap: .5rem; margin-bottom: .6rem; }}
.bar-label {{ font-size: 12.5px; color: var(--ink-soft); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.bar-track {{ height: 8px; background: var(--line); border-radius: 999px; overflow: hidden; }}
.bar-fill {{ height: 100%; border-radius: 999px; }}
.bar-count {{ font-family: var(--font-mono); font-size: 11px; font-weight: 600; text-align: center; border-radius: 5px; padding: .1rem 0; }}
.qa-heading {{ font-family: var(--font-display); font-size: 17px; font-weight: 600; margin: 0 0 1rem; }}
.competency-list {{ margin-bottom: 2rem; }}
.competency-row {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: .85rem 1.1rem; margin-bottom: .75rem; }}
.competency-head {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: .4rem; }}
.competency-name {{ font-size: 13.5px; font-weight: 600; }}
.competency-score {{ font-family: var(--font-mono); font-size: 12px; color: var(--ink-soft); }}
.competency-remark {{ font-size: 13px; color: var(--ink-soft); line-height: 1.5; margin: .5rem 0 0; }}
.interview-list {{ list-style: none; margin: 0; padding: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }}
.interview-list li {{ display: grid; grid-template-columns: 100px minmax(0,1fr) minmax(0,1fr); gap: .75rem; padding: .6rem 1rem; font-size: 12.5px; border-bottom: 1px solid var(--line); }}
.interview-list li:last-child {{ border-bottom: none; }}
.interview-date {{ font-family: var(--font-mono); color: var(--ink-faint); }}
.interview-app {{ color: var(--ink); }}
.interview-status {{ color: var(--ink-soft); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.empty-note {{ font-size: 13.5px; color: var(--ink-faint); }}
@media (max-width: 560px) {{ .columns {{ grid-template-columns: 1fr; }} .interview-list li {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="sheet">
  <div class="masthead">
    <p class="eyebrow">Interview Analyzer &middot; trends</p>
    <h1>{_e(title)}</h1>
    <p class="meta">Updated {_e(updated)} &middot; based on {analyzed_count} analyzed interview(s)</p>
  </div>

  {body}
</div>
</body>
</html>
"""
