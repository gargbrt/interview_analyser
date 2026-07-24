"""Generates:
  1. A per-interview markdown report (questions, issues, suggested rewrites).
  2. A continuously-updated trends.md aggregating recurring issues across
     ALL stored interviews (using only the small analysis_json — never the
     audio), so patterns like "rambling on system design Qs, 6/8 interviews"
     surface automatically as history grows.
"""
from __future__ import annotations

import collections
import datetime as dt
import os
import pathlib
from typing import Optional

from .config_loader import Config
from .confidence import competency_weight, format_confidence, weighted_competency_total
from .db import InterviewRecord
from .profiles import GENERIC_PROFILE, AssessmentProfile, competency_emphasis_map


def _stringify(value) -> str:
    """LLM output doesn't always exactly match the requested JSON schema --
    smaller/local models in particular sometimes return a richer object
    (e.g. {"issue": "...", "detail": "..."}) where a plain string was
    asked for. Coerce defensively into a hashable, displayable string
    instead of crashing report/trend generation on an unexpected shape."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("issue", "strength", "text", "description", "summary", "name"):
            if isinstance(value.get(key), str):
                return value[key]
        return ", ".join(str(v) for v in value.values()) if value else str(value)
    return str(value)


def _interview_report_lines(
    record: InterviewRecord, analysis: dict, profile: Optional[AssessmentProfile] = None,
) -> list[str]:
    """Builds the markdown report's lines for `analysis` scored against
    `record`'s other fields (date/app/id) -- a pure, in-memory function
    shared by write_interview_report (writes analysis=record.analysis to
    disk) and render_interview_report_markdown (previews an arbitrary past
    analysis from db.AnalysisHistoryRecord without writing anything).

    `profile` decides how each competency's weight/emphasis label is shown
    -- defaults to `record.profile` (the snapshot actually used for this
    analysis) so weights shown always match what really produced the
    scores, falling back to GENERIC_PROFILE for pre-profile-feature
    interviews."""
    if profile is None:
        profile = record.profile or GENERIC_PROFILE
    date_str = record.started_at.split("T")[0]
    qa_pairs = analysis.get("qa_pairs", [])
    summary = analysis.get("session_summary", {})

    lines = [
        f"# Interview Report — {record.source_app or 'Unknown app'} — {date_str}",
        "",
        f"_Interview #{record.id} · started {record.started_at}_",
        "",
    ]

    if analysis.get("parse_error"):
        lines += [
            "> ⚠️ The analyzer output could not be parsed as structured JSON. "
            "Raw output is included below for reference.",
            "",
            "```",
            analysis.get("raw", ""),
            "```",
        ]
        return lines

    if analysis.get("no_speech_detected"):
        lines += [
            "> No speech was detected in this recording — it may have captured "
            "silence or background noise only. There's nothing to analyze.",
        ]
        return lines

    lines += ["## Session Summary", ""]
    lines.append(f"**Confidence in this assessment:** {format_confidence(analysis.get('confidence_info'))}")
    lines.append("")
    selection_probability = analysis.get("selection_probability")
    if selection_probability and selection_probability.get("percent") is not None:
        label = selection_probability.get("label")
        label_suffix = f" ({label})" if label else ""
        lines.append(f"**Estimated selection probability:** {selection_probability['percent']}%{label_suffix}")
        if selection_probability.get("binary_recommendation"):
            lines.append(f"**Recommendation:** {selection_probability['binary_recommendation']}")
        if selection_probability.get("basis"):
            lines.append(f"_{selection_probability['basis']}_")
        lines.append("")
    hire_recommendation = summary.get("hire_recommendation")
    if isinstance(hire_recommendation, dict) and hire_recommendation.get("level"):
        lines.append(f"**Hire recommendation:** {hire_recommendation['level']}")
        if hire_recommendation.get("rationale"):
            lines.append(hire_recommendation["rationale"])
        lines.append("")
    competency_scores = summary.get("competency_scores")
    if competency_scores:
        emphasis_map = competency_emphasis_map(profile)
        lines.append("**Competency scores:**")
        for entry in competency_scores:
            if not isinstance(entry, dict):
                continue
            name = _stringify(entry.get("name", ""))
            score = entry.get("score")
            score_text = f" — {score}/100" if isinstance(score, (int, float)) and not isinstance(score, bool) else ""
            emphasis = emphasis_map.get(name)
            weight_text = f" ({emphasis} weight)" if emphasis else ""
            lines.append(f"- **{name}**{weight_text}{score_text}: {entry.get('remark', '')}")
        overall = weighted_competency_total(competency_scores, profile)
        if overall is not None:
            lines.append(f"- **Overall competency score** — {round(overall)}/100")
        lines.append("")
    if summary.get("top_strengths"):
        lines.append("**Top strengths:**")
        lines += [f"- {_stringify(s)}" for s in summary["top_strengths"]]
        lines.append("")
    if summary.get("top_issues"):
        lines.append("**Top issues:**")
        lines += [f"- {_stringify(s)}" for s in summary["top_issues"]]
        lines.append("")
    if summary.get("one_thing_to_practice_next"):
        lines.append(f"**Focus for next practice:** {summary['one_thing_to_practice_next']}")
        lines.append("")

    lines += ["## Question-by-question breakdown", ""]
    for i, qa in enumerate(qa_pairs, 1):
        lines.append(f"### Q{i}. {qa.get('question', '(question)')}")
        lines.append(f"**Your answer (summary):** {qa.get('answer_summary', '')}")
        issues = qa.get("issues", [])
        if issues:
            lines.append("**Issues:**")
            for issue in issues:
                if isinstance(issue, dict):
                    lines.append(f"- _{issue.get('category', '')}_: {issue.get('detail', '')}")
                    excerpt = issue.get("excerpt")
                    if excerpt:
                        lines.append(f'> "{excerpt}"')
                else:
                    lines.append(f"- {_stringify(issue)}")
        if qa.get("suggested_improvement"):
            lines.append(f"**Suggested improvement:** {qa['suggested_improvement']}")
        lines.append("")

    return lines


def render_interview_report_markdown(
    record: InterviewRecord, analysis: dict, profile: Optional[AssessmentProfile] = None,
) -> str:
    """In-memory rendering of `analysis` as `record`'s report -- used to
    preview a past assessment from db.AnalysisHistoryRecord (see
    dashboard.py's "Previous assessments" section) without writing a file.
    Pass the historical entry's own `profile` explicitly so weights shown
    match whatever was active when that past analysis actually ran."""
    return "\n".join(_interview_report_lines(record, analysis, profile))


def write_interview_report(record: InterviewRecord, cfg: Config) -> pathlib.Path:
    out_dir = cfg.resolve(cfg.output.get("output_dir", "output")) / cfg.output.get(
        "reports_subdir", "reports"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    date_str = record.started_at.split("T")[0]
    report_path = out_dir / f"{date_str}_{record.source_app or 'interview'}_{record.id}.md"

    analysis = record.analysis or {}
    lines = _interview_report_lines(record, analysis)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def trends_report_path(cfg: Config, user_id: Optional[int] = None) -> pathlib.Path:
    """Where a given user's trends report lives. Each local profile gets
    its own file (`trends_<user_id>.md`) -- a single shared `trends.md`
    used to mean whichever profile's interview finished processing most
    recently silently overwrote every other profile's trends, and every
    profile's dashboard read back whatever was in it regardless of who was
    actually logged in. `user_id=None` is only for contexts with no login
    concept at all (not used by the normal tray+dashboard app)."""
    out_dir = cfg.resolve(cfg.output.get("output_dir", "output"))
    base = cfg.output.get("trends_filename", "trends.md")
    if user_id is None:
        return out_dir / base
    stem, ext = os.path.splitext(base)
    return out_dir / f"{stem}_user{user_id}{ext}"


def aggregate_trends(records: list[InterviewRecord]) -> dict:
    """Counts recurring issues/strengths across every analyzed interview in
    `records` -- shared by write_trends_report (markdown) and
    infographic.py's write_trends_infographic (HTML bar charts), so the two
    can never disagree about what "most frequent" means. Records with no
    usable analysis (parse_error, no_speech_detected, or none at all) are
    skipped, same as an individual report would treat them."""
    issue_counter: collections.Counter[str] = collections.Counter()
    strength_counter: collections.Counter[str] = collections.Counter()
    # {competency name: [score, score, ...]} across every interview that
    # scored it -- callers compute averages/sample counts as needed (see
    # write_trends_report's "Competency averages" section and
    # infographic.py's matching bar chart).
    competency_scores: dict[str, list[float]] = collections.defaultdict(list)
    analyzed_count = 0

    for record in records:
        analysis = record.analysis
        if not analysis or analysis.get("parse_error") or analysis.get("no_speech_detected"):
            continue
        analyzed_count += 1
        summary = analysis.get("session_summary", {})
        for issue in summary.get("top_issues", []):
            issue_counter[_stringify(issue)] += 1
        for strength in summary.get("top_strengths", []):
            strength_counter[_stringify(strength)] += 1
        for qa in analysis.get("qa_pairs", []):
            for issue in qa.get("issues", []):
                category = issue.get("category", "unspecified") if isinstance(issue, dict) else _stringify(issue)
                issue_counter[category] += 1
        for entry in summary.get("competency_scores") or []:
            if not isinstance(entry, dict):
                continue
            name = _stringify(entry.get("name", ""))
            score = entry.get("score")
            if name and isinstance(score, (int, float)) and not isinstance(score, bool):
                competency_scores[name].append(score)

    return {
        "issue_counter": issue_counter,
        "strength_counter": strength_counter,
        "competency_scores": dict(competency_scores),
        "analyzed_count": analyzed_count,
    }


def write_trends_report(records: list[InterviewRecord], cfg: Config, user_id: Optional[int] = None) -> pathlib.Path:
    out_dir = cfg.resolve(cfg.output.get("output_dir", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    trends_path = trends_report_path(cfg, user_id)

    agg = aggregate_trends(records)
    issue_counter = agg["issue_counter"]
    strength_counter = agg["strength_counter"]
    competency_scores = agg["competency_scores"]
    analyzed_count = agg["analyzed_count"]

    lines = [
        "# Recurring Issues Across Interviews",
        "",
        f"_Updated {dt.datetime.now().isoformat()} · based on {analyzed_count} analyzed interview(s)_",
        "",
    ]

    if analyzed_count == 0:
        lines.append("No analyzed interviews yet.")
    else:
        lines += ["## Most frequent issues", ""]
        for issue, count in issue_counter.most_common(10):
            lines.append(f"- **{issue}** — flagged in {count} instance(s)")
        lines += ["", "## Most frequent strengths", ""]
        for strength, count in strength_counter.most_common(10):
            lines.append(f"- **{strength}** — noted {count} time(s)")
        if competency_scores:
            # weakest average first -- the recurring areas most worth
            # practicing, same spirit as "most frequent issues" above
            averages = sorted(
                ((name, sum(scores) / len(scores), len(scores)) for name, scores in competency_scores.items()),
                key=lambda t: t[1],
            )
            lines += ["", "## Competency averages", ""]
            for name, avg, n in averages:
                lines.append(f"- **{name}** — {avg:.0f}/100 average, across {n} interview(s)")
        lines += ["", "## All interviews", ""]
        for record in records:
            date_str = record.started_at.split("T")[0]
            report_link = record.report_path or "(not yet generated)"
            lines.append(f"- {date_str} — {record.source_app or 'unknown'} — {report_link}")

    trends_path.write_text("\n".join(lines), encoding="utf-8")
    return trends_path
