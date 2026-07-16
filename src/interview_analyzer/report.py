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
import pathlib

from .config_loader import Config
from .confidence import format_confidence
from .db import InterviewRecord


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


def write_interview_report(record: InterviewRecord, cfg: Config) -> pathlib.Path:
    out_dir = cfg.resolve(cfg.output.get("output_dir", "output")) / cfg.output.get(
        "reports_subdir", "reports"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    date_str = record.started_at.split("T")[0]
    report_path = out_dir / f"{date_str}_{record.source_app or 'interview'}_{record.id}.md"

    analysis = record.analysis or {}
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
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    if analysis.get("no_speech_detected"):
        lines += [
            "> No speech was detected in this recording — it may have captured "
            "silence or background noise only. There's nothing to analyze.",
        ]
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    lines += ["## Session Summary", ""]
    lines.append(f"**Confidence in this assessment:** {format_confidence(analysis.get('confidence_info'))}")
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

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_trends_report(records: list[InterviewRecord], cfg: Config) -> pathlib.Path:
    out_dir = cfg.resolve(cfg.output.get("output_dir", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    trends_path = out_dir / cfg.output.get("trends_filename", "trends.md")

    issue_counter: collections.Counter[str] = collections.Counter()
    strength_counter: collections.Counter[str] = collections.Counter()
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
        lines += ["", "## All interviews", ""]
        for record in records:
            date_str = record.started_at.split("T")[0]
            report_link = record.report_path or "(not yet generated)"
            lines.append(f"- {date_str} — {record.source_app or 'unknown'} — {report_link}")

    trends_path.write_text("\n".join(lines), encoding="utf-8")
    return trends_path
