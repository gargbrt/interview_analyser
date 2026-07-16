"""Tests for the pure markdown-line-parsing used to render reports inside
the dashboard (no Tk display needed here -- only the Text-widget adapter
in report_view.render_into_text_widget touches Tkinter)."""
from __future__ import annotations

from interview_analyzer.report_view import parse_markdown_lines

SAMPLE = """\
# Interview Report — Zoom — 2026-07-16

_Interview #42 · started 2026-07-16T09:30:00_

## Session Summary

**Top strengths:**
- Comfortable, conversational tone throughout

**Top issues:**
- Answers lack a clear structure

## Question-by-question breakdown

### Q1. Tell me about a time you disagreed with a teammate.
**Your answer (summary):** Vague, no clear structure.
**Issues:**
- _structure_: No STAR structure.
"""


def test_headings_are_tagged_by_level():
    lines = parse_markdown_lines(SAMPLE)
    assert ("h1", "Interview Report — Zoom — 2026-07-16") in lines
    assert ("h2", "Session Summary") in lines
    assert ("h3", "Q1. Tell me about a time you disagreed with a teammate.") in lines


def test_bullets_are_tagged_and_stripped_of_markers():
    lines = parse_markdown_lines(SAMPLE)
    assert ("bullet", "Comfortable, conversational tone throughout") in lines
    # "_structure_: No STAR structure." -> italics marker stripped
    assert ("bullet", "structure: No STAR structure.") in lines


def test_bold_paragraph_markers_are_stripped_but_line_kept_as_text():
    lines = parse_markdown_lines(SAMPLE)
    assert ("text", "Top strengths:") in lines
    assert ("text", "Your answer (summary): Vague, no clear structure.") in lines


def test_blank_lines_preserved_for_spacing():
    lines = parse_markdown_lines("# Title\n\nBody text")
    assert lines == [("h1", "Title"), ("blank", ""), ("text", "Body text")]


def test_empty_markdown_returns_empty_list():
    assert parse_markdown_lines("") == []
