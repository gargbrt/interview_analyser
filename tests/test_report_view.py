"""Tests for the pure markdown-line-parsing used to render reports inside
the dashboard (no Tk display needed here -- only the Text-widget adapter
in report_view.render_into_text_widget touches Tkinter)."""
from __future__ import annotations

from interview_analyzer.report_view import _score_to_color, parse_markdown_lines

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


class TestScoreToColor:
    def test_zero_is_red_hundred_is_green(self):
        assert _score_to_color(0) == "#c0392b"
        assert _score_to_color(100) == "#3d7a4a"

    def test_fifty_is_amber(self):
        assert _score_to_color(50) == "#b5701f"

    def test_clamps_out_of_range_scores(self):
        assert _score_to_color(-20) == _score_to_color(0)
        assert _score_to_color(150) == _score_to_color(100)

    def test_higher_score_is_greener_lower_is_redder(self):
        low = _score_to_color(20)
        high = _score_to_color(80)
        assert low != high
        assert low == _score_to_color(20)  # deterministic


class TestColorCodedLines:
    """report.py's competency-score/hire-recommendation/selection-
    probability lines get a color-coded tag (red=worst, green=best) --
    every other line's tag is completely unaffected (see
    test_bullets_are_tagged_and_stripped_of_markers etc. above, which still
    pass unchanged)."""

    def test_competency_score_bullet_is_color_coded(self):
        markdown = "- **Leadership** — 80/100: Strong ownership shown."
        lines = parse_markdown_lines(markdown)
        assert len(lines) == 1
        tag, content = lines[0]
        assert tag.startswith("bullet|color:")
        assert content == "Leadership — 80/100: Strong ownership shown."

    def test_low_competency_score_is_reddish_high_is_greenish(self):
        low_tag, _ = parse_markdown_lines("- **Execution** — 10/100: Needs work.")[0]
        high_tag, _ = parse_markdown_lines("- **Execution** — 95/100: Excellent.")[0]
        low_color = low_tag.split("|color:")[1]
        high_color = high_tag.split("|color:")[1]
        assert low_color == _score_to_color(10)
        assert high_color == _score_to_color(95)

    def test_ordinary_bullet_is_not_color_coded(self):
        """Regression guard: a plain bullet that merely happens to contain
        a dash and a number must not be mistaken for a competency line."""
        tag, content = parse_markdown_lines("- Mentioned working 100/100 days on the project.")[0]
        assert tag == "bullet"
        assert content == "Mentioned working 100/100 days on the project."

    def test_hire_recommendation_line_is_color_coded(self):
        markdown = "**Hire recommendation:** Strong Hire"
        tag, content = parse_markdown_lines(markdown)[0]
        assert tag.startswith("text|color:")
        assert content == "Hire recommendation: Strong Hire"

    def test_low_hire_level_is_reddish_high_is_greenish(self):
        low_tag, _ = parse_markdown_lines("**Hire recommendation:** Strong No Hire")[0]
        high_tag, _ = parse_markdown_lines("**Hire recommendation:** Exceptional")[0]
        assert low_tag.split("|color:")[1] == "#c0392b"
        assert high_tag.split("|color:")[1] == "#3d7a4a"

    def test_unrecognized_hire_level_is_not_color_coded(self):
        tag, _ = parse_markdown_lines("**Hire recommendation:** Some Unknown Level")[0]
        assert tag == "text"

    def test_selection_probability_line_is_color_coded(self):
        markdown = "**Estimated selection probability:** 72% (Hire)"
        tag, content = parse_markdown_lines(markdown)[0]
        assert tag.startswith("text|color:")
        assert content == "Estimated selection probability: 72% (Hire)"
        assert tag.split("|color:")[1] == _score_to_color(72)

    def test_ordinary_text_line_is_not_color_coded(self):
        tag, _ = parse_markdown_lines("Just a plain paragraph.")[0]
        assert tag == "text"

    def test_recommended_binary_recommendation_is_green(self):
        markdown = "**Recommendation:** Recommended"
        tag, content = parse_markdown_lines(markdown)[0]
        assert content == "Recommendation: Recommended"
        assert tag.split("|color:")[1] == "#3d7a4a"

    def test_not_recommended_binary_recommendation_is_red(self):
        markdown = "**Recommendation:** Not Recommended"
        tag, _ = parse_markdown_lines(markdown)[0]
        assert tag.split("|color:")[1] == "#c0392b"

    def test_overall_competency_score_bullet_is_color_coded(self):
        markdown = "- **Overall competency score** — 75/100"
        tag, content = parse_markdown_lines(markdown)[0]
        assert content == "Overall competency score — 75/100"
        assert tag.split("|color:")[1] == _score_to_color(75)
