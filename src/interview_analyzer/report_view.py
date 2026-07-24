"""Turns the small markdown subset report.py actually emits (headings,
bullets, bold/italic emphasis, paragraphs) into a readable in-app view for
the dashboard's History/Trends tabs -- no external markdown/HTML renderer
needed for our own controlled output.

Also color-codes the specific lines report.py emits for the assessment-
profile feature (competency scores, hire recommendation, selection
probability) on a red (worst) -> amber -> green (best) scale, matching the
same palette infographic.py uses, so the in-app text view and the HTML
infographic never disagree about what "good" looks like.

The parsing is a pure function (`parse_markdown_lines`) so it's testable
without a Tk display; `render_into_text_widget` is the thin Tk-specific
adapter the dashboard actually calls.
"""
from __future__ import annotations

import re

_INLINE_MARKERS = re.compile(r"\*\*(.+?)\*\*|_(.+?)_")

# Matches report.py's "- **{name}** [({emphasis} weight)] — {score}/100: {remark}"
# competency bullet line, AFTER inline markers have already been stripped
# (so no `**` around name here).
_COMPETENCY_LINE_RE = re.compile(r"^.+? — (\d+)/100: .*$")
# report.py's "- **Overall competency score** — {total}/100" bullet -- same
# shape but with no ": {remark}" suffix.
_OVERALL_SCORE_RE = re.compile(r"^Overall competency score — (\d+)/100$")
# report.py's "**Hire recommendation:** {level}", "**Estimated selection
# probability:** {percent}%...", and "**Recommendation:** {Recommended|Not
# Recommended}" lines, same post-stripping shape.
_HIRE_RECOMMENDATION_RE = re.compile(r"^Hire recommendation: (.+)$")
_SELECTION_PROBABILITY_RE = re.compile(r"^Estimated selection probability: (\d+)%")
_BINARY_RECOMMENDATION_RE = re.compile(r"^Recommendation: (Recommended|Not Recommended)$")

# Kept in sync with infographic.py's palette (_WATCH/_GOOD) so the in-app
# text view and the HTML infographic agree on what "good" vs "needs work"
# looks like.
_RED = (0xc0, 0x39, 0x2b)
_AMBER = (0xb5, 0x70, 0x1f)
_GREEN = (0x3d, 0x7a, 0x4a)

# The reference framework's 7-point hire scale (rubric.HIRE_RECOMMENDATION_LEVELS,
# not imported directly to avoid this pure-parsing module depending on the
# rubric/profile machinery) -- position within it anchors a 0-100 score for
# coloring purposes only.
_HIRE_LEVELS_ORDER = [
    "Strong No Hire", "No Hire", "Lean No Hire", "Lean Hire", "Hire", "Strong Hire", "Exceptional",
]


def _strip_inline_markers(line: str) -> str:
    return _INLINE_MARKERS.sub(lambda m: m.group(1) or m.group(2), line)


def _score_to_color(score: float) -> str:
    """Red (0) -> amber (50) -> green (100)."""
    score = max(0.0, min(100.0, score))
    if score <= 50:
        t = score / 50
        c1, c2 = _RED, _AMBER
    else:
        t = (score - 50) / 50
        c1, c2 = _AMBER, _GREEN
    r = round(c1[0] + (c2[0] - c1[0]) * t)
    g = round(c1[1] + (c2[1] - c1[1]) * t)
    b = round(c1[2] + (c2[2] - c1[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _hire_level_to_color(level: str) -> str | None:
    try:
        index = _HIRE_LEVELS_ORDER.index(level)
    except ValueError:
        return None
    score = index / (len(_HIRE_LEVELS_ORDER) - 1) * 100
    return _score_to_color(score)


def _colored(base_tag: str, color_hex: str) -> str:
    """Encodes a color into the tag string itself (rather than changing
    parse_markdown_lines's 2-tuple return shape) -- render_into_text_widget
    splits it back apart. Keeps every other line's tag exactly as before
    (e.g. plain "bullet"), so existing callers/tests are unaffected."""
    return f"{base_tag}|color:{color_hex}"


def parse_markdown_lines(markdown: str) -> list[tuple[str, str]]:
    """Return [(tag, display_text), ...] for each line, where tag is one of
    "h1", "h2", "h3", "bullet", "quote", "blank", or "text" -- or, for a
    competency score / hire recommendation / selection probability line,
    that same base tag with "|color:#rrggbb" appended (see _colored)."""
    lines: list[tuple[str, str]] = []
    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            lines.append(("blank", ""))
        elif line.startswith("### "):
            lines.append(("h3", _strip_inline_markers(line[4:])))
        elif line.startswith("## "):
            lines.append(("h2", _strip_inline_markers(line[3:])))
        elif line.startswith("# "):
            lines.append(("h1", _strip_inline_markers(line[2:])))
        elif line.startswith("- "):
            content = _strip_inline_markers(line[2:])
            match = _COMPETENCY_LINE_RE.match(content) or _OVERALL_SCORE_RE.match(content)
            if match:
                lines.append((_colored("bullet", _score_to_color(float(match.group(1)))), content))
            else:
                lines.append(("bullet", content))
        elif line.startswith("> "):
            lines.append(("quote", _strip_inline_markers(line[2:])))
        elif line.startswith("```"):
            continue  # code fences aren't used in our own report output
        else:
            content = _strip_inline_markers(line)
            hire_match = _HIRE_RECOMMENDATION_RE.match(content)
            selection_match = _SELECTION_PROBABILITY_RE.match(content)
            binary_match = _BINARY_RECOMMENDATION_RE.match(content)
            color = None
            if hire_match:
                color = _hire_level_to_color(hire_match.group(1))
            elif selection_match:
                color = _score_to_color(float(selection_match.group(1)))
            elif binary_match:
                rgb = _GREEN if binary_match.group(1) == "Recommended" else _RED
                color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            lines.append((_colored("text", color) if color else "text", content))
    return lines


def render_into_text_widget(text_widget, markdown: str) -> None:
    """Populate a Tkinter Text widget (must already have the tags below
    configured, see dashboard.py) with a readable rendering of `markdown`.
    Leaves the widget in its normal (editable) state when done is up to
    the caller -- this only inserts content."""
    text_widget.delete("1.0", "end")
    existing_tags = set(text_widget.tag_names())
    for raw_tag, content in parse_markdown_lines(markdown):
        base_tag, _, color_part = raw_tag.partition("|color:")
        tk_tags = [base_tag]
        if color_part:
            color_tag = f"dyncolor_{color_part.lstrip('#')}"
            if color_tag not in existing_tags:
                text_widget.tag_configure(color_tag, foreground=color_part)
                existing_tags.add(color_tag)
            tk_tags.append(color_tag)

        if base_tag == "blank":
            text_widget.insert("end", "\n")
        elif base_tag == "bullet":
            text_widget.insert("end", f"  •  {content}\n", tuple(tk_tags))
        elif base_tag == "quote":
            text_widget.insert("end", f"{content}\n", tuple(tk_tags))
        else:
            text_widget.insert("end", f"{content}\n", tuple(tk_tags))
