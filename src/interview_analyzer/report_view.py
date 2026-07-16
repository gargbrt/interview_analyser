"""Turns the small markdown subset report.py actually emits (headings,
bullets, bold/italic emphasis, paragraphs) into a readable in-app view for
the dashboard's History/Trends tabs -- no external markdown/HTML renderer
needed for our own controlled output.

The parsing is a pure function (`parse_markdown_lines`) so it's testable
without a Tk display; `render_into_text_widget` is the thin Tk-specific
adapter the dashboard actually calls.
"""
from __future__ import annotations

import re

_INLINE_MARKERS = re.compile(r"\*\*(.+?)\*\*|_(.+?)_")


def _strip_inline_markers(line: str) -> str:
    return _INLINE_MARKERS.sub(lambda m: m.group(1) or m.group(2), line)


def parse_markdown_lines(markdown: str) -> list[tuple[str, str]]:
    """Return [(tag, display_text), ...] for each line, where tag is one of
    "h1", "h2", "h3", "bullet", "quote", "blank", or "text"."""
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
            lines.append(("bullet", _strip_inline_markers(line[2:])))
        elif line.startswith("> "):
            lines.append(("quote", _strip_inline_markers(line[2:])))
        elif line.startswith("```"):
            continue  # code fences aren't used in our own report output
        else:
            lines.append(("text", _strip_inline_markers(line)))
    return lines


def render_into_text_widget(text_widget, markdown: str) -> None:
    """Populate a Tkinter Text widget (must already have the tags below
    configured, see dashboard.py) with a readable rendering of `markdown`.
    Leaves the widget in its normal (editable) state when done is up to
    the caller -- this only inserts content."""
    text_widget.delete("1.0", "end")
    for tag, content in parse_markdown_lines(markdown):
        if tag == "blank":
            text_widget.insert("end", "\n")
        elif tag == "bullet":
            text_widget.insert("end", f"  •  {content}\n", ("bullet",))
        elif tag == "quote":
            text_widget.insert("end", f"{content}\n", ("quote",))
        else:
            text_widget.insert("end", f"{content}\n", (tag,))
