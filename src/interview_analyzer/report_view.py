"""Turns the small markdown subset report.py actually emits (headings,
bullets, bold/italic emphasis, paragraphs) into a readable in-app view for
the dashboard's History/Trends tabs -- no external markdown/HTML renderer
needed for our own controlled output.

Also color-codes the specific lines report.py emits for the assessment-
profile feature (competency scores, hire recommendation, selection
probability) on a red (worst) -> amber -> green (best) scale, matching the
same palette infographic.py uses, so the in-app text view and the HTML
infographic never disagree about what "good" looks like.

Also understands two more of report.py's markdown conventions well enough
to make the Score Summary table clickable in-app, not just in a real
markdown viewer: pipe-delimited tables (header + "---" separator row,
skipped, + body rows), and `[label](#anchor)` links, which `render_into_
text_widget` binds to a Tk mark set at the matching "### {heading}"'s
slug (see _slugify) so clicking a table row -- or a "back to top" link --
scrolls the Text widget there via `.see()`. A literal `<a id="...">
</a>` line (report.py's #top anchor, for real markdown viewers) is
recognized and simply dropped -- Tk's "top" mark is always the buffer
start regardless, so the raw HTML tag has nothing useful to render here.

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

# report.py's pipe-delimited Score Summary table: a "| --- | --- |"-style
# separator row (any mix of dashes/colons/pipes/whitespace) is recognized
# and dropped entirely -- it's a markdown-table-syntax marker, not content.
_TABLE_SEPARATOR_RE = re.compile(r"^\|?[\s:|-]+\|?$")
# report.py's own bare HTML anchor convention (`<a id="top"></a>`) for real
# markdown viewers -- Tk always has a "top" mark at the buffer start
# regardless, so this line is recognized and simply dropped here.
_HTML_ANCHOR_RE = re.compile(r'^<a id="[a-z0-9\-]+"></a>$')
# [label](#anchor) link syntax -- can appear inside a table cell or as a
# standalone line (e.g. report.py's "[↑ Back to top](#top)").
_LINK_RE = re.compile(r"\[([^\]]+)\]\(#([a-z0-9\-]+)\)")
# report.py's collapsible-section sentinels around a competency's "Related
# questions" list (e.g. "<!-- collapsible:problem-solving:start -->") --
# invisible HTML comments in a real markdown viewer (so that list there is
# just always shown), but recognized here to make the matching "[View
# details](#toggle-{slug})" link a genuine expand/collapse toggle in Tk.
_COLLAPSIBLE_START_RE = re.compile(r"^<!-- collapsible:([a-z0-9\-]+):start -->$")
_COLLAPSIBLE_END_RE = re.compile(r"^<!-- collapsible:([a-z0-9\-]+):end -->$")

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


def _slugify(text: str) -> str:
    """Same algorithm as report.py's own `_slugify` (duplicated rather than
    imported, same reasoning as the duplicated color palette above) --
    this is what turns a "### {name}" heading into the Tk mark name a
    matching [name](#slug) link jumps to, so the two MUST agree."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "section"


def _split_table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _colored(base_tag: str, color_hex: str) -> str:
    """Encodes a color into the tag string itself (rather than changing
    parse_markdown_lines's 2-tuple return shape) -- render_into_text_widget
    splits it back apart. Keeps every other line's tag exactly as before
    (e.g. plain "bullet"), so existing callers/tests are unaffected."""
    return f"{base_tag}|color:{color_hex}"


def parse_markdown_lines(markdown: str) -> list[tuple[str, str]]:
    """Return [(tag, display_text), ...] for each line, where tag is one of
    "h1", "h2", "h3", "bullet", "quote", "blank", "text", "table_header",
    "table_row", "collapsible_start", or "collapsible_end" -- or, for a
    competency score / hire recommendation / selection probability line,
    that same base tag with "|color:#rrggbb" appended (see _colored). A
    table_header/table_row's display_text is its cells tab-joined; a cell
    (or any other line) may itself contain raw "[label](#anchor)" link
    syntax, left as-is here for render_into_text_widget to turn into a
    clickable jump (see module docstring). collapsible_start/end's
    display_text is the section's slug (see _COLLAPSIBLE_START_RE)."""
    lines: list[tuple[str, str]] = []
    table_started = False
    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            table_started = False
            lines.append(("blank", ""))
            continue
        if _HTML_ANCHOR_RE.match(line.strip()):
            continue
        start_match = _COLLAPSIBLE_START_RE.match(line.strip())
        if start_match:
            lines.append(("collapsible_start", start_match.group(1)))
            continue
        end_match = _COLLAPSIBLE_END_RE.match(line.strip())
        if end_match:
            lines.append(("collapsible_end", end_match.group(1)))
            continue
        if line.lstrip().startswith("|") and line.rstrip().endswith("|"):
            if _TABLE_SEPARATOR_RE.match(line.strip()):
                continue
            cells = _split_table_cells(_strip_inline_markers(line))
            tag = "table_row" if table_started else "table_header"
            table_started = True
            lines.append((tag, "\t".join(cells)))
            continue
        table_started = False
        if line.startswith("### "):
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


def _jump_to_anchor(text_widget, anchor: str) -> None:
    """Handles both link kinds report.py emits: "#toggle-{slug}" expands
    or collapses that competency's "Related questions" section (see
    render_into_text_widget's collapsible_start/end handling); anything
    else scrolls to that heading's mark, same as before."""
    if anchor.startswith("toggle-"):
        slug = anchor[len("toggle-"):]
        tag = f"collapsible_{slug}"
        if tag not in text_widget.tag_names():
            return
        expanded_state = getattr(text_widget, "_collapsible_expanded", None)
        if expanded_state is None:
            expanded_state = {}
            text_widget._collapsible_expanded = expanded_state
        expanded = not expanded_state.get(tag, False)
        expanded_state[tag] = expanded
        text_widget.tag_configure(tag, elide=not expanded)
        return
    mark_name = f"anchor_{anchor}"
    if mark_name in text_widget.mark_names():
        text_widget.see(mark_name)


def _insert_with_links(text_widget, existing_tags: set, text: str, base_tags: tuple) -> None:
    """Inserts `text`, turning any "[label](#anchor)" substrings into a
    clickable jump to that anchor's Tk mark -- everything else is inserted
    as plain tagged text. Shared by every line type (table cells, bullets,
    standalone "back to top" links) since a link can appear in any of
    them, not just one specific tag."""
    pos = 0
    for m in _LINK_RE.finditer(text):
        if m.start() > pos:
            text_widget.insert("end", text[pos:m.start()], base_tags)
        label, anchor = m.group(1), m.group(2)
        link_tag = f"link_to_{anchor}"
        if link_tag not in existing_tags:
            text_widget.tag_configure(link_tag, foreground="#0f6e77", underline=True)
            text_widget.tag_bind(link_tag, "<Button-1>", lambda e, a=anchor: _jump_to_anchor(e.widget, a))
            text_widget.tag_bind(link_tag, "<Enter>", lambda e: e.widget.config(cursor="hand2"))
            text_widget.tag_bind(link_tag, "<Leave>", lambda e: e.widget.config(cursor=""))
            existing_tags.add(link_tag)
        text_widget.insert("end", label, base_tags + (link_tag,))
        pos = m.end()
    if pos < len(text):
        text_widget.insert("end", text[pos:], base_tags)


def render_into_text_widget(text_widget, markdown: str) -> None:
    """Populate a Tkinter Text widget (must already have the tags below
    configured, see dashboard.py) with a readable rendering of `markdown`.
    Leaves the widget in its normal (editable) state when done is up to
    the caller -- this only inserts content.

    Also sets up navigation: a "top" mark at the buffer start, and a mark
    per heading keyed by its slug (see _slugify) -- report.py's Score
    Summary table links and "back to top" lines are turned into real
    clickable jumps to these marks by _insert_with_links, working even
    though these Text widgets are left in state="disabled" after
    rendering (tag bindings and .see() aren't gated by that option).

    Also handles report.py's collapsible "Related questions" sections
    (collapsible_start/end, see parse_markdown_lines): everything between
    a start/end pair gets an extra `collapsible_{slug}` tag, initially
    elided (hidden) -- clicking the matching "[View details](#toggle-
    {slug})" link (handled by _jump_to_anchor) flips that tag's elide
    state to show/hide the whole section as one block. Expand/collapse
    state resets to collapsed every render, same as a fresh page load."""
    text_widget.delete("1.0", "end")
    text_widget._collapsible_expanded = {}
    existing_tags = set(text_widget.tag_names())
    text_widget.mark_set("anchor_top", "1.0")
    text_widget.mark_gravity("anchor_top", "left")
    current_collapsible: str | None = None
    for raw_tag, content in parse_markdown_lines(markdown):
        base_tag, _, color_part = raw_tag.partition("|color:")
        tk_tags = [base_tag]
        if color_part:
            color_tag = f"dyncolor_{color_part.lstrip('#')}"
            if color_tag not in existing_tags:
                text_widget.tag_configure(color_tag, foreground=color_part)
                existing_tags.add(color_tag)
            tk_tags.append(color_tag)

        if base_tag == "collapsible_start":
            current_collapsible = content
            continue
        if base_tag == "collapsible_end":
            current_collapsible = None
            continue
        if current_collapsible is not None:
            collapsible_tag = f"collapsible_{current_collapsible}"
            if collapsible_tag not in existing_tags:
                text_widget.tag_configure(collapsible_tag, elide=True)
                existing_tags.add(collapsible_tag)
            tk_tags.append(collapsible_tag)

        if base_tag == "blank":
            # tagged (not a bare insert) so a blank line inside a
            # collapsible section is elided along with the rest of it --
            # untagged text is immune to elide and would otherwise leave a
            # stray visible empty line even while "collapsed"
            text_widget.insert("end", "\n", tuple(tk_tags))
        elif base_tag in ("h1", "h2", "h3"):
            mark_name = f"anchor_{_slugify(content)}"
            text_widget.mark_set(mark_name, "end")
            text_widget.mark_gravity(mark_name, "left")
            _insert_with_links(text_widget, existing_tags, f"{content}\n", tuple(tk_tags))
        elif base_tag == "bullet":
            _insert_with_links(text_widget, existing_tags, f"  •  {content}\n", tuple(tk_tags))
        else:
            _insert_with_links(text_widget, existing_tags, f"{content}\n", tuple(tk_tags))
