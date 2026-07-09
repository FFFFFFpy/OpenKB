"""Split a Markdown article into H2-delimited sections (pure, no I/O).

``#`` (H1) is the document root title, not a section. ``##`` is the section
boundary. ``###``/``####`` and everything else stay inside the section they
appear in. When the document has no ``##`` heading, a single
``sections/00_document.md`` covers the whole body.

Line numbers are 1-indexed, inclusive, and match the physical lines of the
input text exactly - these are the evidence coordinates the LLM cites and the
``sources/article.md`` reader sees, so they must not drift.
"""

from __future__ import annotations

import re

from openkb.okf.schema import SectionSpec

# A line that starts (after optional spaces) with exactly two ``#`` followed
# by a space or end-of-line. ``(?<!#)`` prevents ``###`` from matching as a
# ``##`` (a common off-by-one when the third hash is consumed greedily), and
# the negative lookahead ``(?!#)`` does the same for the ``##`` end.
_H2_RE = re.compile(r"(?m)^[ \t]{0,3}##(?!#)[ \t]*(.*?)[ \t]*$")
# ``#`` (exactly one) - the document root title.
_H1_RE = re.compile(r"(?m)^[ \t]{0,3}#(?!#)[ \t]*(.*?)[ \t]*$")


def extract_title(md: str) -> str | None:
    """Return the first H1 heading text, stripped; ``None`` when absent."""
    m = _H1_RE.search(md)
    if m is None:
        return None
    title = m.group(1).strip()
    return title or None


def split_sections(md: str) -> list[SectionSpec]:
    """Split ``md`` into ``SectionSpec`` records by H2 boundaries.

    The H1 root title (if any) is folded into the first section's body rather
    than becoming its own section - the document root is not a section. H3/H4
    and all body content stay in whichever section they appear under. A
    document with zero ``##`` headings yields a single section spanning the
    whole body (``00_document.md``).

    ``line_start``/``line_end`` are 1-indexed and inclusive and count every
    physical line (CRLF/LF normalized via :func:`str.splitlines` semantics of
    the regex operating on the raw string - we index by character offsets then
    map to line numbers).
    """
    if md == "":
        return [_whole_doc_section(md, 0)]

    h2_matches = list(_H2_RE.finditer(md))
    if not h2_matches:
        return [_whole_doc_section(md, 0)]

    sections: list[SectionSpec] = []
    # Heading path for a single-article bundle: the H2 title alone (no parent
    # H1 segment) keeps evidence coordinates simple and unambiguous. We do not
    # prepend the H1 title because the H1 is the document root, not a section
    # ancestor.
    for i, m in enumerate(h2_matches):
        title = (m.group(1) or "").strip() or "(untitled)"
        start = _line_at(md, m.start())
        if i + 1 < len(h2_matches):
            end = _line_at(md, h2_matches[i + 1].start()) - 1
        else:
            end = _line_count(md)
        if end < start:
            end = start
        body = md[m.start() : (h2_matches[i + 1].start() if i + 1 < len(h2_matches) else len(md))]
        sections.append(
            SectionSpec(
                index=i,
                title=title,
                heading_path=title,
                line_start=start,
                line_end=end,
                body=_strip_trailing_newline(body),
            )
        )
    return sections


def _whole_doc_section(md: str, index: int) -> SectionSpec:
    """The single section used when the document has no H2 headings."""
    line_count = _line_count(md)
    return SectionSpec(
        index=index,
        title="document",
        heading_path="document",
        line_start=1,
        line_end=max(line_count, 1),
        body=md,
    )


def _line_at(md: str, offset: int) -> int:
    """1-indexed line number of the character at ``offset``."""
    if offset <= 0:
        return 1
    return md.count("\n", 0, offset) + 1


def _line_count(md: str) -> int:
    """Number of lines (a trailing newline => one more empty line is NOT counted)."""
    if md == "":
        return 0
    # splitlines handles trailing newline correctly: "a\n" -> ["a"] (1 line).
    # For evidence, a final unterminated line still counts as a line.
    if md.endswith("\n"):
        return md.count("\n")
    return md.count("\n") + 1


def _strip_trailing_newline(text: str) -> str:
    if text.endswith("\r\n"):
        return text[:-2]
    if text.endswith("\n"):
        return text[:-1]
    return text
