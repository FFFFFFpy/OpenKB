"""Conservative Markdown sectioning for OKF bundles.

Section boundaries come only from ATX headings (``##`` through ``######``).
Pseudo headings such as ``|title`` or ``**|title**`` are recorded as anchors
inside the current section; they never split sections.

Line numbers are 1-indexed, inclusive, and match the physical lines of the
input text exactly. These are the evidence coordinates the LLM cites and the
``sources/article.md`` reader sees, so they must not drift.
"""

from __future__ import annotations

import re

from openkb.okf.schema import SectionAnchor, SectioningResult, SectionSpec

_ATX_RE = re.compile(r"^[ \t]{0,3}(#{1,6})(?!#)[ \t]*(.*?)[ \t]*#*[ \t]*$")
_H1_RE = re.compile(r"(?m)^[ \t]{0,3}#(?!#)[ \t]*(.*?)[ \t]*$")

_BOLD_PIPE_RE = re.compile(r"^\*\*\s*\|(.+?)\s*\*\*$")
_BARE_PIPE_RE = re.compile(r"^\|(.+)$")
_BOLD_RE = re.compile(r"^\*\*(.+?)\*\*$")
_ORDINAL_RE = re.compile(
    r"^(?:第[一二三四五六七八九十百千万]+[：:]|[一二三四五六七八九十]+、|\d+[.、]\s+)(.+)$"
)

_PLAYER_TEXT = {
    "follow",
    "replay",
    "share",
    "like",
    "close",
    "your browser does not support video tags",
}


def extract_title(md: str) -> str | None:
    """Return the first H1 heading text, stripped; ``None`` when absent."""
    m = _H1_RE.search(md)
    if m is None:
        return None
    title = m.group(1).strip()
    return title or None


def split_sections(md: str, strategy: str = "auto") -> SectioningResult:
    """Split ``md`` into conservative ``SectionSpec`` records.

    ``strategy`` currently supports only ``auto``: choose the first ATX level
    from H2..H6 with at least two headings. H1 is the document title, not a
    section boundary. When no level has enough headings, the result is a
    single whole-document section.
    """
    lines = md.splitlines(keepends=True)
    h_counts, headings = _scan_atx(lines)
    effective_level = _effective_level(h_counts, strategy)

    if effective_level is None:
        section = _whole_doc_section(md, h_counts)
        return _result([section], h_counts, None)

    boundary_indexes = [h["line_index"] for h in headings if h["level"] == effective_level]
    sections: list[SectionSpec] = []

    first_boundary = boundary_indexes[0]
    if first_boundary > 0 and _has_effective_preamble(lines[:first_boundary]):
        body = "".join(lines[:first_boundary])
        sections.append(
            SectionSpec(
                index=0,
                title="preamble",
                heading_path="preamble",
                line_start=1,
                line_end=first_boundary,
                body=_strip_trailing_newline(body),
                section_id="",
                markdown_level=None,
                boundary_kind="preamble",
            )
        )

    for boundary_pos, start_index in enumerate(boundary_indexes):
        next_start = (
            boundary_indexes[boundary_pos + 1]
            if boundary_pos + 1 < len(boundary_indexes)
            else len(lines)
        )
        heading = _heading_at(headings, start_index)
        title = heading["title"] or "(untitled)"
        section_index = len(sections)
        body = "".join(lines[start_index:next_start])
        section = SectionSpec(
            index=section_index,
            title=title,
            heading_path=title,
            line_start=start_index + 1,
            line_end=max(next_start, start_index + 1),
            body=_strip_trailing_newline(body),
            section_id="",
            markdown_level=effective_level,
            boundary_kind="atx",
        )
        sections.append(section)

    _assign_ids_and_anchors(sections, lines, effective_level)
    return _result(sections, h_counts, effective_level)


def _scan_atx(lines: list[str]) -> tuple[dict[int, int], list[dict]]:
    h_counts = {level: 0 for level in range(2, 7)}
    headings: list[dict] = []
    for i, raw_line in enumerate(lines):
        match = _ATX_RE.match(raw_line.rstrip("\r\n"))
        if match is None:
            continue
        level = len(match.group(1))
        title = (match.group(2) or "").strip() or "(untitled)"
        headings.append({"line_index": i, "level": level, "title": title, "raw": raw_line})
        if 2 <= level <= 6:
            h_counts[level] += 1
    return h_counts, headings


def _effective_level(h_counts: dict[int, int], strategy: str) -> int | None:
    if strategy not in {"auto", "auto_atx"}:
        raise ValueError(f"unknown sectioning strategy: {strategy!r}")
    for level in range(2, 7):
        if h_counts.get(level, 0) >= 2:
            return level
    return None


def _result(
    sections: list[SectionSpec], h_counts: dict[int, int], effective_level: int | None
) -> SectioningResult:
    _assign_ids_and_anchors(sections, None, effective_level)
    anchor_count = sum(len(s.anchors) for s in sections)
    return SectioningResult(
        sections=sections,
        strategy="auto_atx",
        effective_level=effective_level,
        h_counts=dict(h_counts),
        section_count=len(sections),
        anchor_count=anchor_count,
    )


def _whole_doc_section(md: str, h_counts: dict[int, int]) -> SectionSpec:
    line_count = _line_count(md)
    section = SectionSpec(
        index=0,
        title="document",
        heading_path="document",
        line_start=1,
        line_end=max(line_count, 1),
        body=md,
        section_id="s0001",
        markdown_level=None,
        boundary_kind="whole_doc",
    )
    _assign_ids_and_anchors([section], md.splitlines(keepends=True), None)
    return section


def _heading_at(headings: list[dict], line_index: int) -> dict:
    for heading in headings:
        if heading["line_index"] == line_index:
            return heading
    raise ValueError(f"missing heading at line index {line_index}")


def _assign_ids_and_anchors(
    sections: list[SectionSpec], lines: list[str] | None, effective_level: int | None
) -> None:
    anchor_seq = 1
    for i, section in enumerate(sections, start=1):
        section.index = i - 1
        section.section_id = f"s{i:04d}"
        if lines is None:
            continue
        start = section.line_start - 1
        end = min(section.line_end, len(lines))
        anchors: list[SectionAnchor] = []
        for line_index in range(start, end):
            if line_index == start and section.boundary_kind == "atx":
                continue
            anchor = _anchor_from_line(
                lines[line_index],
                line_no=line_index + 1,
                anchor_id=f"a{anchor_seq:04d}",
                effective_level=effective_level,
            )
            if anchor is None:
                continue
            anchors.append(anchor)
            anchor_seq += 1
        section.anchors = anchors


def _anchor_from_line(
    raw_line: str, *, line_no: int, anchor_id: str, effective_level: int | None
) -> SectionAnchor | None:
    raw = raw_line.rstrip("\r\n")
    stripped = raw.strip()
    if _reject_anchor_line(stripped):
        return None

    atx = _ATX_RE.match(raw)
    if atx is not None:
        level = len(atx.group(1))
        if level == 1 or (effective_level is not None and level == effective_level):
            return None
        title = _clean_title(atx.group(2))
        return _anchor(anchor_id, title, raw, line_no, "atx_subheading", level)

    match = _BOLD_PIPE_RE.match(stripped)
    if match is not None:
        return _anchor(anchor_id, _clean_title("|" + match.group(1)), raw, line_no, "bold_pipe")

    match = _BARE_PIPE_RE.match(stripped)
    if match is not None:
        return _anchor(anchor_id, _clean_title("|" + match.group(1)), raw, line_no, "bare_pipe")

    match = _BOLD_RE.match(stripped)
    if match is not None:
        return _anchor(anchor_id, _clean_title(match.group(1)), raw, line_no, "bold")

    match = _ORDINAL_RE.match(stripped)
    if match is not None:
        return _anchor(anchor_id, _clean_title(stripped), raw, line_no, "ordinal")

    return None


def _anchor(
    anchor_id: str,
    title: str,
    raw: str,
    line_no: int,
    kind: str,
    markdown_level: int | None = None,
) -> SectionAnchor | None:
    if not title or len(title) > 80:
        return None
    return SectionAnchor(
        anchor_id=anchor_id,
        title=title,
        raw=raw,
        line_no=line_no,
        kind=kind,
        markdown_level=markdown_level,
    )


def _reject_anchor_line(stripped: str) -> bool:
    if not stripped:
        return True
    lower = stripped.lower()
    if lower in _PLAYER_TEXT:
        return True
    if stripped.startswith(("!", ">", "- [", "* [")):
        return True
    if stripped.startswith("|") and stripped.count("|") >= 2:
        return True
    if re.match(r"^[\-*_]{3,}$", stripped):
        return True
    if re.search(r"全文约?\d+字|阅读用时|相关阅读|本文完", stripped):
        return True
    return False


def _has_effective_preamble(lines: list[str]) -> bool:
    for raw_line in lines:
        stripped = raw_line.strip()
        if _reject_anchor_line(stripped):
            continue
        atx = _ATX_RE.match(raw_line.rstrip("\r\n"))
        if atx is not None:
            if len(atx.group(1)) == 1:
                continue
            return True
        if len(stripped) >= 12:
            return True
    return False


def _clean_title(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("**") and t.endswith("**") and len(t) >= 4:
        t = t[2:-2].strip()
    if t.startswith("|"):
        t = t[1:].strip()
    return t.strip()


def _line_count(md: str) -> int:
    if md == "":
        return 0
    if md.endswith("\n"):
        return md.count("\n")
    return md.count("\n") + 1


def _strip_trailing_newline(text: str) -> str:
    if text.endswith("\r\n"):
        return text[:-2]
    if text.endswith("\n"):
        return text[:-1]
    return text
