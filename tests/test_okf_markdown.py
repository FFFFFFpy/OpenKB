"""Tests for openkb.okf.markdown - H2 section splitting.

Covers (#1): H2 splitting, inclusive line_start/line_end, H1 as document
root (not a section), no-H2 -> 00_document.md, and H3/H4 staying in their
parent section.
"""

from __future__ import annotations

from openkb.okf.markdown import extract_title, split_sections
from openkb.okf.schema import Evidence, validate_evidence


def test_h2_split_line_numbers_inclusive():
    md = "# Title\n\nIntro.\n\n## First\n\nA\n\nB\n\n## Second\n\nC\n"
    # L1 # Title | L2 blank | L3 Intro. | L4 blank | L5 ## First | L6 blank |
    # L7 A | L8 blank | L9 B | L10 blank | L11 ## Second | L12 blank | L13 C
    sections = split_sections(md)
    assert [s.title for s in sections] == ["First", "Second"]
    assert sections[0].line_start == 5  # ## First on L5
    assert sections[0].line_end == 10  # last line before ## Second (L11) is the blank L10
    assert sections[1].line_start == 11  # ## Second
    assert sections[1].line_end == 13  # C on L13


def test_h1_is_root_not_a_section():
    md = "# Root\n\n## Section\n\nBody.\n\n## Next\n\nMore.\n"
    sections = split_sections(md)
    assert len(sections) == 2
    assert sections[0].title == "Section"
    assert extract_title(md) == "Root"


def test_no_h2_yields_single_document_section():
    md = "# Just Title\n\nPara one.\nPara two.\n"
    sections = split_sections(md)
    assert len(sections) == 1
    assert sections[0].title == "document"
    assert sections[0].filename == "sections/00_document.md"
    assert sections[0].line_start == 1
    assert sections[0].line_end >= 3


def test_one_h2_but_multiple_h3_uses_h3_sections():
    md = "# T\n\n## Parent\n\n### A\n\nbody\n\n### B\n\nbody\n"
    sectioning = split_sections(md)

    assert sectioning.effective_level == 3
    assert [s.title for s in sectioning.sections if s.boundary_kind == "atx"] == ["A", "B"]
    assert all(s.markdown_level == 3 for s in sectioning.sections if s.boundary_kind == "atx")


def test_no_h2_but_multiple_h3_uses_h3_sections():
    md = "# T\n\n### A\n\nbody\n\n### B\n\nbody\n"
    sectioning = split_sections(md)

    assert sectioning.effective_level == 3
    assert [s.title for s in sectioning.sections] == ["A", "B"]


def test_pipe_and_bold_pipe_are_anchors_not_sections():
    md = (
        "# T\n\n"
        "### A\n\n"
        "**|Bold Pipe Anchor**\n\n"
        "|Bare Pipe Anchor\n\n"
        "### B\n\n"
        "body\n"
    )
    sections = split_sections(md)

    assert [s.title for s in sections] == ["A", "B"]
    anchors = sections[0].anchors
    assert [(a.title, a.kind) for a in anchors] == [
        ("Bold Pipe Anchor", "bold_pipe"),
        ("Bare Pipe Anchor", "bare_pipe"),
    ]


def test_mixed_wechat_style_sections_follow_h3_not_pipe():
    md = (
        "# Title\n\n"
        "### First Main\n\n"
        "body\n\n"
        "### Second Main\n\n"
        "## |Pipe-looking ATX but H2 count is too small\n\n"
        "body\n\n"
        "**|Bold pipe anchor**\n\n"
        "body\n\n"
        "### Third Main\n\n"
        "|Bare pipe anchor\n\n"
        "body\n"
    )
    sectioning = split_sections(md)

    assert sectioning.effective_level == 3
    assert [s.title for s in sectioning.sections] == ["First Main", "Second Main", "Third Main"]
    assert sectioning.anchor_count == 3
    assert [a.kind for s in sectioning.sections for a in s.anchors] == [
        "atx_subheading",
        "bold_pipe",
        "bare_pipe",
    ]


def test_h3_h4_stay_in_parent_section():
    md = (
        "# T\n\n## Parent\n\nIntro.\n\n### Child\n\nsub.\n\n"
        "#### Grandchild\n\nsubsub.\n\n## Next\n\nx\n"
    )
    sections = split_sections(md)
    assert [s.title for s in sections] == ["Parent", "Next"]
    parent_body = sections[0].body
    assert "### Child" in parent_body
    assert "#### Grandchild" in parent_body
    # the child headings must NOT start a new section
    assert sections[1].title == "Next"


def test_empty_document():
    sections = split_sections("")
    assert len(sections) == 1
    assert sections[0].title == "document"
    assert sections[0].filename == "sections/00_document.md"


def test_section_filename_slug():
    md = "## Hello, World!\n\nbody\n\n## Next\n\nmore\n"
    sections = split_sections(md)
    assert sections[0].filename == "sections/00_hello-world.md"


def test_h2_with_no_title():
    md = "# T\n\n##\n\nbody\n\n## Real\n\nx\n"
    sections = split_sections(md)
    assert len(sections) == 2
    # empty H2 title -> "(untitled)"
    assert sections[0].title == "(untitled)"
    assert sections[1].title == "Real"


def test_h2_preceded_by_content():
    # Leading body before the first effective heading becomes a preamble.
    md = "# T\n\nLead paragraph before any section.\n\n## First\n\nbody\n\n## Second\n\nmore\n"
    sections = split_sections(md)
    assert len(sections) == 3
    assert sections[0].title == "preamble"
    assert sections[0].boundary_kind == "preamble"
    assert sections[1].title == "First"
    assert sections[1].line_start == 5  # ## First on line 5


def test_line_numbers_match_sources_article_verbatim():
    # The evidence contract: line_start/line_end must point at the exact
    # physical lines of sources/article.md (the verbatim input).
    md = "line1\n## S\nline3\n## T\nline5\n"
    sections = split_sections(md)
    assert sections[0].line_start == 2  # "## S"
    assert sections[0].line_end == 3  # "line3"
    # the section body begins with the heading line
    assert sections[0].body.startswith("## S")


def test_validate_evidence_accepts_later_duplicate_heading():
    md = "# T\n\n## Repeat\n\nfirst\n\n## Repeat\n\nsecond\n"
    sections = split_sections(md)

    assert [s.heading_path for s in sections] == ["Repeat", "Repeat"]
    assert validate_evidence(Evidence("Repeat", line_start=7, line_end=9), sections, 9)


def test_validate_evidence_prefers_section_id_and_rejects_unknown():
    md = "# T\n\n## Repeat\n\nfirst\n\n## Repeat\n\nsecond\n"
    sections = split_sections(md)

    assert sections[0].section_id == "s0001"
    assert sections[1].section_id == "s0002"
    assert validate_evidence(
        Evidence("Repeat", line_start=7, line_end=9, section_id="s0002"), sections, 9
    )
    assert not validate_evidence(
        Evidence("Repeat", line_start=7, line_end=9, section_id="s9999"), sections, 9
    )
    assert not validate_evidence(
        Evidence("Repeat", line_start=7, line_end=9, section_id="s0001"), sections, 9
    )
