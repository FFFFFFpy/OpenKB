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
    md = "# Root\n\n## Section\n\nBody.\n"
    sections = split_sections(md)
    assert len(sections) == 1
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
    md = "## Hello, World!\n\nbody\n"
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
    # Leading body before the first H2 is not its own section; the first
    # section still starts at the first ## line.
    md = "# T\n\nLead paragraph before any section.\n\n## First\n\nbody\n"
    sections = split_sections(md)
    assert len(sections) == 1
    assert sections[0].title == "First"
    assert sections[0].line_start == 5  # ## First on line 5


def test_line_numbers_match_sources_article_verbatim():
    # The evidence contract: line_start/line_end must point at the exact
    # physical lines of sources/article.md (the verbatim input).
    md = "line1\n## S\nline3\n"
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
