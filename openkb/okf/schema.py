"""OKF Bundle schema constants and shared dataclasses.

Single source of truth for the OKF Bundle layout path strings, the
``okf.yaml`` / ``manifest.json`` skeletons, and the small dataclasses that
flow between ``markdown`` / ``assets`` / ``render`` / ``compiler``.

This is a subtraction bundle: every compile is fresh and reads/writes nothing
outside its own zip. The manifest ``compiler`` block records
``global_read=false`` / ``global_write=false`` so consumers can trust the
isolation guarantee from the manifest alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- OKF format identity -------------------------------------------------

OKF_FORMAT = "okf-bundle"
OKF_VERSION = 1
OKF_BUNDLE_KIND = "markdown_article"
OKF_BUNDLE_TYPE = "markdown_article"  # manifest.json bundle_type (aligned with kind)

# The compiler identity recorded in manifest.json. ``mode`` marks this as a
# fresh single-Markdown compile (no global KB state touched).
COMPILER_NAME = "openkb-okf-compiler"
COMPILER_MODE = "fresh-single-md"

# --- ZIP layout paths (single source of truth) ----------------------------
# Forward-slash, zip-relative. Every writer reads these so the on-disk tree
# and the manifest ``entry``/``main_source`` pointers never drift apart.

OKF_YAML = "okf.yaml"
MANIFEST_JSON = "manifest.json"
INDEX_MD = "index.md"
LOG_MD = "log.md"

SOURCES_DIR = "sources"
ARTICLE_MD = "sources/article.md"  # verbatim copy of the input Markdown

SECTIONS_DIR = "sections"
EXTRACTS_DIR = "extracts"
SUMMARY_MD = "extracts/summary.md"
CONCEPTS_DIR = "extracts/concepts"
ENTITIES_DIR = "extracts/entities"
CLAIMS_DIR = "extracts/claims"  # reserved, empty in v1
RELATIONS_DIR = "relations"
PROPOSED_EDGES_JSONL = "relations/proposed_edges.jsonl"

ASSETS_DIR = "assets"
IMAGES_DIR = "assets/images"

SOURCE_MAP_JSON = "source_map.json"

# Section filename zero-pad width (00_document.md, 01_..., 02_...).
SECTION_INDEX_WIDTH = 2

# Section frontmatter ``type`` value.
SECTION_TYPE = "Section"


@dataclass
class SectionAnchor:
    """A section-internal anchor discovered from subheadings or pseudo headings."""

    anchor_id: str
    title: str
    raw: str
    line_no: int
    kind: str
    markdown_level: int | None = None


@dataclass
class SectionSpec:
    """One H2-delimited section of the source Markdown.

    ``line_start``/``line_end`` are 1-indexed and inclusive, spanning the
    section's ``## `` heading line through the last line before the next
    ``## `` (or end of document). They are the evidence coordinates the LLM
    cites against, so they must match what a reader sees in ``sources/article.md``.
    """

    index: int
    title: str
    heading_path: str
    line_start: int
    line_end: int
    body: str
    section_id: str = ""
    markdown_level: int | None = None
    boundary_kind: str = "atx"
    anchors: list[SectionAnchor] = field(default_factory=list)

    @property
    def filename(self) -> str:
        """Zip-relative path: ``sections/NN_slug.md``."""
        slug = _slugify(self.title) or "section"
        return f"{SECTIONS_DIR}/{self.index:0{SECTION_INDEX_WIDTH}d}_{slug}.md"


@dataclass
class SectioningResult:
    """Result of conservative Markdown sectioning.

    It intentionally behaves like a list for existing callers that only need
    the sections, while carrying metadata for manifest/source-map consumers.
    """

    sections: list[SectionSpec]
    strategy: str
    effective_level: int | None
    h_counts: dict[int, int]
    section_count: int
    anchor_count: int

    def __iter__(self):
        return iter(self.sections)

    def __len__(self) -> int:
        return len(self.sections)

    def __getitem__(self, index):
        return self.sections[index]

    def __bool__(self) -> bool:
        return bool(self.sections)

    def to_manifest(self) -> dict:
        data = {
            "strategy": self.strategy,
            "effective_level": self.effective_level,
            "section_count": self.section_count,
            "anchor_count": self.anchor_count,
        }
        for level in range(2, 7):
            data[f"h{level}_count"] = self.h_counts.get(level, 0)
        return data


@dataclass
class ImageRef:
    """A relative image discovered in the Markdown and copied into the bundle.

    ``dest_name`` is the filename actually written under ``assets/images/``
    (may carry a hash prefix when two sources share a basename).
    ``found`` is False when the source file was missing - the link is left
    unchanged and the caller records a warning + bumps ``missing_assets``.
    """

    original_ref: str  # the raw ``(path)` substring from the Markdown link
    dest_name: str  # filename under assets/images/
    source_path: str  # absolute source path (for diagnostics only - never written)
    found: bool
    alt: str = ""


@dataclass
class Evidence:
    """Line-anchored provenance for an LLM extract (concept/entity/relation).

    Every concept/entity/relation MUST carry one. Extracts without evidence
    are dropped (with a recorded warning) rather than written, so a consumer
    can always trace a claim back to a span of the source article.
    """

    heading_path: str
    line_start: int
    line_end: int
    section_id: str = ""

    def is_valid(self) -> bool:
        """Structural validity only (no section-map check).

        Use :func:`validate_evidence` for the full check against the article's
        section map - that's what the extractor calls before keeping an extract.
        """
        return bool(self.heading_path) and self.line_start >= 1 and self.line_end >= self.line_start


def validate_evidence(
    evidence: Evidence | None, sections: list[SectionSpec], total_lines: int
) -> bool:
    """Full evidence check against the article's section map.

    Returns True only when:
      * ``evidence`` is present and structurally valid;
      * ``heading_path`` matches a known ``SectionSpec.heading_path``;
      * ``line_start``/``line_end`` fall within that section's
        ``[section.line_start, section.line_end]`` range;
      * ``line_end >= line_start``.

    ``total_lines`` bounds the document so an out-of-section but in-range
    citation against a heading whose range we lost can't slip through; the
    section-range check is the primary gate, ``total_lines`` is a backstop.
    """
    if evidence is None or not evidence.is_valid():
        return False
    if evidence.section_id:
        for sec in sections:
            if sec.section_id != evidence.section_id:
                continue
            return (
                sec.line_start <= evidence.line_start <= sec.line_end
                and sec.line_start <= evidence.line_end <= sec.line_end
                and evidence.line_end >= evidence.line_start
            )
        return False
    for sec in sections:
        if sec.heading_path != evidence.heading_path:
            continue
        if (
            sec.line_start <= evidence.line_start <= sec.line_end
            and sec.line_start <= evidence.line_end <= sec.line_end
            and evidence.line_end >= evidence.line_start
        ):
            return True
    # No matching section. If there are no sections at all (whole-doc), accept
    # any in-document range; otherwise reject.
    if not sections:
        return 1 <= evidence.line_start <= total_lines and evidence.line_end <= total_lines
    return False


@dataclass
class ConceptExtract:
    """A local concept extracted from one article (NOT a global wiki page)."""

    name: str
    description: str
    evidence: Evidence | None = None
    confidence: float | None = None

    def is_valid(self) -> bool:
        return bool(self.name and self.name.strip()) and bool(
            self.evidence and self.evidence.is_valid()
        )


@dataclass
class EntityExtract:
    """A local entity (person/org/place/...) extracted from one article."""

    name: str
    entity_type: str
    description: str
    aliases: list[str] = field(default_factory=list)
    evidence: Evidence | None = None
    confidence: float | None = None

    def is_valid(self) -> bool:
        return (
            bool(self.name and self.name.strip())
            and bool(self.entity_type and self.entity_type.strip())
            and bool(self.evidence and self.evidence.is_valid())
        )


@dataclass
class ProposedEdge:
    """A proposed relation between two local extracts (or typed nodes).

    Written only to ``relations/proposed_edges.jsonl`` - never materialized as
    a bidirectional link in the article body. Consumers decide whether to
    accept the edge.
    """

    subject: str
    relation: str
    object: str
    evidence: Evidence | None = None
    note: str = ""

    def is_valid(self) -> bool:
        return (
            bool(self.subject.strip())
            and bool(self.relation.strip())
            and bool(self.object.strip())
            and bool(self.evidence and self.evidence.is_valid())
        )


@dataclass
class Extracts:
    """All LLM-produced extracts for one article.

    Built by ``llm.extract`` (or empty when ``--no-llm``). Carries the
    warnings recorded during extraction (evidence-less drops, parse errors)
    so the renderer can surface them in ``manifest.json``.
    """

    summary: str = ""
    concepts: list[ConceptExtract] = field(default_factory=list)
    entities: list[EntityExtract] = field(default_factory=list)
    relations: list[ProposedEdge] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def okf_yaml_text(title: str) -> str:
    """Render the canonical ``okf.yaml`` for a bundle.

    ``title`` is the document root title (first H1) or the input stem. Kept a
    plain YAML literal (no JSON quoting) since the values here are all simple
    scalars with no escaping needs.

    ``compiler.global_read`` / ``compiler.global_write`` use the dotted-key
    form the OKF spec mandates, so a consumer can grep the literal
    ``compiler.global_read: false`` string from the file.
    """
    safe_title = _yaml_scalar(title) if title else ""
    return (
        f"format: {OKF_FORMAT}\n"
        f"version: {OKF_VERSION}\n"
        f"kind: {OKF_BUNDLE_KIND}\n"
        f"title: {safe_title}\n"
        f"entry: {INDEX_MD}\n"
        f"main_source: {ARTICLE_MD}\n"
        f"compiler.name: {COMPILER_NAME}\n"
        f"compiler.mode: {COMPILER_MODE}\n"
        f"compiler.global_read: false\n"
        f"compiler.global_write: false\n"
    )


def manifest_compiler_block(*, llm_enabled: bool, model: str | None) -> dict:
    """Build the ``compiler`` sub-object of ``manifest.json``.

    ``api_key`` is deliberately never included here (or anywhere in the bundle).
    ``model`` is recorded when an LLM ran so consumers know which model produced
    the extracts.
    """
    block: dict = {
        "name": COMPILER_NAME,
        "mode": COMPILER_MODE,
        "global_read": False,
        "global_write": False,
        "llm_enabled": bool(llm_enabled),
    }
    if model:
        block["model"] = model
    return block


def _slugify(text: str) -> str:
    """Lowercase, alnum+dashes only, trimmed. Empty string when nothing usable."""
    out = []
    for ch in text.strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    slug = "".join(out).strip("-")
    return slug[:64]


def _yaml_scalar(value: str) -> str:
    """Quote a YAML scalar only when it needs it; bare otherwise."""
    v = value.strip()
    if v and not any(c in v for c in ":#{}\n'\"") and not v.startswith(("-", " ", "?")):
        return v
    # Fallback: JSON-style quoted (strict YAML subset), escapes safely.
    import json

    return json.dumps(v, ensure_ascii=False)
