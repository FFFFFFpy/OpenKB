"""Render an OKF Bundle tree into a workdir.

Every write goes through :func:`openkb.locks.atomic_write_text` /
:func:`openkb.locks.atomic_write_json` (the golden-principle "no ad-hoc
writes" rule). Those helpers are stateless - they swap a temp file into
place under a process-wide file lock and never touch KB mutation journals -
so rendering here cannot leak into the long-lived KB state.

The tree mirrors the OKF ZIP layout in :mod:`openkb.okf.schema`; this module
is the single place that turns in-memory dataclasses into files on disk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from openkb import frontmatter
from openkb.locks import atomic_write_json, atomic_write_text
from openkb.okf.llm import redact_secrets
from openkb.okf.schema import (
    ARTICLE_MD,
    ASSETS_DIR,
    CLAIMS_DIR,
    CONCEPTS_DIR,
    ENTITIES_DIR,
    EXTRACTS_DIR,
    IMAGES_DIR,
    INDEX_MD,
    LOG_MD,
    MANIFEST_JSON,
    OKF_BUNDLE_KIND,
    OKF_BUNDLE_TYPE,
    OKF_FORMAT,
    OKF_VERSION,
    OKF_YAML,
    PROPOSED_EDGES_JSONL,
    RELATIONS_DIR,
    SECTION_TYPE,
    SECTIONS_DIR,
    SOURCE_MAP_JSON,
    SOURCES_DIR,
    SUMMARY_MD,
    ConceptExtract,
    EntityExtract,
    Extracts,
    ImageRef,
    ProposedEdge,
    SectionSpec,
    manifest_compiler_block,
    okf_yaml_text,
)

logger = logging.getLogger(__name__)


def render_bundle(
    workdir: Path,
    *,
    markdown: str,
    sections: list[SectionSpec],
    image_refs: list[ImageRef],
    extracts: Extracts,
    original_filename: str,
    title: str | None,
    language: str,
    llm_enabled: bool,
    model: str | None,
    warnings: list[str],
) -> dict:
    """Write the full OKF tree into ``workdir`` and return the manifest dict.

    ``workdir`` is created if missing. ``assets/images/`` is assumed already
    populated by :mod:`openkb.okf.assets` (the actual byte copy happens there
    because it needs the source files); this function only writes text/JSON.

    ``original_filename`` is the input Markdown's basename (e.g.
    ``article.md``); it is recorded in the manifest so consumers know the
    source filename without an absolute local path leaking in.

    Returns the ``manifest.json`` dict (also written to disk) so the caller
    can include it in a batch report without re-parsing.
    """
    _safe_title = (title or _stem_from(original_filename) or "untitled").strip() or "untitled"
    _ensure_dirs(workdir)

    # --- okf.yaml ---
    atomic_write_text(workdir / OKF_YAML, okf_yaml_text(_safe_title))

    # --- sources/article.md (verbatim) ---
    atomic_write_text(workdir / ARTICLE_MD, markdown)

    # --- sections/*.md (with frontmatter) ---
    for sec in sections:
        _write_section(workdir, sec)

    # --- extracts/ ---
    _write_summary(workdir, extracts, _safe_title)
    _write_concepts(workdir, extracts.concepts, sections)
    _write_entities(workdir, extracts.entities, sections)
    # extracts/claims/ is reserved (empty in v1); create the dir so consumers
    # see the expected tree even with no claims.
    (workdir / CLAIMS_DIR).mkdir(parents=True, exist_ok=True)

    # --- relations/proposed_edges.jsonl ---
    _write_relations(workdir, extracts.relations)

    # --- source_map.json (sections + images + all extract evidence) ---
    source_map = _build_source_map(sections, image_refs, extracts)
    atomic_write_json(workdir / SOURCE_MAP_JSON, source_map)

    # --- index.md + log.md ---
    atomic_write_text(workdir / INDEX_MD, _index_md(_safe_title, sections, extracts))
    atomic_write_text(workdir / LOG_MD, _log_md(_safe_title, llm_enabled, warnings))

    # --- manifest.json (last, so it reflects what was actually written) ---
    counts = {
        "sections": len(sections),
        "concepts": len(extracts.concepts),
        "entities": len(extracts.entities),
        "relations": len(extracts.relations),
        "images": sum(1 for r in image_refs if r.found),
        "missing_assets": sum(1 for r in image_refs if not r.found),
    }
    manifest = {
        "format": OKF_FORMAT,
        "version": OKF_VERSION,
        "bundle_type": OKF_BUNDLE_TYPE,
        "kind": OKF_BUNDLE_KIND,
        "title": _safe_title,
        "entry": INDEX_MD,
        "main_source": ARTICLE_MD,
        "language": language or "en",
        "created_at": _now_iso(),
        "counts": counts,
        # Portable inputs: a filename + byte count, never an absolute path.
        "inputs": {
            "source_md": ARTICLE_MD,
            "original_filename": original_filename,
            "markdown_bytes": len(markdown.encode("utf-8")),
        },
        "compiler": manifest_compiler_block(llm_enabled=llm_enabled, model=model),
        "warnings": list(warnings) + list(extracts.warnings),
    }
    atomic_write_json(workdir / MANIFEST_JSON, manifest)
    return manifest


def _ensure_dirs(workdir: Path) -> None:
    for sub in (
        SOURCES_DIR,
        SECTIONS_DIR,
        EXTRACTS_DIR,
        CONCEPTS_DIR,
        ENTITIES_DIR,
        CLAIMS_DIR,
        RELATIONS_DIR,
        ASSETS_DIR,
        IMAGES_DIR,
    ):
        (workdir / sub).mkdir(parents=True, exist_ok=True)


def _write_section(workdir: Path, sec: SectionSpec) -> None:
    """Write one ``sections/NN_slug.md`` with evidence-matching frontmatter."""
    lines = [
        frontmatter.kv_line("type", SECTION_TYPE),
        frontmatter.kv_line("title", sec.title),
        frontmatter.kv_line("source", ARTICLE_MD),
        frontmatter.kv_line("heading_path", sec.heading_path),
        f"line_start: {sec.line_start}",
        f"line_end: {sec.line_end}",
    ]
    atomic_write_text(workdir / sec.filename, frontmatter.block(lines) + sec.body + "\n")


def _write_summary(workdir: Path, extracts: Extracts, title: str) -> None:
    """Write ``extracts/summary.md`` with a Document-Summary frontmatter.

    Frontmatter carries ``type``/``title``/``source`` so a consumer can identify
    the file without parsing the body. The summary itself is whole-article, so
    no line evidence is attached.
    """
    body = extracts.summary.strip() if extracts.summary else "_(no summary available)_"
    lines = [
        frontmatter.kv_line("type", "Document Summary"),
        frontmatter.kv_line("title", title),
        frontmatter.kv_line("source", ARTICLE_MD),
    ]
    atomic_write_text(workdir / SUMMARY_MD, frontmatter.block(lines) + f"{body}\n")


def _write_concepts(
    workdir: Path, concepts: list[ConceptExtract], sections: list[SectionSpec]
) -> None:
    for c in concepts:
        slug = _slugify(c.name) or "concept"
        lines = [
            frontmatter.kv_line("type", "Local Concept"),
            frontmatter.kv_line("title", c.name),
            frontmatter.kv_line("source", ARTICLE_MD),
            frontmatter.kv_line("scope", "local"),
            frontmatter.kv_line("heading_path", c.evidence.heading_path if c.evidence else ""),
            f"line_start: {c.evidence.line_start if c.evidence else 0}",
            f"line_end: {c.evidence.line_end if c.evidence else 0}",
        ]
        if c.confidence is not None:
            lines.append(f"confidence: {c.confidence}")
        body = c.description or ""
        content = frontmatter.block(lines) + f"{body}\n" if body else frontmatter.block(lines)
        atomic_write_text(workdir / CONCEPTS_DIR / f"{slug}.md", content)


def _write_entities(
    workdir: Path, entities: list[EntityExtract], sections: list[SectionSpec]
) -> None:
    for e in entities:
        slug = _slugify(e.name) or "entity"
        lines = [
            frontmatter.kv_line("type", f"Local {_capitalize(e.entity_type)}"),
            frontmatter.kv_line("title", e.name),
            frontmatter.kv_line("source", ARTICLE_MD),
            frontmatter.kv_line("scope", "local"),
            frontmatter.list_line("aliases", e.aliases)
            if e.aliases
            else frontmatter.kv_line("aliases", ""),
            frontmatter.kv_line("heading_path", e.evidence.heading_path if e.evidence else ""),
            f"line_start: {e.evidence.line_start if e.evidence else 0}",
            f"line_end: {e.evidence.line_end if e.evidence else 0}",
        ]
        if e.confidence is not None:
            lines.append(f"confidence: {e.confidence}")
        body = e.description or ""
        content = frontmatter.block(lines) + f"{body}\n" if body else frontmatter.block(lines)
        atomic_write_text(workdir / ENTITIES_DIR / f"{slug}.md", content)


def _write_relations(workdir: Path, relations: list[ProposedEdge]) -> None:
    """One JSON object per line (JSONL). Empty file when there are none."""
    path = workdir / PROPOSED_EDGES_JSONL
    if not relations:
        atomic_write_text(path, "")
        return
    lines = []
    for r in relations:
        obj = {
            "subject": r.subject,
            "relation": r.relation,
            "object": r.object,
            "evidence": (
                {
                    "heading_path": r.evidence.heading_path,
                    "line_start": r.evidence.line_start,
                    "line_end": r.evidence.line_end,
                }
                if r.evidence
                else None
            ),
        }
        if r.note:
            obj["note"] = r.note
        lines.append(json.dumps(obj, ensure_ascii=False))
    atomic_write_text(path, "\n".join(lines) + "\n")


def _build_source_map(
    sections: list[SectionSpec],
    image_refs: list[ImageRef],
    extracts: Extracts,
) -> dict:
    """Map every emitted artifact back to its provenance in ``sources/article.md``.

    Sections and images map to their spans/refs; each LLM extract
    (summary/concepts/entities/relations) carries its evidence so a consumer
    can trace any claim back to a section + line range of the source article.
    """
    article = ARTICLE_MD
    return {
        "article": {"file": article, "sections": len(sections)},
        "sections": [
            {
                "file": sec.filename,
                "source": article,
                "title": sec.title,
                "heading_path": sec.heading_path,
                "line_start": sec.line_start,
                "line_end": sec.line_end,
            }
            for sec in sections
        ],
        "images": [
            {
                "dest": f"{IMAGES_DIR}/{r.dest_name}" if r.found else None,
                "source": article if r.found else None,
                "original_ref": r.original_ref,
                "found": r.found,
            }
            for r in image_refs
        ],
        "summary": {
            "file": SUMMARY_MD,
            "source": article,
            # whole-article; no line range
        },
        "concepts": [
            {
                "file": f"{CONCEPTS_DIR}/{_slugify(c.name)}.md",
                "source": article,
                "name": c.name,
                "heading_path": c.evidence.heading_path if c.evidence else "",
                "line_start": c.evidence.line_start if c.evidence else 0,
                "line_end": c.evidence.line_end if c.evidence else 0,
            }
            for c in extracts.concepts
        ],
        "entities": [
            {
                "file": f"{ENTITIES_DIR}/{_slugify(e.name)}.md",
                "source": article,
                "name": e.name,
                "type": e.entity_type,
                "heading_path": e.evidence.heading_path if e.evidence else "",
                "line_start": e.evidence.line_start if e.evidence else 0,
                "line_end": e.evidence.line_end if e.evidence else 0,
            }
            for e in extracts.entities
        ],
        "relations": [
            {
                "file": PROPOSED_EDGES_JSONL,
                "source": article,
                "subject": r.subject,
                "relation": r.relation,
                "object": r.object,
                "heading_path": r.evidence.heading_path if r.evidence else "",
                "line_start": r.evidence.line_start if r.evidence else 0,
                "line_end": r.evidence.line_end if r.evidence else 0,
            }
            for r in extracts.relations
        ],
    }


def _index_md(title: str, sections: list[SectionSpec], extracts: Extracts) -> str:
    lines = [f"# {title}", "", "## Sections", ""]
    for sec in sections:
        lines.append(f"- [{sec.title}]({sec.filename})")
    lines += ["", "## Extracts", ""]
    lines.append(f"- [Summary]({SUMMARY_MD})")
    if extracts.concepts:
        lines.append(f"- Concepts ({len(extracts.concepts)})")
        for c in extracts.concepts:
            lines.append(f"  - [{c.name}](concepts/{_slugify(c.name)}.md)")
    if extracts.entities:
        lines.append(f"- Entities ({len(extracts.entities)})")
        for e in extracts.entities:
            lines.append(f"  - [{e.name}](entities/{_slugify(e.name)}.md)")
    if extracts.relations:
        lines.append(
            f"- Relations ({len(extracts.relations)}) -> "
            f"[{PROPOSED_EDGES_JSONL}]({PROPOSED_EDGES_JSONL})"
        )
    return "\n".join(lines) + "\n"


def _log_md(title: str, llm_enabled: bool, warnings: list[str]) -> str:
    status = "llm" if llm_enabled else "no-llm"
    header = f"## [compile] fresh single-Markdown compile ({status})"
    lines = ["# OKF Bundle Operations Log", "", header, ""]
    lines.append(f"Compiled article: {title}")
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in warnings:
            # Redact in case a thrown exception carried a key/token into a warning.
            lines.append(f"- {redact_secrets(w)}")
    return "\n".join(lines) + "\n"


def _slugify(text: str) -> str:
    out = []
    for ch in text.strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-")[:64]


def _capitalize(text: str) -> str:
    """Title-case an entity type for the frontmatter ``type`` field."""
    t = text.strip()
    return t[:1].upper() + t[1:] if t else t


def _stem_from(path_str: str) -> str:
    from pathlib import PurePosixPath

    return PurePosixPath(path_str).stem


def _now_iso() -> str:
    """ISO-8601 UTC timestamp without depending on the scheduler's frozen clock."""
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
