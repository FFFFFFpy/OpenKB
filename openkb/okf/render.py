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
    source_path: str,
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

    Returns the ``manifest.json`` dict (also written to disk) so the caller
    can include it in a batch report without re-parsing.
    """
    _safe_title = (title or _stem_from(source_path) or "untitled").strip() or "untitled"
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
    _write_concepts(workdir, extracts.concepts)
    _write_entities(workdir, extracts.entities)
    # extracts/claims/ is reserved (empty in v1); create the dir so consumers
    # see the expected tree even with no claims.
    (workdir / CLAIMS_DIR).mkdir(parents=True, exist_ok=True)

    # --- relations/proposed_edges.jsonl ---
    _write_relations(workdir, extracts.relations)

    # --- source_map.json ---
    source_map = _build_source_map(sections, image_refs)
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
        "inputs": {"source": source_path, "markdown_bytes": len(markdown.encode("utf-8"))},
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
    body = extracts.summary.strip() if extracts.summary else "_(no summary available)_"
    content = f"# {title}\n\n{body}\n"
    atomic_write_text(workdir / SUMMARY_MD, content)


def _write_concepts(workdir: Path, concepts: list[ConceptExtract]) -> None:
    for c in concepts:
        slug = _slugify(c.name) or "concept"
        lines = [
            frontmatter.kv_line("type", "Concept"),
            frontmatter.kv_line("name", c.name),
            frontmatter.kv_line("scope", "local"),
            frontmatter.kv_line("heading_path", c.evidence.heading_path if c.evidence else ""),
            f"line_start: {c.evidence.line_start if c.evidence else 0}",
            f"line_end: {c.evidence.line_end if c.evidence else 0}",
        ]
        body = c.description or ""
        content = frontmatter.block(lines) + f"{body}\n" if body else frontmatter.block(lines)
        atomic_write_text(workdir / CONCEPTS_DIR / f"{slug}.md", content)


def _write_entities(workdir: Path, entities: list[EntityExtract]) -> None:
    for e in entities:
        slug = _slugify(e.name) or "entity"
        lines = [
            frontmatter.kv_line("type", _capitalize(e.entity_type)),
            frontmatter.kv_line("name", e.name),
            frontmatter.kv_line("scope", "local"),
            frontmatter.list_line("aliases", e.aliases)
            if e.aliases
            else frontmatter.kv_line("aliases", ""),
            frontmatter.kv_line("heading_path", e.evidence.heading_path if e.evidence else ""),
            f"line_start: {e.evidence.line_start if e.evidence else 0}",
            f"line_end: {e.evidence.line_end if e.evidence else 0}",
        ]
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


def _build_source_map(sections: list[SectionSpec], image_refs: list[ImageRef]) -> dict:
    """Map each section file and each image to its source provenance."""
    return {
        "sections": [
            {
                "file": sec.filename,
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
                "original_ref": r.original_ref,
                "found": r.found,
            }
            for r in image_refs
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
            lines.append(f"- {w}")
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
