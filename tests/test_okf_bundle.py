"""Tests for openkb.okf bundle + render - zip hygiene guarantees.

Covers (#5): zip has no absolute paths, no ``..`` segments, no
``raw/original.mhtml``; the structure matches the spec tree; okf.yaml and
manifest.json carry the required fields. (#12): the api_key never appears in
any file inside the zip.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from openkb.okf.bundle import write_zip
from openkb.okf.markdown import split_sections
from openkb.okf.render import render_bundle
from openkb.okf.schema import (
    ConceptExtract,
    EntityExtract,
    Evidence,
    Extracts,
    ProposedEdge,
)

_MD = "# My Article\n\n## Section A\n\nBody.\n\n## Section B\n\nMore.\n"


def _build_zip(tmp_path: Path, *, llm_enabled: bool = False, model: str | None = None) -> Path:
    workdir = tmp_path / "wd"
    render_bundle(
        workdir,
        markdown=_MD,
        sections=split_sections(_MD),
        image_refs=[],
        extracts=Extracts(),
        original_filename="article.md",
        title="My Article",
        language="zh",
        llm_enabled=llm_enabled,
        model=model,
        warnings=[],
    )
    out = tmp_path / "out.okf.zip"
    return write_zip(workdir, out)


def test_zip_has_no_absolute_or_traversal_paths(tmp_path):
    out = _build_zip(tmp_path)
    with zipfile.ZipFile(out) as zf:
        for name in zf.namelist():
            assert not name.startswith("/"), name
            assert not name.startswith("\\"), name
            assert ".." not in name.split("/"), name


def test_zip_has_no_raw_mhtml(tmp_path):
    out = _build_zip(tmp_path)
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    assert not any("raw/" in n or "original.mhtml" in n for n in names)


def test_zip_structure_matches_spec(tmp_path):
    out = _build_zip(tmp_path)
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    # required top-level files
    for required in ("okf.yaml", "manifest.json", "index.md", "log.md", "source_map.json"):
        assert required in names, required
    assert "sources/article.md" in names
    assert "sections/00_section-a.md" in names
    assert "sections/01_section-b.md" in names
    assert "extracts/summary.md" in names
    assert "relations/proposed_edges.jsonl" in names
    # reserved empty dir
    assert "extracts/claims/" in names
    assert "assets/images/" in names


def test_okf_yaml_required_fields(tmp_path):
    out = _build_zip(tmp_path)
    with zipfile.ZipFile(out) as zf:
        text = zf.read("okf.yaml").decode()
    assert "format: okf-bundle" in text
    assert "version: 1" in text
    assert "kind: markdown_article" in text
    assert "entry: index.md" in text
    assert "main_source: sources/article.md" in text
    assert "compiler.global_read: false" in text
    assert "compiler.global_write: false" in text


def test_manifest_required_fields(tmp_path):
    out = _build_zip(tmp_path, llm_enabled=True, model="openai/gpt-4o")
    with zipfile.ZipFile(out) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    for key in (
        "format",
        "version",
        "bundle_type",
        "title",
        "entry",
        "main_source",
        "language",
        "created_at",
        "counts",
        "inputs",
        "compiler",
        "warnings",
    ):
        assert key in manifest, key
    compiler = manifest["compiler"]
    assert compiler["name"] == "openkb-okf-compiler"
    assert compiler["mode"] == "fresh-single-md"
    assert compiler["global_read"] is False
    assert compiler["global_write"] is False
    assert compiler["llm_enabled"] is True
    assert compiler["model"] == "openai/gpt-4o"
    # api_key must NEVER appear in the compiler block
    assert "api_key" not in compiler
    # bundle_type aligned with kind (markdown_article, not single-article)
    assert manifest["bundle_type"] == "markdown_article"
    # inputs must be portable (no absolute local path)
    inputs = manifest["inputs"]
    assert inputs["source_md"] == "sources/article.md"
    assert inputs["original_filename"] == "article.md"
    assert isinstance(inputs["markdown_bytes"], int)
    assert "source" not in inputs  # old absolute-path field removed
    counts = manifest["counts"]
    for key in ("sections", "concepts", "entities", "relations", "images", "missing_assets"):
        assert key in counts, key
    assert counts["sections"] == 2


def test_api_key_absent_from_zip_contents(tmp_path):
    """The api_key must never be written into any file in the zip.

    Render is api_key-agnostic: ``render_bundle`` takes no api_key argument,
    and ``manifest_compiler_block`` records only the model. So a bundle built
    while a real api_key is in flight cannot leak it through the renderer.
    We assert the bundle contains no ``api_key`` key anywhere and that the
    (fake) key string is absent from every entry.
    """
    fake_key = "sk-FAKE-KEY-DO-NOT-LEAK-9876543210"
    workdir = tmp_path / "wd"
    # Build an Extracts whose summary intentionally does NOT contain the key
    # (extracts never see the key), to confirm the renderer doesn't inject it.
    render_bundle(
        workdir,
        markdown=_MD,
        sections=split_sections(_MD),
        image_refs=[],
        extracts=Extracts(summary="A clean summary."),
        original_filename="article.md",
        title="My Article",
        language="zh",
        llm_enabled=True,
        model="openai/gpt-4o",
        warnings=[],
    )
    out = tmp_path / "out.okf.zip"
    write_zip(workdir, out)
    with zipfile.ZipFile(out) as zf:
        for name in zf.namelist():
            data = zf.read(name).decode("utf-8", errors="replace")
            assert fake_key not in data, f"api_key leaked into {name}"
    # The manifest must not carry an api_key key (or value) anywhere.
    with zipfile.ZipFile(out) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert "api_key" not in json.dumps(manifest)


def test_manifest_compiler_has_no_api_key_field(tmp_path):
    out = _build_zip(tmp_path, llm_enabled=True, model="openai/gpt-4o")
    with zipfile.ZipFile(out) as zf:
        manifest = json.loads(zf.read("manifest.json"))

    # exhaustive: no key anywhere in the manifest whose value is the api_key
    def _walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert k.lower() != "api_key", k
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)

    _walk(manifest)


def test_forbidden_path_raises(tmp_path):
    """A raw/original.mhtml in the workdir must block the zip write."""
    workdir = tmp_path / "wd"
    (workdir / "raw").mkdir(parents=True)
    (workdir / "raw" / "original.mhtml").write_text("should not be here")
    out = tmp_path / "bad.okf.zip"
    try:
        write_zip(workdir, out)
        raise AssertionError("expected ValueError for forbidden raw/ path")
    except ValueError as exc:
        assert "raw" in str(exc).lower() or "mhtml" in str(exc).lower()


def _extracts_with_evidence() -> Extracts:
    ev = Evidence(heading_path="Section A", line_start=3, line_end=4)
    return Extracts(
        summary="A summary.",
        concepts=[ConceptExtract(name="ConceptA", description="d", evidence=ev, confidence=0.9)],
        entities=[EntityExtract(name="EntityA", entity_type="org", description="d", evidence=ev)],
        relations=[
            ProposedEdge(subject="ConceptA", relation="mentions", object="EntityA", evidence=ev)
        ],
    )


def test_source_map_contains_extract_evidence(tmp_path):
    """source_map.json must map each extract back to sources/article.md + evidence."""
    workdir = tmp_path / "wd"
    render_bundle(
        workdir,
        markdown=_MD,
        sections=split_sections(_MD),
        image_refs=[],
        extracts=_extracts_with_evidence(),
        original_filename="article.md",
        title="My Article",
        language="zh",
        llm_enabled=True,
        model="openai/gpt-4o",
        warnings=[],
    )
    sm = json.loads((workdir / "source_map.json").read_text(encoding="utf-8"))
    # every extract category is present and points at the article source
    assert sm["summary"]["source"] == "sources/article.md"
    assert sm["concepts"] and sm["concepts"][0]["source"] == "sources/article.md"
    assert sm["concepts"][0]["name"] == "ConceptA"
    assert sm["concepts"][0]["heading_path"] == "Section A"
    assert sm["entities"] and sm["entities"][0]["source"] == "sources/article.md"
    assert sm["entities"][0]["name"] == "EntityA"
    assert sm["relations"] and sm["relations"][0]["source"] == "sources/article.md"
    assert sm["relations"][0]["subject"] == "ConceptA"
    assert sm["relations"][0]["object"] == "EntityA"


def test_summary_and_concept_frontmatter(tmp_path):
    """summary.md has Document-Summary frontmatter; concepts are Local Concept."""
    workdir = tmp_path / "wd"
    render_bundle(
        workdir,
        markdown=_MD,
        sections=split_sections(_MD),
        image_refs=[],
        extracts=_extracts_with_evidence(),
        original_filename="article.md",
        title="My Article",
        language="zh",
        llm_enabled=True,
        model="openai/gpt-4o",
        warnings=[],
    )
    summary = (workdir / "extracts" / "summary.md").read_text(encoding="utf-8")
    assert 'type: "Document Summary"' in summary
    assert 'source: "../sources/article.md"' in summary
    concept = (workdir / "extracts" / "concepts" / "concepta.md").read_text(encoding="utf-8")
    assert 'type: "Local Concept"' in concept
    assert 'source: "../../sources/article.md"' in concept
    assert "confidence: 0.9" in concept


def test_no_absolute_paths_in_zip_text_files(tmp_path):
    """No drive-letter or rooted absolute path in any zip text file."""
    import re

    out = _build_zip(tmp_path, llm_enabled=True, model="openai/gpt-4o")
    abs_re = re.compile(r"[A-Za-z]:[\\/]|^/")
    with zipfile.ZipFile(out) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            data = zf.read(name).decode("utf-8", errors="replace")
            assert not abs_re.search(data), f"absolute path in {name}"
