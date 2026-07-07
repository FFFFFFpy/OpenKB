"""Tests for MHTML → PageIndex integration.

Covers:
  * ``openkb.mhtml`` MIME unpack + CID/Content-Location image resolution + the
    HTML→Markdown structure conversion.
  * ``convert_document`` routing an ``.mhtml``/``.mht`` input to the long-doc
    pipeline (``is_long_doc=True``, ``pageindex_source`` set).
  * ``index_mhtml_document`` feeding the prepared Markdown to PageIndex and
    writing the long-doc wiki artifacts.
  * CLI ``add``/``remove``/``recompile`` recognizing ``mhtml``/``mht`` registry
    types.
"""

from __future__ import annotations

import base64
import email.policy
import json
from email.message import EmailMessage
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from openkb.mhtml import (
    MHTMLPrepareResult,
    html_to_markdown,
    prepare_mhtml_for_pageindex,
    unpack_mhtml,
)

# A 1x1 transparent PNG for image-bearing MIME parts.
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _build_mhtml(
    *,
    html: str,
    cid_images: dict[str, bytes] | None = None,
    location_images: dict[str, bytes] | None = None,
) -> bytes:
    """Build an MHTML archive (multipart/related root = a text/html part).

    ``cid_images`` maps a Content-ID (no angle brackets) → image bytes, attached
    as ``cid:<id>`` references. ``location_images`` maps a Content-Location URL
    → image bytes, attached as ``<img src="<url>">`` references.
    """
    root_cid = "<root-html>"
    related = MIMEMultipart("related", start=root_cid)
    related.attach(MIMEText(html, "html", _charset="utf-8"))

    for cid, data in (cid_images or {}).items():
        img = MIMEImage(data, _subtype="png")
        img.add_header("Content-ID", f"<{cid}>")
        img.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
        related.attach(img)

    for url, data in (location_images or {}).items():
        img = MIMEImage(data, _subtype="png")
        img.add_header("Content-Location", url)
        img.add_header("Content-Disposition", "inline", filename=Path(url).name)
        related.attach(img)

    wrapper = EmailMessage()
    wrapper["From"] = "saved@example.com"
    wrapper["Subject"] = "Saved Page"
    wrapper["MIME-Version"] = "1.0"
    wrapper.make_mixed()
    wrapper.attach(related)
    return wrapper.as_bytes(policy=email.policy.default)


# ---------------------------------------------------------------------------
# openkb.mhtml — HTML→Markdown
# ---------------------------------------------------------------------------


class TestHtmlToMarkdown:
    def test_preserves_heading_hierarchy(self):
        md = html_to_markdown("<h1>Title</h1><h2>Section</h2><h3>Sub</h3><p>Body.</p>")
        assert "# Title" in md
        assert "## Section" in md
        assert "### Sub" in md
        assert "Body." in md

    def test_lists_and_tables(self):
        md = html_to_markdown(
            "<ul><li>a</li><li>b</li></ul>"
            "<table><tr><th>X</th><th>Y</th></tr><tr><td>1</td><td>2</td></tr></table>"
        )
        assert "- a\n- b" in md
        assert "| X | Y |" in md
        assert "| --- | --- |" in md
        assert "| 1 | 2 |" in md

    def test_ignores_head_title_and_style(self):
        md = html_to_markdown(
            "<head><title>Should Not Appear</title>"
            "<style>body{color:red}</style></head>"
            "<body><h1>Real Title</h1><p>Visible.</p></body>"
        )
        assert "Should Not Appear" not in md
        assert "color:red" not in md
        assert "# Real Title" in md
        assert "Visible." in md

    def test_inline_emphasis_and_links(self):
        md = html_to_markdown(
            "<p>Some <strong>bold</strong> and <em>italic</em> "
            '<a href="https://example.com">link</a>.</p>'
        )
        assert "**bold**" in md
        assert "*italic*" in md
        assert "[link](https://example.com)" in md

    def test_promotes_standalone_bold_numbered_titles(self):
        # Web archives (e.g. WeChat articles) mark section headings as
        # <strong>01 Title</strong> instead of <h2>; promoting these to real
        # headings is what keeps PageIndex's tree multi-section rather than
        # a single flattened node.
        md = html_to_markdown(
            "<h1>Article</h1>"
            "<p><strong>01 Intro</strong></p>"
            "<p>Body text.</p>"
            "<p><strong>4.1 Subsection</strong></p>"
            "<p><strong>02 Conclusion</strong></p>"
        )
        assert "## 01 Intro" in md
        assert "### 4.1 Subsection" in md  # dotted → level 3
        assert "## 02 Conclusion" in md

    def test_inline_bold_runs_not_promoted_to_headings(self):
        # Bold runs *inside* a paragraph must NOT become headings — only
        # standalone bolded lines that are pure section numbers.
        md = html_to_markdown(
            "<p>See <strong>01</strong> for context and <strong>3 reasons</strong> why.</p>"
        )
        assert "## 01" not in md
        assert "**01**" in md  # preserved as inline bold


# ---------------------------------------------------------------------------
# openkb.mhtml — unpack_mhtml + image resolution
# ---------------------------------------------------------------------------


class TestUnpackMhtml:
    def test_simple_page(self, tmp_path):
        html = (
            "<html><head><title>Sample Article</title></head><body>"
            "<h1>Sample Article</h1>"
            "<p>An intro paragraph.</p>"
            '<img src="cid:image001" alt="cid">'
            "</body></html>"
        )
        mhtml = tmp_path / "article.mhtml"
        mhtml.write_bytes(_build_mhtml(html=html, cid_images={"image001": _PNG_BYTES}))

        out = tmp_path / "out"
        result = unpack_mhtml(mhtml, out)

        assert isinstance(result, MHTMLPrepareResult)
        assert result.markdown_path.exists()
        assert result.html_path.exists()
        assert result.image_dir.is_dir()
        md = result.markdown_path.read_text(encoding="utf-8")
        assert "# Sample Article" in md
        assert "An intro paragraph." in md
        # image extracted and rewritten to ./images/img001.png
        images = sorted(p.name for p in result.image_dir.iterdir())
        assert images == ["img001.png"]
        assert "./images/img001.png" in md

    def test_cid_image_ref_rewritten(self, tmp_path):
        html = '<html><body><img src="cid:image001"></body></html>'
        mhtml = tmp_path / "x.mhtml"
        mhtml.write_bytes(_build_mhtml(html=html, cid_images={"image001": _PNG_BYTES}))

        result = unpack_mhtml(mhtml, tmp_path / "out")
        md = result.markdown_path.read_text(encoding="utf-8")
        assert "cid:image001" not in md
        assert "./images/img001.png" in md

    def test_content_location_ref_rewritten(self, tmp_path):
        url = "https://cdn.example.com/imgs/photo.png"
        html = f'<html><body><img src="{url}"></body></html>'
        mhtml = tmp_path / "x.mhtml"
        mhtml.write_bytes(_build_mhtml(html=html, location_images={url: _PNG_BYTES}))

        result = unpack_mhtml(mhtml, tmp_path / "out")
        md = result.markdown_path.read_text(encoding="utf-8")
        assert "https://cdn.example.com" not in md
        assert "./images/img001.png" in md

    def test_large_structured_page_preserves_hierarchy(self, tmp_path):
        html = (
            "<html><body>"
            "<h1>Guide</h1>"
            "<h2>Part One</h2>"
            "<p>Intro.</p>"
            '<img src="cid:a">'
            "<table><tr><th>k</th><th>v</th></tr><tr><td>1</td><td>2</td></tr></table>"
            "<h2>Part Two</h2>"
            "<ul><li>x</li><li>y</li></ul>"
            '<img src="cid:b">'
            "</body></html>"
        )
        mhtml = tmp_path / "guide.mhtml"
        mhtml.write_bytes(_build_mhtml(html=html, cid_images={"a": _PNG_BYTES, "b": _PNG_BYTES}))

        result = unpack_mhtml(mhtml, tmp_path / "out")
        md = result.markdown_path.read_text(encoding="utf-8")
        # section order + levels preserved
        assert md.index("# Guide") < md.index("## Part One") < md.index("## Part Two")
        assert "| k | v |" in md
        assert "- x\n- y" in md
        assert sum(1 for _ in result.image_dir.iterdir()) == 2

    def test_no_images_still_indexes_doc(self, tmp_path):
        html = "<html><body><h1>Only Text</h1><p>No images here.</p></body></html>"
        mhtml = tmp_path / "t.mhtml"
        mhtml.write_bytes(_build_mhtml(html=html))
        result = unpack_mhtml(mhtml, tmp_path / "out")
        md = result.markdown_path.read_text(encoding="utf-8")
        assert "# Only Text" in md
        assert not any(result.image_dir.iterdir())

    def test_mht_extension_supported(self, tmp_path):
        # .mht is the same format as .mhtml — is_mhtml must accept it.
        from openkb.mhtml import is_mhtml

        assert is_mhtml(Path("a.mhtml"))
        assert is_mhtml(Path("a.mht"))
        assert not is_mhtml(Path("a.html"))

    def test_prepare_uses_mhtml_assets_root(self, tmp_path, monkeypatch):
        # Images + markdown land under kb_dir/.openkb/mhtml_assets/<doc_name>/,
        # NOT wiki/sources/images (the short-doc tree).
        html = "<html><body><h1>Doc</h1><p>x</p></body></html>"
        mhtml = tmp_path / "src.mhtml"
        mhtml.write_bytes(_build_mhtml(html=html))

        # minimal kb_dir
        kb_dir = tmp_path / "kb"
        (kb_dir / ".openkb").mkdir(parents=True)

        result = prepare_mhtml_for_pageindex(mhtml, kb_dir)
        expected_root = kb_dir / ".openkb" / "mhtml_assets" / "src"
        assert result.markdown_path.parent == expected_root
        assert result.image_dir.parent == expected_root
        assert not (kb_dir / "wiki" / "sources" / "images").exists()


# ---------------------------------------------------------------------------
# openkb.converter — MHTML routing
# ---------------------------------------------------------------------------


class TestConvertDocumentMhtml:
    def _setup_kb(self, tmp_path):
        (tmp_path / "raw").mkdir()
        (tmp_path / "wiki" / "sources" / "images").mkdir(parents=True)
        (tmp_path / "wiki" / "summaries").mkdir(parents=True)
        openkb_dir = tmp_path / ".openkb"
        openkb_dir.mkdir()
        (openkb_dir / "config.yaml").write_text("model: gpt-4o-mini\n")
        (openkb_dir / "hashes.json").write_text(json.dumps({}))
        return tmp_path

    def test_mhtml_routes_to_long_doc_pipeline(self, tmp_path):
        from openkb.converter import convert_document

        kb_dir = self._setup_kb(tmp_path)
        mhtml = tmp_path / "article.mhtml"
        mhtml.write_bytes(
            _build_mhtml(
                html="<html><body><h1>Title</h1><p>Body.</p></body></html>",
                cid_images={"image001": _PNG_BYTES},
            )
        )

        result = convert_document(mhtml, kb_dir)

        assert result.skipped is False
        assert result.is_long_doc is True
        assert result.pageindex_source is not None
        assert result.pageindex_source.suffix == ".md"
        assert result.pageindex_source.exists()
        # raw mhtml was copied
        assert result.raw_path is not None and result.raw_path.exists()
        # never touched the short-doc markdown tree
        assert not (kb_dir / "wiki" / "sources" / f"{result.doc_name}.md").exists()

    def test_mht_extension_also_long_doc(self, tmp_path):
        from openkb.converter import convert_document

        kb_dir = self._setup_kb(tmp_path)
        mhtml = tmp_path / "page.mht"
        mhtml.write_bytes(_build_mhtml(html="<html><body><h1>Page</h1></body></html>"))

        result = convert_document(mhtml, kb_dir)
        assert result.is_long_doc is True
        assert result.pageindex_source is not None


# ---------------------------------------------------------------------------
# openkb.indexer — index_mhtml_document
# ---------------------------------------------------------------------------


class TestIndexMhtmlDocument:
    def _make_fake_collection(self, doc_id: str, structure: list, pages: list):
        col = MagicMock()
        col.add.return_value = doc_id
        col.get_document.return_value = {
            "doc_id": doc_id,
            "doc_name": "article",
            "doc_description": "A saved web page.",
            "structure": structure,
        }
        col.get_page_content.return_value = pages
        return col

    def test_calls_col_add_with_markdown_path(self, kb_dir, tmp_path):
        from openkb.indexer import index_mhtml_document

        # Write a prepared markdown file as convert_document would.
        md_path = kb_dir / ".openkb" / "mhtml_assets" / "article" / "document.md"
        md_path.parent.mkdir(parents=True)
        md_path.write_text("# Article\n\nBody.", encoding="utf-8")

        structure = [{"title": "Article", "start_index": 1, "end_index": 2, "nodes": []}]
        pages = [{"page": 1, "content": "# Article\n\nBody."}]
        col = self._make_fake_collection("doc-1", structure, pages)
        client = MagicMock()
        client.collection.return_value = col

        with patch("openkb.indexer.PageIndexClient", return_value=client):
            result = index_mhtml_document(md_path, kb_dir, doc_name="article")

        # PageIndex was handed the MARKDOWN path, not an HTML/raw path.
        col.add.assert_called_once_with(str(md_path))
        assert result.doc_id == "doc-1"
        # long-doc artifacts written (keyed by the explicit doc_name, not the
        # markdown file's stem "document")
        assert (kb_dir / "wiki" / "summaries" / "article.md").exists()
        json_file = kb_dir / "wiki" / "sources" / "article.json"
        assert json_file.exists()
        assert "Body." in json_file.read_text(encoding="utf-8")

    def test_windows_page_fetch_over_1000_cap(self, kb_dir, tmp_path):
        """A markdown doc whose node indices exceed 1000 must fetch in windows
        (parse_pages rejects >1000-page ranges), concatenating every page."""
        from openkb.indexer import index_mhtml_document

        md_path = kb_dir / ".openkb" / "mhtml_assets" / "big" / "document.md"
        md_path.parent.mkdir(parents=True)
        md_path.write_text("# Big\n\n.", encoding="utf-8")

        # Nodes at line indices 1 and 1500 → forces a 2-window fetch.
        structure = [
            {"title": "A", "start_index": 1, "end_index": 500, "nodes": []},
            {"title": "B", "start_index": 1500, "end_index": 1600, "nodes": []},
        ]

        def fake_get_page_content(doc_id, rng):
            start, end = (int(x) for x in rng.split("-"))
            assert end - start + 1 <= 1000  # never an oversized range
            return [
                {"page": p, "content": f"p{p}"} for p in range(start, end + 1) if p in (1, 1500)
            ]

        col = MagicMock()
        col.add.return_value = "doc-big"
        col.get_document.return_value = {
            "doc_name": "big",
            "doc_description": "",
            "structure": structure,
        }
        col.get_page_content.side_effect = fake_get_page_content
        client = MagicMock()
        client.collection.return_value = col

        with patch("openkb.indexer.PageIndexClient", return_value=client):
            result = index_mhtml_document(md_path, kb_dir, doc_name="big")

        ranges = [c.args[1] for c in col.get_page_content.call_args_list]
        # windowed around the 1000-page cap; max_index=1600 → windows end at 1000 then 1600
        assert all(int(r.split("-")[1]) - int(r.split("-")[0]) + 1 <= 1000 for r in ranges)
        json_file = kb_dir / "wiki" / "sources" / "big.json"
        data = json.loads(json_file.read_text(encoding="utf-8"))
        assert {p["page"] for p in data} == {1, 1500}
        assert result.doc_id == "doc-big"

    def test_deletes_pageindex_doc_when_post_add_fails(self, kb_dir, tmp_path):
        """Mirrors index_long_document's invariant: if a step after col.add()
        raises, the freshly-added doc must be deleted so its blob can't leak."""
        from openkb.indexer import index_mhtml_document

        md_path = kb_dir / ".openkb" / "mhtml_assets" / "x" / "document.md"
        md_path.parent.mkdir(parents=True)
        md_path.write_text("# x\n", encoding="utf-8")

        col = MagicMock()
        col.add.return_value = "doc-x"
        col.get_document.side_effect = RuntimeError("get_document blew up")
        client = MagicMock()
        client.collection.return_value = col

        with patch("openkb.indexer.PageIndexClient", return_value=client):
            with pytest.raises(RuntimeError, match="get_document blew up"):
                index_mhtml_document(md_path, kb_dir)

        col.delete_document.assert_called_once_with("doc-x")


# ---------------------------------------------------------------------------
# CLI add — MHTML end-to-end (mocked PageIndex + compile)
# ---------------------------------------------------------------------------


def _setup_kb(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki" / "sources" / "images").mkdir(parents=True)
    (tmp_path / "wiki" / "summaries").mkdir(parents=True)
    (tmp_path / "wiki" / "concepts").mkdir(parents=True)
    openkb_dir = tmp_path / ".openkb"
    openkb_dir.mkdir()
    (openkb_dir / "config.yaml").write_text("model: gpt-4o-mini\n")
    (openkb_dir / "hashes.json").write_text(json.dumps({}))
    return tmp_path


class TestCliAddMhtml:
    def test_add_mhtml_dispatches_long_doc_pipeline(self, tmp_path):
        """`openkb add article.mhtml` must route through index_mhtml_document +
        compile_long_doc — NOT compile_short_doc."""
        from openkb.cli import add_single_file
        from openkb.indexer import IndexResult

        kb_dir = _setup_kb(tmp_path)
        mhtml = tmp_path / "article.mhtml"
        mhtml.write_bytes(
            _build_mhtml(
                html="<html><body><h1>Title</h1><p>Body.</p></body></html>",
                cid_images={"image001": _PNG_BYTES},
            )
        )

        with (
            patch(
                "openkb.indexer.index_mhtml_document",
                return_value=IndexResult(doc_id="mhtml-1", description="d", tree={"structure": []}),
            ) as mock_index,
            patch("openkb.indexer.index_long_document") as mock_long_index,
            patch(
                "openkb.agent.compiler.compile_long_doc", new_callable=AsyncMock
            ) as mock_compile_long,
            patch(
                "openkb.agent.compiler.compile_short_doc", new_callable=AsyncMock
            ) as mock_compile_short,
            patch("openkb.cli.time.sleep"),
            patch("openkb.cli._setup_llm_key"),
        ):
            outcome = add_single_file(mhtml, kb_dir)

        assert outcome == "added"
        mock_index.assert_called_once()
        # the markdown path was passed, not the raw mhtml
        passed_md = mock_index.call_args.args[0]
        assert passed_md.suffix == ".md" and passed_md.exists()
        mock_long_index.assert_not_called()
        mock_compile_long.assert_called_once()
        mock_compile_short.assert_not_called()

        # registry recorded an mhtml long-doc entry with a doc_id + web_archive origin
        from openkb.state import HashRegistry

        meta = next(iter(HashRegistry(kb_dir / ".openkb" / "hashes.json").all_entries().values()))
        assert meta["type"] == "mhtml"
        assert meta["source_format"] == "web_archive"
        assert meta["doc_id"] == "mhtml-1"
        assert meta["doc_name"] == "article"

    def test_add_mhtml_failure_rolls_back_blob(self, tmp_path):
        """A failed MHTML add must roll back the PageIndex blob it created
        under .openkb/files, mirroring the long-PDF rollback invariant."""
        from openkb.cli import add_single_file
        from openkb.indexer import IndexResult

        kb_dir = _setup_kb(tmp_path)
        files = kb_dir / ".openkb" / "files" / "default"
        files.mkdir(parents=True)
        new_id = "33333333-3333-3333-3333-333333333333"

        def fake_index(md_path, kb_dir_arg, doc_name=None):
            (files / f"{new_id}.md").write_bytes(b"new-blob")
            return IndexResult(doc_id=new_id, description="", tree={"structure": []})

        mhtml = tmp_path / "fail.mhtml"
        mhtml.write_bytes(_build_mhtml(html="<html><body><h1>Fail</h1></body></html>"))

        with (
            patch("openkb.indexer.index_mhtml_document", side_effect=fake_index),
            patch("openkb.agent.compiler.compile_long_doc", side_effect=RuntimeError("boom")),
            patch("openkb.cli.time.sleep"),
            patch("openkb.cli._setup_llm_key"),
        ):
            outcome = add_single_file(mhtml, kb_dir)

        assert outcome == "failed"
        assert not (files / f"{new_id}.md").exists()


# ---------------------------------------------------------------------------
# CLI remove / recompile — mhtml type recognition
# ---------------------------------------------------------------------------


class TestCliRemoveRecompileMhtml:
    def test_remove_mhtml_invokes_pageindex_delete(self, tmp_path):
        from openkb.cli import cli
        from openkb.state import HashRegistry

        kb_dir = _setup_kb(tmp_path)
        # Seed an mhtml registry entry + a stub summary so remove() has a file
        # to delete and a doc_id to hand to PageIndex cleanup.
        (kb_dir / "wiki" / "summaries").mkdir(parents=True, exist_ok=True)
        (kb_dir / "wiki" / "summaries" / "article.md").write_text(
            "---\ntype: Summary\n---\n# Article\n", encoding="utf-8"
        )
        reg = HashRegistry(kb_dir / ".openkb" / "hashes.json")
        reg.add(
            "deadbeef",
            {
                "name": "article.mhtml",
                "doc_name": "article",
                "type": "mhtml",
                "source_format": "web_archive",
                "doc_id": "mhtml-doc-1",
                "raw_path": "raw/article.mhtml",
            },
        )
        # raw file present (remove deletes it unless --keep-raw)
        (kb_dir / "raw").mkdir(exist_ok=True)
        (kb_dir / "raw" / "article.mhtml").write_bytes(b"mhtml bytes")
        # pageindex.db present so the cleanup branch fires
        (kb_dir / ".openkb" / "pageindex.db").write_bytes(b"stub")

        runner = CliRunner()
        with (
            patch("openkb.cli._find_kb_dir", return_value=kb_dir),
            patch("openkb.cli._cleanup_pageindex", return_value=(True, "deleted")) as mock_cleanup,
            patch("openkb.cli._setup_llm_key"),
        ):
            result = runner.invoke(cli, ["remove", "article.mhtml", "--yes"])

        assert result.exit_code == 0, result.output
        mock_cleanup.assert_called_once()
        # doc_id flowed through to cleanup
        assert mock_cleanup.call_args.args[3] == "mhtml-doc-1"

    def test_recompile_mhtml_dispatches_compile_long_doc(self, tmp_path):
        """A registered mhtml doc must recompile via compile_long_doc (long-doc
        layout + doc_id), not be misrouted to compile_short_doc."""
        from openkb.cli import cli
        from openkb.state import HashRegistry

        kb_dir = _setup_kb(tmp_path)
        (kb_dir / "wiki" / "summaries").mkdir(parents=True, exist_ok=True)
        summary = kb_dir / "wiki" / "summaries" / "page.md"
        summary.write_text("---\ntype: Summary\n---\n# Page\n", encoding="utf-8")
        reg = HashRegistry(kb_dir / ".openkb" / "hashes.json")
        reg.add(
            "h1",
            {
                "name": "page.mhtml",
                "doc_name": "page",
                "type": "mhtml",
                "doc_id": "mhtml-doc-9",
            },
        )

        runner = CliRunner()
        with (
            patch("openkb.cli._find_kb_dir", return_value=kb_dir),
            patch("openkb.agent.compiler.compile_long_doc", new_callable=AsyncMock) as mock_long,
            patch("openkb.agent.compiler.compile_short_doc", new_callable=AsyncMock) as mock_short,
            patch("openkb.cli._setup_llm_key"),
        ):
            result = runner.invoke(cli, ["recompile", "page", "--yes"])

        assert result.exit_code == 0, result.output
        mock_long.assert_called_once()
        # doc_id from the registry flowed into compile_long_doc
        assert mock_long.call_args.args[2] == "mhtml-doc-9"
        mock_short.assert_not_called()


# ---------------------------------------------------------------------------
# Type-set unit tests (mirrors test_recompile's cloud coverage)
# ---------------------------------------------------------------------------


def test_type_sets_and_display_type_cover_mhtml():
    from openkb.cli import _LONG_DOC_TYPES, _TYPE_DISPLAY_MAP, _display_type, _is_long_doc

    assert "mhtml" in _LONG_DOC_TYPES
    assert "mht" in _LONG_DOC_TYPES
    assert _is_long_doc({"type": "mhtml"}) is True
    assert _is_long_doc({"type": "mht"}) is True
    assert _display_type("mhtml") == "pageindex"
    assert _display_type("mht") == "pageindex"
    assert _TYPE_DISPLAY_MAP["mhtml"] == "pageindex"
