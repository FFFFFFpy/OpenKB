"""PageIndex indexer for long documents."""

from __future__ import annotations

import json as json_mod
import logging
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pageindex import IndexConfig, PageIndexClient

from openkb.config import load_config
from openkb.tree_renderer import render_summary_md

logger = logging.getLogger(__name__)


@dataclass
class IndexResult:
    """Result of indexing a long document via PageIndex."""

    doc_id: str
    description: str
    tree: dict


@dataclass
class CloudImportResult:
    """Result of importing an existing PageIndex Cloud document."""

    doc_id: str
    doc_name: str  # collision-resistant wiki slug
    name: str  # cloud display name (original filename in the cloud)
    description: str


@dataclass
class CloudImportData:
    """A fetched cloud doc + its resolved wiki name, before any KB write.

    Returned by :func:`prepare_cloud_import` so the caller can snapshot this
    doc's specific paths (O(1)) before :func:`_write_long_doc_artifacts` writes
    them — instead of copying the whole summaries/sources trees on every import.
    """

    doc_id: str
    doc_name: str  # collision-resistant wiki slug (resolved, not yet written)
    cloud_name: str  # cloud display name (original filename in the cloud)
    description: str
    tree: dict
    all_pages: list


def _cloud_display_stem(cloud_name: str, fallback: str) -> str:
    """Return a platform-independent stem for a PageIndex Cloud display name."""
    normalized = cloud_name.replace("\\", "/").rstrip("/")
    leaf = normalized.rsplit("/", 1)[-1] if normalized else ""
    return PurePosixPath(leaf).stem or fallback


def _normalize_page_content(raw_pages: Any) -> list[dict[str, Any]]:
    """Normalize PageIndex/local PDF page content into OpenKB's JSON shape."""
    if not isinstance(raw_pages, list):
        return []

    pages: list[dict[str, Any]] = []
    for index, item in enumerate(raw_pages, start=1):
        if isinstance(item, str):
            content = item.strip()
            if content:
                pages.append({"page": index, "content": content, "images": []})
            continue

        if not isinstance(item, dict):
            continue

        raw_page = item.get("page", item.get("page_number", item.get("page_num", index)))
        try:
            page_number = int(raw_page)
        except (TypeError, ValueError):
            page_number = index
        if page_number < 1:
            page_number = index

        content = item.get("content", item.get("markdown", item.get("text", "")))
        if content is None:
            content = ""
        content = str(content).strip()

        images = item.get("images", [])
        if not isinstance(images, list):
            images = []
        normalized_images = [
            image
            for image in images
            if isinstance(image, dict) and isinstance(image.get("path"), str)
        ]

        if content or normalized_images:
            pages.append(
                {
                    "page": page_number,
                    "content": content,
                    "images": normalized_images,
                }
            )

    return pages


def _get_pdf_page_count(pdf_path: Path) -> int:
    from openkb.converter import get_pdf_page_count

    return get_pdf_page_count(pdf_path)


def _convert_pdf_to_pages(pdf_path: Path, doc_name: str, images_dir: Path) -> list[dict[str, Any]]:
    from openkb.images import convert_pdf_to_pages

    return convert_pdf_to_pages(pdf_path, doc_name, images_dir)


def _write_long_doc_artifacts(
    tree: dict,
    pages: list[dict[str, Any]],
    doc_name: str,
    doc_id: str,
    kb_dir: Path,
    description: str = "",
) -> Path:
    """Write ``wiki/sources/<doc_name>.json`` + ``wiki/summaries/<doc_name>.md``.

    Returns the summary path. Shared by :func:`index_long_document` (local)
    and :func:`import_cloud_document` (cloud) so both produce identical
    artifacts. Page images, when present, are written separately by the
    caller's page extractor — this helper only persists page text + summary.
    """
    sources_dir = kb_dir / "wiki" / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    (sources_dir / f"{doc_name}.json").write_text(
        json_mod.dumps(pages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summaries_dir = kb_dir / "wiki" / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summaries_dir / f"{doc_name}.md"
    summary_path.write_text(
        render_summary_md(tree, doc_name, doc_id, description=description), encoding="utf-8"
    )
    return summary_path


def index_long_document(pdf_path: Path, kb_dir: Path, doc_name: str | None = None) -> IndexResult:
    """Index a long PDF document using PageIndex and write wiki pages.

    ``doc_name`` is the collision-resistant wiki name used for all written
    artifacts; defaults to the PDF's stem for backward compatibility.
    """
    source_name = doc_name or pdf_path.stem
    openkb_dir = kb_dir / ".openkb"
    config = load_config(openkb_dir / "config.yaml")

    model: str = config.get("model", "gpt-5.4")
    pageindex_api_key = os.environ.get("PAGEINDEX_API_KEY", "")

    index_config = IndexConfig(
        if_add_node_text=True,
        if_add_node_summary=True,
        if_add_doc_description=True,
    )

    client = PageIndexClient(
        api_key=pageindex_api_key or None,
        model=model,
        storage_path=str(openkb_dir),
        index_config=index_config,
    )
    col = client.collection()

    # Add PDF (retry up to 3 times — PageIndex TOC accuracy is stochastic)
    max_retries = 3
    doc_id = None
    for attempt in range(1, max_retries + 1):
        try:
            doc_id = col.add(str(pdf_path))
            logger.info(
                "PageIndex added %s → doc_id=%s (attempt %d)", pdf_path.name, doc_id, attempt
            )
            break
        except Exception as exc:
            logger.warning(
                "PageIndex attempt %d/%d failed for %s: %s",
                attempt,
                max_retries,
                pdf_path.name,
                exc,
            )
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to index {pdf_path.name} after {max_retries} attempts: {exc}"
                ) from exc

    # The PageIndex blob for doc_id is now durably on disk. The add mutation no
    # longer eagerly snapshots .openkb/files — it registers the new blob via
    # snapshot.track_new() only on a successful return — so if any step below
    # fails, delete the document we just added. Otherwise the blob leaks as an
    # orphan that pageindex.db (rolled back by the snapshot) no longer refs and
    # no reaper reclaims.
    try:
        # Fetch complete document (metadata + structure + text)
        doc = col.get_document(doc_id, include_text=True)
        indexed_doc_name: str = doc.get("doc_name", pdf_path.stem)
        description: str = doc.get("doc_description", "")
        structure: list = doc.get("structure", [])

        # Debug: print doc keys and page_count to diagnose get_page_content range
        logger.info("Doc keys: %s", list(doc.keys()))
        logger.info("page_count from doc: %s", doc.get("page_count", "NOT PRESENT"))

        tree = {
            "doc_name": indexed_doc_name,
            "doc_description": description,
            "structure": structure,
        }

        # Write wiki/sources/ — per-page content
        sources_dir = kb_dir / "wiki" / "sources"
        sources_dir.mkdir(parents=True, exist_ok=True)
        images_dir = sources_dir / "images" / source_name

        all_pages: list[dict[str, Any]] = []
        if pageindex_api_key:
            # Cloud mode: fetch OCR'd markdown from PageIndex. get_page_content
            # requires a page range, so pass "1-N".
            page_count = _get_pdf_page_count(pdf_path)
            try:
                all_pages = _normalize_page_content(col.get_page_content(doc_id, f"1-{page_count}"))
            except Exception as exc:
                logger.warning("Cloud get_page_content failed for %s: %s", pdf_path.name, exc)

        if not all_pages:
            if pageindex_api_key:
                logger.warning(
                    "Cloud returned no pages for %s; falling back to local pymupdf", pdf_path.name
                )
            all_pages = _normalize_page_content(
                _convert_pdf_to_pages(pdf_path, source_name, images_dir)
            )

        if not all_pages:
            raise RuntimeError(f"No page content extracted for {pdf_path.name}")

        _write_long_doc_artifacts(
            tree, all_pages, source_name, doc_id, kb_dir, description=description
        )
        return IndexResult(doc_id=doc_id, description=description, tree=tree)
    except BaseException:
        # Best-effort: remove the blob this add created. A failure here (e.g. a
        # second interrupt) only means the blob may stay orphaned — the original
        # error still propagates so the caller (mutation coordinator) rolls back
        # everything else it snapshotted.
        try:
            col.delete_document(doc_id)
        except Exception:
            logger.warning(
                "PageIndex cleanup of %s failed after error; blob may be orphaned", doc_id
            )
        raise


def _max_page_index(structure: list) -> int:
    """Return the largest ``start_index``/``end_index`` in a PageIndex tree.

    Markdown-input trees key their nodes by source line number, which can far
    exceed a doc's section count. ``get_page_content`` rejects ranges over 1000
    pages (``parse_pages``), so windowing the fetch against this upper bound
    keeps it from truncating without requesting an oversized range.
    """
    best = 0
    for node in structure or []:
        if not isinstance(node, dict):
            continue
        for key in ("start_index", "end_index"):
            try:
                best = max(best, int(node.get(key)))
            except (TypeError, ValueError):
                pass
        best = max(best, _max_page_index(node.get("nodes", [])))
    return best


def index_mhtml_document(
    markdown_path: Path, kb_dir: Path, doc_name: str | None = None
) -> IndexResult:
    """Index an MHTML archive's prepared Markdown via PageIndex.

    *markdown_path* is the PageIndex-ready Markdown produced by
    :func:`openkb.mhtml.prepare_mhtml_for_pageindex`. PageIndex's
    :class:`MarkdownParser` builds the structure tree from its headings —
    preserving the page's section hierarchy that the short-doc path would
    flatten. Per-page content is read back from PageIndex's cached markdown
    pages (windowed around the 1000-page ``parse_pages`` cap), so unlike the
    PDF path there is no cloud-OCR fallback or per-page image extractor.
    """
    source_name = doc_name or markdown_path.stem
    openkb_dir = kb_dir / ".openkb"
    config = load_config(openkb_dir / "config.yaml")

    model: str = config.get("model", "gpt-5.4")
    pageindex_api_key = os.environ.get("PAGEINDEX_API_KEY", "")

    index_config = IndexConfig(
        if_add_node_text=True,
        if_add_node_summary=True,
        if_add_doc_description=True,
    )
    client = PageIndexClient(
        api_key=pageindex_api_key or None,
        model=model,
        storage_path=str(openkb_dir),
        index_config=index_config,
    )
    col = client.collection()

    max_retries = 3
    doc_id = None
    for attempt in range(1, max_retries + 1):
        try:
            doc_id = col.add(str(markdown_path))
            logger.info(
                "PageIndex added %s → doc_id=%s (attempt %d)",
                markdown_path.name,
                doc_id,
                attempt,
            )
            break
        except Exception as exc:
            logger.warning(
                "PageIndex attempt %d/%d failed for %s: %s",
                attempt,
                max_retries,
                markdown_path.name,
                exc,
            )
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to index {markdown_path.name} after {max_retries} attempts: {exc}"
                ) from exc

    try:
        doc = col.get_document(doc_id, include_text=True)
        indexed_doc_name: str = doc.get("doc_name", markdown_path.stem)
        description: str = doc.get("doc_description", "")
        structure: list = doc.get("structure", [])

        logger.info("Doc keys: %s", list(doc.keys()))
        logger.info("page_count from doc: %s", doc.get("page_count", "NOT PRESENT"))

        tree = {
            "doc_name": indexed_doc_name,
            "doc_description": description,
            "structure": structure,
        }

        # Markdown pages are cached by MarkdownParser; fetch them windowed
        # around the 1000-page parse_pages cap. Node indices are line numbers,
        # so the upper bound comes from the tree's own start/end_index values.
        # Unlike the cloud PDF path, markdown node indices are SPARSE (a section
        # at line 1500 leaves lines 2-1499 without a node), so a short window is
        # NOT a stop signal — we drive the loop purely by max_index.
        max_index = _max_page_index(structure) or 1
        all_pages: list[dict[str, Any]] = []
        start = 1
        while start <= max_index:
            end = min(start + _CLOUD_PAGE_WINDOW - 1, max_index)
            # A window read must not silently truncate the document: if the
            # first few windows succeed and a later one fails, breaking here
            # would write partial page content and return success — the user
            # gets an index that's missing sections with no signal. Retry the
            # window a couple of times (PageIndex's parse can be transient),
            # then raise so the caller rolls back the whole add.
            window: list[dict[str, Any]] = []
            last_exc: Exception | None = None
            for attempt in range(1, _WINDOW_MAX_RETRIES + 1):
                try:
                    window = _normalize_page_content(col.get_page_content(doc_id, f"{start}-{end}"))
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        "MHTML get_page_content failed for %s (%d-%d), attempt %d/%d: %s",
                        markdown_path.name,
                        start,
                        end,
                        attempt,
                        _WINDOW_MAX_RETRIES,
                        exc,
                    )
            if last_exc is not None:
                raise RuntimeError(
                    f"PageIndex page read failed for {markdown_path.name} "
                    f"(pages {start}-{end}) after {_WINDOW_MAX_RETRIES} attempts: {last_exc}"
                ) from last_exc
            all_pages.extend(window)
            start = end + 1

        if not all_pages:
            raise RuntimeError(f"No page content extracted for {markdown_path.name}")

        _write_long_doc_artifacts(
            tree, all_pages, source_name, doc_id, kb_dir, description=description
        )
        return IndexResult(doc_id=doc_id, description=description, tree=tree)
    except BaseException:
        try:
            col.delete_document(doc_id)
        except Exception:
            logger.warning(
                "PageIndex cleanup of %s failed after error; blob may be orphaned", doc_id
            )
        raise


# PageIndex's get_page_content rejects a single page range covering more than
# this many pages (``parse_pages`` raises "Page range too large (max 1000)"),
# so cloud page fetches are windowed in chunks of this size.
_CLOUD_PAGE_WINDOW = 1000
# Safety bound on the windowed fetch (in pages) in case a backend never returns
# a short window — caps the loop at _CLOUD_PAGE_MAX / _CLOUD_PAGE_WINDOW calls.
_CLOUD_PAGE_MAX = 1_000_000
# Retries for a single MHTML page window before failing the whole index. A
# failing window must NOT silently truncate the document — exhaust these, then
# raise so the caller rolls back the add instead of writing partial content.
_WINDOW_MAX_RETRIES = 3


def _fetch_cloud_pages(col, doc_id: str) -> list[dict[str, Any]]:
    """Fetch all OCR pages of a cloud doc, windowing around the 1000-page cap.

    ``get_page_content`` returns the whole document and uses its ``pages`` arg
    only as a client-side filter that ``parse_pages`` caps at 1000 pages — so a
    single ``"1-<N>"`` request fails for any doc over 1000 pages. Request fixed
    ``1000``-page windows and stop as soon as a window comes back SHORT (fewer
    than a full window): PageIndex page numbers are sequential, so a short window
    means we've passed the last page. This is what makes the common (≤1000-page)
    doc a single request, while still fetching every page of a larger one — and,
    unlike bounding the loop by the tree's max page index, it never truncates a
    doc whose tree under-reports its page count (a real case: a paper whose tree
    stops a couple pages short of the references). A wide safety bound guards
    against a backend that never narrows the window.
    """
    pages: list[dict[str, Any]] = []
    start = 1
    while start <= _CLOUD_PAGE_MAX:
        window = _normalize_page_content(
            col.get_page_content(doc_id, f"{start}-{start + _CLOUD_PAGE_WINDOW - 1}")
        )
        pages.extend(window)
        if len(window) < _CLOUD_PAGE_WINDOW:
            break
        start += _CLOUD_PAGE_WINDOW
    return pages


def prepare_cloud_import(doc_id: str, kb_dir: Path, path_key: str) -> CloudImportData:
    """Fetch a PageIndex Cloud doc and resolve its wiki name WITHOUT writing.

    Cloud fetch + collision-resistant name resolution only — no KB mutation —
    so the caller knows ``doc_name`` before writing and can snapshot just this
    doc's paths instead of copying the whole summaries/sources trees. Name
    resolution reads the registry but does not mutate it.
    """
    from openkb.converter import resolve_doc_name_from_key
    from openkb.state import HashRegistry

    pageindex_api_key = os.environ.get("PAGEINDEX_API_KEY", "")
    if not pageindex_api_key:
        raise RuntimeError(
            "Importing from PageIndex Cloud requires the PAGEINDEX_API_KEY environment variable."
        )

    client = PageIndexClient(api_key=pageindex_api_key)
    col = client.collection()

    doc = col.get_document(doc_id, include_text=True)
    cloud_name: str = doc.get("doc_name") or doc_id
    description: str = doc.get("doc_description", "")
    structure: list = doc.get("structure", [])

    registry = HashRegistry(kb_dir / ".openkb" / "hashes.json")
    stem = _cloud_display_stem(cloud_name, doc_id)
    doc_name = resolve_doc_name_from_key(stem, path_key, registry)

    tree = {
        "doc_name": cloud_name,
        "doc_description": description,
        "structure": structure,
    }

    all_pages = _fetch_cloud_pages(col, doc_id)
    if not all_pages:
        raise RuntimeError(f"No page content returned from PageIndex Cloud for doc_id={doc_id}")

    return CloudImportData(
        doc_id=doc_id,
        doc_name=doc_name,
        cloud_name=cloud_name,
        description=description,
        tree=tree,
        all_pages=all_pages,
    )


def import_cloud_document(doc_id: str, kb_dir: Path, path_key: str) -> CloudImportResult:
    """Import an already-indexed PageIndex Cloud document by ``doc_id``.

    Fetches structure + OCR'd page content from the cloud (no local PDF) and
    writes the same wiki artifacts as :func:`index_long_document`. Requires
    ``PAGEINDEX_API_KEY``. ``path_key`` is the synthetic identity key
    (``pageindex-cloud:<doc_id>``) used to resolve a collision-resistant
    wiki name.

    Writes immediately. Callers that need to snapshot before writing (e.g. the
    crash-safe CLI path) should call :func:`prepare_cloud_import` then
    :func:`_write_long_doc_artifacts`, so the snapshot can cover only this
    doc's paths.
    """
    cloud = prepare_cloud_import(doc_id, kb_dir, path_key)
    _write_long_doc_artifacts(
        cloud.tree,
        cloud.all_pages,
        cloud.doc_name,
        cloud.doc_id,
        kb_dir,
        description=cloud.description,
    )
    return CloudImportResult(
        doc_id=cloud.doc_id,
        doc_name=cloud.doc_name,
        name=cloud.cloud_name,
        description=cloud.description,
    )
