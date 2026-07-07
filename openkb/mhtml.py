"""MHTML → PageIndex adapter.

Unpacks a browser-saved ``.mhtml``/``.mht`` web archive into a single
structural Markdown file plus a sibling ``images/`` directory, the shape
PageIndex's :class:`MarkdownParser` consumes. Heading hierarchy, tables, code
blocks, links, and inline images are preserved so the long-doc pipeline can
build a real structure tree — this is *not* the short-doc markdown path.

Standard library only (``email.parser`` for MIME, ``html.parser`` for the
HTML walk): no new dependencies, per the repo's supply-chain invariant.
"""

from __future__ import annotations

import email.policy
import logging
import mimetypes
import re
import unicodedata
from dataclasses import dataclass
from email.message import Message
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

logger = logging.getLogger(__name__)

_MHTML_SUFFIXES = {".mhtml", ".mht"}


@dataclass
class MHTMLPrepareResult:
    """Outcome of preparing an MHTML archive for PageIndex.

    ``markdown_path`` is the PageIndex input — a structural Markdown rendering
    of the archive's HTML body with images rewritten to ``./images/...``.
    ``image_dir`` is the sibling images directory the markdown references.
    """

    html_path: Path
    markdown_path: Path
    image_dir: Path
    doc_name: str


# ---------------------------------------------------------------------------
# MIME unpacking
# ---------------------------------------------------------------------------


def _strip_cid(value: str) -> str:
    """``<image001>`` → ``image001``; bare id returned unchanged."""
    v = value.strip()
    if v.startswith("<") and v.endswith(">"):
        return v[1:-1]
    return v


def _ext_for_content_type(content_type: str) -> str:
    """Return a dotted extension (``.png``) for a MIME ``content_type``.

    Falls back to ``.bin`` when the type is unknown or missing the subtype.
    """
    if not content_type:
        return ".bin"
    ext = mimetypes.guess_extension(content_type.split(";", 1)[0].strip().lower())
    return ext or ".bin"


def _iter_parts(msg: Message):
    """Yield every leaf MIME part under ``msg`` (depth-first, non-multipart)."""
    if msg.is_multipart():
        payload = msg.get_payload()
        if isinstance(payload, list):
            for part in payload:
                if isinstance(part, Message):
                    yield from _iter_parts(part)
    else:
        yield msg


def _html_part_and_parts(msg: Message) -> tuple[Message | None, list[Message]]:
    """Return ``(html_part, all_leaf_parts)`` for an MHTML message.

    The HTML part is chosen by, in order: the ``multipart/related`` ``start``
    parameter (a Content-ID), then the first ``text/html`` leaf, then the first
    ``text/plain`` leaf as a last resort. ``all_leaf_parts`` is returned so the
    caller can scan for image resources without re-walking the tree.
    """
    parts = list(_iter_parts(msg))

    # 1. multipart/related start=CID
    start_cid = None
    if msg.get_content_type() == "multipart/related":
        start = msg.get_param("start")
        if isinstance(start, str):
            start_cid = _strip_cid(start)

    for part in parts:
        if start_cid and _strip_cid(part.get("Content-ID", "")) == start_cid:
            return part, parts

    # 2. first text/html
    for part in parts:
        if part.get_content_type() == "text/html":
            return part, parts

    # 3. text/plain fallback
    for part in parts:
        if part.get_content_type() == "text/plain":
            return part, parts

    return None, parts


def _normalize_url(url: str) -> str:
    """Strip the query string and fragment from *url* for matching.

    Browser-saved MHTML commonly stores a part's ``Content-Location`` with
    cache-busting query params (``?wx_fmt=png&...``) and a fragment
    (``#imgIndex=3``) that the HTML ``<img src>`` omits — so the two never
    compare equal as raw strings. Normalizing to scheme+host+path (decoded)
    makes them line up.
    """
    decoded = unquote(url)
    s = urlsplit(decoded)
    if s.scheme:
        return f"{s.scheme}://{s.netloc}{s.path}"
    return s.path


def _image_resource_keys(part: Message) -> list[str]:
    """Return every lookup key an HTML ``<img src>`` might use to refer to *part*.

    A part can be referenced by its Content-ID (``cid:image001``), its full
    Content-Location URL, the URL minus its query/fragment, or — as a last
    resort — its basename. We return all of these so the resolver can match
    whichever spelling the HTML happens to use. Basenames frequently collide
    across images (e.g. a CDN serving every image from ``.../640``), so the
    basename is a FALLBACK only; the normalized URL is the primary key.
    """
    ctype = part.get_content_type()
    if not (ctype.startswith("image/") or ctype == "application/octet-stream"):
        return []

    keys: list[str] = []
    cid = part.get("Content-ID")
    if cid:
        keys.append(_strip_cid(cid))

    loc = part.get("Content-Location")
    if loc:
        keys.append(loc)
        keys.append(_normalize_url(loc))
        base = Path(_normalize_url(loc)).name
        if base:
            keys.append(base)

    # Dedup while preserving order (primary keys first).
    seen: set[str] = set()
    unique: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


def _payload_bytes(part: Message) -> bytes:
    """Decode a MIME part's payload to raw bytes (handles base64/quoted-printable)."""
    payload = part.get_payload(decode=True)
    if payload is None:
        # get_payload(decode=True) returns None for non-multipart text payloads
        # whose CTE it couldn't reverse; fall back to the raw string body.
        raw = part.get_payload()
        if isinstance(raw, str):
            return raw.encode(part.get_content_charset() or "utf-8", errors="replace")
        return b""
    assert isinstance(payload, bytes)
    return payload


# ---------------------------------------------------------------------------
# HTML → Markdown
# ---------------------------------------------------------------------------


class _HtmlToMarkdown(HTMLParser):
    """Stream an HTML document into structural Markdown.

    Preserves heading levels (``<h1>``→``#`` … ``<h6>``→``######``),
    paragraphs, ordered/unordered lists (nested), tables (pipe tables),
    blockquotes, preformatted blocks, inline links and images, and
    ``<strong>``/``<em>``. Inline styling is kept minimal — the goal is a
    structure tree for PageIndex, not pixel-perfect prose.
    """

    _BLOCK_TAGS = {"p", "div", "section", "article", "header", "footer", "main"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._out: list[str] = []
        self._list_stack: list[str] = []  # "ul"/"ol" entries
        self._list_counters: list[int] = []
        self._pre_depth = 0
        self._table_rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] = []
        self._in_cell = False
        self._skip = 0  # depth of <script>/<style>/<head>/<title>/...> to ignore

    # --- block flow helpers ---
    def _blank(self) -> None:
        if self._out and self._out[-1] != "\n":
            self._out.append("\n")

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        # ``meta``/``link`` are void elements with no end tag, so they must NOT
        # enter the skip counter — only content-bearing containers (script,
        # style, noscript, head, title) do, otherwise their missing </tag>
        # leaves _skip stuck and skips the entire document body.
        if t in ("script", "style", "noscript", "head", "title"):
            self._skip += 1
            return
        if self._skip:
            return
        attrs_d = dict(attrs)
        if t in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._blank()
            level = int(t[1])
            self._out.append("#" * level + " ")
        elif t == "p":
            self._blank()
        elif t in self._BLOCK_TAGS:
            self._blank()
        elif t == "br":
            self._out.append("\n")
        elif t == "hr":
            self._blank()
            self._out.append("---\n")
        elif t == "ul":
            self._list_stack.append("ul")
            self._list_counters.append(0)
        elif t == "ol":
            self._list_stack.append("ol")
            self._list_counters.append(0)
        elif t == "li":
            indent = "  " * (len(self._list_stack) - 1)
            if self._list_stack and self._list_stack[-1] == "ol":
                self._list_counters[-1] += 1
                self._out.append(f"{indent}{self._list_counters[-1]}. ")
            else:
                self._out.append(f"{indent}- ")
        elif t == "blockquote":
            self._out.append("> ")
        elif t == "pre":
            self._blank()
            self._out.append("```\n")
            self._pre_depth += 1
        elif t == "code" and not self._pre_depth:
            self._out.append("`")
        elif t == "strong" or t == "b":
            self._out.append("**")
        elif t == "em" or t == "i":
            self._out.append("*")
        elif t == "a":
            self._out.append("[")
            self._set_placeholder("link", attrs_d.get("href", ""))
        elif t == "img":
            self._emit_image(attrs_d)
        elif t == "table":
            self._blank()
            self._table_rows = []
        elif t == "tr":
            self._current_row = []
        elif t in ("td", "th"):
            self._in_cell = True
            self._current_cell = []

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in ("script", "style", "noscript", "head", "title"):
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if t in ("h1", "h2", "h3", "h4", "h5", "h6", "p"):
            self._out.append("\n\n")
        elif t in self._BLOCK_TAGS:
            self._out.append("\n")
        elif t == "li":
            self._out.append("\n")
        elif t in ("ul", "ol"):
            if self._list_stack:
                self._list_stack.pop()
                self._list_counters.pop()
            self._blank()
        elif t == "blockquote":
            self._out.append("\n")
        elif t == "pre":
            self._out.append("```\n\n")
            self._pre_depth = max(0, self._pre_depth - 1)
        elif t == "code" and not self._pre_depth:
            self._out.append("`")
        elif t in ("strong", "b"):
            self._out.append("**")
        elif t in ("em", "i"):
            self._out.append("*")
        elif t == "a":
            href = self._pop_placeholder("link")
            if href:
                self._out.append(f"]({href})")
            else:
                self._out.append("]")
        elif t in ("td", "th"):
            if self._current_row is not None:
                cell = "".join(self._current_cell).strip().replace("\n", " ")
                self._current_row.append(cell)
            self._in_cell = False
        elif t == "tr":
            if self._current_row is not None:
                self._table_rows.append(self._current_row)
                self._current_row = None
        elif t == "table":
            self._emit_table()

    def handle_startendtag(self, tag, attrs):
        """Handle self-closing tags written as ``<x/>`` (HTML/XHTML).

        HTMLParser calls this instead of ``handle_starttag`` for ``<br/>``,
        ``<img/>``, ``<hr/>``, ``<input/>``, etc. Void elements with no content
        still emit their markdown effect (a break, an image, a rule), so
        forward to the start-tag logic for the cases that matter; everything
        else is a no-op (a self-closing container has no body to render).
        """
        t = tag.lower()
        if t in ("br", "hr", "img", "input", "wbr"):
            self.handle_starttag(tag, attrs)
            if t == "hr":
                self.handle_endtag(tag)

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_cell:
            self._current_cell.append(data)
            return
        if self._pre_depth:
            self._out.append(data)
            return
        # collapse runs of whitespace; drop pure-whitespace text between
        # block boundaries so it doesn't leak stray spaces into the markdown.
        collapsed = re.sub(r"\s+", " ", data)
        if collapsed.strip():
            self._out.append(collapsed)

    # --- placeholders let <a> capture its inner text before emitting href ---
    _PLACEHOLDER_KEY: str = "_openkb_link_href"

    def _set_placeholder(self, _name: str, value: str) -> None:
        # The href is read in endtag; stash nothing inline — track via a stack.
        self._out.append("")  # no-op marker to keep append surface uniform
        self.__dict__["_pending_href"] = value

    def _pop_placeholder(self, _name: str) -> str:
        return self.__dict__.pop("_pending_href", "")

    def _emit_image(self, attrs: dict[str, str | None]) -> None:
        src = (attrs.get("src") or "").strip()
        alt = (attrs.get("alt") or "").strip()
        if not src:
            return
        self._out.append(f"\n![{alt}]({src})\n")

    def _emit_table(self) -> None:
        rows = self._table_rows
        self._table_rows = []
        if not rows:
            return
        # First row is the header; pad to max column count.
        width = max(len(r) for r in rows)
        for r in rows:
            while len(r) < width:
                r.append("")
        header = rows[0]
        body = rows[1:]
        self._out.append("| " + " | ".join(header) + " |\n")
        self._out.append("| " + " | ".join("---" for _ in header) + " |\n")
        for r in body:
            self._out.append("| " + " | ".join(r) + " |\n")
        self._out.append("\n")

    def markdown(self) -> str:
        text = "".join(self._out)
        # collapse 3+ blank lines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip() + "\n"


# A line that is *exactly* ``**NN Title**`` (or ``**N.M Title**``): the whole
# line is bolded and the content starts with a numeric section number. Web
# archives (e.g. WeChat articles) routinely mark section headings this way
# instead of using <h2>/<h3>, which would otherwise flatten PageIndex's tree
# to a single node. Promoting these to real headings restores the hierarchy.
_BOLD_TITLE_RE = re.compile(r"^\*\*\s*((?:\d+\.)*\d+)[\s\.\、\)]+(.+?)\s*\*\*$")


def _promote_bold_section_titles(markdown: str) -> str:
    """Promote standalone ``**NN Title**`` lines to Markdown headings.

    Top-level numbers (``01``, ``10``) become ``##``; dotted sub-numbers
    (``4.1``, ``8.2``) become ``###``. Only lines whose entire content is the
    bolded title are touched — inline bold runs inside a paragraph are left
    alone (they aren't standalone lines).
    """
    out: list[str] = []
    for line in markdown.split("\n"):
        m = _BOLD_TITLE_RE.match(line.strip())
        if m and line.strip() == line.strip().strip():
            number, title = m.group(1), m.group(2)
            level = "###" if "." in number else "##"
            out.append(f"{level} {number} {title}")
        else:
            out.append(line)
    return "\n".join(out)


def html_to_markdown(html: str) -> str:
    """Convert an HTML string to structural Markdown."""
    parser = _HtmlToMarkdown()
    parser.feed(html)
    parser.close()
    return _promote_bold_section_titles(parser.markdown())


# ---------------------------------------------------------------------------
# Image rewriting
# ---------------------------------------------------------------------------


_IMG_SRC_RE = re.compile(
    r"""<img\b[^>]*\bsrc\s*=\s*["']?([^"'>\s]+)["']?[^>]*>""",
    re.IGNORECASE | re.DOTALL,
)


def _rewrite_image_sources(html: str, ref_to_local: dict[str, str]) -> str:
    """Rewrite ``<img src="cid:...">`` / ``<img src="url|basename">`` to local paths.

    ``ref_to_local`` is keyed by both Content-ID (``image001``) and the
    Content-Location basename/full URL, so cid: refs and URL refs both resolve.
    Unknown sources are left untouched (PageIndex will still accept the md).
    """

    def repl(match: re.Match[str]) -> str:
        tag = match.group(0)
        src = match.group(1).strip()
        local = _resolve_ref(src, ref_to_local)
        if local is None:
            return tag
        return re.sub(
            r"""(\bsrc\s*=\s*["']?)([^"'>\s]+)(["']?)""",
            lambda m: f"{m.group(1)}{local}{m.group(3)}",
            tag,
            count=1,
            flags=re.IGNORECASE,
        )

    return _IMG_SRC_RE.sub(repl, html)


def _resolve_ref(src: str, ref_to_local: dict[str, str]) -> str | None:
    """Map an ``<img src>`` value to its local image path, if known.

    Tries, in order: the CID (for ``cid:`` refs), the raw src, the decoded src,
    the normalized URL (query/fragment stripped — this is what actually matches
    browser-saved Content-Location values), and finally the URL basename as a
    last resort.
    """
    if src.lower().startswith("cid:"):
        cid = _strip_cid(src[4:])
        return ref_to_local.get(cid)
    if src in ref_to_local:
        return ref_to_local[src]
    decoded = unquote(src)
    if decoded in ref_to_local:
        return ref_to_local[decoded]
    normalized = _normalize_url(src)
    if normalized in ref_to_local:
        return ref_to_local[normalized]
    base = Path(normalized).name
    if base and base in ref_to_local:
        return ref_to_local[base]
    return None


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _sanitize_doc_name(stem: str) -> str:
    normalized = unicodedata.normalize("NFKC", stem)
    cleaned = re.sub(r"[^\w\-]+", "-", normalized).strip("-")
    return cleaned or "document"


def unpack_mhtml(
    mhtml_path: Path, out_dir: Path, *, doc_name: str | None = None
) -> MHTMLPrepareResult:
    """Unpack an MHTML archive into ``out_dir/document.md`` + ``out_dir/images/``.

    Writes the PageIndex-ready Markdown and returns its path. ``out_dir`` is
    created if missing; existing contents (a prior unpack of the same source)
    are overwritten in place so retries keep a stable name.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    resolved_doc_name = doc_name or _sanitize_doc_name(mhtml_path.stem)

    with mhtml_path.open("rb") as fh:
        # email.policy.default is typed as EmailPolicy[EmailMessage], but
        # BytesParser's policy param is typed as Policy[Message[str, str]] — a
        # stdlib stubs mismatch we silence rather than per-module disable.
        msg = BytesParser(policy=email.policy.default).parse(fh)  # type: ignore[arg-type]

    html_part, parts = _html_part_and_parts(msg)

    # Extract every image-bearing part: each gets a unique imgNNN file, then is
    # registered under EVERY key an <img src> might spell (Content-ID, full
    # Content-Location URL, normalized URL, basename). The normalized URL is
    # the primary key — basenames often collide across images (CDNs serving
    # every image from .../<n>), so basename uses setdefault: first image wins
    # that fallback slot, but the URL-keyed entries are unique per image.
    ref_to_local: dict[str, str] = {}
    counter = 0
    for part in parts:
        keys = _image_resource_keys(part)
        if not keys:
            continue
        counter += 1
        ext = _ext_for_content_type(part.get_content_type())
        filename = f"img{counter:03d}{ext}"
        (image_dir / filename).write_bytes(_payload_bytes(part))
        rel = f"./images/{filename}"
        for key in keys:
            ref_to_local.setdefault(key, rel)

    if html_part is None:
        html = ""
        logger.warning("MHTML %s has no text/html or text/plain part", mhtml_path.name)
    else:
        charset = html_part.get_content_charset() or "utf-8"
        raw = _payload_bytes(html_part)
        try:
            html = raw.decode(charset, errors="replace")
        except (LookupError, TypeError):
            html = raw.decode("utf-8", errors="replace")

    html = _rewrite_image_sources(html, ref_to_local)
    markdown = html_to_markdown(html)

    # Front-load a level-1 title from the archive's stem so MarkdownParser
    # always produces at least one structural node, even for unstructured pages.
    if not re.search(r"(?m)^#{1}\s", markdown):
        markdown = f"# {resolved_doc_name}\n\n" + markdown

    html_path = out_dir / "document.html"
    html_path.write_text(html, encoding="utf-8")
    markdown_path = out_dir / "document.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    return MHTMLPrepareResult(
        html_path=html_path,
        markdown_path=markdown_path,
        image_dir=image_dir,
        doc_name=resolved_doc_name,
    )


def prepare_mhtml_for_pageindex(
    mhtml_path: Path, kb_dir: Path, *, doc_name: str | None = None
) -> MHTMLPrepareResult:
    """Prepare an MHTML archive for PageIndex consumption.

    Stages the unpacked Markdown + images under
    ``<kb>/.openkb/mhtml_assets/<doc_name>/`` — a PageIndex-managed artifact
    root, deliberately *not* ``wiki/sources/images`` (the short-doc image tree).
    PageIndex owns the lifecycle of this input.
    """
    resolved_doc_name = doc_name or _sanitize_doc_name(mhtml_path.stem)
    out_dir = kb_dir / ".openkb" / "mhtml_assets" / resolved_doc_name
    return unpack_mhtml(mhtml_path, out_dir, doc_name=resolved_doc_name)


def is_mhtml(path: Path) -> bool:
    """True when *path* is an MHTML web archive (by extension)."""
    return path.suffix.lower() in _MHTML_SUFFIXES


__all__ = [
    "MHTMLPrepareResult",
    "html_to_markdown",
    "is_mhtml",
    "prepare_mhtml_for_pageindex",
    "unpack_mhtml",
]
