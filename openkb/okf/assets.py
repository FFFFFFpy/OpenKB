"""Scan a Markdown article's sections for relative image refs and copy them
into the bundle's ``assets/images/`` directory, rewriting the in-section
links to point at ``../assets/images/<name>``.

Mirrors :func:`openkb.images.copy_relative_images` (the dedup + rewrite
pattern from PR dce4972): an ``assigned`` dict keys a resolved source path
to its destination name so the same image referenced twice isn't copied
twice, and a ``taken`` set ensures two *different* sources that share a
basename (``a/logo.png`` and ``b/logo.png``) don't clobber each other.

Diverges from ``openkb.images`` in two ways, both deliberate:
  * disambiguation uses a short SHA-1 prefix of the source path rather than
    an ``_n`` counter, so names stay stable across re-runs (the same source
    maps to the same dest name every time);
  * only Markdown ``![alt](rel)`` syntax is handled - no HTML ``<img>``
    (explicitly out of scope for v1).

Missing source files do not fail the compile: the link is left unchanged and
a warning is recorded for the manifest + ``counts.missing_assets`` is bumped.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from pathlib import Path

from openkb.okf.schema import IMAGES_DIR, ImageRef

logger = logging.getLogger(__name__)

# Matches: ![alt](relative/path) - excludes http(s):// and data: URIs.
# Copied verbatim from openkb/images.py:19 so the two pipelines agree on what
# a "relative image reference" is.
_RELATIVE_RE = re.compile(r"!\[([^\]]*)\]\((?!https?://|data:)([^)]+)\)")

# The in-section link target after copy. ``../`` climbs out of ``sections/``
# (or ``extracts/``) into the bundle root, then into ``assets/images/``.
_REWRITTEN_PREFIX = f"../{IMAGES_DIR}/"


def collect_images(
    section_bodies: list[str],
    source_dir: Path,
    images_dir: Path,
) -> tuple[list[str], list[ImageRef], list[str]]:
    """Copy referenced relative images and rewrite section links.

    Args:
        section_bodies: the body text of each section (already split).
        source_dir: the directory the input Markdown lives in (image paths
            are resolved relative to this).
        images_dir: the bundle's ``assets/images/`` directory; created if any
            image is found.

    Returns:
        ``(rewritten_bodies, image_refs, warnings)``. ``rewritten_bodies``
        is the input list with each found link replaced; missing-image links
        are left untouched. ``warnings`` carries one human-readable line per
        missing image (and per escaped path).
    """
    warnings: list[str] = []
    # source-resolved-path -> dest filename, so the same image referenced
    # twice (across sections or within one) is copied once and both links
    # resolve to the same dest.
    assigned: dict[Path, str] = {}
    # dest filenames already claimed, so two sources sharing a basename
    # don't clobber each other.
    taken: set[str] = set()
    refs: list[ImageRef] = []

    rewritten_bodies: list[str] = []
    for body in section_bodies:
        rewritten = body
        for match in _RELATIVE_RE.finditer(body):
            alt, rel_path = match.group(1), match.group(2)
            src = _resolve_under(rel_path, source_dir)
            if src is None:
                warnings.append(f"image path escapes source dir: {rel_path}")
                continue
            if not src.exists():
                warnings.append(f"relative image not found: {rel_path}")
                refs.append(
                    ImageRef(
                        original_ref=rel_path,
                        dest_name="",
                        source_path=str(src),
                        found=False,
                        alt=alt,
                    )
                )
                continue

            dest_name = assigned.get(src)
            if dest_name is None:
                dest_name = _unique_name(src, taken)
                assigned[src] = dest_name
                taken.add(dest_name)
                images_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, images_dir / dest_name)
            refs.append(
                ImageRef(
                    original_ref=rel_path,
                    dest_name=dest_name,
                    source_path=str(src),
                    found=True,
                    alt=alt,
                )
            )
            new_ref = f"![{alt}]({_REWRITTEN_PREFIX}{dest_name})"
            rewritten = rewritten.replace(match.group(0), new_ref, 1)
        rewritten_bodies.append(rewritten)

    return rewritten_bodies, refs, warnings


def count_missing(refs: list[ImageRef]) -> int:
    """Number of referenced images whose source file was absent."""
    return sum(1 for r in refs if not r.found)


def _resolve_under(rel_path: str, source_dir: Path) -> Path | None:
    """Resolve ``rel_path`` under ``source_dir``, rejecting escapes.

    A ``..`` that climbs above ``source_dir`` is dropped (recorded as a
    warning by the caller) so a malicious or broken path can't read outside
    the article's directory. Absolute paths are likewise rejected.
    """
    rel_path = rel_path.strip().strip('"').strip("'")
    if not rel_path or rel_path.startswith(("/", "\\")):
        return None
    candidate = (source_dir / rel_path).resolve()
    try:
        candidate.relative_to(source_dir.resolve())
    except ValueError:
        return None
    return candidate


def _unique_name(src: Path, taken: set[str]) -> str:
    """A destination filename for ``src``, disambiguated on basename clash.

    When the bare ``src.name`` is free it is used as-is. When another source
    already claimed that basename, the new name is prefixed with the first 8
    hex chars of a SHA-1 of the resolved source path - stable across re-runs,
    so the same source always lands at the same dest name.
    """
    base = src.name
    if base not in taken:
        return base
    digest = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:8]
    candidate = f"{digest}_{base}"
    # Extremely unlikely (two distinct sources hashing to the same 8-char
    # prefix AND sharing a basename), but guard against an infinite reuse.
    n = 1
    while candidate in taken:
        candidate = f"{digest}_{n}_{base}"
        n += 1
    return candidate
