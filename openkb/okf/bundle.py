"""Assemble a workdir tree into an ``.okf.zip``.

Walks the rendered tree and writes every file into a ZIP_DEFLATED archive
with forward-slash, zip-relative paths in a stable, sorted order (so the
archive layout is consistent across platforms and easy to diff). Note the
archive is NOT byte-identical across runs: ``manifest.json`` records a real
``created_at`` timestamp on purpose, and ZIP entries carry file mtimes.

Defensive guards (the spec requires these even though the renderer never
produces them):
  * no absolute paths (drive letters or leading ``/``) ever enter the zip;
  * no ``..`` segment ever enters the zip;
  * ``raw/original.mhtml`` is asserted absent (this is a Markdown-only
    compiler - the raw/ tree must not exist).
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Paths that must never appear in an OKF Bundle. The renderer doesn't create
# them, but a consumer can trust the guarantee by checking the zip - and the
# test suite asserts it - so a future regression can't silently leak raw
# document state into a "fresh" bundle.
FORBIDDEN_PATHS = {"raw/original.mhtml", "raw/"}


def write_zip(workdir: Path, out_path: Path) -> Path:
    """Write ``workdir``'s tree into ``out_path`` (a ``.okf.zip``).

    Returns ``out_path`` (resolved). Overwrites an existing file at that path.
    Raises :class:`ValueError` if a forbidden path is encountered.
    """
    workdir = workdir.resolve()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    entries = _collect_entries(workdir)
    _assert_clean(entries)

    out_path = out_path.resolve()
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel, abs_path in entries:
            zf.write(abs_path, rel)
    logger.info("OKF bundle written: %s (%d entries)", out_path, len(entries))
    return out_path


def _collect_entries(workdir: Path) -> list[tuple[str, Path]]:
    """Gather ``(zip-relative-path, absolute-path)`` pairs, sorted by path.

    Directories are represented by a trailing-slash entry so empty reserved
    dirs (e.g. ``extracts/claims/``) survive in the zip and consumers see the
    expected tree shape.
    """
    entries: list[tuple[str, Path]] = []
    for p in sorted(workdir.rglob("*")):
        rel = p.relative_to(workdir).as_posix()
        if p.is_dir():
            entries.append((rel + "/", p))
        else:
            entries.append((rel, p))
    # Sort by path so the archive order is deterministic across platforms.
    entries.sort(key=lambda e: e[0])
    return entries


def _assert_clean(entries: list[tuple[str, Path]]) -> None:
    """Reject absolute paths, ``..`` segments, and forbidden files."""
    for rel, _ in entries:
        # Absolute drive-letter or leading-slash path.
        if rel.startswith(("/", "\\")) or (len(rel) >= 2 and rel[1] == ":"):
            raise ValueError(f"refusing to write absolute path into zip: {rel!r}")
        # Any ``..`` component.
        if any(seg == ".." for seg in rel.split("/")):
            raise ValueError(f"refusing to write path-traversal segment into zip: {rel!r}")
        # Forbidden raw-document state.
        for forbidden in FORBIDDEN_PATHS:
            if rel == forbidden or rel.startswith(forbidden) or rel == forbidden.rstrip("/"):
                raise ValueError(
                    f"forbidden path {rel!r} in OKF bundle "
                    "(Markdown-only compiler must not emit raw/)"
                )
