"""OKF compiler orchestrator.

``compile_one`` turns a single Markdown file into one ``.okf.zip``. Every
compile is fresh and isolated: it reads only its input file, writes only its
workdir + output zip, and never touches the long-lived KB state.

``compile_dir`` is a *thin* orchestrator over ``compile_one``: it enumerates
input Markdown files (flat / wechat / auto) and loops ``compile_one`` per
file with **no shared context between files** - per the subtraction principle,
there is no cross-article state, no global concept merge, no batch bundle.
One file failing does not abort the others unless ``--fail-fast`` is set.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from openkb.okf.assets import collect_images, count_missing
from openkb.okf.bundle import write_zip
from openkb.okf.llm import LLMClient, LLMConfig, extract
from openkb.okf.markdown import extract_title, split_sections
from openkb.okf.render import render_bundle
from openkb.okf.schema import Extracts, SectionSpec

logger = logging.getLogger(__name__)

OKF_ZIP_SUFFIX = ".okf.zip"
DEFAULT_REPORT_NAME = "batch_report.json"


@dataclass
class CompileOptions:
    """Per-compile knobs shared by ``compile_one`` and ``compile_dir``.

    ``llm_config`` is ``None`` for ``--no-llm`` runs; when set, an LLM run is
    attempted (and the ``api_key`` inside it is transient - never persisted).
    """

    workdir: Path | None = None
    keep_workdir: bool = False
    overwrite: bool = False
    no_llm: bool = False
    language: str = "en"
    max_concepts: int = 12
    max_entities: int = 12
    llm_config: LLMConfig | None = None


@dataclass
class CompileResult:
    """Outcome of compiling one Markdown file."""

    input_path: Path
    output_path: Path | None
    ok: bool
    skipped: bool = False
    error: str = ""
    manifest: dict | None = None
    warnings: list[str] = field(default_factory=list)

    def to_report(self) -> dict:
        """Compact dict for the batch report (no api_key, ever)."""
        return {
            "input": _posix(self.input_path),
            "output": _posix(self.output_path) if self.output_path else None,
            "status": ("skipped" if self.skipped else ("ok" if self.ok else "failed")),
            "error": self.error or None,
            "counts": (self.manifest or {}).get("counts"),
            "warnings": list(self.warnings),
        }


def compile_one(input_md: Path, out: Path, opts: CompileOptions) -> CompileResult:
    """Compile a single Markdown file into ``out`` (a ``.okf.zip``).

    ``out`` should be the final zip path; a temp workdir is created (or the
    caller-provided ``opts.workdir`` is used) and removed unless
    ``opts.keep_workdir``. Overwrite handling: an existing ``out`` is an error
    unless ``opts.overwrite``.
    """
    input_md = Path(input_md).resolve()
    out = Path(out).resolve()
    warnings: list[str] = []

    if not input_md.exists():
        return CompileResult(input_md, out, ok=False, error=f"input not found: {input_md}")
    if out.exists() and not opts.overwrite:
        return CompileResult(
            input_md,
            out,
            ok=False,
            skipped=True,
            error="output exists (use --overwrite)",
        )

    workdir = _resolve_workdir(opts)
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        markdown = input_md.read_text(encoding="utf-8")
        sections = split_sections(markdown)
        if not sections:
            sections = [
                SectionSpec(
                    index=0,
                    title="document",
                    heading_path="document",
                    line_start=1,
                    line_end=max(markdown.count("\n") + 1, 1),
                    body=markdown,
                )
            ]
        title = extract_title(markdown) or input_md.stem

        # Images: copy into workdir/assets/images and rewrite section bodies.
        images_dir = workdir / "assets" / "images"
        rewritten, image_refs, asset_warnings = collect_images(
            [s.body for s in sections], input_md.parent, images_dir
        )
        warnings.extend(asset_warnings)
        for sec, body in zip(sections, rewritten):
            sec.body = body

        # LLM extracts (only when configured and not --no-llm).
        extracts = Extracts()
        llm_enabled = False
        model_recorded: str | None = None
        if not opts.no_llm and opts.llm_config is not None and opts.llm_config.is_configured():
            try:
                client = LLMClient(opts.llm_config)
                model_recorded = client.model
                extracts = extract(
                    client,
                    markdown,
                    sections,
                    language=opts.language,
                    max_concepts=opts.max_concepts,
                    max_entities=opts.max_entities,
                )
                llm_enabled = True
            except Exception as exc:  # noqa: BLE001 - a bad config/call degrades to no-llm, not a crash
                warnings.append(f"llm: disabled for this article ({type(exc).__name__}: {exc})")
                llm_enabled = False
        # Surface missing-asset count as a warning line for visibility.
        missing = count_missing(image_refs)
        if missing:
            warnings.append(f"assets: {missing} referenced image(s) not found")

        manifest = render_bundle(
            workdir,
            markdown=markdown,
            sections=sections,
            image_refs=image_refs,
            extracts=extracts,
            source_path=_posix(input_md),
            title=title,
            language=opts.language,
            llm_enabled=llm_enabled,
            model=model_recorded,
            warnings=warnings,
        )
        write_zip(workdir, out)
        return CompileResult(
            input_md,
            out,
            ok=True,
            manifest=manifest,
            warnings=list(extracts.warnings) + list(warnings),
        )
    except Exception as exc:  # noqa: BLE001 - never crash the batch; report the failure
        logger.debug("compile_one failed for %s", input_md, exc_info=True)
        return CompileResult(input_md, out, ok=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        if not opts.keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


@dataclass
class BatchReport:
    """Aggregate result of ``compile_dir``."""

    total: int = 0
    ok: int = 0
    skipped: int = 0
    failed: int = 0
    results: list[CompileResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "ok": self.ok,
            "skipped": self.skipped,
            "failed": self.failed,
            "results": [r.to_report() for r in self.results],
        }


def compile_dir(
    input_dir: Path,
    out_dir: Path,
    opts: CompileOptions,
    *,
    mode: str = "auto",
    glob_pattern: str = "*.md",
    recursive: bool = True,
    skip_existing: bool = True,
    max_workers: int = 1,
    fail_fast: bool = False,
    report_path: Path | None = None,
) -> BatchReport:
    """Compile every Markdown file under ``input_dir`` into its own ``.okf.zip``.

    Thin orchestrator: enumerate inputs per ``mode`` (flat/wechat/auto), then
    run ``compile_one`` per file. No cross-file context is shared. Failures are
    isolated unless ``fail_fast``. Writes ``batch_report.json`` to
    ``report_path`` (defaults to ``<out_dir>/batch_report.json``) when set or
    when the caller passes an explicit path.
    """
    input_dir = Path(input_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = enumerate_inputs(input_dir, mode=mode, glob_pattern=glob_pattern, recursive=recursive)
    report = BatchReport(total=len(inputs))

    if not inputs:
        logger.warning("compile-dir: no Markdown files found under %s (mode=%s)", input_dir, mode)

    # When skip_existing is on, drop already-compiled targets up front so they
    # don't consume a worker slot (and so the report reflects the real work).
    # Output names are disambiguated: when two inputs share a stem (common in
    # wechat mode where every article dir has an ``index.md``), the parent dir
    # name is prefixed so neither clobbers the other.
    pending: list[tuple[Path, Path]] = []
    used_names: set[str] = set()
    for md in inputs:
        zip_name = _output_name(md, used_names, relative_to=input_dir)
        used_names.add(zip_name)
        zip_path = out_dir / zip_name
        if zip_path.exists() and skip_existing and not opts.overwrite:
            report.results.append(
                CompileResult(md, zip_path, ok=False, skipped=True, error="already compiled")
            )
            report.skipped += 1
            continue
        pending.append((md, zip_path))

    def _do(item: tuple[Path, Path]) -> CompileResult:
        md, zip_path = item
        return compile_one(md, zip_path, opts)

    failed_hard = False
    if max_workers <= 1:
        for item in pending:
            r = _do(item)
            report.results.append(r)
            if r.ok:
                report.ok += 1
            elif r.skipped:
                report.skipped += 1
            else:
                report.failed += 1
                if fail_fast:
                    failed_hard = True
                    break
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_do, item): item for item in pending}
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                except Exception as exc:  # noqa: BLE001 - worker blew up
                    md = futures[fut][0]
                    r = CompileResult(md, None, ok=False, error=f"{type(exc).__name__}: {exc}")
                report.results.append(r)
                if r.ok:
                    report.ok += 1
                elif r.skipped:
                    report.skipped += 1
                else:
                    report.failed += 1
                    if fail_fast:
                        failed_hard = True
                        # cancel remaining
                        for f in futures:
                            f.cancel()
                        break

    # Sort results by input path for a stable report.
    report.results.sort(key=lambda r: _posix(r.input_path))

    if report_path is not None:
        _write_report(report_path, report.to_dict())
    if failed_hard:
        logger.warning("compile-dir: aborted early (--fail-fast) after a failure")
    return report


# ---------------------------------------------------------------------------
# input enumeration (flat / wechat / auto)
# ---------------------------------------------------------------------------

_MD_SUFFIXES = {".md", ".markdown"}


def _output_name(md: Path, used: set[str], *, relative_to: Path) -> str:
    """Compute a unique output ``.okf.zip`` filename for ``md``.

    Default ``<stem>.okf.zip``. When that name is already taken (two inputs
    sharing a stem, e.g. wechat article dirs each with ``index.md``), prefix
    the parent directory stem to disambiguate: ``<parent>__<stem>.okf.zip``.
    The parent is computed relative to ``relative_to`` (the input root) so
    only the meaningful article-dir segment is used, not the full path.
    """
    base = md.stem + OKF_ZIP_SUFFIX
    if base not in used:
        return base
    try:
        rel = md.relative_to(relative_to)
    except ValueError:
        rel = md
    parent = rel.parent.name
    if parent:
        candidate = f"{parent}__{md.stem}{OKF_ZIP_SUFFIX}"
    else:
        candidate = f"{md.stem}_{len(used)}{OKF_ZIP_SUFFIX}"
    # last-resort numeric suffix if still colliding
    n = 1
    while candidate in used:
        if parent:
            candidate = f"{parent}__{md.stem}_{n}{OKF_ZIP_SUFFIX}"
        else:
            candidate = f"{md.stem}_{n}{OKF_ZIP_SUFFIX}"
        n += 1
    return candidate


def enumerate_inputs(
    input_dir: Path, *, mode: str, glob_pattern: str, recursive: bool
) -> list[Path]:
    """Return the Markdown files to compile under ``input_dir`` per ``mode``.

    - ``flat``: every ``.md``/``.markdown`` file (recursively when ``recursive``).
    - ``wechat``: detects ``wechat_article_to_markdown`` output layout - a
      tree of article directories each containing one main ``.md`` (plus
      assets). Only the main article Markdown is selected per directory.
    - ``auto``: try ``wechat`` first; fall back to ``flat`` when no article
      dirs are detected.
    """
    if not input_dir.is_dir():
        return []
    mode = (mode or "auto").lower()
    if mode == "flat":
        return _flat_inputs(input_dir, glob_pattern, recursive)
    if mode == "wechat":
        return _wechat_inputs(input_dir)
    if mode == "auto":
        wechat = _wechat_inputs(input_dir)
        if wechat:
            return wechat
        return _flat_inputs(input_dir, glob_pattern, recursive)
    raise ValueError(f"unknown mode: {mode!r} (expected auto|flat|wechat)")


def _flat_inputs(input_dir: Path, glob_pattern: str, recursive: bool) -> list[Path]:
    """All Markdown files under ``input_dir`` matching ``glob_pattern``."""
    # ``glob_pattern`` filters the basename (e.g. ``*.md``); suffix is the
    # real gate so a ``*.markdown`` glob still picks up ``.markdown`` files.
    iterator = input_dir.rglob(glob_pattern) if recursive else input_dir.glob(glob_pattern)
    out = [p for p in iterator if p.is_file() and p.suffix.lower() in _MD_SUFFIXES]
    out.sort()
    return out


def _wechat_inputs(input_dir: Path) -> list[Path]:
    """Pick the main article ``.md`` in each wechat_article_to_markdown dir.

    The wechat layout puts each article in its own subdirectory alongside an
    ``assets/`` folder; the main Markdown is the (typically largest) ``.md``
    directly in the article dir. We pick the largest ``.md`` by byte size as
    a stable heuristic when more than one is present (sometimes a README or
    index sneaks in). Returns ``[]`` when no article dirs are detected, so
    ``auto`` mode can fall back to flat.
    """
    article_dirs = []
    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        # An article dir has at least one .md and (typically) an assets/ dir.
        mds = [p for p in child.glob("*.md") if p.is_file()]
        if mds:
            article_dirs.append((child, mds))
    if not article_dirs:
        return []
    out = []
    for _dir, mds in article_dirs:
        mds.sort(key=lambda p: p.stat().st_size, reverse=True)
        out.append(mds[0])
    out.sort()
    return out


def _resolve_workdir(opts: CompileOptions) -> Path:
    """Use the caller-provided workdir, else a fresh temp dir."""
    if opts.workdir is not None:
        return Path(opts.workdir).resolve()
    return Path(tempfile.mkdtemp(prefix="okf-compile-"))


def _write_report(report_path: Path, data: dict) -> None:
    """Write the batch report JSON (atomic; no api_key present in ``data``)."""
    from openkb.locks import atomic_write_json

    atomic_write_json(Path(report_path), data)


def _posix(path: Path) -> str:
    return Path(path).as_posix()
