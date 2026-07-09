"""Click CLI for the ``openkb okf`` command group.

Three subcommands: ``compile`` (one Markdown -> one ``.okf.zip``),
``compile-dir`` (a directory of Markdown -> one zip per file), and
``test-llm`` (round-trip a trivial JSON request against the configured
OpenAI-compatible endpoint).

This group is deliberately **KB-independent**: it never calls
``openkb.cli._find_kb_dir``, never takes the KB lock, and never calls
``_setup_llm_key``. LLM access is its own 3-param client
(``--base-url`` / ``--model`` / ``--api-key`` / ``--timeout``) resolved CLI >
``OPENKB_LLM_*`` env > ``.env`` > default. ``--no-llm`` skips all LLM config
validation and requests and still emits a basic bundle.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click

from openkb.okf.compiler import (
    DEFAULT_REPORT_NAME,
    OKF_ZIP_SUFFIX,
    BatchReport,
    CompileOptions,
    compile_dir,
    compile_one,
)
from openkb.okf.llm import LLMClient, LLMConfig, load_dotenv_values, resolve_config

# Shared LLM option flags attached to ``compile`` / ``compile-dir`` / ``test-llm``.
# Defined once so the three commands agree on names and help text.


def _llm_options(fn):
    """Attach the four LLM options to a command (used as a decorator stack)."""
    fn = click.option(
        "--timeout",
        "timeout",
        type=float,
        default=None,
        help="Request timeout (seconds). Env: OPENKB_LLM_TIMEOUT.",
    )(fn)
    fn = click.option(
        "--api-key",
        "api_key",
        default=None,
        help="LLM API key. Env: OPENKB_LLM_API_KEY. Never written to output.",
    )(fn)
    fn = click.option(
        "--model",
        "model",
        default=None,
        help="LLM model (bare name auto-prefixed openai/ when --base-url set). "
        "Env: OPENKB_LLM_MODEL.",
    )(fn)
    fn = click.option(
        "--base-url",
        "base_url",
        default=None,
        help="OpenAI-compatible base URL. Env: OPENKB_LLM_BASE_URL.",
    )(fn)
    return fn


@click.group()
def okf() -> None:
    """Compile Markdown into self-contained OKF Bundles (fresh, per-article)."""


@okf.command("compile")
@click.argument("input_md", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--out",
    "out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output .okf.zip path. Default: <input-stem>.okf.zip next to the input.",
)
@click.option(
    "--workdir",
    "workdir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Staging workdir for the bundle tree. Default: a temp dir, removed after.",
)
@click.option(
    "--keep-workdir",
    is_flag=True,
    default=False,
    help="Keep the workdir after writing the zip (for debugging).",
)
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite an existing output zip.")
@click.option(
    "--no-llm",
    "no_llm",
    is_flag=True,
    default=False,
    help="Skip LLM extraction; emit a basic bundle (sections + assets only).",
)
@click.option(
    "--language",
    "language",
    default="zh",
    show_default=True,
    help="Language for the generated summary.",
)
@click.option(
    "--max-concepts",
    "max_concepts",
    type=int,
    default=12,
    show_default=True,
    help="Maximum number of concepts to extract.",
)
@click.option(
    "--max-entities",
    "max_entities",
    type=int,
    default=12,
    show_default=True,
    help="Maximum number of entities to extract.",
)
@_llm_options
def compile_cmd(
    input_md,
    out,
    workdir,
    keep_workdir,
    overwrite,
    no_llm,
    language,
    max_concepts,
    max_entities,
    base_url,
    model,
    api_key,
    timeout,
):
    """Compile INPUT.md into an OKF Bundle (.okf.zip)."""
    out = _resolve_single_out(input_md, out)
    config = _maybe_config(no_llm, base_url, model, api_key, timeout)
    opts = CompileOptions(
        workdir=workdir,
        keep_workdir=keep_workdir,
        overwrite=overwrite,
        no_llm=no_llm,
        language=language,
        max_concepts=max_concepts,
        max_entities=max_entities,
        llm_config=config,
    )
    result = compile_one(input_md, out, opts)
    _print_result(result)
    if not result.ok:
        sys.exit(1)


@okf.command("compile-dir")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--out",
    "out_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory for the per-file .okf.zip bundles.",
)
@click.option(
    "--mode",
    "mode",
    type=click.Choice(["auto", "flat", "wechat"]),
    default="auto",
    show_default=True,
    help="Input enumeration mode: flat (all .md), wechat (per-article dirs), auto.",
)
@click.option(
    "--glob",
    "glob_pattern",
    default="*.md",
    show_default=True,
    help="Glob pattern for flat/auto input enumeration.",
)
@click.option(
    "--recursive/--no-recursive",
    "recursive",
    default=True,
    help="Recurse into subdirectories (flat/auto mode).",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Recompile even when an output zip already exists.",
)
@click.option(
    "--skip-existing/--no-skip-existing",
    "skip_existing",
    default=True,
    help="Skip inputs whose output zip already exists (default: on).",
)
@click.option(
    "--no-llm",
    "no_llm",
    is_flag=True,
    default=False,
    help="Skip LLM extraction; emit basic bundles.",
)
@click.option(
    "--max-workers",
    "max_workers",
    type=int,
    default=1,
    show_default=True,
    help="Parallel compiles across files (LLM calls are blocking).",
)
@click.option(
    "--fail-fast",
    is_flag=True,
    default=False,
    help="Abort the batch on the first failure (default: continue).",
)
@click.option(
    "--report",
    "report_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=f"Write a JSON batch report here (default: <OUT>/{DEFAULT_REPORT_NAME}).",
)
@click.option("--language", "language", default="zh", show_default=True)
@click.option("--max-concepts", "max_concepts", type=int, default=12, show_default=True)
@click.option("--max-entities", "max_entities", type=int, default=12, show_default=True)
@_llm_options
def compile_dir_cmd(
    input_dir,
    out_dir,
    mode,
    glob_pattern,
    recursive,
    overwrite,
    skip_existing,
    no_llm,
    max_workers,
    fail_fast,
    report_path,
    language,
    max_concepts,
    max_entities,
    base_url,
    model,
    api_key,
    timeout,
):
    """Compile every Markdown file under INPUT_DIR into its own .okf.zip."""
    config = _maybe_config(no_llm, base_url, model, api_key, timeout)
    opts = CompileOptions(
        overwrite=overwrite,
        no_llm=no_llm,
        language=language,
        max_concepts=max_concepts,
        max_entities=max_entities,
        llm_config=config,
    )
    if report_path is None:
        report_path = Path(out_dir) / DEFAULT_REPORT_NAME
    report: BatchReport = compile_dir(
        input_dir,
        out_dir,
        opts,
        mode=mode,
        glob_pattern=glob_pattern,
        recursive=recursive,
        skip_existing=skip_existing,
        max_workers=max_workers,
        fail_fast=fail_fast,
        report_path=report_path,
    )
    click.echo(
        f"Done: {report.ok} ok, {report.skipped} skipped, "
        f"{report.failed} failed ({report.total} total)."
    )
    click.echo(f"Report: {report_path}")
    if report.failed and fail_fast:
        sys.exit(1)


@okf.command("test-llm")
@_llm_options
def test_llm_cmd(base_url, model, api_key, timeout):
    """Round-trip a trivial JSON request to verify the LLM endpoint."""
    config = _maybe_config(
        no_llm=False, base_url=base_url, model=model, api_key=api_key, timeout=timeout
    )
    if config is None or not config.is_configured():
        click.echo("[ERROR] No --model (or OPENKB_LLM_MODEL) set. Nothing to test.", err=True)
        sys.exit(1)
    try:
        client = LLMClient(config)
    except ValueError as exc:
        click.echo(f"[ERROR] {exc}", err=True)
        sys.exit(1)
    click.echo(f"Testing model={client.model} base_url={config.base_url or '(default)'}...")
    try:
        content = client.test()
        click.echo(f"[OK] LLM responded: {content[:200]}")
    except Exception as exc:  # noqa: BLE001 - surface the real failure
        click.echo(f"[ERROR] LLM request failed: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _maybe_config(no_llm: bool, base_url, model, api_key, timeout) -> LLMConfig | None:
    """Build an LLMConfig from CLI/env/.env, or return None for --no-llm.

    In ``--no-llm`` mode we return ``None`` without touching env/.env so no
    key ever needs to be present - the compile proceeds with sections + assets.
    """
    if no_llm:
        return None
    dotenv = load_dotenv_values(_local_dotenv())
    return resolve_config(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
        env=dict(os.environ),
        dotenv_values=dotenv,
    )


def _local_dotenv() -> Path:
    """The ``.env`` to consult for OPENKB_LLM_* values (cwd-local, best-effort)."""
    return Path.cwd() / ".env"


def _resolve_single_out(input_md: Path, out: Path | None) -> Path:
    """Default output path: ``<input-stem>.okf.zip`` next to the input."""
    if out is not None:
        return Path(out)
    return input_md.with_suffix(OKF_ZIP_SUFFIX)


def _print_result(result) -> None:
    if result.skipped:
        click.echo(f"[SKIP] {result.input_path.name}: {result.error}")
        return
    if not result.ok:
        click.echo(f"[FAIL] {result.input_path.name}: {result.error}", err=True)
        return
    click.echo(f"[OK] {result.input_path.name} -> {result.output_path}")
    if result.workdir_path is not None:
        click.echo(f"     workdir={result.workdir_path}")
    if result.manifest:
        counts = result.manifest.get("counts", {})
        click.echo(
            f"     sections={counts.get('sections', 0)} "
            f"concepts={counts.get('concepts', 0)} "
            f"entities={counts.get('entities', 0)} "
            f"relations={counts.get('relations', 0)} "
            f"images={counts.get('images', 0)} "
            f"missing={counts.get('missing_assets', 0)}"
        )
    for w in result.warnings:
        click.echo(f"     [WARN] {w}")
