"""Tests for `openkb okf compile-dir` - batch compilation.

Covers (#7): flat/wechat/auto produce multiple independent .okf.zip files.
(#8): existing outputs are skipped by default; --overwrite regenerates.
(#9): one file failing does not abort the others; batch_report records the
failed entry. Also asserts no shared context between files (independent zips).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from click.testing import CliRunner

from openkb.cli import cli


def _make_md(path: Path, title: str) -> None:
    path.write_text(f"# {title}\n\n## Body\n\ncontent for {title}.\n", encoding="utf-8")


def test_flat_mode_multiple_zips(tmp_path):
    indir = tmp_path / "in"
    indir.mkdir()
    for i in range(3):
        _make_md(indir / f"a{i}.md", f"Doc {i}")
    outdir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "okf",
            "compile-dir",
            str(indir),
            "--out",
            str(outdir),
            "--no-llm",
            "--mode",
            "flat",
            "--no-recursive",
        ],
    )
    assert result.exit_code == 0, result.output
    zips = sorted(outdir.glob("*.okf.zip"))
    assert len(zips) == 3
    # each is an independent, valid bundle
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            assert "manifest.json" in zf.namelist()


def test_wechat_mode_picks_main_md_per_dir(tmp_path):
    indir = tmp_path / "in"
    indir.mkdir()
    # two article dirs, each with a main .md + assets/
    for n in ("art1", "art2"):
        d = indir / n
        d.mkdir()
        _make_md(d / "index.md", n)
        (d / "assets").mkdir()
    # a stray top-level md that wechat must NOT pick
    _make_md(indir / "stray.md", "Stray")
    outdir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["okf", "compile-dir", str(indir), "--out", str(outdir), "--no-llm", "--mode", "wechat"],
    )
    assert result.exit_code == 0, result.output
    zips = sorted(p.name for p in outdir.glob("*.okf.zip"))
    # exactly the two article zips (index stem), not stray
    assert "index.okf.zip" in zips
    assert len(zips) == 2
    assert "stray.okf.zip" not in zips


def test_auto_mode_falls_back_to_flat(tmp_path):
    indir = tmp_path / "in"
    indir.mkdir()
    _make_md(indir / "x.md", "X")
    _make_md(indir / "y.md", "Y")
    outdir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["okf", "compile-dir", str(indir), "--out", str(outdir), "--no-llm", "--mode", "auto"]
    )
    assert result.exit_code == 0, result.output
    assert len(list(outdir.glob("*.okf.zip"))) == 2


def test_existing_output_skipped_by_default(tmp_path):
    indir = tmp_path / "in"
    indir.mkdir()
    _make_md(indir / "a.md", "A")
    outdir = tmp_path / "out"
    runner = CliRunner()
    r1 = runner.invoke(cli, ["okf", "compile-dir", str(indir), "--out", str(outdir), "--no-llm"])
    assert r1.exit_code == 0
    first = (outdir / "a.okf.zip").read_bytes()
    # rerun: skipped by default
    r2 = runner.invoke(cli, ["okf", "compile-dir", str(indir), "--out", str(outdir), "--no-llm"])
    assert r2.exit_code == 0
    assert (outdir / "a.okf.zip").read_bytes() == first  # unchanged


def test_overwrite_regenerates(tmp_path):
    indir = tmp_path / "in"
    indir.mkdir()
    _make_md(indir / "a.md", "A")
    outdir = tmp_path / "out"
    runner = CliRunner()
    runner.invoke(cli, ["okf", "compile-dir", str(indir), "--out", str(outdir), "--no-llm"])
    runner.invoke(
        cli, ["okf", "compile-dir", str(indir), "--out", str(outdir), "--no-llm", "--overwrite"]
    )
    # overwrite path: file still present and valid
    assert (outdir / "a.okf.zip").exists()


def test_one_failure_does_not_abort_others(tmp_path):
    indir = tmp_path / "in"
    indir.mkdir()
    _make_md(indir / "good1.md", "G1")
    _make_md(indir / "good2.md", "G2")
    # invalid utf-8 -> decode fails
    (indir / "bad.md").write_bytes(b"\xff\xfe# Bad\n")
    outdir = tmp_path / "out"
    report = tmp_path / "report.json"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "okf",
            "compile-dir",
            str(indir),
            "--out",
            str(outdir),
            "--no-llm",
            "--report",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    # the two good files succeeded
    assert (outdir / "good1.okf.zip").exists()
    assert (outdir / "good2.okf.zip").exists()
    assert not (outdir / "bad.okf.zip").exists()
    # batch_report records the failed entry
    data = json.loads(report.read_text())
    statuses = {r["status"] for r in data["results"]}
    assert "failed" in statuses
    assert "ok" in statuses
    assert data["failed"] == 1
    assert data["ok"] == 2


def test_batch_report_written_by_default(tmp_path):
    indir = tmp_path / "in"
    indir.mkdir()
    _make_md(indir / "a.md", "A")
    outdir = tmp_path / "out"
    runner = CliRunner()
    runner.invoke(cli, ["okf", "compile-dir", str(indir), "--out", str(outdir), "--no-llm"])
    # default report path is <out>/batch_report.json
    default_report = outdir / "batch_report.json"
    assert default_report.exists()
    data = json.loads(default_report.read_text())
    assert data["total"] == 1
    assert data["ok"] == 1


def test_zips_are_independent_no_shared_context(tmp_path):
    """Two articles compile to two zips with distinct content (no cross-contamination)."""
    indir = tmp_path / "in"
    indir.mkdir()
    _make_md(indir / "alpha.md", "Alpha")
    _make_md(indir / "beta.md", "Beta")
    outdir = tmp_path / "out"
    runner = CliRunner()
    runner.invoke(cli, ["okf", "compile-dir", str(indir), "--out", str(outdir), "--no-llm"])
    with zipfile.ZipFile(outdir / "alpha.okf.zip") as zf:
        alpha_article = zf.read("sources/article.md").decode()
    with zipfile.ZipFile(outdir / "beta.okf.zip") as zf:
        beta_article = zf.read("sources/article.md").decode()
    assert "Alpha" in alpha_article and "Beta" not in alpha_article
    assert "Beta" in beta_article and "Alpha" not in beta_article


def test_markdown_extension_discovered_by_default(tmp_path):
    """flat mode with the default *.md glob must also pick up .markdown files."""
    indir = tmp_path / "in"
    indir.mkdir()
    _make_md(indir / "a.md", "A")
    (indir / "b.markdown").write_text("# B\n\n## S\n\nbody\n", encoding="utf-8")
    outdir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["okf", "compile-dir", str(indir), "--out", str(outdir), "--no-llm", "--mode", "flat"]
    )
    assert result.exit_code == 0, result.output
    assert (outdir / "a.okf.zip").exists()
    assert (outdir / "b.okf.zip").exists()


def test_wechat_ambiguous_dir_skipped_not_guessed(tmp_path):
    """A wechat dir with multiple .md and no dir-stem match is skipped, not guessed."""
    indir = tmp_path / "in"
    indir.mkdir()
    # clean dir: one md matching the dir name -> selected
    d1 = indir / "art1"
    d1.mkdir()
    _make_md(d1 / "art1.md", "Art1")
    (d1 / "assets").mkdir()
    # ambiguous dir: two md, neither matches dir name
    d2 = indir / "ambiguous"
    d2.mkdir()
    (d2 / "a.md").write_text("# A\n\n## S\n\nx\n", encoding="utf-8")
    (d2 / "b.md").write_text("# B\n\n## S\n\nx\n", encoding="utf-8")
    outdir = tmp_path / "out"
    report = tmp_path / "report.json"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "okf",
            "compile-dir",
            str(indir),
            "--out",
            str(outdir),
            "--no-llm",
            "--mode",
            "wechat",
            "--report",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    # the clean dir compiled; the ambiguous dir was skipped (no zip for it)
    assert (outdir / "art1.okf.zip").exists()
    data = json.loads(report.read_text(encoding="utf-8"))
    skip = [r for r in data["results"] if r["status"] == "skipped"]
    assert any(r["input"].endswith("ambiguous") for r in skip), skip


def test_batch_report_no_absolute_paths(tmp_path):
    """batch_report.json input/output paths are relative to input/out dirs."""
    indir = tmp_path / "in"
    indir.mkdir()
    _make_md(indir / "a.md", "A")
    outdir = tmp_path / "out"
    report = tmp_path / "report.json"
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "okf",
            "compile-dir",
            str(indir),
            "--out",
            str(outdir),
            "--no-llm",
            "--report",
            str(report),
        ],
    )
    data = json.loads(report.read_text(encoding="utf-8"))
    for r in data["results"]:
        # relative -> no drive letter, no leading slash
        assert ":" not in r["input"], r
        assert not r["input"].startswith("/"), r
        assert r["output"] is None or (":" not in r["output"]), r


def test_exception_with_api_key_redacted_from_report(tmp_path, monkeypatch):
    """An LLM error echoing the api_key must be redacted in the batch report.

    Patches ``compile_one`` to raise an exception whose message carries the
    api_key, so the test is deterministic (no real network call) and fast.
    The redaction gate in ``compile_one``'s except clause must strip the key
    from the recorded error before it reaches ``batch_report.json``.
    """
    import openkb.okf.compiler as okf_compiler

    secret = "sk-SECRET-XYZ-9876543210"

    def _boom(input_md, out, opts):
        raise RuntimeError(f"auth failed for key {secret}")

    monkeypatch.setattr(okf_compiler, "compile_one", _boom)

    indir = tmp_path / "in"
    indir.mkdir()
    _make_md(indir / "a.md", "A")
    outdir = tmp_path / "out"
    report = tmp_path / "report.json"
    runner = CliRunner()
    # Pass the secret as the api_key so the redaction gate has it to strip.
    runner.invoke(
        cli,
        [
            "okf",
            "compile-dir",
            str(indir),
            "--out",
            str(outdir),
            "--api-key",
            secret,
            "--model",
            "m",
            "--report",
            str(report),
        ],
    )
    data = json.loads(report.read_text(encoding="utf-8"))
    blob = json.dumps(data)
    assert secret not in blob, f"api_key leaked into batch report: {blob}"
    # the failed result recorded an error (redacted)
    failed = [r for r in data["results"] if r["status"] == "failed"]
    assert failed, data
