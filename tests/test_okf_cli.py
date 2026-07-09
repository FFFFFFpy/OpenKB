"""Tests for `openkb okf compile` via the Click CLI.

Covers (#4): ``--no-llm`` makes no LLM request and still emits a valid basic
bundle. (#6): ``compile`` on a single file produces a ``.okf.zip``. Uses the
CliRunner against the real ``openkb.cli:cli`` group, so registration is
exercised end-to-end.
"""

from __future__ import annotations

import json
import zipfile
from unittest.mock import patch

from click.testing import CliRunner

from openkb.cli import cli


def _write_md(tmp_path, name="article.md", body="# T\n\n## S\n\nHello.\n"):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_compile_single_file_produces_zip(tmp_path):
    md = _write_md(tmp_path)
    out = tmp_path / "out.okf.zip"
    runner = CliRunner()
    result = runner.invoke(cli, ["okf", "compile", str(md), "--no-llm", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "manifest.json" in names
        assert "okf.yaml" in names
        assert "sources/article.md" in names
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["compiler"]["llm_enabled"] is False
        assert manifest["counts"]["sections"] == 1


def test_no_llm_makes_no_llm_request(tmp_path):
    """--no-llm must not touch litellm at all; the basic bundle still works."""
    md = _write_md(tmp_path, body="# T\n\n## S\n\nbody\n")
    out = tmp_path / "out.okf.zip"
    runner = CliRunner()
    # Patch litellm.completion so the test fails loudly if the no-llm path
    # ever reaches it.
    with patch("litellm.completion") as mock_completion:
        result = runner.invoke(cli, ["okf", "compile", str(md), "--no-llm", "--out", str(out)])
        assert result.exit_code == 0, result.output
        assert mock_completion.call_count == 0
    assert out.exists()


def test_compile_default_out_path(tmp_path):
    """Without --out, the zip lands at <input-stem>.okf.zip next to the input."""
    md = _write_md(tmp_path, name="doc.md")
    runner = CliRunner()
    result = runner.invoke(cli, ["okf", "compile", str(md), "--no-llm"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "doc.okf.zip").exists()


def test_compile_overwrite_required(tmp_path):
    """An existing output is skipped unless --overwrite is passed."""
    md = _write_md(tmp_path)
    out = tmp_path / "out.okf.zip"
    runner = CliRunner()
    r1 = runner.invoke(cli, ["okf", "compile", str(md), "--no-llm", "--out", str(out)])
    assert r1.exit_code == 0
    first = out.read_bytes()
    # second run without --overwrite -> skipped (non-zero exit, [SKIP])
    r2 = runner.invoke(cli, ["okf", "compile", str(md), "--no-llm", "--out", str(out)])
    assert r2.exit_code != 0
    assert "[SKIP]" in r2.output or "exists" in r2.output
    assert out.read_bytes() == first  # untouched
    # --overwrite regenerates
    r3 = runner.invoke(
        cli, ["okf", "compile", str(md), "--no-llm", "--out", str(out), "--overwrite"]
    )
    assert r3.exit_code == 0


def test_workdir_option_is_base_and_child_is_cleaned(tmp_path):
    """--workdir is a staging base; compile must not remove the user base dir."""
    md = _write_md(tmp_path)
    out = tmp_path / "out.okf.zip"
    base = tmp_path / "staging-base"
    base.mkdir()
    (base / "sentinel.txt").write_text("keep me", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["okf", "compile", str(md), "--no-llm", "--out", str(out), "--workdir", str(base)],
    )

    assert result.exit_code == 0, result.output
    assert base.exists()
    assert (base / "sentinel.txt").read_text(encoding="utf-8") == "keep me"
    assert not list(base.glob("okf-compile-*"))


def test_keep_workdir_preserves_child_and_prints_path(tmp_path):
    """--keep-workdir keeps the unique child, not just the base."""
    md = _write_md(tmp_path)
    out = tmp_path / "out.okf.zip"
    base = tmp_path / "staging-base"
    base.mkdir()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "okf",
            "compile",
            str(md),
            "--no-llm",
            "--out",
            str(out),
            "--workdir",
            str(base),
            "--keep-workdir",
        ],
    )

    assert result.exit_code == 0, result.output
    children = list(base.glob("okf-compile-*"))
    assert len(children) == 1
    assert (children[0] / "manifest.json").exists()
    assert str(children[0]) in result.output


def test_compile_rejects_non_markdown_file(tmp_path):
    txt = tmp_path / "not-markdown.txt"
    txt.write_text("plain text", encoding="utf-8")
    out = tmp_path / "out.okf.zip"

    runner = CliRunner()
    result = runner.invoke(cli, ["okf", "compile", str(txt), "--no-llm", "--out", str(out)])

    assert result.exit_code != 0
    assert "markdown" in result.output.lower()
    assert not out.exists()


def test_okf_group_registered():
    """The okf group must be wired into the main cli (regression for add_command)."""
    runner = CliRunner()
    result = runner.invoke(cli, ["okf", "--help"])
    assert result.exit_code == 0
    assert "compile" in result.output
    assert "compile-dir" in result.output
    assert "test-llm" in result.output


def test_test_llm_missing_model_errors(tmp_path):
    """test-llm without a model configured exits non-zero (no key, no call)."""
    runner = CliRunner()
    # No --model, no env: nothing to test.
    result = runner.invoke(cli, ["okf", "test-llm"])
    assert result.exit_code != 0
    assert "model" in result.output.lower()
