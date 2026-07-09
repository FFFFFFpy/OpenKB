"""Tests for openkb.okf.assets - image copy + section path rewrite.

Covers (#2): existing image copied to assets/images and section path
rewritten to ../assets/images/<name>. (#3): missing image does not fail;
the warning is recorded and counts.missing_assets increments. Plus a
basename-collision disambiguation check.
"""

from __future__ import annotations

from openkb.okf.assets import collect_images, count_missing


def test_existing_image_copied_and_rewritten(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "logo.png").write_bytes(b"PNG")
    images_dir = tmp_path / "bundle" / "assets" / "images"

    bodies = ["## S\n\n![logo](logo.png)\n"]
    rewritten, refs, warnings = collect_images(bodies, src_dir, images_dir)

    assert "../assets/images/logo.png" in rewritten[0]
    assert len(refs) == 1
    assert refs[0].found is True
    assert refs[0].dest_name == "logo.png"
    assert warnings == []
    assert (images_dir / "logo.png").read_bytes() == b"PNG"
    assert count_missing(refs) == 0


def test_missing_image_does_not_fail(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    images_dir = tmp_path / "bundle" / "assets" / "images"

    bodies = ["## S\n\n![missing](nope.png)\n"]
    rewritten, refs, warnings = collect_images(bodies, src_dir, images_dir)

    # the link is left unchanged - no rewrite, no copy, no crash
    assert "![missing](nope.png)" in rewritten[0]
    assert "../assets/images/" not in rewritten[0]
    assert len(refs) == 1
    assert refs[0].found is False
    assert refs[0].dest_name == ""
    assert any("nope.png" in w for w in warnings)
    assert count_missing(refs) == 1
    # no images dir created (nothing copied)
    assert not images_dir.exists()


def test_missing_image_recorded_in_manifest_counts(tmp_path):
    # The compiler surfaces missing assets as counts.missing_assets; verify
    # count_missing is the source the manifest uses.
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "ok.png").write_bytes(b"PNG")
    images_dir = tmp_path / "bundle" / "assets" / "images"

    bodies = ["![ok](ok.png) ![gone](gone.png)\n"]
    _rewritten, refs, _warnings = collect_images(bodies, src_dir, images_dir)
    assert count_missing(refs) == 1  # exactly one missing
    assert sum(1 for r in refs if r.found) == 1


def test_basename_collision_disambiguated(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "logo.png").write_bytes(b"ROOT")
    sub = src_dir / "sub"
    sub.mkdir()
    (sub / "logo.png").write_bytes(b"NESTED")
    images_dir = tmp_path / "bundle" / "assets" / "images"

    bodies = ["![a](logo.png) ![b](sub/logo.png)\n"]
    rewritten, refs, warnings = collect_images(bodies, src_dir, images_dir)

    assert warnings == []
    # both copied under different names, both point at ../assets/images/<name>
    dests = {r.dest_name for r in refs if r.found}
    assert len(dests) == 2
    assert "logo.png" in dests  # the first one keeps the bare name
    # the second carries an 8-hex prefix (e.g. ebea2706_logo.png)
    other_name = (dests - {"logo.png"}).pop()
    assert other_name.endswith("_logo.png")
    assert len(other_name.split("_")[0]) == 8  # 8-hex prefix
    # bytes preserved (no clobber)
    files = {p.name: p.read_bytes() for p in images_dir.iterdir()}
    assert len(files) == 2
    assert set(files.values()) == {b"ROOT", b"NESTED"}
    # both links rewritten
    assert rewritten[0].count("../assets/images/") == 2


def test_same_image_reused_not_duplicated(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "shared.png").write_bytes(b"IMG")
    images_dir = tmp_path / "bundle" / "assets" / "images"

    bodies = ["![a](shared.png)\n\n![b](shared.png)\n"]
    _rewritten, refs, _warnings = collect_images(bodies, src_dir, images_dir)
    # referenced twice, copied once
    assert len([r for r in refs if r.found]) == 2
    assert len(list(images_dir.iterdir())) == 1
    assert all(r.dest_name == "shared.png" for r in refs)


def test_http_and_data_uris_left_untouched(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    images_dir = tmp_path / "bundle" / "assets" / "images"

    bodies = [
        "![remote](https://example.com/a.png)\n"
        "![data](data:image/png;base64,iVBOR=)\n"
        "![local](local.png)\n"
    ]
    rewritten, refs, warnings = collect_images(bodies, src_dir, images_dir)
    # http + data URIs are not relative refs -> not copied, not rewritten, not missing
    assert "https://example.com/a.png" in rewritten[0]
    assert "data:image/png;base64,iVBOR=" in rewritten[0]
    # local.png is missing -> warning, link left
    assert "![local](local.png)" in rewritten[0]
    assert any("local.png" in w for w in warnings)
    assert count_missing(refs) == 1


def test_path_escape_rejected(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    images_dir = tmp_path / "bundle" / "assets" / "images"

    bodies = ["![evil](../../etc/passwd)\n"]
    _rewritten, refs, warnings = collect_images(bodies, src_dir, images_dir)
    # escaped path is dropped (not treated as found, not as missing-file)
    assert all(not r.found for r in refs)
    assert any("escapes source dir" in w for w in warnings)
