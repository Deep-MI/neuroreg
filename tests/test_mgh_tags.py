"""Tests for the MGH tag section and ``mri tag``.

Fixtures are packed by hand with ``struct`` following FreeSurfer's layout
(``utils/fstagsio.cpp``, ``utils/colortab.cpp``) rather than produced by the
module's own encoder, so the encoder is checked against the format and not
against itself. The tag mix mirrors what FreeSurfer writes into ``filled.mgz``:
xform, PE direction, field strength, colour table and command lines.
"""

from __future__ import annotations

import gzip
import logging
import os
import struct
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neuroreg.cli.mri import main as mri_main
from neuroreg.image import (
    ColorTable,
    ColorTableEntry,
    read_color_lut,
    read_mgh_tags,
    relative_xform_path,
    write_mgh_tags,
)
from neuroreg.image.mgh_tags import decode_color_table, encode_color_table

DATA = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)


def _body() -> bytes:
    """Header, voxels and scan parameters of a small volume, as nibabel writes them."""
    affine = np.array([[-1.0, 0, 0, 1], [0, 0, 1, -2], [0, -1, 0, 3], [0, 0, 0, 1]])
    return nib.MGHImage(DATA, affine).to_bytes()


def _sized(tag: int, payload: bytes) -> bytes:
    return struct.pack(">iq", tag, len(payload)) + payload


def _ctab_v2(fname: bytes, entries: list[tuple[int, bytes, int, int, int, int]], nentries: int) -> bytes:
    out = struct.pack(">iii", -2, nentries, len(fname) + 1) + fname + b"\0" + struct.pack(">i", len(entries))
    for index, name, r, g, b, t in entries:
        out += struct.pack(">ii", index, len(name) + 1) + name + b"\0" + struct.pack(">4i", r, g, b, t)
    return out


CTAB_ENTRIES = [(0, b"Unknown", 0, 0, 0, 0), (127, b"Right-Mass", 0, 180, 0, 0), (255, b"Left-Mass", 0, 0, 180, 0)]
CTAB = _ctab_v2(b"SubCorticalMassLUT.txt", CTAB_ENTRIES, 256)
XFORM_TAG = _sized(31, b"/old/subject/mri/transforms/talairach.xfm\0")
PEDIR_TAG = _sized(41, b"UNKNOWN")
FIELD_TAG = _sized(43, b"\0\0\0\0")
CTAB_TAG = struct.pack(">i", 1) + CTAB
CMD_TAGS = _sized(3, b"mri_normalize -mprage in.mgz out.mgz\0") + _sized(3, b"mri_fill -a log in.mgz out.mgz\0")


def _write(path: Path, raw: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(raw) if path.suffix == ".mgz" else raw)
    return path


def _raw(path: Path) -> bytes:
    data = path.read_bytes()
    return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data


def _tagged(path: Path, *, xform: bytes | None = None, ctab: bytes | None = None) -> Path:
    """Write a volume holding the given xform path and/or binary colour table."""
    tags = b"" if xform is None else _sized(31, xform + b"\0")
    tags += PEDIR_TAG
    if ctab is not None:
        tags += struct.pack(">i", 1) + ctab
    return _write(path, _body() + tags)


# ── reading and writing MGH tags ──────────────────────────────────────────────


def test_reads_every_tag_of_a_freesurfer_style_file(tmp_path: Path):
    path = _write(tmp_path / "filled.mgz", _body() + XFORM_TAG + PEDIR_TAG + FIELD_TAG + CTAB_TAG + CMD_TAGS)

    tags = read_mgh_tags(path)

    assert [tag for tag, _ in tags.tags] == [31, 41, 43, 1, 3, 3]
    assert tags.xform == "/old/subject/mri/transforms/talairach.xfm"
    assert tags.command_lines == ["mri_normalize -mprage in.mgz out.mgz", "mri_fill -a log in.mgz out.mgz"]
    assert tags.color_table.fname == "SubCorticalMassLUT.txt"
    assert tags.color_table.entries[127] == ColorTableEntry("Right-Mass", (0, 180, 0), 0)


def test_the_last_of_duplicate_tags_is_in_effect(tmp_path: Path):
    # FreeSurfer applies tags in file order, so a later duplicate overrides
    # an earlier one; reading the first would compare and copy a stale value.
    newer_ctab = _ctab_v2(b"newer.txt", [(0, b"Unknown", 0, 0, 0, 0)], 1)
    raw = _body() + _sized(31, b"old.xfm\0") + CTAB_TAG + _sized(31, b"new.xfm\0") + struct.pack(">i", 1) + newer_ctab
    path = _write(tmp_path / "nu.mgz", raw)

    tags = read_mgh_tags(path)

    assert tags.xform == "new.xfm"
    assert tags.color_table.fname == "newer.txt"


def test_decodes_a_version_1_colour_table():
    # Version 1 starts with the entry count and leaves label values implicit;
    # its length has to be right, or every tag after it is misread.
    v1 = struct.pack(">ii", 2, 4) + b"lut\0"
    v1 += struct.pack(">i", 5) + b"zero\0" + struct.pack(">4i", 1, 2, 3, 0)
    v1 += struct.pack(">i", 4) + b"one\0" + struct.pack(">4i", 4, 5, 6, 7)

    table, end = decode_color_table(v1)

    assert end == len(v1)
    assert table.entries == {0: ColorTableEntry("zero", (1, 2, 3), 0), 1: ColorTableEntry("one", (4, 5, 6), 7)}


@pytest.mark.parametrize(
    ("tags", "message"),
    [
        # Tags 10 to 12 carry no length field, so nothing after them can be found.
        (struct.pack(">i", 10) + b"opaque geometry", "TAG_GCAMORPH_GEOM"),
        (_sized(3, b"a command line\0")[:-5], "truncated"),
    ],
)
def test_refuses_tags_it_cannot_step_over(tmp_path: Path, tags: bytes, message: str):
    path = _write(tmp_path / "bad.mgz", _body() + tags)

    with pytest.raises(ValueError, match=message):
        read_mgh_tags(path)


def test_replacing_the_xform_changes_nothing_else(tmp_path: Path):
    # The point of a byte-level rewrite: header, voxels, scan parameters and
    # every other tag survive exactly.
    rest = PEDIR_TAG + FIELD_TAG + CTAB_TAG + CMD_TAGS
    source = _write(tmp_path / "filled.mgz", _body() + XFORM_TAG + rest)
    out = tmp_path / "tagged.mgz"

    write_mgh_tags(source, out, xform="transforms/talairach.xfm")

    assert _raw(out) == _body() + _sized(31, b"transforms/talairach.xfm\0") + rest


def test_replacing_the_xform_leaves_no_stale_duplicate(tmp_path: Path):
    # FreeSurfer applies tags in order, so a leftover legacy tag after the new
    # one would win and resurrect the old path.
    legacy = struct.pack(">ii", 30, 8) + b"old.xfm\0"
    source = _write(tmp_path / "in.mgz", _body() + XFORM_TAG + PEDIR_TAG + legacy)

    write_mgh_tags(source, tmp_path / "out.mgz", xform="new.xfm")

    tags = read_mgh_tags(tmp_path / "out.mgz")
    assert [tag for tag, _ in tags.tags] == [31, 41]
    assert tags.xform == "new.xfm"


def test_output_compression_follows_the_extension(tmp_path: Path):
    source = _write(tmp_path / "in.mgz", _body())

    write_mgh_tags(source, tmp_path / "out.mgh", xform="t.xfm")
    write_mgh_tags(source, tmp_path / "out.mgz", xform="t.xfm")

    assert (tmp_path / "out.mgh").read_bytes()[:2] != b"\x1f\x8b"
    assert _raw(tmp_path / "out.mgh") == _raw(tmp_path / "out.mgz")


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no POSIX permission bits to keep")
def test_tags_in_place_and_keeps_the_file_mode(tmp_path: Path):
    # The new file is created private; the replaced file's mode has to carry
    # over, or tagging a volume in a shared subject directory hides it.
    path = _write(tmp_path / "nu.mgz", _body())
    os.chmod(path, 0o644)

    write_mgh_tags(path, path, xform="transforms/talairach.xfm")

    assert read_mgh_tags(path).xform == "transforms/talairach.xfm"
    assert path.stat().st_mode & 0o777 == 0o644
    assert [p.name for p in tmp_path.iterdir()] == ["nu.mgz"]


def test_a_kept_relative_xform_is_rebased_into_another_directory(tmp_path: Path, caplog):
    # A relative path resolves from the volume's own directory, so a byte copy
    # elsewhere would name another file. FreeSurfer, though, resolves anything
    # but transforms/talairach.xfm from its current directory, which is said.
    source = _tagged(tmp_path / "subject" / "mri" / "nu.mgz", xform=b"transforms/talairach.xfm")
    out = tmp_path / "elsewhere" / "nu.mgz"
    out.parent.mkdir()

    with caplog.at_level(logging.WARNING, logger="neuroreg.image.mgh_tags"):
        write_mgh_tags(source, out, color_table=ColorTable({0: ColorTableEntry("Unknown", (0, 0, 0))}))

    assert read_mgh_tags(out).xform == "../subject/mri/transforms/talairach.xfm"
    assert "--abs-xform-tag" in caplog.text


def test_in_place_leaves_an_unusual_relative_path_alone(tmp_path: Path):
    # Rebasing within one directory would only normalise the spelling, which
    # an in-place rewrite must not do.
    raw = _body() + _sized(31, b"./transforms/../transforms/talairach.xfm\0")
    path = _write(tmp_path / "nu.mgz", raw)

    write_mgh_tags(path, path)

    assert _raw(path) == raw


def test_a_transform_sharing_only_the_root_is_recorded_absolute():
    # Climbing to the root and back down breaks as soon as either side moves.
    target = os.path.abspath(os.path.join(os.sep, "fs_subjects", "bert", "mri", "transforms", "talairach.xfm"))
    out_dir = os.path.abspath(os.path.join(os.sep, "scratch", "run1", "mri"))

    assert relative_xform_path(target, out_dir) == Path(target).as_posix()


# ── colour lookup tables ──────────────────────────────────────────────────────


def test_encodes_a_colour_table_exactly_as_freesurfer_does():
    table = ColorTable(
        {
            0: ColorTableEntry("Unknown", (0, 0, 0)),
            127: ColorTableEntry("Right-Mass", (0, 180, 0)),
            255: ColorTableEntry("Left-Mass", (0, 0, 180)),
        },
        "SubCorticalMassLUT.txt",
    )

    assert encode_color_table(table) == CTAB


def test_reads_an_ascii_lut_like_freesurfer(tmp_path: Path):
    lut = tmp_path / "lut.txt"
    lut.write_text(
        "#$Id: FreeSurferColorLUT.txt\n"
        "\n"
        "#No. Label Name:                R   G   B   A\n"
        "  2  Left-Cerebral-White-Matter 245 245 245 0   1\n"
        "  0  Unknown                    0   0   0   0\n"
        " 17  Left-Hippocampus           220 216 20  10\n"
    )

    table = read_color_lut(lut)

    assert list(table.entries) == [0, 2, 17]
    assert table.entries[2] == ColorTableEntry("Left-Cerebral-White-Matter", (245, 245, 245), 0)
    assert table.entries[17].transparency == 10


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("3 Short 1 2 3\n", "expected 'index name R G B A'"),
        ("0 Unknown 0 0 0 0\n0 Again 1 1 1 0\n", "defined twice"),
        ("# only a comment\n", "no colour table entries"),
    ],
)
def test_rejects_a_malformed_lut(tmp_path: Path, line: str, message: str):
    lut = tmp_path / "lut.txt"
    lut.write_text(line)

    with pytest.raises(ValueError, match=message):
        read_color_lut(lut)


# ── mri tag ───────────────────────────────────────────────────────────────────


@pytest.fixture
def subject(tmp_path: Path) -> Path:
    """A subject's mri directory holding nu.mgz and transforms/talairach.xfm."""
    mri = tmp_path / "subject" / "mri"
    (mri / "transforms").mkdir(parents=True)
    (mri / "transforms" / "talairach.xfm").write_text("MNI Transform File\n")
    _write(mri / "nu.mgz", _body() + PEDIR_TAG)
    return mri


def test_records_the_transform_relative_to_the_output_wherever_it_runs(subject: Path, monkeypatch, capsys):
    # Relative to where the volume lives, the only place it can be resolved
    # from later, not to wherever the command happened to run.
    monkeypatch.chdir(subject.parent.parent)
    nu = os.path.relpath(subject / "nu.mgz")

    mri_main(["tag", nu, nu, "--xform", os.path.relpath(subject / "transforms" / "talairach.xfm")])

    assert read_mgh_tags(subject / "nu.mgz").xform == "transforms/talairach.xfm"
    assert "WARNING" not in capsys.readouterr().err


def test_abs_xform_tag_rewrites_the_same_transform_without_force(subject: Path):
    # The volume already names this transform, relatively. Asking for the
    # absolute form of the same file is a rewrite, not a conflict.
    path = _tagged(subject / "nu.mgz", xform=b"transforms/talairach.xfm")
    xform = subject / "transforms" / "talairach.xfm"

    mri_main(["tag", str(path), str(path), "--xform", str(xform), "--abs-xform-tag"])

    assert read_mgh_tags(path).xform == xform.resolve().as_posix()


def test_warns_when_freesurfer_would_not_find_a_relative_path(subject: Path, capsys):
    other = subject / "transforms" / "other.xfm"
    other.write_text("MNI Transform File\n")

    mri_main(["tag", str(subject / "nu.mgz"), str(subject / "nu.mgz"), "--xform", str(other)])

    assert read_mgh_tags(subject / "nu.mgz").xform == "transforms/other.xfm"
    assert "--abs-xform-tag" in capsys.readouterr().err


def test_records_a_missing_transform_with_a_warning(subject: Path, capsys):
    # Like mri_add_xform_to_header -c: the transform may be written later.
    missing = subject / "transforms" / "talairach.xfm"
    missing.unlink()

    mri_main(["tag", str(subject / "nu.mgz"), str(subject / "nu.mgz"), "--xform", str(missing)])

    assert read_mgh_tags(subject / "nu.mgz").xform == "transforms/talairach.xfm"
    assert "does not exist" in capsys.readouterr().err


def test_embeds_a_colour_table_and_info_reports_both_tags(subject: Path, capsys):
    lut = subject.parent / "lut.txt"
    lut.write_text("0 Unknown 0 0 0 0\n2 Left-Cerebral-White-Matter 245 245 245 0\n")
    path = str(subject / "nu.mgz")

    mri_main(["tag", path, path, "--xform", str(subject / "transforms" / "talairach.xfm"), "--ctab", str(lut)])
    capsys.readouterr()
    mri_main(["info", path])

    info = capsys.readouterr().out
    assert "talairach xfm: transforms/talairach.xfm" in info
    assert "color table: 2 entries" in info


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["tag", "in.mgz", "out.mgz"], "needs --xform/--xform-from and/or --ctab/--ctab-from"),
        (["tag", "in.mgz", "out.mgz", "--ctab", "lut.txt", "--abs-xform-tag"], "only says how to record a transform"),
        (["tag", "in.mgz", "out.mgz", "--xform", "t.xfm", "--xform-from", "nu.mgz"], "not allowed with"),
        (["tag", "in.img", "out.mgz", "--xform", "t.xfm"], "reads MGH or NIfTI input only"),
        (["tag", "in.mgz", "out.nii.gz", "--xform", "t.xfm"], "keeps its format; vol2vol converts"),
    ],
)
def test_rejects_requests_it_cannot_honour(argv: list[str], message: str, capsys):
    with pytest.raises(SystemExit):
        mri_main(argv)

    assert message in capsys.readouterr().err


def test_xform_from_resolves_the_path_where_it_was_recorded(subject: Path, tmp_path: Path):
    # The copied path is relative to nu.mgz's directory, so a volume elsewhere
    # needs it re-expressed relative to its own.
    _tagged(subject / "nu.mgz", xform=b"transforms/talairach.xfm")
    other = _tagged(tmp_path / "other" / "seg.mgz")

    mri_main(["tag", str(other), str(other), "--xform-from", str(subject / "nu.mgz")])

    assert read_mgh_tags(other).xform == "../subject/mri/transforms/talairach.xfm"


def test_xform_from_an_absolute_freesurfer_path_is_recorded_relative(subject: Path):
    # recon-all records absolute paths; copied, they are recorded like --xform.
    _tagged(subject / "nu.mgz", xform=str(subject / "transforms" / "talairach.xfm").encode())
    _tagged(subject / "seg.mgz")

    mri_main(["tag", str(subject / "seg.mgz"), str(subject / "seg.mgz"), "--xform-from", str(subject / "nu.mgz")])

    assert read_mgh_tags(subject / "seg.mgz").xform == "transforms/talairach.xfm"


def test_copying_a_tag_the_source_lacks_is_an_error(subject: Path, capsys):
    with pytest.raises(SystemExit):
        mri_main(["tag", str(subject / "nu.mgz"), str(subject / "out.mgz"), "--xform-from", str(subject / "nu.mgz")])

    assert "records no talairach xform to copy" in capsys.readouterr().err
    assert not (subject / "out.mgz").exists()


def test_refuses_to_replace_a_different_xform_without_force(subject: Path, capsys):
    path = _tagged(subject / "nu.mgz", xform=b"transforms/other.xfm")
    before = path.read_bytes()
    xform = str(subject / "transforms" / "talairach.xfm")

    with pytest.raises(SystemExit):
        mri_main(["tag", str(path), str(path), "--xform", xform])

    assert "--force" in capsys.readouterr().err
    assert path.read_bytes() == before

    mri_main(["tag", str(path), str(path), "--xform", xform, "--force"])

    assert read_mgh_tags(path).xform == "transforms/talairach.xfm"


def test_colour_tables_conflict_by_entries_not_by_lut_file(subject: Path, capsys):
    path = _tagged(subject / "aseg.mgz", ctab=CTAB)
    before = path.read_bytes()
    same = subject.parent / "copy_of_lut.txt"
    same.write_text("0 Unknown 0 0 0 0\n127 Right-Mass 0 180 0 0\n255 Left-Mass 0 0 180 0\n")
    different = subject.parent / "lut.txt"
    different.write_text("0 Unknown 0 0 0 0\n2 Left-Cerebral-White-Matter 245 245 245 0\n")

    # The same entries loaded from another file are the same table.
    mri_main(["tag", str(path), str(path), "--ctab", str(same)])
    assert path.read_bytes() == before

    with pytest.raises(SystemExit):
        mri_main(["tag", str(path), str(path), "--ctab", str(different)])
    assert "embeds a different colour table (3 entries" in capsys.readouterr().err

    mri_main(["tag", str(path), str(path), "--ctab", str(different), "--force"])
    assert list(read_mgh_tags(path).color_table.entries) == [0, 2]


def test_repeating_a_step_leaves_the_file_untouched(subject: Path, capsys):
    # FastSurfer reruns steps; asking for what a volume already holds must
    # neither fail nor rewrite it.
    path = _tagged(subject / "nu.mgz", xform=b"transforms/talairach.xfm")
    stamp = path.stat().st_mtime_ns

    mri_main(["tag", str(path), str(path), "--xform", str(subject / "transforms" / "talairach.xfm")])

    assert "unchanged" in capsys.readouterr().out
    assert path.stat().st_mtime_ns == stamp
