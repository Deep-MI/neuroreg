"""Tests for the MGH tag section: the talairach transform path and the colour table.

Fixtures are packed by hand with ``struct`` following FreeSurfer's layout
(``utils/fstagsio.cpp``, ``utils/colortab.cpp``) rather than produced by the
module's own encoder, so the encoder is checked against the format and not
against itself. The tag mix mirrors what FreeSurfer writes into ``filled.mgz``:
xform, PE direction, field strength, colour table and command lines.
"""

from __future__ import annotations

import gzip
import os
import struct
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neuroreg.cli.mri import main as mri_main
from neuroreg.image import ColorTable, ColorTableEntry, read_color_lut, read_mgh_tags, write_mgh_tags
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
    path.write_bytes(gzip.compress(raw) if path.suffix == ".mgz" else raw)
    return path


def _raw(path: Path) -> bytes:
    data = path.read_bytes()
    return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data


# ── reading ───────────────────────────────────────────────────────────────────


def test_reads_every_tag_of_a_freesurfer_style_file(tmp_path: Path):
    path = _write(tmp_path / "filled.mgz", _body() + XFORM_TAG + PEDIR_TAG + FIELD_TAG + CTAB_TAG + CMD_TAGS)

    tags = read_mgh_tags(path)

    assert [tag for tag, _ in tags.tags] == [31, 41, 43, 1, 3, 3]
    assert tags.xform == "/old/subject/mri/transforms/talairach.xfm"
    assert tags.command_lines == ["mri_normalize -mprage in.mgz out.mgz", "mri_fill -a log in.mgz out.mgz"]
    table = tags.color_table
    assert table is not None
    assert table.fname == "SubCorticalMassLUT.txt"
    assert table.entries == {
        0: ColorTableEntry("Unknown", (0, 0, 0), 0),
        127: ColorTableEntry("Right-Mass", (0, 180, 0), 0),
        255: ColorTableEntry("Left-Mass", (0, 0, 180), 0),
    }
    assert tags.trailer == b""


def test_a_volume_nibabel_wrote_has_no_tags(tmp_path: Path):
    path = _write(tmp_path / "plain.mgz", _body())

    tags = read_mgh_tags(path)

    assert tags.tags == []
    assert tags.xform is None
    assert tags.color_table is None


def test_reads_the_legacy_xform_tag(tmp_path: Path):
    # Tag 30 carries an int32 length that counts the terminating NUL.
    legacy = struct.pack(">ii", 30, 13) + b"talairach.xfm"
    path = _write(tmp_path / "legacy.mgh", _body() + legacy)

    assert read_mgh_tags(path).xform == "talairach.xfm"


def test_decodes_a_version_1_colour_table():
    # Version 1 starts with the entry count and leaves label values implicit.
    v1 = struct.pack(">ii", 2, 4) + b"lut\0"
    v1 += struct.pack(">i", 5) + b"zero\0" + struct.pack(">4i", 1, 2, 3, 0)
    v1 += struct.pack(">i", 4) + b"one\0" + struct.pack(">4i", 4, 5, 6, 7)

    table, end = decode_color_table(v1)

    assert end == len(v1)
    assert table.fname == "lut"
    assert table.entries == {0: ColorTableEntry("zero", (1, 2, 3), 0), 1: ColorTableEntry("one", (4, 5, 6), 7)}


def test_refuses_a_warp_field_tag_it_cannot_step_over(tmp_path: Path):
    # Tags 10 to 12 carry no length field, so nothing after them can be found.
    path = _write(tmp_path / "warp.mgz", _body() + struct.pack(">i", 10) + b"opaque geometry")

    with pytest.raises(ValueError, match="TAG_GCAMORPH_GEOM"):
        read_mgh_tags(path)


def test_refuses_a_truncated_tag(tmp_path: Path):
    path = _write(tmp_path / "cut.mgz", _body() + _sized(3, b"a command line\0")[:-5])

    with pytest.raises(ValueError, match="truncated"):
        read_mgh_tags(path)


def test_keeps_whatever_follows_a_zero_tag(tmp_path: Path):
    # FreeSurfer stops reading at a zero id; a rewrite must not lose those bytes.
    trailer = struct.pack(">i", 0) + b"ignored by FreeSurfer"
    source = _write(tmp_path / "zero.mgz", _body() + PEDIR_TAG + trailer)
    out = tmp_path / "out.mgz"

    write_mgh_tags(source, out, xform="transforms/talairach.xfm")

    assert _raw(out) == _body() + _sized(31, b"transforms/talairach.xfm\0") + PEDIR_TAG + trailer


# ── writing ───────────────────────────────────────────────────────────────────


def test_replacing_the_xform_changes_nothing_else(tmp_path: Path):
    # The whole point of a byte-level rewrite: header, voxels, scan parameters
    # and every other tag survive exactly, and the xform keeps its position.
    rest = PEDIR_TAG + FIELD_TAG + CTAB_TAG + CMD_TAGS
    source = _write(tmp_path / "filled.mgz", _body() + XFORM_TAG + rest)
    out = tmp_path / "tagged.mgz"

    write_mgh_tags(source, out, xform="transforms/talairach.xfm")

    assert _raw(out) == _body() + _sized(31, b"transforms/talairach.xfm\0") + rest


def test_a_rewrite_without_changes_is_byte_identical(tmp_path: Path):
    raw = _body() + XFORM_TAG + PEDIR_TAG + FIELD_TAG + CTAB_TAG + CMD_TAGS
    source = _write(tmp_path / "filled.mgz", raw)
    out = tmp_path / "copy.mgz"

    write_mgh_tags(source, out)

    assert _raw(out) == raw


def test_a_new_xform_goes_first_and_a_new_colour_table_last(tmp_path: Path):
    # FreeSurfer writes the xform first; the reader does not care about order.
    source = _write(tmp_path / "in.mgz", _body() + PEDIR_TAG + CMD_TAGS)
    out = tmp_path / "out.mgz"
    table = ColorTable({0: ColorTableEntry("Unknown", (0, 0, 0)), 2: ColorTableEntry("WM", (245, 245, 245))}, "lut")

    write_mgh_tags(source, out, xform="transforms/talairach.xfm", color_table=table)

    tags = read_mgh_tags(out)
    assert [tag for tag, _ in tags.tags] == [31, 41, 3, 3, 1]
    assert tags.color_table == table


def test_replacing_the_colour_table_keeps_its_position(tmp_path: Path):
    source = _write(tmp_path / "in.mgz", _body() + PEDIR_TAG + CTAB_TAG + CMD_TAGS)
    out = tmp_path / "out.mgz"
    table = ColorTable({7: ColorTableEntry("Seven", (7, 7, 7))}, "other.txt")

    write_mgh_tags(source, out, color_table=table)

    assert [tag for tag, _ in read_mgh_tags(out).tags] == [41, 1, 3, 3]
    assert read_mgh_tags(out).color_table == table


def test_replacing_the_xform_leaves_no_stale_duplicate(tmp_path: Path):
    # FreeSurfer applies tags in order, so a leftover legacy tag after the new
    # one would win and resurrect the old path.
    legacy = struct.pack(">ii", 30, 8) + b"old.xfm\0"
    source = _write(tmp_path / "in.mgz", _body() + XFORM_TAG + PEDIR_TAG + legacy)
    out = tmp_path / "out.mgz"

    write_mgh_tags(source, out, xform="new.xfm")

    tags = read_mgh_tags(out)
    assert [tag for tag, _ in tags.tags] == [31, 41]
    assert tags.xform == "new.xfm"


def test_the_volume_still_loads_with_identical_voxels_and_header(tmp_path: Path):
    source = _write(tmp_path / "in.mgz", _body())
    out = tmp_path / "out.mgz"

    write_mgh_tags(source, out, xform="transforms/talairach.xfm")

    before, after = nib.load(str(source)), nib.load(str(out))
    assert np.array_equal(np.asanyarray(after.dataobj), DATA)
    assert np.array_equal(after.affine, before.affine)
    assert after.header.binaryblock == before.header.binaryblock


def test_supplies_missing_scan_parameters_before_the_tags(tmp_path: Path):
    # Tags can only follow the scan parameters, so a file that stops right
    # after its voxels gets the zeros FreeSurfer assumes when they are absent.
    body = _body()
    source = _write(tmp_path / "short.mgh", body[:-20])
    out = tmp_path / "out.mgh"

    write_mgh_tags(source, out, xform="t.xfm")

    assert _raw(out) == body[:-20] + bytes(20) + _sized(31, b"t.xfm\0")


def test_output_compression_follows_the_extension(tmp_path: Path):
    source = _write(tmp_path / "in.mgz", _body())

    write_mgh_tags(source, tmp_path / "out.mgh", xform="t.xfm")
    write_mgh_tags(source, tmp_path / "out.mgz", xform="t.xfm")

    assert (tmp_path / "out.mgh").read_bytes()[:2] != b"\x1f\x8b"
    assert (tmp_path / "out.mgz").read_bytes()[:2] == b"\x1f\x8b"
    assert _raw(tmp_path / "out.mgh") == _raw(tmp_path / "out.mgz")


def test_tags_in_place_and_keeps_the_file_mode(tmp_path: Path):
    # The new file is created private; the replaced file's mode has to carry
    # over, or tagging a volume in a shared subject directory hides it.
    path = _write(tmp_path / "nu.mgz", _body())
    os.chmod(path, 0o644)

    write_mgh_tags(path, path, xform="transforms/talairach.xfm")

    assert read_mgh_tags(path).xform == "transforms/talairach.xfm"
    assert path.stat().st_mode & 0o777 == 0o644
    assert [p.name for p in tmp_path.iterdir()] == ["nu.mgz"]


def test_refuses_a_non_mgh_output(tmp_path: Path):
    source = _write(tmp_path / "in.mgz", _body())

    with pytest.raises(ValueError, match="only be stored in MGH"):
        write_mgh_tags(source, tmp_path / "out.nii.gz", xform="t.xfm")


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

    assert table.fname == str(lut)
    assert list(table.entries) == [0, 2, 17]
    assert table.entries[2] == ColorTableEntry("Left-Cerebral-White-Matter", (245, 245, 245), 0)
    assert table.entries[17].transparency == 10


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("3 Short 1 2 3\n", "expected 'index name R G B A'"),
        ("3 Bad 1 two 3 0\n", "must be integers"),
        ("-1 Negative 1 2 3 0\n", "must not be negative"),
        ("0 Unknown 0 0 0 0\n0 Again 1 1 1 0\n", "defined twice"),
        ("# only a comment\n", "no colour table entries"),
    ],
)
def test_rejects_a_malformed_lut(tmp_path: Path, line: str, message: str):
    lut = tmp_path / "lut.txt"
    lut.write_text(line)

    with pytest.raises(ValueError, match=message):
        read_color_lut(lut)


# ── CLI ───────────────────────────────────────────────────────────────────────


@pytest.fixture
def subject(tmp_path: Path) -> Path:
    """A subject's mri directory holding nu.mgz and transforms/talairach.xfm."""
    mri = tmp_path / "subject" / "mri"
    (mri / "transforms").mkdir(parents=True)
    (mri / "transforms" / "talairach.xfm").write_text("MNI Transform File\n")
    _write(mri / "nu.mgz", _body() + PEDIR_TAG)
    return mri


def test_cli_records_the_standard_transform_relative_to_the_output(subject: Path, capsys):
    out = subject / "tagged.mgz"

    mri_main(["tag", str(subject / "nu.mgz"), str(out), "--xform", str(subject / "transforms" / "talairach.xfm")])

    assert read_mgh_tags(out).xform == os.path.join("transforms", "talairach.xfm")
    captured = capsys.readouterr()
    assert "xform tag: transforms/talairach.xfm" in captured.out
    assert "WARNING" not in captured.err


def test_cli_relative_path_does_not_depend_on_the_current_directory(subject: Path, monkeypatch):
    # The path is relative to where the volume lives, the only place it can be
    # resolved from later, not to wherever the command happened to run.
    monkeypatch.chdir(subject.parent.parent)
    xform = os.path.relpath(subject / "transforms" / "talairach.xfm")

    mri_main(["tag", os.path.relpath(subject / "nu.mgz"), os.path.relpath(subject / "nu.mgz"), "--xform", xform])

    assert read_mgh_tags(subject / "nu.mgz").xform == os.path.join("transforms", "talairach.xfm")


def test_cli_abs_xform_tag_records_the_absolute_path(subject: Path):
    xform = subject / "transforms" / "talairach.xfm"

    mri_main(["tag", str(subject / "nu.mgz"), str(subject / "nu.mgz"), "--xform", str(xform), "--abs-xform-tag"])

    assert read_mgh_tags(subject / "nu.mgz").xform == str(xform.resolve())


def test_cli_warns_when_freesurfer_would_not_find_a_relative_path(subject: Path, capsys):
    other = subject / "transforms" / "other.xfm"
    other.write_text("MNI Transform File\n")

    mri_main(["tag", str(subject / "nu.mgz"), str(subject / "nu.mgz"), "--xform", str(other)])

    assert read_mgh_tags(subject / "nu.mgz").xform == os.path.join("transforms", "other.xfm")
    assert "--abs-xform-tag" in capsys.readouterr().err


def test_cli_records_a_missing_transform_with_a_warning(subject: Path, capsys):
    # Like mri_add_xform_to_header -c: the transform may be written later.
    missing = subject / "transforms" / "talairach.xfm"
    missing.unlink()

    mri_main(["tag", str(subject / "nu.mgz"), str(subject / "nu.mgz"), "--xform", str(missing)])

    assert read_mgh_tags(subject / "nu.mgz").xform == os.path.join("transforms", "talairach.xfm")
    assert "does not exist" in capsys.readouterr().err


def test_cli_embeds_a_colour_table_and_info_reports_both_tags(subject: Path, capsys):
    lut = subject.parent / "lut.txt"
    lut.write_text("0 Unknown 0 0 0 0\n2 Left-Cerebral-White-Matter 245 245 245 0\n")
    path = str(subject / "nu.mgz")

    mri_main(["tag", path, path, "--xform", str(subject / "transforms" / "talairach.xfm"), "--ctab", str(lut)])
    capsys.readouterr()
    mri_main(["info", path])

    info = capsys.readouterr().out
    assert "talairach xfm: transforms/talairach.xfm" in info
    assert "color table: 2 entries" in info
    assert [tag for tag, _ in read_mgh_tags(path).tags] == [31, 41, 1]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["tag", "in.mgz", "out.mgz"], "needs --xform and/or --ctab"),
        (["tag", "in.mgz", "out.mgz", "--ctab", "lut.txt", "--abs-xform-tag"], "only says how to record --xform"),
        (["tag", "in.nii.gz", "out.mgz", "--xform", "t.xfm"], "reads MGH/MGZ input only"),
        (["tag", "in.mgz", "out.nii.gz", "--xform", "t.xfm"], "needs a .mgz or .mgh extension"),
    ],
)
def test_cli_rejects_requests_it_cannot_honour(argv: list[str], message: str, capsys):
    with pytest.raises(SystemExit):
        mri_main(argv)

    assert message in capsys.readouterr().err
