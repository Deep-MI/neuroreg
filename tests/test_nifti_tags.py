"""FreeSurfer tags in NIfTI: the header extension with code 14.

The fixtures follow FreeSurfer 8's ``niiWrite`` byte for byte: a four-byte
header, tags as big-endian int32 id, int64 length and payload, an end tag, and
zero padding to a multiple of 16. They are packed by hand so the codec is
checked against the format rather than against itself.
"""

from __future__ import annotations

import gzip
import logging
import os
import struct
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from nibabel.nifti1 import Nifti1Extension

from neuroreg.cli.mri import main as mri_main
from neuroreg.cli.vol2vol import main as vol2vol_main
from neuroreg.image import MGHTags, ScanParameters, inherit_tags, read_image_tags, read_mgh_tags, save_image
from neuroreg.image.mgh_tags import (
    _decode_matrix,
    _encode_matrix,
    decode_nifti_extension,
    encode_nifti_extension,
    read_nifti_tags,
)

DATA = np.zeros((6, 6, 6), dtype=np.uint8)
DATA[2:4, 2:4, 2:4] = 17
AFFINE = np.array([[-1.0, 0, 0, 3], [0, 0, 1, -3], [0, -1, 0, 3], [0, 0, 0, 1]])
ORIG_RAS2VOX = np.array([[-1.0, 0, 0, 128], [0, 0, -1, 128], [0, 1, 0, 128], [0, 0, 0, 1]])


def _tag(tag: int, payload: bytes) -> bytes:
    return struct.pack(">iq", tag, len(payload)) + payload


def _ctab() -> bytes:
    entries = [(0, b"Unknown", 0, 0, 0, 0), (17, b"Left-Hippocampus", 220, 216, 20, 0)]
    out = struct.pack(">iii", -2, 18, 8) + b"lut.txt\0" + struct.pack(">i", len(entries))
    for index, name, r, g, b, t in entries:
        out += struct.pack(">ii", index, len(name) + 1) + name + b"\0" + struct.pack(">4i", r, g, b, t)
    return out


END = _tag(-1, b"*")
XFORM = _tag(31, b"transforms/talairach.xfm\0")
CTAB = _tag(1, _ctab())
HISTORY = _tag(3, b"ProgramName: mri_convert  ProgramArguments: mri_convert a.mgz b.nii.gz\0")
ORIG = _tag(44, struct.pack(">16f", *ORIG_RAS2VOX.ravel()))


def _freesurfer_extension(*tags: bytes, end: bool = True) -> bytes:
    """Extension content as FreeSurfer writes it, padding included."""
    content = b">\x00\x00\x01" + b"".join(tags) + (END if end else b"")
    return content + bytes(-(len(content) + 8) % 16)


def _tagged_nifti(path: Path, content: bytes, *extra: Nifti1Extension) -> Path:
    image = nib.Nifti1Image(DATA, AFFINE)
    for extension in (*extra, Nifti1Extension(14, content)):
        image.header.extensions.append(extension)
    nib.save(image, str(path))
    return path


def _ids(tags: MGHTags) -> list[int]:
    return [tag for tag, _ in tags.tags]


# ── the codec ─────────────────────────────────────────────────────────────────


def test_reads_an_extension_as_freesurfer_writes_it():
    # FreeSurfer also writes DOF (7), its geometry copy (8, 9) and the scan
    # parameters (45); the image header holds those, so they are not kept.
    content = _freesurfer_extension(
        _tag(7, b"\0\0\0\0"), XFORM, _tag(8, bytes(48)), _tag(9, bytes(28)), CTAB, ORIG, HISTORY,
        _tag(45, bytes(30)),
    )

    tags = decode_nifti_extension(content)

    assert _ids(tags) == [31, 1, 44, 3]
    assert tags.xform == "transforms/talairach.xfm"
    assert list(tags.color_table.entries) == [0, 17]
    assert tags.command_lines[0].startswith("ProgramName: mri_convert")
    assert _decode_matrix(dict(tags.tags)[44]) == pytest.approx(ORIG_RAS2VOX)


def test_matrices_move_between_mgh_text_and_extension_floats():
    tags = MGHTags([(44, _encode_matrix(ORIG_RAS2VOX)), (33, _encode_matrix(np.eye(4)))])

    content = encode_nifti_extension(tags)

    assert _tag(44, struct.pack(">16f", *ORIG_RAS2VOX.ravel())) in content
    assert decode_nifti_extension(content).tags == tags.tags


def test_writes_freesurfers_layout_and_ends_with_the_end_tag():
    tags = MGHTags([(31, b"transforms/talairach.xfm\0"), (1, _ctab()), (3, b"made here\0")])

    content = encode_nifti_extension(tags)

    assert content == b">\x00\x00\x01" + XFORM + CTAB + _tag(3, b"made here\0") + END


def test_frames_are_not_written_and_unknown_acquisition_values_add_nothing(caplog):
    # FreeSurfer writes PE direction UNKNOWN and field strength 0 into every
    # MGH; carrying them into NIfTI would say nothing.
    tags = MGHTags([(41, b"UNKNOWN"), (43, b"\0\0\0\0"), (42, b"frames"), (3, b"x\0")])

    with caplog.at_level(logging.INFO, logger="neuroreg.image.mgh_tags"):
        content = encode_nifti_extension(tags)

    assert content == b">\x00\x00\x01" + _tag(3, b"x\0") + END
    assert "Not storing in the NIfTI extension: frames." in caplog.text


# ── acquisition parameters ────────────────────────────────────────────────────

ROW = b"ROW\0"
TESLA_3 = struct.pack("<f", 3.0)  # MGH tag 43: native byte order, little-endian


def test_acquisition_values_go_into_freesurfers_scan_parameter_tag():
    tags = MGHTags([(41, ROW), (43, TESLA_3), (3, b"x\0")], scan=ScanParameters(2300.0, 2.98, 900.0, 0.157))

    content = encode_nifti_extension(tags)

    # FStagsIO::write_scan_parameters: TE, TI as float32, flip as float64,
    # field strength as float32, then the PE direction. TR is not part of it.
    expected = _tag(45, struct.pack(">ffdf", 2.98, 900.0, 0.157, 3.0) + ROW)
    assert content == b">\x00\x00\x01" + expected + _tag(3, b"x\0") + END


def test_the_scan_parameter_tag_reads_back_into_mgh_layout():
    payload = struct.pack(">ffdf", 2.98, 900.0, 0.157, 3.0) + ROW

    tags = decode_nifti_extension(_freesurfer_extension(_tag(45, payload)))

    assert tags.tags == [(41, ROW), (43, TESLA_3)]
    assert (tags.scan.te, tags.scan.ti, tags.scan.flip_angle) == pytest.approx((2.98, 900.0, 0.157))


def test_a_freesurfer_scan_parameter_tag_holding_only_defaults_adds_no_tags():
    payload = struct.pack(">ffdf", 0.0, 0.0, 0.0, 0.0) + b"UNKNOWN"

    tags = decode_nifti_extension(_freesurfer_extension(_tag(45, payload), HISTORY))

    assert _ids(tags) == [3]


@pytest.mark.parametrize(("unit", "pixdim", "expected"), [("sec", 2.3, 2300.0), ("msec", 2300.0, 2300.0)])
def test_tr_is_read_from_pixdim_in_milliseconds(tmp_path: Path, unit: str, pixdim: float, expected: float):
    image = nib.Nifti1Image(np.zeros((4, 4, 4, 2), np.float32), AFFINE)
    image.header.set_xyzt_units("mm", unit)
    image.header["pixdim"][4] = pixdim
    nib.save(image, str(tmp_path / "bold.nii.gz"))

    assert read_nifti_tags(tmp_path / "bold.nii.gz").scan.tr == pytest.approx(expected)


def test_nibabels_default_pixdim_is_not_a_tr(tmp_path: Path):
    # nibabel writes pixdim[4] = 1 with unknown units; FreeSurfer reads unknown
    # units as no TR, and so does this.
    nib.save(nib.Nifti1Image(DATA, AFFINE), str(tmp_path / "plain.nii.gz"))

    assert read_nifti_tags(tmp_path / "plain.nii.gz").scan is None


def test_save_image_writes_tr_where_nifti_keeps_it_and_restores_the_header(tmp_path: Path):
    image = nib.Nifti1Image(DATA, AFFINE)
    before = image.header.binaryblock

    save_image(image, tmp_path / "out.nii.gz", tags=MGHTags(scan=ScanParameters(tr=2300.0)))

    header = nib.load(str(tmp_path / "out.nii.gz")).header
    assert float(header["pixdim"][4]) == pytest.approx(2.3)
    assert header.get_xyzt_units() == ("mm", "sec")
    assert header.extensions == []
    assert image.header.binaryblock == before


def test_vol2vol_keeps_every_acquisition_value_from_mgh_to_nifti_and_back(tmp_path: Path):
    mgh = nib.MGHImage(DATA, AFFINE)
    for key, value in {"tr": 2300.0, "te": 2.98, "ti": 900.0, "flip_angle": 0.157}.items():
        mgh.header[key] = value
    source = tmp_path / "t1.mgz"
    source.write_bytes(gzip.compress(mgh.to_bytes() + _tag(41, ROW) + _tag(43, TESLA_3)))

    vol2vol_main(["--in", str(source), "--out", str(tmp_path / "t1.nii.gz")])
    vol2vol_main(["--in", str(tmp_path / "t1.nii.gz"), "--out", str(tmp_path / "back.mgz")])

    back = nib.load(str(tmp_path / "back.mgz")).header
    assert [float(back[k]) for k in ("tr", "te", "ti", "flip_angle")] == pytest.approx([2300.0, 2.98, 900.0, 0.157])
    tags = read_mgh_tags(tmp_path / "back.mgz")
    assert (41, ROW) in tags.tags
    assert (43, TESLA_3) in tags.tags


def test_mri_tag_on_nifti_leaves_tr_and_its_units_alone(tmp_path: Path):
    path = tmp_path / "bold.nii.gz"
    image = nib.Nifti1Image(np.zeros((4, 4, 4, 2), np.float32), AFFINE)
    image.header.set_xyzt_units("mm", "msec")
    image.header["pixdim"][4] = 2300.0
    nib.save(image, str(path))
    lut = tmp_path / "lut.txt"
    lut.write_text("0 Unknown 0 0 0 0\n")

    mri_main(["tag", str(path), str(path), "--ctab", str(lut)])

    header = nib.load(str(path)).header
    assert header.get_xyzt_units() == ("mm", "msec")
    assert float(header["pixdim"][4]) == pytest.approx(2300.0)


def test_mgh_scan_parameters_are_read_from_the_footer(tmp_path: Path):
    mgh = nib.MGHImage(DATA, AFFINE)
    mgh.header["tr"], mgh.header["te"], mgh.header["ti"], mgh.header["flip_angle"] = 2300.0, 2.98, 900.0, 0.157
    nib.save(mgh, str(tmp_path / "t1.mgz"))

    scan = read_mgh_tags(tmp_path / "t1.mgz").scan

    assert (scan.tr, scan.te, scan.ti, scan.flip_angle) == pytest.approx((2300.0, 2.98, 900.0, 0.157))


def test_nothing_to_store_writes_no_extension():
    assert encode_nifti_extension(MGHTags([(41, b"UNKNOWN")])) is None


def test_the_legacy_xform_tag_is_written_as_tag_31():
    content = encode_nifti_extension(MGHTags([(30, b"talairach.xfm\0")]))

    assert decode_nifti_extension(content).tags == [(31, b"talairach.xfm\0")]


def test_a_last_tag_shortened_by_nibabel_is_restored():
    # nibabel strips trailing zero bytes from extension content. Without an end
    # tag, the history's terminating NUL goes with them.
    content = (b">\x00\x00\x01" + HISTORY).rstrip(b"\0")

    tags = decode_nifti_extension(content)

    assert tags.tags == [(3, HISTORY[12:])]


def test_rejects_a_code_14_extension_that_is_not_freesurfers():
    with pytest.raises(ValueError, match="big-endian marker"):
        decode_nifti_extension(b"something else entirely")


# ── files ─────────────────────────────────────────────────────────────────────


def test_reads_the_tags_of_a_freesurfer_nifti(tmp_path: Path):
    path = _tagged_nifti(tmp_path / "aseg.nii.gz", _freesurfer_extension(XFORM, CTAB, HISTORY))

    tags = read_image_tags(path)

    assert _ids(tags) == [31, 1, 3]


def test_a_nifti_without_the_extension_has_no_tags(tmp_path: Path):
    path = tmp_path / "plain.nii.gz"
    nib.save(nib.Nifti1Image(DATA, AFFINE), str(path))

    assert read_nifti_tags(path).tags == []


def test_save_image_stores_tags_in_the_extension_and_nothing_else_changes(tmp_path: Path):
    tags = MGHTags([(31, b"transforms/talairach.xfm\0"), (1, _ctab()), (3, b"made here\0")])

    save_image(nib.Nifti1Image(DATA, AFFINE), tmp_path / "tagged.nii.gz", tags=tags)
    save_image(nib.Nifti1Image(DATA, AFFINE), tmp_path / "plain.nii.gz")

    tagged, plain = nib.load(str(tmp_path / "tagged.nii.gz")), nib.load(str(tmp_path / "plain.nii.gz"))
    assert [e.get_code() for e in tagged.header.extensions] == [14]
    assert read_nifti_tags(tmp_path / "tagged.nii.gz").tags == tags.tags
    assert np.array_equal(np.asanyarray(tagged.dataobj), np.asanyarray(plain.dataobj))
    assert tagged.affine == pytest.approx(plain.affine)
    assert tagged.header.get_sform(coded=True)[1] == plain.header.get_sform(coded=True)[1]


def test_save_image_replaces_a_carried_extension_and_keeps_others(tmp_path: Path):
    # A FreeSurfer extension carried in a source header describes the source;
    # it is replaced, never duplicated, and unrelated extensions survive.
    stale = _tagged_nifti(
        tmp_path / "source.nii.gz", _freesurfer_extension(XFORM), Nifti1Extension(6, b"a comment")
    )
    image = nib.load(str(stale))

    save_image(image, tmp_path / "out.nii.gz", tags=MGHTags([(3, b"fresh\0")]))

    out = nib.load(str(tmp_path / "out.nii.gz"))
    assert sorted(e.get_code() for e in out.header.extensions) == [6, 14]
    assert read_nifti_tags(tmp_path / "out.nii.gz").tags == [(3, b"fresh\0")]
    # The caller's image is left as it was.
    assert sorted(e.get_code() for e in image.header.extensions) == [6, 14]


def test_save_image_with_no_tags_removes_a_carried_extension(tmp_path: Path):
    stale = _tagged_nifti(tmp_path / "source.nii.gz", _freesurfer_extension(XFORM))

    save_image(nib.load(str(stale)), tmp_path / "decided.nii.gz", tags=MGHTags())
    save_image(nib.load(str(stale)), tmp_path / "undecided.nii.gz")

    assert [e.get_code() for e in nib.load(str(tmp_path / "decided.nii.gz")).header.extensions] == []
    assert [e.get_code() for e in nib.load(str(tmp_path / "undecided.nii.gz")).header.extensions] == [14]


def test_inherit_tags_reads_a_nifti_source(tmp_path: Path):
    source = _tagged_nifti(tmp_path / "aseg.nii.gz", _freesurfer_extension(XFORM, CTAB))

    assert _ids(inherit_tags(source, tmp_path / "out.mgz")) == [31, 1]


# ── commands ──────────────────────────────────────────────────────────────────


def test_vol2vol_carries_tags_from_mgh_to_nifti_and_back(tmp_path: Path):
    raw = nib.MGHImage(DATA, AFFINE).to_bytes() + _tag(31, b"transforms/talairach.xfm\0")
    raw += struct.pack(">i", 1) + _ctab() + _tag(3, b"origin\0")
    source = tmp_path / "aseg.mgz"
    source.write_bytes(gzip.compress(raw))

    vol2vol_main(["--in", str(source), "--out", str(tmp_path / "aseg.nii.gz")])
    vol2vol_main(["--in", str(tmp_path / "aseg.nii.gz"), "--out", str(tmp_path / "back.mgz")])

    as_nifti, back = read_image_tags(tmp_path / "aseg.nii.gz"), read_mgh_tags(tmp_path / "back.mgz")
    original = read_mgh_tags(source)
    for tags in (as_nifti, back):
        assert tags.xform == "transforms/talairach.xfm"
        assert tags.color_table == original.color_table
    assert back.command_lines[0] == "origin"
    assert [line.split("  ")[0] for line in back.command_lines[1:]] == ["ProgramName: vol2vol"] * 2


def test_mri_tag_tags_a_nifti_in_place(tmp_path: Path, capsys):
    path = tmp_path / "aseg.nii.gz"
    nib.save(nib.Nifti1Image(DATA, AFFINE), str(path))
    lut = tmp_path / "lut.txt"
    lut.write_text("0 Unknown 0 0 0 0\n17 Left-Hippocampus 220 216 20 0\n")
    (tmp_path / "transforms").mkdir()
    (tmp_path / "transforms" / "talairach.xfm").write_text("MNI Transform File\n")

    xform = str(tmp_path / "transforms" / "talairach.xfm")
    mri_main(["tag", str(path), str(path), "--ctab", str(lut), "--xform", xform])

    tags = read_image_tags(path)
    assert tags.xform == os.path.join("transforms", "talairach.xfm")
    assert list(tags.color_table.entries) == [0, 17]
    image = nib.load(str(path))
    assert np.array_equal(np.asanyarray(image.dataobj), DATA)
    assert image.affine == pytest.approx(AFFINE)

    capsys.readouterr()
    mri_main(["tag", str(path), str(path), "--ctab", str(lut)])
    assert "unchanged" in capsys.readouterr().out

    mri_main(["info", str(path)])
    info = capsys.readouterr().out
    assert "talairach xfm: transforms/talairach.xfm" in info
    assert "color table: 2 entries" in info


def test_mri_tag_copies_a_colour_table_from_nifti_into_mgh(tmp_path: Path):
    source = _tagged_nifti(tmp_path / "seg.nii.gz", _freesurfer_extension(CTAB))
    target = tmp_path / "aseg.mgz"
    nib.save(nib.MGHImage(DATA, AFFINE), str(target))

    mri_main(["tag", str(target), str(target), "--ctab-from", str(source)])

    assert read_mgh_tags(target).color_table == read_image_tags(source).color_table
