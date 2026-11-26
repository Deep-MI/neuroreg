"""FreeSurfer tags in NIfTI: the header extension with code 14, and TR in pixdim.

The fixtures follow FreeSurfer 8's ``niiWrite`` byte for byte: a four-byte
header, tags as big-endian int32 id, int64 length and payload, an end tag, and
zero padding to a multiple of 16. They are packed by hand so the codec is
checked against the format rather than against itself.
"""

from __future__ import annotations

import gzip
import logging
import struct
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from nibabel.nifti1 import Nifti1Extension

from neuroreg.cli.mri import main as mri_main
from neuroreg.cli.vol2vol import main as vol2vol_main
from neuroreg.image import MGHTags, ScanParameters, read_image_tags, read_mgh_tags, save_image
from neuroreg.image.mgh_tags import (
    _decode_matrix,
    _encode_matrix,
    _extension_bytes,
    _split_nifti_extension,
    decode_nifti_extension,
    encode_nifti_extension,
    read_nifti_tags,
)

DATA = np.zeros((6, 6, 6), dtype=np.uint8)
DATA[2:4, 2:4, 2:4] = 17
AFFINE = np.array([[-1.0, 0, 0, 3], [0, 0, 1, -3], [0, -1, 0, 3], [0, 0, 0, 1]])
ORIG_RAS2VOX = np.array([[-1.0, 0, 0, 128], [0, 0, -1, 128], [0, 1, 0, 128], [0, 0, 0, 1]])
ROW = b"ROW\0"
TESLA_3 = struct.pack("<f", 3.0)  # MGH tag 43: native byte order, little-endian


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
    # FreeSurfer also writes DOF (7) and its geometry copy (8, 9); the image
    # header holds those, so they are not kept.
    content = _freesurfer_extension(
        _tag(7, b"\0\0\0\0"), XFORM, _tag(8, bytes(48)), _tag(9, bytes(28)), CTAB, ORIG, HISTORY
    )

    tags = decode_nifti_extension(content)

    assert _ids(tags) == [31, 1, 44, 3]
    assert tags.xform == "transforms/talairach.xfm"
    assert list(tags.color_table.entries) == [0, 17]
    assert _decode_matrix(dict(tags.tags)[44]) == pytest.approx(ORIG_RAS2VOX)


def test_writes_freesurfers_layout_and_ends_with_the_end_tag():
    # Matrices are text in MGH but 16 big-endian floats in the extension.
    tags = MGHTags([(31, b"transforms/talairach.xfm\0"), (1, _ctab()), (44, _encode_matrix(ORIG_RAS2VOX))])

    assert encode_nifti_extension(tags) == b">\x00\x00\x01" + XFORM + CTAB + ORIG + END


def test_frames_are_not_written_and_unknown_acquisition_values_add_nothing(caplog):
    # FreeSurfer writes PE direction UNKNOWN and field strength 0 into every
    # MGH; carrying them into NIfTI would say nothing.
    defaults = [(41, b"UNKNOWN"), (43, b"\0\0\0\0")]

    with caplog.at_level(logging.INFO, logger="neuroreg.image.mgh_tags"):
        content = encode_nifti_extension(MGHTags([*defaults, (42, b"frames"), (3, b"x\0")]))

    assert content == b">\x00\x00\x01" + _tag(3, b"x\0") + END
    assert "Not storing in the NIfTI extension: frames." in caplog.text
    assert encode_nifti_extension(MGHTags(defaults)) is None


@pytest.mark.parametrize(
    ("header", "accepted"),
    [
        (b">\x00\x00\x01", True),
        (b">\x00\x06\x01", True),  # a non-zero intent, as FreeSurfer writes for warps
        (b"<\x00\x00\x01", False),
        (b">\x00\x00\x02", False),
    ],
)
def test_the_extension_header_is_checked_for_byte_order_and_version_not_intent(header: bytes, accepted: bool):
    content = header + XFORM + END

    if accepted:
        assert decode_nifti_extension(content).xform == "transforms/talairach.xfm"
    else:
        with pytest.raises(ValueError, match="FreeSurfer's header"):
            decode_nifti_extension(content)


def test_a_last_tag_shortened_by_nibabel_is_restored():
    # nibabel strips trailing zero bytes from extension content. Without an end
    # tag, the history's terminating NUL goes with them.
    content = (b">\x00\x00\x01" + HISTORY).rstrip(b"\0")

    assert decode_nifti_extension(content).tags == [(3, HISTORY[12:])]


def test_a_colour_table_that_does_not_fill_its_tag_is_dropped(caplog):
    # MGH writes this tag without a length, so trailing bytes copied there
    # would be read as the start of the next tag. The other tags are kept.
    content = b">\x00\x00\x01" + XFORM + _tag(1, _ctab() + b"junk") + END

    with caplog.at_level(logging.WARNING, logger="neuroreg.image.mgh_tags"):
        tags = decode_nifti_extension(content)

    assert _ids(tags) == [31]
    assert "malformed colour table" in caplog.text


def test_a_volume_too_large_for_nifti1_is_saved_as_nifti2_with_its_tags(tmp_path: Path):
    # nib.save falls back to NIfTI-2 when a dimension exceeds int16; the tagged
    # path has to make the same choice.
    out = tmp_path / "long.nii"
    tags = MGHTags([(31, b"transforms/talairach.xfm\0")])

    save_image(nib.MGHImage(np.zeros((40000, 2, 1), np.uint8), AFFINE), out, tags=tags)

    assert isinstance(nib.load(str(out)), nib.Nifti2Image)
    assert read_image_tags(out).xform == "transforms/talairach.xfm"


def test_a_corrupt_tag_length_is_rejected_rather_than_padded():
    # Padding up to a length read from the file would let one bad int64 ask
    # for exabytes; anything beyond stripped zeros is an error.
    content = b">\x00\x00\x01" + struct.pack(">iq", 3, 2**62) + b"x\0"

    with pytest.raises(ValueError, match="declares"):
        decode_nifti_extension(content)


# ── acquisition parameters ────────────────────────────────────────────────────


def test_acquisition_values_go_into_freesurfers_scan_parameter_tag():
    tags = MGHTags([(41, ROW), (43, TESLA_3)], scan=ScanParameters(2300.0, 2.98, 900.0, 0.157))

    content = encode_nifti_extension(tags)

    # FStagsIO::write_scan_parameters: TE, TI as float32, flip as float64,
    # field strength as float32, then the PE direction. TR is not part of it.
    expected = _tag(45, struct.pack(">ffdf", 2.98, 900.0, 0.157, 3.0) + ROW)
    assert content == b">\x00\x00\x01" + expected + END


def test_the_scan_parameter_tag_reads_back_into_mgh_layout():
    payload = struct.pack(">ffdf", 2.98, 900.0, 0.157, 3.0) + ROW
    defaults = struct.pack(">ffdf", 0.0, 0.0, 0.0, 0.0) + b"UNKNOWN"

    tags = decode_nifti_extension(_freesurfer_extension(_tag(45, payload)))

    assert tags.tags == [(41, ROW), (43, TESLA_3)]
    assert (tags.scan.te, tags.scan.ti, tags.scan.flip_angle) == pytest.approx((2.98, 900.0, 0.157))
    assert _ids(decode_nifti_extension(_freesurfer_extension(_tag(45, defaults)))) == []


@pytest.mark.parametrize(("unit", "expected"), [("sec", 2300.0), ("unknown", None)])
def test_tr_is_read_from_pixdim_only_with_a_time_unit(tmp_path: Path, unit: str, expected: float | None):
    # nibabel writes pixdim[4] = 1 with an unknown unit by default; FreeSurfer
    # reads that as no TR, and so does this.
    image = nib.Nifti1Image(np.zeros((4, 4, 4, 2), np.float32), AFFINE)
    image.header.set_xyzt_units("mm", unit)
    image.header["pixdim"][4] = 2.3
    nib.save(image, str(tmp_path / "bold.nii.gz"))

    scan = read_nifti_tags(tmp_path / "bold.nii.gz").scan

    if expected is None:
        assert scan is None
    else:
        assert scan.tr == pytest.approx(expected)


# ── files and commands ────────────────────────────────────────────────────────


def test_save_image_stores_tags_and_tr_and_leaves_the_callers_header_alone(tmp_path: Path):
    image = nib.Nifti1Image(DATA, AFFINE)
    before = image.header.binaryblock
    tags = MGHTags([(31, b"transforms/talairach.xfm\0"), (1, _ctab())], scan=ScanParameters(tr=2300.0))

    save_image(image, tmp_path / "out.nii.gz", tags=tags)

    out = nib.load(str(tmp_path / "out.nii.gz"))
    assert [e.get_code() for e in out.header.extensions] == [14]
    assert read_nifti_tags(tmp_path / "out.nii.gz").tags == tags.tags
    assert float(out.header["pixdim"][4]) == pytest.approx(2.3)
    assert out.header.get_xyzt_units() == ("mm", "sec")
    assert out.affine == pytest.approx(AFFINE)
    assert image.header.binaryblock == before


def test_a_carried_extension_is_replaced_or_removed_but_never_duplicated(tmp_path: Path):
    # A FreeSurfer extension carried in a source header describes the source.
    stale = _tagged_nifti(tmp_path / "source.nii.gz", _freesurfer_extension(XFORM), Nifti1Extension(6, b"a note"))

    save_image(nib.load(str(stale)), tmp_path / "new.nii.gz", tags=MGHTags([(3, b"fresh\0")]))
    save_image(nib.load(str(stale)), tmp_path / "none.nii.gz", tags=MGHTags())

    new = nib.load(str(tmp_path / "new.nii.gz")).header.extensions
    assert sorted(e.get_code() for e in new) == [6, 14]
    assert read_nifti_tags(tmp_path / "new.nii.gz").tags == [(3, b"fresh\0")]
    assert [e.get_code() for e in nib.load(str(tmp_path / "none.nii.gz")).header.extensions] == [6]


def test_vol2vol_keeps_tags_and_acquisition_values_from_mgh_to_nifti_and_back(tmp_path: Path):
    mgh = nib.MGHImage(DATA, AFFINE)
    for key, value in {"tr": 2300.0, "te": 2.98, "ti": 900.0, "flip_angle": 0.157}.items():
        mgh.header[key] = value
    raw = mgh.to_bytes() + XFORM + _tag(41, ROW) + _tag(43, TESLA_3) + struct.pack(">i", 1) + _ctab()
    source = tmp_path / "t1.mgz"
    source.write_bytes(gzip.compress(raw))

    vol2vol_main(["--in", str(source), "--out", str(tmp_path / "t1.nii.gz")])
    vol2vol_main(["--in", str(tmp_path / "t1.nii.gz"), "--out", str(tmp_path / "back.mgz")])

    header = nib.load(str(tmp_path / "back.mgz")).header
    assert [float(header[k]) for k in ("tr", "te", "ti", "flip_angle")] == pytest.approx([2300.0, 2.98, 900.0, 0.157])
    back = read_mgh_tags(tmp_path / "back.mgz")
    assert back.xform == "transforms/talairach.xfm"
    assert back.color_table == read_mgh_tags(source).color_table
    assert (41, ROW) in back.tags and (43, TESLA_3) in back.tags


def test_mri_tag_tags_a_nifti_in_place_and_leaves_tr_alone(tmp_path: Path, capsys):
    path = tmp_path / "bold.nii.gz"
    image = nib.Nifti1Image(np.zeros((6, 6, 6, 2), np.float32), AFFINE)
    image.header.set_xyzt_units("mm", "msec")
    image.header["pixdim"][4] = 2300.0
    nib.save(image, str(path))
    lut = tmp_path / "lut.txt"
    lut.write_text("0 Unknown 0 0 0 0\n17 Left-Hippocampus 220 216 20 0\n")
    (tmp_path / "transforms").mkdir()
    xform = tmp_path / "transforms" / "talairach.xfm"
    xform.write_text("MNI Transform File\n")

    mri_main(["tag", str(path), str(path), "--ctab", str(lut), "--xform", str(xform)])

    tags = read_image_tags(path)
    assert tags.xform == "transforms/talairach.xfm"
    assert list(tags.color_table.entries) == [0, 17]
    header = nib.load(str(path)).header
    assert header.get_xyzt_units() == ("mm", "msec")
    assert float(header["pixdim"][4]) == pytest.approx(2300.0)

    capsys.readouterr()
    mri_main(["tag", str(path), str(path), "--ctab", str(lut)])
    assert "unchanged" in capsys.readouterr().out


def test_mri_tag_on_nifti_changes_only_the_requested_tag(tmp_path: Path):
    # The MGH view leaves out the NIfTI-only DOF tag and the frame metadata,
    # so rebuilding the extension from it would silently drop them, and with
    # them the intent FreeSurfer stores in the extension header.
    dof, frames = _tag(7, b"\0\0\0\x05"), _tag(42, b"frame records")
    content = b">\x00\x06\x01" + dof + XFORM + frames + HISTORY + END
    path = _tagged_nifti(tmp_path / "warp.nii.gz", content + bytes(-(len(content) + 8) % 16))
    lut = tmp_path / "lut.txt"
    lut.write_text("0 Unknown 0 0 0 0\n17 Left-Hippocampus 220 216 20 0\n")

    mri_main(["tag", str(path), str(path), "--ctab", str(lut)])

    stored = [e for e in nib.load(str(path)).header.extensions if e.get_code() == 14]
    header, tags = _split_nifti_extension(_extension_bytes(stored[0]))
    assert header == b">\x00\x06\x01"
    assert [tag for tag, _ in tags] == [7, 31, 42, 3, 1]
    assert [_tag(tag, payload) for tag, payload in tags[:4]] == [dof, XFORM, frames, HISTORY]


def test_mri_tag_copies_a_colour_table_from_nifti_into_mgh(tmp_path: Path):
    source = _tagged_nifti(tmp_path / "seg.nii.gz", _freesurfer_extension(CTAB))
    target = tmp_path / "aseg.mgz"
    nib.save(nib.MGHImage(DATA, AFFINE), str(target))

    mri_main(["tag", str(target), str(target), "--ctab-from", str(source)])

    assert read_mgh_tags(target).color_table == read_image_tags(source).color_table
