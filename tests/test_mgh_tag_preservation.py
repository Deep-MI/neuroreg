"""Which MGH tags an output carries, and from where.

The rules: tags describing the voxel values (colour table, phase-encode
direction, field strength, frames) follow the values; tags tied to the
coordinate frame (talairach path, AutoAlign, unknown ids) survive only when the
output stays in the source frame; the original RAS-to-voxel matrix is
re-expressed in a new frame; history comes from the value source with the
writing command appended; volumes with new values start fresh.
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
import torch

from neuroreg.cli.coreg import main as coreg_main
from neuroreg.cli.mri import main as mri_main
from neuroreg.cli.multireg import main as multireg_main
from neuroreg.cli.vol2vol import main as vol2vol_main
from neuroreg.image import MGHTags, history_entry, read_mgh_tags
from neuroreg.image.mgh_tags import _decode_matrix
from neuroreg.multireg import MultiRegResult
from neuroreg.transforms import LTA

DATA = np.zeros((6, 6, 6), dtype=np.uint8)
DATA[2:4, 2:4, 2:4] = 17


def _sized(tag: int, payload: bytes) -> bytes:
    return struct.pack(">iq", tag, len(payload)) + payload


def _ctab(fname: bytes) -> bytes:
    entries = [(0, b"Unknown", 0, 0, 0, 0), (17, b"Left-Hippocampus", 220, 216, 20, 0)]
    out = struct.pack(">iii", -2, 18, len(fname) + 1) + fname + b"\0" + struct.pack(">i", len(entries))
    for index, name, r, g, b, t in entries:
        out += struct.pack(">ii", index, len(name) + 1) + name + b"\0" + struct.pack(">4i", r, g, b, t)
    return out


def _matrix_text(matrix: np.ndarray) -> bytes:
    """A matrix tag laid out as FreeSurfer writes it: text in a 1600-byte buffer."""
    return ("Matrix " + " ".join(f"{v:10f}" for v in matrix.ravel())).encode().ljust(1600, b"\0")


def _tagged_volume(path: Path, *, xform: str = "transforms/talairach.xfm", lut: bytes = b"lut.txt") -> Path:
    """A label volume with the tags FreeSurfer writes, in FreeSurfer's order."""
    raw = nib.MGHImage(DATA, np.eye(4)).to_bytes()
    raw += _sized(31, xform.encode() + b"\0")
    raw += _sized(41, b"UNKNOWN")
    raw += struct.pack(">i", 1) + _ctab(lut)
    raw += _sized(3, b"ProgramName: mri_convert  ProgramArguments: mri_convert in.mgz out.mgz\0")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(raw))
    return path


def _ids(path: Path) -> list[int]:
    return [tag for tag, _ in read_mgh_tags(path).tags]


def _rotation_and_shift() -> np.ndarray:
    angle = np.deg2rad(20.0)
    matrix = np.eye(4)
    matrix[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    matrix[:3, 3] = [3.0, -2.0, 5.0]
    return matrix


# ── the rules ─────────────────────────────────────────────────────────────────

ORIG_RAS2VOX = np.array([[-1.0, 0, 0, 128], [0, 0, -1, 128], [0, 1, 0, 128], [0, 0, 0, 1]])
ALL_KINDS = MGHTags(
    [
        (31, b"transforms/talairach.xfm\0"),
        (41, b"UNKNOWN"),
        (43, b"\0\0\0\0"),
        (33, _matrix_text(np.eye(4))),
        (99, b"unknown payload"),
        (1, _ctab(b"lut.txt")),
        (42, b"frame data"),
        (3, b"history\0"),
        (44, _matrix_text(ORIG_RAS2VOX)),
    ]
)


def test_frame_tags_survive_only_while_the_output_stays_in_the_source_frame(caplog):
    assert ALL_KINDS.derive(r2r=np.eye(4)).tags == ALL_KINDS.tags

    with caplog.at_level(logging.INFO, logger="neuroreg.image.mgh_tags"):
        derived = ALL_KINDS.derive(r2r=_rotation_and_shift())

    assert [tag for tag, _ in derived.tags] == [41, 43, 1, 42, 3, 44]
    assert "talairach xform path" in caplog.text


def test_the_original_ras2vox_still_finds_the_same_original_voxel():
    # The point of updating rather than dropping it: a point in the new frame
    # must map to the same original voxel its old position did.
    moved = _rotation_and_shift()
    payload = dict(ALL_KINDS.derive(r2r=moved).tags)[44]
    point_old = np.array([10.0, -20.0, 30.0, 1.0])

    assert _decode_matrix(payload) @ (moved @ point_old) == pytest.approx(ORIG_RAS2VOX @ point_old, abs=1e-5)
    assert len(payload) == 1600  # FreeSurfer's fixed text buffer


def test_history_entry_follows_freesurfers_layout_without_user_or_machine():
    entry = history_entry("vol2vol", ["--in", "a b.mgz", "--out", "c.mgz"])

    assert entry.startswith("ProgramName: vol2vol  ProgramArguments: vol2vol --in 'a b.mgz' --out c.mgz  ")
    assert "ProgramVersion: neuroreg " in entry
    assert "User:" not in entry and "Machine:" not in entry


# ── commands ──────────────────────────────────────────────────────────────────


def test_vol2vol_conversion_keeps_the_tags_and_rebases_the_xform(tmp_path: Path):
    source = _tagged_volume(tmp_path / "subject" / "mri" / "aseg.mgz")
    out = tmp_path / "elsewhere" / "aseg.mgz"
    out.parent.mkdir()

    vol2vol_main(["--in", str(source), "--out", str(out)])

    tags = read_mgh_tags(out)
    assert [tag for tag, _ in tags.tags] == [31, 41, 1, 3, 3]
    assert tags.xform == os.path.join("..", "subject", "mri", "transforms", "talairach.xfm")
    assert tags.color_table == read_mgh_tags(source).color_table
    assert tags.command_lines[1].startswith("ProgramName: vol2vol")


def test_vol2vol_header_only_into_a_new_frame_drops_the_xform(tmp_path: Path):
    source = _tagged_volume(tmp_path / "aseg.mgz")
    image = nib.load(str(source))
    transform = tmp_path / "move.lta"
    LTA.from_matrix(_rotation_and_shift(), str(source), image, str(source), image, lta_type=1).write(transform)

    vol2vol_main(
        ["--in", str(source), "--transform", str(transform), "--header-only", "--out", str(tmp_path / "o.mgz")]
    )

    assert _ids(tmp_path / "o.mgz") == [41, 1, 3, 3]


def test_mri_binarize_drops_the_colour_table(tmp_path: Path):
    source = _tagged_volume(tmp_path / "aseg.mgz")

    mri_main(["binarize", "--i", str(source), "--o", str(tmp_path / "bin.mgz"), "--match", "17"])

    assert _ids(tmp_path / "bin.mgz") == [31, 41, 3, 3]


def test_mri_geom_records_only_how_it_was_made(tmp_path: Path):
    mri_main(["geom", "--o", str(tmp_path / "grid.mgz"), "--fov", "8", "--vox-size", "1", "--orientation", "LIA"])

    assert _ids(tmp_path / "grid.mgz") == [3]


def test_coreg_mapmov_carries_the_moving_images_labels_but_not_its_frame(monkeypatch, tmp_path: Path):
    mov = _tagged_volume(tmp_path / "mov.mgz")
    ref = tmp_path / "ref.mgz"
    nib.save(nib.MGHImage(np.zeros((6, 6, 6), np.float32), np.eye(4)), ref)

    class _Result:
        def detach(self):
            return self

        def cpu(self):
            return torch.as_tensor(_rotation_and_shift())

    monkeypatch.setattr("neuroreg.imreg.coreg.coreg", lambda *args, **kwargs: _Result())
    coreg_main(
        [
            "--mov", str(mov), "--ref", str(ref), "--out", str(tmp_path / "reg.lta"),
            "--mapmov", str(tmp_path / "mapped.mgz"), "--mapmovhdr", str(tmp_path / "hdr.mgz"),
        ]
    )

    for output in ("mapped.mgz", "hdr.mgz"):
        assert _ids(tmp_path / output) == [41, 1, 3, 3]


def test_multireg_template_starts_fresh_and_time_points_keep_their_own_tags(monkeypatch, tmp_path: Path):
    movs = [_tagged_volume(tmp_path / f"tp{i}.mgz", lut=f"lut{i}.txt".encode()) for i in (1, 2)]

    def fake_multireg(*args, **kwargs):
        image = nib.MGHImage(DATA, np.eye(4))
        return MultiRegResult(
            template_image=image,
            transforms_r2r=[np.eye(4), _rotation_and_shift()],
            ltas=[],
            initial_target_index=0,
            seed=1,
            mapped_images=[image, image],
            template_iterations_run=0,
            iteration_distances=[],
        )

    monkeypatch.setattr("neuroreg.cli.multireg.multireg", fake_multireg)
    mapped = [tmp_path / "tp1_base.mgz", tmp_path / "tp2_base.mgz"]
    multireg_main(
        ["--mov", *map(str, movs), "--template", str(tmp_path / "base.mgz"), "--mapmov", *map(str, mapped), "--noit"]
    )

    assert _ids(tmp_path / "base.mgz") == [3]
    # tp1 sits in template space by identity, so it keeps its xform; tp2 moved.
    assert _ids(mapped[0]) == [31, 41, 1, 3, 3]
    assert _ids(mapped[1]) == [41, 1, 3, 3]
    assert read_mgh_tags(mapped[1]).color_table.fname == "lut2.txt"
