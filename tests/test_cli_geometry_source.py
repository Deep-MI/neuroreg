"""Geometry options accept a target JSON wherever they accept an image.

``--ref``, ``--template-geom`` and ``--dst-img`` all read only the header of
what they are given, so each also takes a centroid target JSON carrying a
``geometry`` block. These tests pin that the JSON and the equivalent image are
interchangeable, which is the whole point of the option.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neuroreg.cli._args import load_geometry_source
from neuroreg.cli.lta import main as lta_main
from neuroreg.cli.mri import main as mri_main
from neuroreg.cli.multireg import main as multireg_main
from neuroreg.cli.vol2vol import main as vol2vol_main
from neuroreg.segreg.atlas import affine_from_header, load_atlas_target
from neuroreg.transforms import LTA

MNI_ATLAS_NAME = "mni_icbm152_t1_tal_nlin_asym_09c"


def _atlas_json() -> str:
    with resources.as_file(resources.files("neuroreg.segreg").joinpath("data", f"{MNI_ATLAS_NAME}.json")) as path:
        return str(path)


def _equivalent_image(path: Path) -> Path:
    """Write the image the bundled target's geometry block describes."""
    geometry = load_atlas_target(MNI_ATLAS_NAME).geometry
    assert geometry is not None
    dims = tuple(int(v) for v in geometry["dims"])
    nib.save(nib.MGHImage(np.zeros(dims, dtype=np.uint8), affine_from_header(geometry)), path)
    return path


def _write_mov(path: Path) -> Path:
    affine = np.eye(4, dtype=np.float32)
    affine[:3, 3] = (-8.0, -9.0, -10.0)
    nib.save(nib.MGHImage(np.ones((16, 16, 16), dtype=np.float32), affine), path)
    return path


def test_load_geometry_source_matches_the_image_it_describes(tmp_path: Path):
    from_json = load_geometry_source(_atlas_json(), flag="--ref")
    from_image = load_geometry_source(_equivalent_image(tmp_path / "geom.mgz"), flag="--ref")

    assert from_json.shape[:3] == from_image.shape[:3] == (256, 256, 256)
    assert np.asarray(from_json.affine) == pytest.approx(np.asarray(from_image.affine))


def test_load_geometry_source_does_not_allocate_the_voxels():
    # The grid is 256 cubed and nothing reads its voxels, so materialising them
    # would be pure waste on every call.
    image = load_geometry_source(_atlas_json(), flag="--ref")
    assert np.asarray(image.dataobj).strides == (0, 0, 0)


def test_load_geometry_source_rejects_a_target_without_geometry(tmp_path: Path):
    target = tmp_path / "centroids_only.json"
    target.write_text('{"centroids": {"1": [1.0, 2.0, 3.0]}}\n')

    with pytest.raises(ValueError, match="no 'geometry' block"):
        load_geometry_source(target, flag="--ref")


def test_vol2vol_ref_accepts_a_geometry_json(tmp_path: Path):
    mov = _write_mov(tmp_path / "mov.mgz")
    from_json = tmp_path / "from_json.mgz"
    from_image = tmp_path / "from_image.mgz"

    vol2vol_main(["--in", str(mov), "--ref", _atlas_json(), "--out", str(from_json)])
    vol2vol_main(["--in", str(mov), "--ref", str(_equivalent_image(tmp_path / "geom.mgz")), "--out", str(from_image)])

    written = nib.load(str(from_json))
    assert written.shape[:3] == (256, 256, 256)
    assert "".join(nib.aff2axcodes(written.affine)) == "LIA"
    assert np.asarray(written.affine) == pytest.approx(np.asarray(nib.load(str(from_image)).affine))
    assert np.array_equal(np.asarray(written.dataobj), np.asarray(nib.load(str(from_image)).dataobj))


def test_multireg_template_geom_accepts_a_geometry_json(tmp_path: Path):
    mov1 = _write_mov(tmp_path / "tp1.mgz")
    mov2 = _write_mov(tmp_path / "tp2.mgz")
    template = tmp_path / "template.mgz"

    multireg_main(
        [
            "--mov",
            str(mov1),
            str(mov2),
            "--template-geom",
            _atlas_json(),
            "--template",
            str(template),
            "--noit",
            "--nmax",
            "1",
        ]
    )

    written = nib.load(str(template))
    assert written.shape[:3] == (256, 256, 256)
    assert "".join(nib.aff2axcodes(written.affine)) == "LIA"


def test_lta_convert_dst_img_accepts_a_geometry_json(tmp_path: Path):
    mov = _write_mov(tmp_path / "mov.mgz")
    source = tmp_path / "in.lta"
    LTA.from_matrix(np.eye(4), str(mov), nib.load(str(mov)), "atlas", None, lta_type=1).write(source)
    from_json = tmp_path / "from_json.lta"
    from_image = tmp_path / "from_image.lta"

    lta_main(["convert", str(source), str(from_json), "--dst-img", _atlas_json()])
    lta_main(["convert", str(source), str(from_image), "--dst-img", str(_equivalent_image(tmp_path / "geom.mgz"))])

    written = LTA.read(from_json)
    assert written.dst["valid"] == 1
    assert list(written.dst["volume"]) == [256, 256, 256]
    assert written.dst["cras"] == pytest.approx([0.0, 0.0, 0.0])

    from_image_lta = LTA.read(from_image)
    assert list(from_image_lta.dst["volume"]) == list(written.dst["volume"])
    assert from_image_lta.dst["cras"] == pytest.approx(written.dst["cras"])
    assert np.asarray(from_image_lta.r2r()) == pytest.approx(np.asarray(written.r2r()))
    # An image source still records its own path, so the volume-info block keeps
    # naming where the geometry came from whenever there is a file to name.
    assert from_image_lta.dst["filename"].endswith("geom.mgz")


def test_lta_convert_src_img_accepts_a_geometry_json(tmp_path: Path):
    # --src-img is the same kind of header-only option as --dst-img, so it has
    # to accept the same inputs; one of two would be its own trap.
    mov = _write_mov(tmp_path / "mov.mgz")
    source = tmp_path / "in.lta"
    LTA.from_matrix(np.eye(4), str(mov), nib.load(str(mov)), "atlas", None, lta_type=1).write(source)
    out = tmp_path / "out.lta"

    lta_main(["convert", str(source), str(out), "--src-img", _atlas_json()])

    written = LTA.read(out)
    assert written.src["valid"] == 1
    assert list(written.src["volume"]) == [256, 256, 256]


def test_mri_geom_like_accepts_a_geometry_json(tmp_path: Path):
    out = tmp_path / "grid.mgz"

    mri_main(["geom", "--o", str(out), "--like", _atlas_json()])

    written = nib.load(str(out))
    assert written.shape[:3] == (256, 256, 256)
    assert "".join(nib.aff2axcodes(written.affine)) == "LIA"
    assert np.linalg.norm(np.asarray(written.affine)[:3, :3], axis=0) == pytest.approx([1.0, 1.0, 1.0])


def test_mri_geom_like_json_rescales_dimensions_like_an_image(tmp_path: Path):
    # --like supplies an extent, not dimensions, so a finer voxel size grows the
    # grid. The JSON has to behave the same way the equivalent image does.
    from_json = tmp_path / "from_json.mgz"
    from_image = tmp_path / "from_image.mgz"

    mri_main(["geom", "--o", str(from_json), "--like", _atlas_json(), "--vox-size", "0.5"])
    mri_main(
        [
            "geom", "--o", str(from_image),
            "--like", str(_equivalent_image(tmp_path / "geom.mgz")),
            "--vox-size", "0.5",
        ]
    )

    assert nib.load(str(from_json)).shape[:3] == (512, 512, 512)
    assert nib.load(str(from_json)).shape[:3] == nib.load(str(from_image)).shape[:3]
    assert np.asarray(nib.load(str(from_json)).affine) == pytest.approx(
        np.asarray(nib.load(str(from_image)).affine)
    )


def test_vol2vol_ref_reports_a_target_without_geometry(tmp_path: Path, capsys):
    mov = _write_mov(tmp_path / "mov.mgz")
    target = tmp_path / "centroids_only.json"
    target.write_text('{"centroids": {"1": [1.0, 2.0, 3.0]}}\n')

    with pytest.raises(SystemExit):
        vol2vol_main(["--in", str(mov), "--ref", str(target), "--out", str(tmp_path / "out.mgz")])

    out = capsys.readouterr()
    assert "no 'geometry' block" in out.out + out.err
