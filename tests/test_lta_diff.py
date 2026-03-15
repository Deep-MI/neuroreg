"""Tests for the lta-diff CLI and underlying transform-distance functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nireg.cmdline.lta_diff import main
from nireg.transforms import (
    LTA,
    affine_dist,
    corner_dist,
    decompose_transform,
    rigid_dist,
    sphere_dist,
)

# ---------------------------------------------------------------------------
# Shared geometry and helpers
# ---------------------------------------------------------------------------

_GEOM = {
    "dims": [256, 256, 256],
    "delta": [1.0, 1.0, 1.0],
    "Mdc": np.eye(3),
    "Pxyz_c": np.zeros(3),
}

_IDENTITY = np.eye(4)


def _rotation_z(deg: float) -> np.ndarray:
    """Return a 4×4 rotation matrix (z-axis, degrees)."""
    r = np.radians(deg)
    M = np.eye(4)
    M[0, 0] = np.cos(r);  M[0, 1] = -np.sin(r)
    M[1, 0] = np.sin(r);  M[1, 1] =  np.cos(r)
    return M


def _translation(tx: float, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    """Return a 4×4 pure-translation matrix (mm)."""
    M = np.eye(4)
    M[:3, 3] = [tx, ty, tz]
    return M


def _write_lta(path: Path, M: np.ndarray, lta_type: int = 1) -> str:
    """Write a minimal LTA file via the LTA class; return its path as a string."""
    s = str(path)
    LTA.from_matrix(M, "src.mgz", _GEOM, "dst.mgz", _GEOM, lta_type=lta_type).write(s)
    return s


# ---------------------------------------------------------------------------
# rigid_dist
# ---------------------------------------------------------------------------

class TestRigidDist:
    def test_identity(self):
        assert rigid_dist(_IDENTITY) == pytest.approx(0.0, abs=1e-10)

    def test_pure_translation(self):
        assert rigid_dist(_translation(3.0, 4.0)) == pytest.approx(5.0, rel=1e-6)

    def test_pure_rotation_angle(self):
        theta = np.radians(5.0)
        assert rigid_dist(_rotation_z(5.0)) == pytest.approx(np.sqrt(2.0) * theta, rel=1e-6)


# ---------------------------------------------------------------------------
# affine_dist
# ---------------------------------------------------------------------------

class TestAffineDist:
    def test_identity(self):
        assert affine_dist(_IDENTITY) == pytest.approx(0.0, abs=1e-10)

    def test_pure_translation(self):
        assert affine_dist(_translation(3.0, 4.0)) == pytest.approx(5.0, rel=1e-6)

    def test_radius_scales_rotation_part(self):
        R = _rotation_z(5.0)
        assert affine_dist(R, radius=200.0) == pytest.approx(2.0 * affine_dist(R, radius=100.0), rel=1e-6)


# ---------------------------------------------------------------------------
# corner_dist
# ---------------------------------------------------------------------------

class TestCornerDist:
    _SHAPE  = (256, 256, 256)
    _AFFINE = np.eye(4)

    def test_identity(self):
        assert corner_dist(_IDENTITY, self._SHAPE, src_affine=self._AFFINE) == pytest.approx(0.0, abs=1e-10)

    def test_pure_translation_mm(self):
        assert corner_dist(_translation(3.0, 4.0), self._SHAPE, src_affine=self._AFFINE) == pytest.approx(5.0, rel=1e-6)


# ---------------------------------------------------------------------------
# sphere_dist
# ---------------------------------------------------------------------------

class TestSphereDist:
    def test_identity(self):
        assert sphere_dist(_IDENTITY) == pytest.approx(0.0, abs=1e-10)

    def test_pure_translation(self):
        t = np.array([3.0, 4.0, 0.0])
        assert sphere_dist(_translation(*t)) == pytest.approx(float(np.linalg.norm(t)), rel=1e-6)

    def test_radius_affects_rotation_result(self):
        R = _rotation_z(5.0)
        assert sphere_dist(R, radius=100.0) == pytest.approx(2.0 * sphere_dist(R, radius=50.0), rel=1e-4)


# ---------------------------------------------------------------------------
# decompose_transform
# ---------------------------------------------------------------------------

class TestDecomposeTransform:
    def test_identity(self):
        d = decompose_transform(_IDENTITY)
        assert d["rotation"]      == pytest.approx(np.eye(3),         abs=1e-10)
        assert d["translation"]   == pytest.approx([0.0, 0.0, 0.0],   abs=1e-10)
        assert d["scales"]        == pytest.approx([1.0, 1.0, 1.0],   rel=1e-6)
        assert d["rot_angle_deg"] == pytest.approx(0.0,                abs=1e-6)
        assert d["determinant"]   == pytest.approx(1.0,                rel=1e-10)

    def test_pure_rotation(self):
        d = decompose_transform(_rotation_z(30.0))
        assert d["rot_angle_deg"] == pytest.approx(30.0, rel=1e-4)
        assert d["abs_trans"]     == pytest.approx(0.0,  abs=1e-10)


# ---------------------------------------------------------------------------
# LTA class
# ---------------------------------------------------------------------------

class TestLTAClass:
    def test_read_write_roundtrip(self, tmp_path):
        M   = _rotation_z(5.0) @ _translation(2.0, 1.0)
        lta = LTA.from_matrix(M, "src.mgz", _GEOM, "dst.mgz", _GEOM)
        lta.write(tmp_path / "out.lta")
        lta2 = LTA.read(tmp_path / "out.lta")
        assert lta2.matrix == pytest.approx(M,  rel=1e-10)
        assert lta2.type   == 1

    def test_r2r_and_v2v(self, tmp_path):
        # R2R stored: r2r() is identity; v2v() must convert correctly
        p = _write_lta(tmp_path / "id.lta", _IDENTITY, lta_type=1)
        assert LTA.read(p).r2r() == pytest.approx(_IDENTITY, abs=1e-10)
        p = _write_lta(tmp_path / "id_v.lta", _IDENTITY, lta_type=0)
        assert LTA.read(p).v2v() == pytest.approx(_IDENTITY, abs=1e-10)

    def test_invert_restores_identity(self):
        M   = _rotation_z(5.0) @ _translation(2.0, 1.0)
        lta = LTA.from_matrix(M, "src.mgz", _GEOM, "dst.mgz", _GEOM)
        assert lta.invert().r2r() @ lta.r2r() == pytest.approx(np.eye(4), abs=1e-10)

    def test_write_type_conversion(self, tmp_path):
        """write(lta_type=0) must produce a V2V file whose r2r() matches the original."""
        M   = _rotation_z(5.0) @ _translation(2.0, 1.0)
        lta = LTA.from_matrix(M, "src.mgz", _GEOM, "dst.mgz", _GEOM, lta_type=1)
        lta.write(tmp_path / "out_v2v.lta", lta_type=0)
        lta2 = LTA.read(tmp_path / "out_v2v.lta")
        assert lta2.type == 0
        assert lta2.r2r() == pytest.approx(lta.r2r(), rel=1e-10)

    def test_corner_dist_method_translation(self, tmp_path):
        p = _write_lta(tmp_path / "t.lta", _translation(3.0, 4.0))
        assert LTA.read(p).corner_dist() == pytest.approx(5.0, rel=1e-5)

    def test_repr(self):
        lta = LTA.from_matrix(_IDENTITY, "src.mgz", _GEOM, "dst.mgz", _GEOM)
        assert "R2R"     in repr(lta)
        assert "src.mgz" in repr(lta)


# ---------------------------------------------------------------------------
# CLI (end-to-end)
# ---------------------------------------------------------------------------

@pytest.fixture()
def identity_lta(tmp_path: Path) -> str:
    return _write_lta(tmp_path / "identity.lta", _IDENTITY)


@pytest.fixture()
def rigid_lta(tmp_path: Path) -> str:
    return _write_lta(tmp_path / "rigid.lta", _rotation_z(5.0) @ _translation(2.0, 1.0, 0.5))


class TestLtaDiffCLI:

    def test_dist2_is_default(self, identity_lta, capsys):
        main([identity_lta])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(0.0, abs=1e-10)

    def test_dist1(self, rigid_lta, capsys):
        main([rigid_lta, "--dist", "1"])
        assert float(capsys.readouterr().out.strip()) > 0.0

    def test_dist3_pure_translation(self, tmp_path, capsys):
        lta = _write_lta(tmp_path / "trans.lta", _translation(3.0, 4.0))
        main([lta, "--dist", "3"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(5.0, rel=1e-5)

    def test_dist4(self, identity_lta, capsys):
        main([identity_lta, "--dist", "4"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(0.0, abs=1e-10)

    def test_dist5_det_is_one(self, rigid_lta, capsys):
        main([rigid_lta, "--dist", "5"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(1.0, rel=1e-4)

    def test_dist7_output_fields(self, identity_lta, capsys):
        main([identity_lta, "--dist", "7"])
        out = capsys.readouterr().out
        for field in ("Rot", "Trans", "Determinant", "Scales", "RotAngle"):
            assert field in out

    def test_normdiv_halves_result(self, rigid_lta, capsys):
        main([rigid_lta, "--dist", "2"])
        d_full = float(capsys.readouterr().out.strip())
        main([rigid_lta, "--dist", "2", "--normdiv", "2"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(d_full / 2.0, rel=1e-6)

    def test_invert1_changes_result(self, rigid_lta, capsys):
        main([rigid_lta, "--dist", "2"])
        d1 = float(capsys.readouterr().out.strip())
        main([rigid_lta, "--dist", "2", "--invert1"])
        d2 = float(capsys.readouterr().out.strip())
        # invert of a non-identity rigid should yield the same magnitude
        assert d1 == pytest.approx(d2, rel=1e-4)

    def test_invert2_requires_second_lta(self, identity_lta):
        with pytest.raises(SystemExit):
            main([identity_lta, "--invert2"])

    def test_vox_nonunit_voxelsize_identity(self, tmp_path, capsys):
        """Regression: identity V2V with non-1 mm voxels must give distance 0."""
        geom_2mm = {"dims": [128, 128, 128], "delta": [2.0, 2.0, 2.0],
                    "Mdc": np.eye(3), "Pxyz_c": np.zeros(3)}
        lta_path = str(tmp_path / "id_2mm.lta")
        LTA.from_matrix(_IDENTITY, "src.mgz", geom_2mm, "dst.mgz", geom_2mm,
                        lta_type=0).write(lta_path)
        main([lta_path, "--dist", "2", "--vox"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(0.0, abs=1e-10)

    def test_missing_lta_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            main([str(tmp_path / "does_not_exist.lta")])
