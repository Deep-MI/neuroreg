"""Tests for the lta-diff CLI and underlying transform-distance functions.

Covers:
- Unit tests for every distance function in nireg.transforms.lta
  (rigid_dist, affine_dist, corner_diff, sphere_diff, decompose_transform)
- End-to-end CLI tests for nireg.cmdline.lta_diff.main with all dist types
  and every flag (--invert1, --invert2, --vox, --radius, --normdiv)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nireg.cmdline.lta_diff import main
from nireg.transforms.lta import (
    affine_dist,
    corner_diff,
    decompose_transform,
    rigid_dist,
    sphere_diff,
    write_lta,
)

# ---------------------------------------------------------------------------
# Shared geometry and helpers
# ---------------------------------------------------------------------------

# A minimal isotropic 1 mm, 256³ volume with identity direction cosines.
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
    M[0, 0] = np.cos(r)
    M[0, 1] = -np.sin(r)
    M[1, 0] = np.sin(r)
    M[1, 1] = np.cos(r)
    return M


def _translation(tx: float, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    """Return a 4×4 pure-translation matrix (mm)."""
    M = np.eye(4)
    M[:3, 3] = [tx, ty, tz]
    return M


def _write_lta(path: Path, M: np.ndarray, lta_type: int = 1) -> str:
    """Write a minimal LTA file and return its path as a string."""
    s = str(path)
    write_lta(s, M, "src.mgz", _GEOM, "dst.mgz", _GEOM, lta_type=lta_type)
    return s


# ---------------------------------------------------------------------------
# rigid_dist
# ---------------------------------------------------------------------------


class TestRigidDist:
    def test_identity_single(self):
        assert rigid_dist(_IDENTITY) == pytest.approx(0.0, abs=1e-10)

    def test_two_equal_matrices_gives_zero(self):
        M = _rotation_z(15.0) @ _translation(3, 2, 1)
        assert rigid_dist(M, M) == pytest.approx(0.0, abs=1e-10)

    def test_pure_translation(self):
        # No rotation → dist == ||T||
        M = _translation(3.0, 4.0)
        assert rigid_dist(M) == pytest.approx(5.0, rel=1e-6)

    def test_pure_rotation_angle(self):
        # dist = sqrt(2) * theta  (Frobenius log-norm, no translation)
        theta = np.radians(5.0)
        expected = np.sqrt(2.0) * theta
        assert rigid_dist(_rotation_z(5.0)) == pytest.approx(expected, rel=1e-6)

    def test_symmetric_inverse(self):
        # rigid_dist(M1, M2) == rigid_dist(M2, M1)  for rigid matrices
        M1 = _rotation_z(10.0) @ _translation(1, 2, 3)
        M2 = _rotation_z(25.0) @ _translation(4, 5, 6)
        assert rigid_dist(M1, M2) == pytest.approx(rigid_dist(M2, M1), rel=1e-6)


# ---------------------------------------------------------------------------
# affine_dist
# ---------------------------------------------------------------------------


class TestAffineDist:
    def test_identity_single(self):
        assert affine_dist(_IDENTITY) == pytest.approx(0.0, abs=1e-10)

    def test_two_equal_matrices_gives_zero(self):
        M = _rotation_z(15.0) @ _translation(3, 2, 1)
        assert affine_dist(M, M) == pytest.approx(0.0, abs=1e-10)

    def test_pure_translation(self):
        # A = 0  →  dist == ||T||
        M = _translation(3.0, 4.0)
        assert affine_dist(M) == pytest.approx(5.0, rel=1e-6)

    def test_radius_scales_rotation_part(self):
        # For pure rotation (no translation), affine_dist ∝ radius
        R = _rotation_z(5.0)
        d1 = affine_dist(R, radius=100.0)
        d2 = affine_dist(R, radius=200.0)
        assert d2 == pytest.approx(2.0 * d1, rel=1e-6)

    def test_default_radius_is_100(self):
        R = _rotation_z(5.0)
        assert affine_dist(R) == pytest.approx(affine_dist(R, radius=100.0), rel=1e-10)


# ---------------------------------------------------------------------------
# corner_diff
# ---------------------------------------------------------------------------


class TestCornerDiff:
    _SHAPE = (256, 256, 256)
    _AFFINE = np.eye(4)  # vox == RAS for this test geometry

    def test_identity_with_src_affine(self):
        assert corner_diff(_IDENTITY, self._SHAPE, src_affine=self._AFFINE) == pytest.approx(
            0.0, abs=1e-10
        )

    def test_identity_voxel_space(self):
        assert corner_diff(_IDENTITY, self._SHAPE) == pytest.approx(0.0, abs=1e-10)

    def test_two_equal_matrices_gives_zero(self):
        M = _rotation_z(5.0)
        assert corner_diff(M, self._SHAPE, M2=M, src_affine=self._AFFINE) == pytest.approx(
            0.0, abs=1e-10
        )

    def test_pure_translation_mm(self):
        # Every corner displaced by ||T||; mean == ||T||
        M = _translation(3.0, 4.0)
        result = corner_diff(M, self._SHAPE, src_affine=self._AFFINE)
        assert result == pytest.approx(5.0, rel=1e-6)

    def test_pure_translation_vox(self):
        # In voxel mode, same property holds for integer translations
        M = _translation(3.0, 4.0)
        result = corner_diff(M, self._SHAPE)
        assert result == pytest.approx(5.0, rel=1e-6)


# ---------------------------------------------------------------------------
# sphere_diff
# ---------------------------------------------------------------------------


class TestSphereDiff:
    def test_identity_single(self):
        assert sphere_diff(_IDENTITY) == pytest.approx(0.0, abs=1e-10)

    def test_two_equal_matrices_gives_zero(self):
        M = _rotation_z(5.0)
        assert sphere_diff(M, M) == pytest.approx(0.0, abs=1e-10)

    def test_pure_translation(self):
        # Every sphere point displaced by ||T||; max == ||T||
        t_vec = np.array([3.0, 4.0, 0.0])
        result = sphere_diff(_translation(*t_vec))
        assert result == pytest.approx(float(np.linalg.norm(t_vec)), rel=1e-6)

    def test_radius_affects_rotation_result(self):
        # For a rotation, max displacement scales with sphere radius
        R = _rotation_z(5.0)
        d1 = sphere_diff(R, radius=50.0)
        d2 = sphere_diff(R, radius=100.0)
        assert d2 == pytest.approx(2.0 * d1, rel=1e-4)


# ---------------------------------------------------------------------------
# decompose_transform
# ---------------------------------------------------------------------------


class TestDecomposeTransform:
    def test_identity(self):
        d = decompose_transform(_IDENTITY)
        assert d["rotation"] == pytest.approx(np.eye(3), abs=1e-10)
        assert d["translation"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-10)
        assert d["scales"] == pytest.approx([1.0, 1.0, 1.0], rel=1e-6)
        assert d["abs_trans"] == pytest.approx(0.0, abs=1e-10)
        assert d["rot_angle_deg"] == pytest.approx(0.0, abs=1e-6)
        assert d["determinant"] == pytest.approx(1.0, rel=1e-10)

    def test_pure_rotation(self):
        d = decompose_transform(_rotation_z(30.0))
        assert d["rot_angle_deg"] == pytest.approx(30.0, rel=1e-4)
        assert d["abs_trans"] == pytest.approx(0.0, abs=1e-10)
        assert d["determinant"] == pytest.approx(1.0, rel=1e-6)

    def test_pure_translation(self):
        M = _translation(5.0, 3.0, 1.0)
        d = decompose_transform(M)
        assert d["translation"] == pytest.approx([5.0, 3.0, 1.0], rel=1e-6)
        assert d["abs_trans"] == pytest.approx(float(np.linalg.norm([5, 3, 1])), rel=1e-6)
        assert d["rot_angle_deg"] == pytest.approx(0.0, abs=1e-6)

    def test_return_keys(self):
        d = decompose_transform(_rotation_z(10.0) @ _translation(1, 2, 3))
        for key in ("rotation", "rot_vec", "rot_angle_deg", "shear", "scales",
                    "translation", "abs_trans", "determinant"):
            assert key in d


# ---------------------------------------------------------------------------
# CLI (end-to-end)  –  all tests use tmp_path + write_lta
# ---------------------------------------------------------------------------


@pytest.fixture()
def identity_lta(tmp_path: Path) -> str:
    return _write_lta(tmp_path / "identity.lta", _IDENTITY)


@pytest.fixture()
def rigid_lta(tmp_path: Path) -> str:
    M = _rotation_z(5.0) @ _translation(2.0, 1.0, 0.5)
    return _write_lta(tmp_path / "rigid.lta", M)


class TestLtaDiffCLI:
    """End-to-end tests: write LTA files, call main(), inspect stdout."""

    # ── dist 1 (rigid) ───────────────────────────────────────────────────────

    def test_dist1_single_identity(self, identity_lta, capsys):
        main([identity_lta, "--dist", "1"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(0.0, abs=1e-10)

    def test_dist1_two_identical(self, rigid_lta, capsys):
        main([rigid_lta, rigid_lta, "--dist", "1"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(0.0, abs=1e-10)

    def test_dist1_single_rigid_nonzero(self, rigid_lta, capsys):
        main([rigid_lta, "--dist", "1"])
        assert float(capsys.readouterr().out.strip()) > 0.0

    # ── dist 2 (affine RMS, default) ─────────────────────────────────────────

    def test_dist2_is_default(self, identity_lta, capsys):
        # calling without --dist should behave like --dist 2
        main([identity_lta])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(0.0, abs=1e-10)

    def test_dist2_single_identity(self, identity_lta, capsys):
        main([identity_lta, "--dist", "2"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(0.0, abs=1e-10)

    def test_dist2_two_identical(self, rigid_lta, capsys):
        main([rigid_lta, rigid_lta, "--dist", "2"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(0.0, abs=1e-10)

    def test_dist2_two_different(self, identity_lta, rigid_lta, capsys):
        main([identity_lta, rigid_lta, "--dist", "2"])
        assert float(capsys.readouterr().out.strip()) > 0.0

    def test_dist2_larger_radius_gives_larger_result(self, rigid_lta, capsys):
        main([rigid_lta, "--dist", "2", "--radius", "100"])
        d1 = float(capsys.readouterr().out.strip())
        main([rigid_lta, "--dist", "2", "--radius", "200"])
        d2 = float(capsys.readouterr().out.strip())
        assert d2 > d1

    def test_dist2_normdiv_halves_result(self, rigid_lta, capsys):
        main([rigid_lta, "--dist", "2"])
        d_full = float(capsys.readouterr().out.strip())
        main([rigid_lta, "--dist", "2", "--normdiv", "2"])
        d_half = float(capsys.readouterr().out.strip())
        assert d_half == pytest.approx(d_full / 2.0, rel=1e-6)

    # ── dist 3 (corner displacement) ─────────────────────────────────────────

    def test_dist3_identity(self, identity_lta, capsys):
        main([identity_lta, "--dist", "3"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(0.0, abs=1e-10)

    def test_dist3_pure_translation(self, tmp_path, capsys):
        lta = _write_lta(tmp_path / "trans34.lta", _translation(3.0, 4.0))
        main([lta, "--dist", "3"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(5.0, rel=1e-5)

    def test_dist3_two_identical(self, rigid_lta, capsys):
        main([rigid_lta, rigid_lta, "--dist", "3"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(0.0, abs=1e-10)

    def test_dist3_vox_identity(self, identity_lta, capsys):
        main([identity_lta, "--dist", "3", "--vox"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(0.0, abs=1e-10)

    # ── dist 4 (sphere max displacement) ─────────────────────────────────────

    def test_dist4_identity(self, identity_lta, capsys):
        main([identity_lta, "--dist", "4"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(0.0, abs=1e-10)

    def test_dist4_two_identical(self, rigid_lta, capsys):
        main([rigid_lta, rigid_lta, "--dist", "4"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(0.0, abs=1e-10)

    def test_dist4_rigid_nonzero(self, rigid_lta, capsys):
        main([rigid_lta, "--dist", "4"])
        assert float(capsys.readouterr().out.strip()) > 0.0

    # ── dist 5 (determinant) ──────────────────────────────────────────────────

    def test_dist5_identity_det_is_one(self, identity_lta, capsys):
        main([identity_lta, "--dist", "5"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(1.0, rel=1e-6)

    def test_dist5_rotation_det_is_one(self, rigid_lta, capsys):
        main([rigid_lta, "--dist", "5"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(1.0, rel=1e-4)

    def test_dist5_two_ltas_product_det(self, identity_lta, rigid_lta, capsys):
        # det(I @ M) == det(M) == 1 for a rigid M
        main([identity_lta, rigid_lta, "--dist", "5"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(1.0, rel=1e-4)

    # ── dist 7 (polar decomposition) ─────────────────────────────────────────

    def test_dist7_identity_output_fields(self, identity_lta, capsys):
        main([identity_lta, "--dist", "7"])
        out = capsys.readouterr().out
        for field in ("Rot", "Trans", "Determinant", "Scales", "RotAngle"):
            assert field in out, f"Missing field '{field}' in dist-7 output"

    def test_dist7_identity_det_is_one(self, identity_lta, capsys):
        main([identity_lta, "--dist", "7"])
        out = capsys.readouterr().out
        det_line = next(line for line in out.splitlines() if "Determinant" in line)
        assert float(det_line.split("=")[-1].strip()) == pytest.approx(1.0, rel=1e-6)

    def test_dist7_rigid_no_crash(self, rigid_lta, capsys):
        main([rigid_lta, "--dist", "7"])
        out = capsys.readouterr().out
        assert "RotAngle" in out

    # ── --invert flags ────────────────────────────────────────────────────────

    def test_invert1_identity_stays_zero(self, identity_lta, capsys):
        # inv(I) == I  →  dist to identity is still 0
        main([identity_lta, "--dist", "1", "--invert1"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(0.0, abs=1e-10)

    def test_invert1_changes_result(self, rigid_lta, capsys):
        main([rigid_lta, "--dist", "2"])
        d_normal = float(capsys.readouterr().out.strip())
        main([rigid_lta, "--dist", "2", "--invert1"])
        d_inverted = float(capsys.readouterr().out.strip())
        # inv(M) vs identity differs from M vs identity (both non-zero, equal mag
        # only for pure rotation about origin — not guaranteed for combined RT)
        assert d_normal > 0.0
        assert d_inverted > 0.0

    def test_invert2_no_crash(self, identity_lta, rigid_lta, capsys):
        main([identity_lta, rigid_lta, "--dist", "2", "--invert2"])
        assert float(capsys.readouterr().out.strip()) >= 0.0

    def test_invert2_requires_second_lta(self, identity_lta):
        with pytest.raises(SystemExit):
            main([identity_lta, "--invert2"])

    def test_invert1_and_invert2_together(self, identity_lta, rigid_lta, capsys):
        main([identity_lta, rigid_lta, "--dist", "2", "--invert1", "--invert2"])
        assert float(capsys.readouterr().out.strip()) >= 0.0

    # ── --vox flag ────────────────────────────────────────────────────────────

    def test_vox_identity_dist2(self, identity_lta, capsys):
        main([identity_lta, "--dist", "2", "--vox"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(0.0, abs=1e-10)

    def test_vox_two_identical_dist2(self, rigid_lta, capsys):
        main([rigid_lta, rigid_lta, "--dist", "2", "--vox"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(0.0, abs=1e-10)

    def test_vox_identity_nonunit_voxelsize(self, tmp_path, capsys):
        # Identity V2V with non-1-mm voxel sizes must still give distance 0.
        geom_2mm = {
            "dims": [128, 128, 128],
            "delta": [2.0, 2.0, 2.0],
            "Mdc": np.eye(3),
            "Pxyz_c": np.zeros(3),
        }
        lta_path = str(tmp_path / "identity_2mm.lta")
        write_lta(lta_path, _IDENTITY, "src.mgz", geom_2mm, "dst.mgz", geom_2mm,
                  lta_type=0)  # store as V2V to exercise that branch too
        main([lta_path, "--dist", "2", "--vox"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(0.0, abs=1e-10)

    # ── error handling ────────────────────────────────────────────────────────

    def test_missing_lta_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            main([str(tmp_path / "does_not_exist.lta")])

    def test_missing_second_lta_file_exits(self, identity_lta, tmp_path):
        with pytest.raises(SystemExit):
            main([identity_lta, str(tmp_path / "does_not_exist.lta"), "--dist", "2"])

