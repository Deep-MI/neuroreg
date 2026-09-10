"""Tests for the target-geometry construction helpers in neuroreg.image.geometry."""

from __future__ import annotations

from itertools import permutations, product

import nibabel as nib
import numpy as np
import pytest

from neuroreg.image import (
    build_grid_affine,
    direction_cosines_from_orientation,
    place_grid_at_cras,
    shape_from_fov,
)


def _all_orientation_codes() -> list[str]:
    """Every valid orientation code: one letter per anatomical axis, any order."""
    codes = []
    for letters in product("RL", "AP", "SI"):
        codes.extend("".join(perm) for perm in permutations(letters))
    return codes


def _cras(affine: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    center_vox = np.array([shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0, 1.0], dtype=np.float64)
    return (np.asarray(affine, dtype=np.float64) @ center_vox)[:3]


def _oblique_cosines() -> np.ndarray:
    rot = np.array(
        [
            [0.9362934, -0.2896295, 0.1986693],
            [0.3129918, 0.9447025, -0.0978434],
            [-0.1593451, 0.1537920, 0.9751703],
        ]
    )
    return rot


class TestDirectionCosinesFromOrientation:
    def test_lia_matches_the_freesurfer_column_convention(self):
        # Letter n is the RAS direction of voxel axis n, as column n.
        cosines = direction_cosines_from_orientation("LIA")

        assert np.array_equal(cosines[:, 0], [-1.0, 0.0, 0.0])  # L
        assert np.array_equal(cosines[:, 1], [0.0, 0.0, -1.0])  # I
        assert np.array_equal(cosines[:, 2], [0.0, 1.0, 0.0])  # A

    def test_ras_is_the_identity(self):
        assert np.array_equal(direction_cosines_from_orientation("RAS"), np.eye(3))

    @pytest.mark.parametrize("code", _all_orientation_codes())
    def test_all_48_codes_round_trip_through_nibabel(self, code: str):
        # The strongest available check on the column/sign convention: nibabel
        # must read back exactly the orientation we claim to have built.
        affine = np.eye(4)
        affine[:3, :3] = direction_cosines_from_orientation(code)

        assert "".join(nib.aff2axcodes(affine)) == code

    def test_accepts_lowercase_and_surrounding_whitespace(self):
        assert np.array_equal(
            direction_cosines_from_orientation(" lia "), direction_cosines_from_orientation("LIA")
        )

    @pytest.mark.parametrize("code", ["LI", "LIAS", "", "XYZ", "LIQ"])
    def test_rejects_codes_that_are_not_three_valid_letters(self, code: str):
        with pytest.raises(ValueError, match="three letters"):
            direction_cosines_from_orientation(code)

    @pytest.mark.parametrize("code", ["LRA", "LIS", "RRR", "API"])
    def test_rejects_codes_that_do_not_name_each_axis_once(self, code: str):
        with pytest.raises(ValueError, match="exactly once"):
            direction_cosines_from_orientation(code)

    def test_columns_are_unit_length(self):
        for code in _all_orientation_codes():
            cosines = direction_cosines_from_orientation(code)
            assert np.linalg.norm(cosines, axis=0) == pytest.approx([1.0, 1.0, 1.0])


class TestShapeFromFov:
    @pytest.mark.parametrize(
        ("vox", "expected"),
        [(1.0, 256), (0.8, 320), (0.5, 512), (2.0, 128)],
    )
    def test_exact_divisions_do_not_gain_a_voxel(self, vox: float, expected: int):
        assert shape_from_fov(np.full(3, 256.0), np.full(3, vox)) == (expected,) * 3

    def test_count_just_above_an_integer_snaps_down_instead_of_ceiling_up(self):
        # The case the epsilon exists for. 256 / 0.8 is exactly 320.0 in
        # float64, so it does not exercise this; a count carrying a small
        # relative residual above the integer does, and plain ceil gives 321.
        fov = np.full(3, 256.0 * (1.0 + 1e-9))
        assert float(fov[0] / 0.8) > 320.0
        assert int(np.ceil(fov[0] / 0.8)) == 321

        assert shape_from_fov(fov, np.full(3, 0.8)) == (320, 320, 320)

    def test_float32_voxel_size_still_gives_the_intended_size(self):
        # A zoom read back from a float32 header is not exactly 0.8.
        vox = np.full(3, float(np.float32(0.8)))

        assert shape_from_fov(np.full(3, 256.0), vox) == (320, 320, 320)

    def test_genuine_partial_voxels_round_up_so_the_fov_is_not_truncated(self):
        # 256 / 1.2 = 213.33; rounding to nearest would drop 0.4mm of FOV.
        assert shape_from_fov(np.full(3, 256.0), np.full(3, 1.2)) == (214, 214, 214)
        assert shape_from_fov(np.full(3, 256.0), np.full(3, 0.7)) == (366, 366, 366)

    def test_supports_anisotropic_fov_and_voxel_size(self):
        assert shape_from_fov(np.array([256.0, 240.0, 180.0]), np.array([1.0, 0.8, 1.2])) == (256, 300, 150)

    @pytest.mark.parametrize(
        ("fov", "vox"),
        [(np.full(3, 0.0), np.full(3, 1.0)), (np.full(3, -256.0), np.full(3, 1.0)), (np.full(3, 256.0), np.zeros(3))],
    )
    def test_rejects_non_positive_inputs(self, fov: np.ndarray, vox: np.ndarray):
        with pytest.raises(ValueError, match="must be positive"):
            shape_from_fov(fov, vox)


class TestPlaceGridAtCras:
    def test_centre_lands_on_the_requested_point(self):
        affine = np.diag([0.8, 0.8, 0.8, 1.0])
        affine[:3, 3] = [-40.0, 17.0, -9.0]
        shape = (10, 10, 10)

        moved = place_grid_at_cras(affine, shape, np.zeros(3))

        assert _cras(moved, shape) == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)

    def test_only_the_translation_column_changes(self):
        affine = np.eye(4)
        affine[:3, :3] = _oblique_cosines() @ np.diag([0.9, 1.3, 2.1])
        affine[:3, 3] = [11.0, -23.0, 7.0]

        moved = place_grid_at_cras(affine, (15, 17, 9), np.array([1.0, 2.0, 3.0]))

        assert np.array_equal(moved[:3, :3], affine[:3, :3])
        assert np.array_equal(moved[3], affine[3])

    def test_odd_anisotropic_oblique_grid_uses_the_shape_over_two_convention(self):
        affine = np.eye(4)
        affine[:3, :3] = _oblique_cosines() @ np.diag([0.9, 1.3, 2.1])
        shape = (15, 17, 9)
        target = np.array([-4.25, 6.5, 0.0])

        moved = place_grid_at_cras(affine, shape, target)

        assert _cras(moved, shape) == pytest.approx(target, abs=1e-9)
        # (shape - 1) / 2 would land somewhere else entirely.
        half_voxel = affine[:3, :3] @ (np.ones(3) / 2.0)
        assert np.linalg.norm(half_voxel) > 1e-3

    def test_does_not_mutate_the_input(self):
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        before = affine.copy()

        place_grid_at_cras(affine, (4, 4, 4), np.array([5.0, 5.0, 5.0]))

        assert np.array_equal(affine, before)


class TestBuildGridAffine:
    def test_assembles_the_requested_components(self):
        shape = (320, 320, 320)
        affine = build_grid_affine(
            cosines=direction_cosines_from_orientation("LIA"),
            vox_size=np.full(3, 0.8),
            shape=shape,
            cras=np.zeros(3),
        )

        assert np.linalg.norm(affine[:3, :3], axis=0) == pytest.approx([0.8, 0.8, 0.8])
        assert "".join(nib.aff2axcodes(affine)) == "LIA"
        assert _cras(affine, shape) == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)

    def test_matches_the_mgh_pxyz_c_field(self):
        # c_ras is exactly what MGH stores as Pxyz_c, so a real header must agree.
        shape = (151, 151, 186)
        cras = np.array([-3.5, 12.0, -7.25])
        affine = build_grid_affine(
            cosines=direction_cosines_from_orientation("LIA"),
            vox_size=np.full(3, 1.0),
            shape=shape,
            cras=cras,
        )

        img = nib.MGHImage(np.zeros(shape, dtype=np.uint8), affine)

        assert np.asarray(img.header["Pxyz_c"], dtype=np.float64) == pytest.approx(cras, abs=1e-3)

    def test_supports_anisotropic_voxel_sizes(self):
        vox = np.array([0.9, 1.3, 2.1])
        affine = build_grid_affine(
            cosines=direction_cosines_from_orientation("RAS"),
            vox_size=vox,
            shape=(8, 9, 10),
            cras=np.zeros(3),
        )

        assert np.linalg.norm(affine[:3, :3], axis=0) == pytest.approx(vox)

    def test_round_trips_an_existing_geometry_through_decompose_and_rebuild(self):
        # This is what "mri geom --like x" does: split an affine into components
        # and reassemble it. The result must be the same grid.
        original = np.eye(4)
        original[:3, :3] = _oblique_cosines() @ np.diag([0.9, 1.3, 2.1])
        original[:3, 3] = [11.0, -23.0, 7.0]
        shape = (15, 17, 9)
        zooms = np.linalg.norm(original[:3, :3], axis=0)

        rebuilt = build_grid_affine(
            cosines=original[:3, :3] / zooms,
            vox_size=zooms,
            shape=shape,
            cras=_cras(original, shape),
        )

        assert rebuilt == pytest.approx(original, abs=1e-12)
