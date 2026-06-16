"""Tests for neuroreg.image: cubic B-spline interpolation and Gaussian pyramid.

Cubic interpolation is designed to match FreeSurfer's ``mri_convert -rt cubic`` /
``mri_vol2vol --interp cubic`` (``MRItoBSpline`` + ``MRIsampleBSpline``), which
prefilters the image into cubic B-spline coefficients and evaluates them with
mirror ("DCT-I") boundary handling. This was validated against locally built
FreeSurfer binaries; the cubic tests instead lock down the same behavior against
``scipy.ndimage.map_coordinates(order=3, mode="mirror")``, which uses the
identical boundary convention, so the regression check has no external
dependency on a FreeSurfer install.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from neuroreg.image import build_gaussian_pyramid
from neuroreg.image.bspline import downsample2_bspline
from neuroreg.image.map import map_r2r
from neuroreg.image.pyramid import _PYRAMID_FILTER, _smooth3d

# ---------------------------------------------------------------------------
# Helpers for cubic interpolation tests
# ---------------------------------------------------------------------------


def _rotation_translation_matrix(
    degrees: float,
    axis: tuple[float, float, float],
    translation: tuple[float, float, float],
) -> np.ndarray:
    theta = np.deg2rad(degrees)
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = translation
    return M


# ---------------------------------------------------------------------------
# Helpers for pyramid / downsampling tests
# ---------------------------------------------------------------------------

_REFERENCE_CENTERED_CUBIC_REDUCE_FILTER = (
    0.708792,
    0.328616,
    -0.165157,
    -0.114448,
    0.0944036,
    0.0543881,
    -0.05193,
    -0.0284868,
    0.0281854,
    0.0152877,
    -0.0152508,
    -0.00825077,
    0.00824629,
    0.00445865,
    -0.0044582,
    -0.00241009,
    0.00241022,
    0.00130278,
    -0.00130313,
    -0.000704109,
    0.000704784,
)


def _pseudo_mirror_index(index: int, length: int) -> int:
    period = 2 * length
    wrapped = index % period
    return wrapped if wrapped < length else period - wrapped - 1


def _reference_reduce_centered_1d(values: torch.Tensor) -> torch.Tensor:
    n_out = values.numel() // 2
    n_even = n_out * 2
    clipped = values[:n_even].double()
    filtered = torch.empty(n_even, dtype=torch.float64)

    for position in range(n_even):
        acc = clipped[position] * _REFERENCE_CENTERED_CUBIC_REDUCE_FILTER[0]
        for offset, coeff in enumerate(_REFERENCE_CENTERED_CUBIC_REDUCE_FILTER[1:], start=1):
            left = _pseudo_mirror_index(position - offset, n_even)
            right = _pseudo_mirror_index(position + offset, n_even)
            acc += coeff * (clipped[left] + clipped[right])
        filtered[position] = acc

    return 0.5 * (filtered[0::2] + filtered[1::2])


# ---------------------------------------------------------------------------
# Cubic interpolation tests
# ---------------------------------------------------------------------------


class TestCubicMode:
    def test_cubic_matches_scipy_spline_interpolation(self):
        rng = np.random.default_rng(0)
        data = rng.random((12, 14, 10)).astype(np.float64)
        src = torch.from_numpy(data).float()
        affine = torch.eye(4, dtype=torch.float32)

        M = _rotation_translation_matrix(4.0, (0.2, 0.5, 0.83), (0.3, -0.2, 0.15))
        r2r = torch.from_numpy(M).float()

        # padding_mode="border" leaves the mirror-extrapolated value everywhere,
        # matching scipy's unconditional "mirror" boundary extension exactly.
        out = map_r2r(
            src,
            r2r,
            source_affine=affine,
            target_affine=affine,
            target_shape=data.shape,
            mode="cubic",
            padding_mode="border",
        )

        v2v = np.linalg.inv(M)  # identity affines => target_vox -> source_vox is just inv(r2r)
        ii, jj, kk = np.meshgrid(*[np.arange(s) for s in data.shape], indexing="ij")
        ones = np.ones_like(ii, dtype=np.float64)
        coords = np.stack([ii, jj, kk, ones]).reshape(4, -1).astype(np.float64)
        src_vox = (v2v @ coords)[:3].reshape(3, *data.shape)
        ref = ndimage.map_coordinates(data, src_vox, order=3, mode="mirror", prefilter=True)

        diff = out.numpy().astype(np.float64) - ref
        assert np.mean(np.abs(diff)) < 1e-5
        assert np.max(np.abs(diff)) < 1e-4

    def test_cubic_identity_transform_reproduces_image(self):
        rng = np.random.default_rng(1)
        data = rng.random((9, 11, 8)).astype(np.float32)
        src = torch.from_numpy(data)
        affine = torch.eye(4, dtype=torch.float32)
        r2r = torch.eye(4, dtype=torch.float32)

        out = map_r2r(src, r2r, source_affine=affine, target_affine=affine, target_shape=data.shape, mode="cubic")

        # An interpolating spline reproduces the original samples exactly at
        # grid points (up to float32 numerical precision).
        np.testing.assert_allclose(out.numpy(), data, atol=1e-4)

    def test_cubic_zeros_padding_masks_out_of_bounds(self):
        data = np.ones((10, 10, 10), dtype=np.float32)
        src = torch.from_numpy(data)
        affine = torch.eye(4, dtype=torch.float32)

        # Shift the target grid far outside the source FOV.
        M = np.eye(4)
        M[:3, 3] = [50.0, 0.0, 0.0]
        r2r = torch.from_numpy(M).float()

        out_zeros = map_r2r(
            src,
            r2r,
            source_affine=affine,
            target_affine=affine,
            target_shape=data.shape,
            mode="cubic",
            padding_mode="zeros",
        )
        assert torch.all(out_zeros == 0)

        out_border = map_r2r(
            src,
            r2r,
            source_affine=affine,
            target_affine=affine,
            target_shape=data.shape,
            mode="cubic",
            padding_mode="border",
        )
        # Mirror-extrapolated values stay close to the (constant) source image
        # rather than being zeroed.
        assert torch.allclose(out_border, torch.ones_like(out_border), atol=1e-3)

    def test_cubic_padding_value_fills_out_of_bounds(self):
        data = np.ones((10, 10, 10), dtype=np.float32)
        src = torch.from_numpy(data)
        affine = torch.eye(4, dtype=torch.float32)

        M = np.eye(4)
        M[:3, 3] = [50.0, 0.0, 0.0]
        r2r = torch.from_numpy(M).float()

        out = map_r2r(
            src,
            r2r,
            source_affine=affine,
            target_affine=affine,
            target_shape=data.shape,
            mode="cubic",
            padding_mode="zeros",
            padding_value=-3.0,
        )
        assert torch.all(out == -3.0)


# ---------------------------------------------------------------------------
# Pyramid / downsampling tests
# ---------------------------------------------------------------------------


def test_downsample2_bspline_matches_centered_cubic_reference_for_odd_length_axis():
    line = torch.tensor([0.0, 1.0, 4.0, 9.0, 16.0], dtype=torch.float32)
    reduced = downsample2_bspline(line[:, None, None])

    expected = _reference_reduce_centered_1d(line)

    assert reduced.shape == (2, 1, 1)
    assert torch.allclose(reduced[:, 0, 0].double(), expected, atol=1e-6, rtol=1e-6)


def test_downsample2_bspline_preserves_constant_values_and_singleton_axes():
    volume = torch.full((1, 5, 7), 3.25, dtype=torch.float32)

    reduced = downsample2_bspline(volume)

    assert reduced.shape == (1, 2, 3)
    assert torch.allclose(reduced, torch.full((1, 2, 3), 3.25, dtype=torch.float32), atol=1e-5, rtol=1e-5)


def test_build_gaussian_pyramid_uses_bspline_reduction_and_centered_affine():
    image = torch.arange(35, dtype=torch.float32).reshape(5, 7, 1)
    affine = torch.eye(4, dtype=torch.float32)

    levels, affines = build_gaussian_pyramid(image, affine, limits=(torch.tensor(0), torch.tensor(1)))

    expected_level = downsample2_bspline(_smooth3d(image, _PYRAMID_FILTER, padding_mode="replicate"))
    expected_affine = torch.eye(4, dtype=torch.float32)
    expected_affine[0, 0] = 2.0
    expected_affine[1, 1] = 2.0
    expected_affine[0, 3] = 0.5
    expected_affine[1, 3] = 0.5

    assert [tuple(int(v) for v in level.shape) for level in levels] == [(5, 7, 1), (2, 3, 1)]
    assert torch.allclose(levels[1], expected_level, atol=1e-6, rtol=1e-6)
    assert torch.allclose(affines[1], expected_affine, atol=1e-6, rtol=1e-6)
