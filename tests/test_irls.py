"""Tests for IRLS rigid registration (neuroreg.imreg.irls)."""

from typing import Any, cast

import pytest
import torch

from neuroreg.imreg.init import get_init_vox2vox, get_ixform_centroids
from neuroreg.imreg.irls import (
    _sqrt_tukey,
    compute_partials,
    construct_Ab,
    irls_inner_loop,
    register_irls,
)
from neuroreg.imreg.robreg import register_irls_pyramid
from neuroreg.transforms import affine_dist, params_to_rigid_matrix

# ---------------------------------------------------------------------------
# _sqrt_tukey
# ---------------------------------------------------------------------------

class TestSqrtTukey:
    def test_zero_residual_gives_one(self):
        r = torch.zeros(10)
        w = _sqrt_tukey(r, sat=4.685)
        assert torch.allclose(w, torch.ones(10))

    def test_large_residual_gives_zero(self):
        r = torch.full((5,), 10.0)
        w = _sqrt_tukey(r, sat=4.685)
        assert torch.all(w == 0)

    def test_intermediate_value(self):
        sat = 4.0
        r = torch.tensor([2.0])
        w = _sqrt_tukey(r, sat=sat)
        expected = 1.0 - (2.0 / 4.0) ** 2  # = 0.75
        assert abs(w.item() - expected) < 1e-5

    def test_no_negative_outputs(self):
        r = torch.linspace(-10, 10, 100)
        w = _sqrt_tukey(r, sat=4.685)
        assert torch.all(w >= 0)


# ---------------------------------------------------------------------------
# params_to_rigid_matrix
# ---------------------------------------------------------------------------

class TestParamsToRigidMatrix:
    def test_zero_params_is_identity(self):
        p = torch.zeros(6)
        T = params_to_rigid_matrix(p)
        assert T.shape == (4, 4)
        assert torch.allclose(T, torch.eye(4), atol=1e-5)

    def test_pure_translation(self):
        p = torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        T = params_to_rigid_matrix(p)
        assert T[2, 3] == pytest.approx(2.0, abs=1e-5)

    def test_rotation_matrix_is_orthogonal(self):
        p = torch.tensor([0.0, 0.0, 0.0, 0.1, 0.2, 0.3])
        T = params_to_rigid_matrix(p)
        R = T[:3, :3]
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)


# ---------------------------------------------------------------------------
# construct_Ab
# ---------------------------------------------------------------------------

class TestConstructAb:
    def test_output_shapes(self):
        src = torch.rand(16, 16, 16)
        trg = torch.rand(16, 16, 16)
        A, b, valid = construct_Ab(src, trg)
        n = valid.sum().item()
        assert A.shape == (n, 6)
        assert b.shape == (n,)
        assert valid.shape == (16 * 16 * 16,)

    def test_valid_mask_excludes_zeros(self):
        src = torch.rand(12, 12, 12)
        trg = torch.rand(12, 12, 12)
        src[:3, :, :] = 0.0
        A, b, valid = construct_Ab(src, trg)
        src_flat = src.reshape(-1)
        assert torch.all(src_flat[valid].abs() > 1e-5)

    def test_compute_partials_matches_internal_dhw_axis_convention(self):
        d, h, w = 9, 10, 11
        zz, yy, xx = torch.meshgrid(
            torch.arange(d, dtype=torch.float32),
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing='ij',
        )

        fx, fy, fz, _ = compute_partials(xx)
        assert fx[4, 5, 6].item() > 0.9
        assert abs(fy[4, 5, 6].item()) < 1e-4
        assert abs(fz[4, 5, 6].item()) < 1e-4

        fx, fy, fz, _ = compute_partials(yy)
        assert abs(fx[4, 5, 6].item()) < 1e-4
        assert fy[4, 5, 6].item() > 0.9
        assert abs(fz[4, 5, 6].item()) < 1e-4

        fx, fy, fz, _ = compute_partials(zz)
        assert abs(fx[4, 5, 6].item()) < 1e-4
        assert abs(fy[4, 5, 6].item()) < 1e-4
        assert fz[4, 5, 6].item() > 0.9


# ---------------------------------------------------------------------------
# irls_inner_loop
# ---------------------------------------------------------------------------

class TestIrlsInnerLoop:

    def test_near_zero_system(self):
        A = torch.randn(300, 6)
        b = torch.zeros(300)
        p, _, _, _ = irls_inner_loop(A, b, sat=4.685)
        assert torch.allclose(p, torch.zeros(6), atol=1e-4)

    def test_outliers_get_zero_weight(self):
        A = torch.randn(500, 6)
        b = torch.randn(500)
        b[:10] = 1000.0
        _, w, _, _ = irls_inner_loop(A, b, sat=4.685)
        assert w[:10].sum() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# register_irls  (single-level)
# ---------------------------------------------------------------------------

class TestRegisterIrls:
    def _make_images(self, size=24):
        torch.manual_seed(0)
        img = torch.rand(size, size, size)
        return img, img.clone()

    def test_info_keys_present(self):
        src, trg = self._make_images()
        _, info = register_irls(src, trg, nmax=2)
        for key in ('iterations', 'converged', 'dists', 'weights', 'valid_mask', 'sigma_hist'):
            assert key in info

    def test_identical_images_near_identity(self):
        src, trg = self._make_images()
        T, _ = register_irls(src, trg, nmax=3)
        assert affine_dist(T, torch.eye(4)) < 1.0

    def test_symmetric_mode_runs(self):
        src, trg = self._make_images()
        T, info = register_irls(src, trg, nmax=2, symmetric=True)
        assert T.shape == (4, 4)
        assert info['iterations'] >= 1

    def test_iterations_respect_nmax(self):
        src, trg = self._make_images()
        _, info = register_irls(src, trg, nmax=2)
        assert info['iterations'] <= 2

    def test_weights_shape_matches_valid_mask(self):
        src, trg = self._make_images()
        _, info = register_irls(src, trg, nmax=2)
        w = info['weights']
        vm = info['valid_mask']
        assert w is not None and vm is not None
        assert w.shape == (vm.sum().item(),)

    def test_first_iteration_weights_are_kept_even_if_error_is_nan(self, monkeypatch: pytest.MonkeyPatch):
        src, trg = self._make_images()

        def fake_irls_inner_loop(A, b, sat=4.685, max_iterations=20, verbose=False):
            return torch.zeros(6), torch.ones(A.shape[0]), 1.0, float("nan")

        monkeypatch.setattr("neuroreg.imreg.irls.irls_inner_loop", fake_irls_inner_loop)

        _, info = register_irls(src, trg, nmax=1)
        w = info['weights']
        vm = info['valid_mask']
        assert w is not None and vm is not None
        assert w.shape == (vm.sum().item(),)
        assert torch.all(w == 1)

    def test_adaptive_sat_runs_and_records_sigma_history(self):
        src, trg = self._make_images()
        T, info = register_irls(
            src,
            trg,
            nmax=2,
            adaptive_sat=True,
            target_outlier_pct=5.0,
        )
        assert T.shape == (4, 4)
        assert len(info['sigma_hist']) == info['iterations']
        assert all(isinstance(v, float) for v in info['sigma_hist'])


# ---------------------------------------------------------------------------
# register_irls_pyramid
# ---------------------------------------------------------------------------

class TestRegisterIrlsPyramid:
    def _aff(self):
        return torch.eye(4)  # 1mm isotropic

    def test_iso_affine_stored_in_info(self):
        torch.manual_seed(2)
        img = torch.rand(20, 20, 20)
        aff = self._aff()
        _, all_info = register_irls_pyramid(
            img, img.clone(), src_affine=aff, trg_affine=aff,
            min_voxels=8, max_voxels=16, nmax=2, isotropic=True,
        )
        for info in all_info:
            assert 'iso_affine' in info

    def test_symmetric_and_directed_both_run(self):
        torch.manual_seed(3)
        img = torch.rand(20, 20, 20)
        aff = self._aff()
        T_dir, _ = register_irls_pyramid(
            img, img.clone(), src_affine=aff, trg_affine=aff,
            min_voxels=8, max_voxels=16, nmax=2, isotropic=False, symmetric=False,
        )
        T_sym, _ = register_irls_pyramid(
            img, img.clone(), src_affine=aff, trg_affine=aff,
            min_voxels=8, max_voxels=16, nmax=2, isotropic=False, symmetric=True,
        )
        assert T_dir.shape == (4, 4)
        assert T_sym.shape == (4, 4)

    def test_isotropic_requires_affines(self):
        img = torch.rand(20, 20, 20)
        with pytest.raises(ValueError, match="src_affine"):
            register_irls_pyramid(img, img.clone(), isotropic=True)

    def test_default_uses_centroid_initialization(self):
        img = torch.rand(20, 20, 20)
        shifted = torch.roll(img, shifts=2, dims=2)
        expected = get_ixform_centroids(img, shifted)
        T, _ = register_irls_pyramid(
            img,
            shifted,
            min_voxels=8,
            max_voxels=16,
            nmax=0,
            isotropic=False,
        )
        assert torch.allclose(T, expected)

    def test_noinit_starts_from_identity(self):
        img = torch.rand(20, 20, 20)
        shifted = torch.roll(img, shifts=2, dims=2)
        T, _ = register_irls_pyramid(
            img,
            shifted,
            init_type="header",
            min_voxels=8,
            max_voxels=16,
            nmax=0,
            isotropic=False,
        )
        assert torch.allclose(T, torch.eye(4), atol=1e-6)

    def test_noinit_uses_header_alignment_when_affines_differ(self):
        img = torch.rand(20, 20, 20)
        src_aff = torch.eye(4)
        trg_aff = torch.eye(4)
        trg_aff[0, 3] = 5.0
        T, _ = register_irls_pyramid(
            img,
            img.clone(),
            src_affine=src_aff,
            trg_affine=trg_aff,
            init_type="header",
            min_voxels=8,
            max_voxels=16,
            nmax=0,
            isotropic=False,
        )
        expected = torch.inverse(trg_aff) @ src_aff
        assert torch.allclose(T, expected)

    def test_explicit_initial_transform_takes_precedence(self):
        img = torch.rand(20, 20, 20)
        shifted = torch.roll(img, shifts=2, dims=2)
        init = torch.eye(4)
        init[2, 3] = 7.0
        T, _ = register_irls_pyramid(
            img,
            shifted,
            initial_transform=init,
            init_type="centroid",
            min_voxels=8,
            max_voxels=16,
            nmax=0,
            isotropic=False,
        )
        assert torch.allclose(T, init)

    def test_max_voxels_none_keeps_original_resolution(self):
        img = torch.zeros(31, 27, 19)
        T, all_info = register_irls_pyramid(
            img,
            img.clone(),
            min_voxels=8,
            max_voxels=None,
            nmax=0,
            isotropic=False,
        )

        assert tuple(T.shape) == (4, 4)
        assert len(all_info) == 2
        assert all_info[-1]["image_shape"] == tuple(img.shape)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestInitHelpers:
    def test_centroid_init_respects_affines(self):
        img = torch.ones(10, 10, 10)
        src_aff = torch.eye(4)
        trg_aff = torch.diag(torch.tensor([2.0, 1.0, 1.0, 1.0]))

        v2v = get_ixform_centroids(img, img, src_aff, trg_aff)

        expected = torch.diag(torch.tensor([0.5, 1.0, 1.0, 1.0]))
        expected[0, 3] = 2.25
        assert torch.allclose(v2v, expected)

    def test_image_center_aligns_image_centers_not_intensity_centroids(self):
        src = torch.zeros(9, 9, 9)
        trg = torch.zeros(9, 9, 9)
        src[1, 1, 1] = 1.0
        trg[3, 1, 1] = 1.0

        centroid_v2v = get_ixform_centroids(src, trg)
        image_center_v2v = get_init_vox2vox(src, trg, init_type="image_center")

        expected_centroid = torch.eye(4)
        expected_centroid[0, 3] = 2.0
        assert torch.allclose(centroid_v2v, expected_centroid)
        assert torch.allclose(image_center_v2v, torch.eye(4))

    def test_invalid_init_type_raises_clear_error(self):
        src = torch.zeros(9, 9, 9)
        trg = torch.zeros(9, 9, 9)

        with pytest.raises(ValueError, match="Unknown init_type"):
            get_init_vox2vox(src, trg, init_type=cast(Any, "bogus"))
