"""Tests for IRLS rigid registration (nireg.transforms.irls)."""

import pytest
import torch

from nireg.transforms.initialize import get_ixform_centroids
from nireg.transforms.irls import (
    _choose_pyramid_levels,
    _sqrt_tukey,
    affine_trans_dist,
    compute_partials,
    construct_Ab,
    irls_inner_loop,
    params_to_rigid_matrix,
    register_irls,
    register_irls_pyramid,
)

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
# affine_trans_dist
# ---------------------------------------------------------------------------

class TestAffineTransDist:
    def test_identity_to_self_is_zero(self):
        T = torch.eye(4)
        assert affine_trans_dist(T, T) == pytest.approx(0.0, abs=1e-6)

    def test_pure_translation(self):
        T1 = torch.eye(4)
        T2 = torch.eye(4)
        T2[0, 3] = 3.0  # 3 mm translation
        assert affine_trans_dist(T1, T2) == pytest.approx(3.0, rel=1e-4)

    def test_symmetry(self):
        T1 = torch.eye(4)
        T1[0, 3] = 5.0
        T2 = torch.eye(4)
        T2[1, 3] = 3.0
        assert affine_trans_dist(T1, T2) == pytest.approx(affine_trans_dist(T2, T1), rel=1e-5)


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

    def test_finite_outputs(self):
        src = torch.rand(8, 8, 8)
        trg = torch.rand(8, 8, 8)
        A, b, valid = construct_Ab(src, trg)
        assert torch.isfinite(A).all()
        assert torch.isfinite(b).all()

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
    def test_returns_correct_shapes(self):
        A = torch.randn(500, 6)
        b = torch.randn(500)
        p, w, sigma, err = irls_inner_loop(A, b, sat=4.685, max_iterations=5)
        assert p.shape == (6,)
        assert w.shape == (500,)
        assert isinstance(sigma, float)
        assert isinstance(err, float)

    def test_weights_in_range(self):
        A = torch.randn(200, 6)
        b = torch.randn(200)
        _, w, _, _ = irls_inner_loop(A, b, sat=4.685)
        assert torch.all(w >= 0)
        assert torch.all(w <= 1)

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

    def test_returns_4x4_transform(self):
        src, trg = self._make_images()
        T, info = register_irls(src, trg, nmax=2)
        assert T.shape == (4, 4)

    def test_info_keys_present(self):
        src, trg = self._make_images()
        _, info = register_irls(src, trg, nmax=2)
        for key in ('iterations', 'converged', 'dists', 'weights', 'valid_mask', 'sigma_hist'):
            assert key in info

    def test_identical_images_near_identity(self):
        src, trg = self._make_images()
        T, _ = register_irls(src, trg, nmax=3)
        assert affine_trans_dist(T, torch.eye(4)) < 1.0

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
        w  = info['weights']
        vm = info['valid_mask']
        assert w is not None and vm is not None
        assert w.shape == (vm.sum().item(),)

    def test_huber_estimator(self):
        src, trg = self._make_images()
        T, info = register_irls(src, trg, nmax=2, sat=1.345)
        assert T.shape == (4, 4)

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

    def test_returns_4x4_and_info_list(self):
        torch.manual_seed(1)
        img = torch.rand(24, 24, 24)
        aff = self._aff()
        T, all_info = register_irls_pyramid(
            img, img.clone(), src_affine=aff, trg_affine=aff,
            min_voxels=8, max_voxels=16, nmax=2, isotropic=False,
        )
        assert T.shape == (4, 4)
        assert isinstance(all_info, list) and len(all_info) >= 1

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
            centroid_init=False,
            min_voxels=8,
            max_voxels=16,
            nmax=0,
            isotropic=False,
        )
        assert torch.allclose(T, torch.eye(4), atol=1e-6)

    def test_explicit_initial_transform_takes_precedence(self):
        img = torch.rand(20, 20, 20)
        shifted = torch.roll(img, shifts=2, dims=2)
        init = torch.eye(4)
        init[2, 3] = 7.0
        T, _ = register_irls_pyramid(
            img,
            shifted,
            initial_transform=init,
            centroid_init=True,
            min_voxels=8,
            max_voxels=16,
            nmax=0,
            isotropic=False,
        )
        assert torch.allclose(T, init)

    def test_choose_pyramid_levels_matches_freesurfer_schedule(self):
        pyramid = [
            torch.zeros(224, 224, 169),
            torch.zeros(112, 112, 84),
            torch.zeros(56, 56, 42),
            torch.zeros(28, 28, 21),
            torch.zeros(14, 14, 10),
            torch.zeros(7, 7, 5),
        ]
        assert _choose_pyramid_levels(pyramid, min_voxels=16, max_voxels=64) == [3, 2, 1, 0]



if __name__ == "__main__":
    pytest.main([__file__, "-v"])

