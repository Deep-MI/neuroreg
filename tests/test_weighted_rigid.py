"""Unit tests for weighted rigid alignment solver."""

import pytest
import torch

from nireg.transforms import get_rotation_euler, rotation_error
from nireg.transforms.weighted_rigid import (
    sample_weighted_voxel_grid,
    solve_weighted_rigid_gpu,
)


def rotation_matrix_3x3(angles: list[float], device: str = "cpu") -> torch.Tensor:
    """Helper: Get 3×3 rotation matrix from Euler angles using existing function."""
    angles_tensor = torch.tensor(angles, device=device)
    R_4x4 = get_rotation_euler(angles_tensor)
    return R_4x4[:3, :3]


class TestSolveWeightedRigidGPU:
    """Test weighted rigid alignment solver."""

    def test_identity_transform(self):
        """Same point sets should give identity transform."""
        points = torch.randn(100, 3)
        weights = torch.ones(100)
        T = solve_weighted_rigid_gpu(points, points, weights)

        # Should be identity
        assert torch.allclose(T, torch.eye(4), atol=1e-5)

    def test_pure_translation(self):
        """Test recovery of pure translation."""
        src = torch.randn(100, 3)
        t_true = torch.tensor([5.0, -3.0, 2.0])
        trg = src + t_true
        weights = torch.ones(100)

        T = solve_weighted_rigid_gpu(src, trg, weights)

        # Check rotation is identity
        R = T[:3, :3]
        assert torch.allclose(R, torch.eye(3), atol=1e-5)

        # Check translation
        t_est = T[:3, 3]
        assert torch.allclose(t_est, t_true, atol=1e-4)

    def test_pure_rotation(self):
        """Test recovery of pure rotation (around origin)."""
        src = torch.randn(100, 3)
        angles = [0.1, 0.2, 0.0]
        R_true = rotation_matrix_3x3(angles)

        trg = (R_true @ src.T).T
        weights = torch.ones(100)

        T = solve_weighted_rigid_gpu(src, trg, weights, center=True)

        R_est = T[:3, :3]
        t_est = T[:3, 3]

        # Check rotation
        assert torch.allclose(R_est, R_true, atol=1e-4)
        # Translation should be small (near zero for centered rotation)
        assert torch.norm(t_est) < 0.1

    def test_rigid_transform_recovery(self):
        """Test recovery of combined rotation + translation."""
        src = torch.randn(100, 3)
        angles = [0.1, 0.2, 0.05]
        R_true = rotation_matrix_3x3(angles)
        t_true = torch.tensor([5.0, -3.0, 2.0])

        trg = (R_true @ src.T).T + t_true
        weights = torch.ones(100)

        T = solve_weighted_rigid_gpu(src, trg, weights)

        R_est = T[:3, :3]
        t_est = T[:3, 3]

        # Check rotation
        assert rotation_error(R_est, R_true) < 0.1  # < 0.1 degrees

        # Check translation
        assert torch.allclose(t_est, t_true, atol=1e-3)

    def test_weighted_alignment_with_outliers(self):
        """Test that weights correctly downweight outliers."""
        torch.manual_seed(42)
        src = torch.randn(100, 3)

        # True transform
        R_true = rotation_matrix_3x3([0.1, 0.2, 0.0])
        t_true = torch.tensor([5.0, -3.0, 2.0])
        trg = (R_true @ src.T).T + t_true

        # Add outliers
        trg[::10] += torch.randn(10, 3) * 20  # 10% outliers with large noise

        # Test 1: Uniform weights (should be affected by outliers)
        weights_uniform = torch.ones(100)
        T_uniform = solve_weighted_rigid_gpu(src, trg, weights_uniform)
        error_uniform = rotation_error(T_uniform[:3, :3], R_true)

        # Test 2: Downweight outliers
        weights_robust = torch.ones(100)
        weights_robust[::10] = 0.01  # Strong downweight
        T_robust = solve_weighted_rigid_gpu(src, trg, weights_robust)
        error_robust = rotation_error(T_robust[:3, :3], R_true)

        # Robust should be better
        assert error_robust < error_uniform
        assert error_robust < 1.0  # Should still recover well

    def test_zero_weights_returns_identity(self):
        """All zero weights should return identity."""
        src = torch.randn(100, 3)
        trg = torch.randn(100, 3)
        weights = torch.zeros(100)

        T = solve_weighted_rigid_gpu(src, trg, weights)

        assert torch.allclose(T, torch.eye(4), atol=1e-6)

    def test_shape_validation(self):
        """Test input validation."""
        # Mismatched shapes
        with pytest.raises(ValueError, match="same shape"):
            solve_weighted_rigid_gpu(torch.randn(100, 3), torch.randn(50, 3), torch.ones(100))

        # Wrong dimensionality
        with pytest.raises(ValueError, match="must be 3D"):
            solve_weighted_rigid_gpu(torch.randn(100, 2), torch.randn(100, 2), torch.ones(100))

        # Wrong number of weights
        with pytest.raises(ValueError, match="must match"):
            solve_weighted_rigid_gpu(torch.randn(100, 3), torch.randn(100, 3), torch.ones(50))


class TestSampleWeightedVoxelGrid:
    """Test voxel grid sampling based on weights."""

    def test_basic_sampling(self):
        """Test basic voxel sampling."""
        # Create small grid
        grid_shape = (10, 10, 10)
        weights = torch.ones(grid_shape)

        voxel_coords, voxel_weights = sample_weighted_voxel_grid(
            grid_shape, weights, sample_fraction=0.5, min_samples=100
        )

        # Check shapes
        assert voxel_coords.ndim == 2 and voxel_coords.shape[1] == 3
        assert voxel_weights.shape[0] == voxel_coords.shape[0]

    def test_weighted_sampling(self):
        """High-weight voxels should be preferentially selected."""
        grid_shape = (20, 20, 20)

        # Create weights with high values in center
        weights = torch.ones(grid_shape) * 0.1
        weights[5:15, 5:15, 5:15] = 1.0  # Center region has high weight

        voxel_coords, voxel_weights = sample_weighted_voxel_grid(
            grid_shape, weights, sample_fraction=0.1, min_samples=100
        )

        # Most selected voxels should have high weight
        assert voxel_weights.mean() > 0.5  # Should be biased toward high weights

    def test_min_samples_respected(self):
        """Minimum samples should be extracted even if fraction is small."""
        grid_shape = (10, 10, 10)  # 1000 voxels
        weights = torch.ones(grid_shape)

        voxel_coords, _ = sample_weighted_voxel_grid(
            grid_shape,
            weights,
            sample_fraction=0.01,
            min_samples=500,  # 1% = 10, but min=500
        )

        assert voxel_coords.shape[0] >= 500


class TestRotationError:
    """Test rotation error metric."""

    def test_identity_zero_error(self):
        """Same rotation should give zero error."""
        R = rotation_matrix_3x3([0.1, 0.2, 0.3])
        error = rotation_error(R, R)
        assert error < 1e-6

    def test_known_angle(self):
        """Test with known rotation angle."""
        R1 = torch.eye(3)
        # 90° rotation around Z
        R2 = rotation_matrix_3x3([0.0, 0.0, torch.pi / 2])
        error = rotation_error(R1, R2)
        # Should be 90 degrees
        assert abs(error - 90.0) < 0.1


    def test_handles_4x4_matrices(self):
        """Should extract 3×3 from 4×4 matrices."""
        R1_4x4 = get_rotation_euler(torch.tensor([0.1, 0.2, 0.0]))
        R2_4x4 = get_rotation_euler(torch.tensor([0.15, 0.25, 0.05]))

        # Should work with 4×4 matrices
        error = rotation_error(R1_4x4, R2_4x4)
        assert isinstance(error, float)
        assert error > 0



class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline_synthetic(self):
        """Test full pipeline: create transform → extract points → solve."""
        torch.manual_seed(42)

        # Create synthetic rigid transform
        R_true = rotation_matrix_3x3([0.1, 0.15, 0.05])
        t_true = torch.tensor([5.0, -3.0, 2.0])

        # Create point cloud
        src = torch.randn(200, 3)
        trg = (R_true @ src.T).T + t_true

        # Add noise and outliers
        trg += torch.randn_like(trg) * 0.01
        trg[::20] += torch.randn(10, 3) * 10  # 5% outliers

        # Compute robust weights (simulated)
        weights = torch.ones(200)
        weights[::20] = 0.01  # Downweight outliers

        # Solve
        T = solve_weighted_rigid_gpu(src, trg, weights)

        # Verify
        R_est = T[:3, :3]
        t_est = T[:3, 3]

        rot_err = rotation_error(R_est, R_true)
        trans_err = torch.norm(t_est - t_true).item()

        print(f"Rotation error: {rot_err:.4f}°")
        print(f"Translation error: {trans_err:.4f}")

        assert rot_err < 1.0  # < 1 degree
        assert trans_err < 0.5  # < 0.5 units


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
