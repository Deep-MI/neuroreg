"""Unit tests for robust M-estimator weighting functions (streamlined)."""

import pytest
import torch

from nireg.imreg.robust import compute_mad, compute_scale_estimate, huber_weights, tukey_weights


class TestTukeyWeights:
    """Test Tukey biweight weighting function (primary estimator for IRLS)."""

    def test_basic_behavior(self):
        """Test zero, inliers, and outliers in one comprehensive test."""
        r = torch.tensor([-10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 7.0, 10.0])
        w = tukey_weights(r, c=6.0)

        # Zero residual → weight 1.0
        assert abs(w[3] - 1.0) < 0.01

        # Small residuals (|r| ≤ 2) → high weight (> 0.75)
        assert torch.all(w[1:6] > 0.75)

        # Large residuals (|r| > 6) → weight ≈ 0
        assert w[0] < 0.01
        assert w[-2] < 0.01
        assert w[-1] < 0.01

    def test_symmetry(self):
        """Weights should be symmetric around zero."""
        r = torch.linspace(-10, 10, 21)
        w = tukey_weights(r, c=6.0)
        assert torch.allclose(w[:10], w[-10:].flip(0), atol=1e-6)

class TestHuberWeights:
    """Test Huber M-estimator (alternative to Tukey)."""

    def test_basic_behavior(self):
        """Test Huber weight function behavior."""
        delta = 1.345
        r = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 4.0])
        w = huber_weights(r, delta=delta)

        # Small residuals (|r| ≤ delta) → weight 1.0
        assert torch.allclose(w[1:4], torch.ones(3))  # Only r=-1,0,1 have |r|≤delta

        # Large residuals → weight = delta / |r|
        assert abs(w[-1] - delta / 4.0) < 1e-6

        # Never exactly zero
        assert torch.all(w > 0)


class TestComputeMAD:
    """Test Median Absolute Deviation (robust scale estimation)."""

    def test_standard_normal(self):
        """MAD of standard normal should be close to 1.0."""
        torch.manual_seed(42)
        r = torch.randn(10000)
        sigma = compute_mad(r)
        assert abs(sigma.item() - 1.0) < 0.05

    def test_robust_to_outliers(self):
        """MAD should be robust to outliers compared to std."""
        torch.manual_seed(42)
        r = torch.randn(10000)
        sigma_clean = compute_mad(r)

        # Add extreme outliers
        r_outliers = r.clone()
        r_outliers[::100] = 100.0  # 1% outliers
        sigma_outliers = compute_mad(r_outliers)

        # MAD should be similar despite outliers
        assert abs(sigma_clean - sigma_outliers) / sigma_clean < 0.1

    def test_constant_residuals(self):
        """MAD of constant residuals should return minimum threshold."""
        r = torch.ones(100) * 5.0
        sigma = compute_mad(r)
        assert 0 < sigma < 0.001  # Clamped to minimum


class TestComputeScaleEstimate:
    """Test flexible scale estimation."""

    def test_mad_method(self):
        """MAD method should match compute_mad."""
        r = torch.randn(1000)
        sigma_mad = compute_scale_estimate(r, method="mad")
        sigma_direct = compute_mad(r)
        assert torch.allclose(sigma_mad, sigma_direct)

    def test_unknown_method_raises(self):
        """Unknown method should raise ValueError."""
        r = torch.randn(100)
        with pytest.raises(ValueError, match="Unknown method"):
            compute_scale_estimate(r, method="invalid")


class TestIntegration:
    """Integration tests for complete IRLS workflow."""

    def test_full_irls_workflow(self):
        """Test complete workflow: residuals → MAD → normalize → weights."""
        torch.manual_seed(42)
        r = torch.randn(1000)
        r[::100] = 20.0  # Add 10 outliers

        # Compute robust scale
        sigma = compute_mad(r)

        # Normalize and compute weights
        r_normalized = r / sigma
        w_tukey = tukey_weights(r_normalized, c=6.0)

        # Verify properties
        assert torch.all(w_tukey >= 0)
        assert torch.all(w_tukey <= 1)

        # Outliers should have low weight
        assert w_tukey[::100].max() < 0.1

        # Inliers should have high weight
        inlier_mask = torch.ones_like(w_tukey, dtype=torch.bool)
        inlier_mask[::100] = False
        assert w_tukey[inlier_mask].mean() > 0.7


    def test_tukey_vs_huber_comparison(self):
        """Compare Tukey (aggressive) vs Huber (conservative) on outliers."""
        torch.manual_seed(42)
        r = torch.randn(10000)
        r[::100] = 50.0  # Add outliers

        sigma = compute_mad(r)
        r_norm = r / sigma

        w_tukey = tukey_weights(r_norm, c=6.0)
        w_huber = huber_weights(r_norm, delta=1.345)

        # Tukey should set outliers to exactly 0
        assert (w_tukey[::100] == 0).any()

        # Huber never reaches exactly zero
        assert torch.all(w_huber > 0)

        # Both should give high weights to inliers
        inlier_idx = torch.tensor([1, 2, 3, 4, 5])
        assert w_tukey[inlier_idx].mean() > 0.8
        assert w_huber[inlier_idx].mean() > 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
