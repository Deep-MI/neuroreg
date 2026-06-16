"""Unit tests for robust M-estimator weighting functions (streamlined)."""

import pytest
import torch

from neuroreg.imreg.robust import compute_mad, compute_scale_estimate, huber_weights, tukey_weights


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

    def test_unknown_method_raises(self):
        """Unknown method should raise ValueError."""
        r = torch.randn(100)
        with pytest.raises(ValueError, match="Unknown method"):
            compute_scale_estimate(r, method="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
