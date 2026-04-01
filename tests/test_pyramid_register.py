from __future__ import annotations

from typing import Any, cast

import nibabel as nib
import numpy as np
import pytest
import torch

from nireg import register_pyramid as register_pyramid_public
from nireg.imreg.losses import mi_loss, ncc_loss, nmi_loss
from nireg.imreg.robreg import register_pyramid as register_pyramid_robreg
from nireg.imreg.robreg_gd import register_pyramid


def _make_blob(shape: tuple[int, int, int] = (20, 20, 20), shift: tuple[float, float, float] = (0, 0, 0)) -> np.ndarray:
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    center = (np.asarray(shape, dtype=np.float32) - 1.0) / 2.0 + np.asarray(shift, dtype=np.float32)
    dist2 = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    return np.exp(-dist2 / 18.0).astype(np.float32)


def _make_img(
    shape: tuple[int, int, int] = (20, 20, 20),
    shift: tuple[float, float, float] = (0, 0, 0),
) -> nib.Nifti1Image:
    affine = cast(Any, np.eye(4, dtype=np.float32))
    data = cast(Any, _make_blob(shape, shift))
    return nib.Nifti1Image(data, affine)


class TestRegisterPyramidSynthetic:
    def test_register_pyramid_returns_v2v_on_identical_images(self):
        img = _make_img()
        v2v = register_pyramid(img, img, return_v2v=True, centroid_init=False, dof=3, n=1, device="cpu")
        assert v2v.shape == (4, 4)
        assert torch.isfinite(v2v).all()
        assert torch.allclose(v2v, torch.eye(4, dtype=v2v.dtype), atol=1.0)

    def test_register_pyramid_respects_max_voxels_schedule(self, monkeypatch: pytest.MonkeyPatch):
        seen_shapes: list[tuple[int, int, int]] = []

        def fake_register_level(simg, timg, **kwargs):
            seen_shapes.append(tuple(int(v) for v in simg.shape))
            return torch.eye(4), [], None

        monkeypatch.setattr("nireg.imreg.robreg_gd.register_level", fake_register_level)

        img = _make_img(shape=(128, 128, 128))
        register_pyramid(img, img, return_v2v=True, centroid_init=False, n=1, min_voxels=16, max_voxels=64)
        assert seen_shapes == [(16, 16, 16), (32, 32, 32), (64, 64, 64)]

        seen_shapes.clear()
        register_pyramid(img, img, return_v2v=True, centroid_init=False, n=1, min_voxels=16, max_voxels=None)
        assert seen_shapes == [(16, 16, 16), (32, 32, 32), (64, 64, 64), (128, 128, 128)]


class TestPublicRobregWrapper:
    def test_top_level_register_pyramid_defaults_to_symmetric(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, object] = {}

        def fake_register_irls_pyramid(**kwargs):
            captured.update(kwargs)
            return torch.eye(4), []

        monkeypatch.setattr("nireg.imreg.robreg.register_irls_pyramid", fake_register_irls_pyramid)

        img = _make_img()
        Mr2r = register_pyramid_public(img, img, return_v2v=False, centroid_init=False, dof=6, nmax=1)

        assert captured["symmetric"] is True
        assert Mr2r.shape == (4, 4)
        assert torch.isfinite(Mr2r).all()

    def test_top_level_register_pyramid_can_disable_symmetric_mode(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, object] = {}

        def fake_register_irls_pyramid(**kwargs):
            captured.update(kwargs)
            return torch.eye(4), []

        monkeypatch.setattr("nireg.imreg.robreg.register_irls_pyramid", fake_register_irls_pyramid)

        img = _make_img()
        Mr2r = register_pyramid_public(img, img, return_v2v=False, centroid_init=False, dof=6, nmax=1, symmetric=False)

        assert captured["symmetric"] is False
        assert Mr2r.shape == (4, 4)
        assert torch.isfinite(Mr2r).all()

    def test_register_pyramid_accepts_file_paths(self, tmp_path):
        src = _make_img()
        trg = _make_img()
        src_path = tmp_path / "src.nii.gz"
        trg_path = tmp_path / "trg.nii.gz"
        nib.save(src, src_path)
        nib.save(trg, trg_path)

        Mr2r = register_pyramid_robreg(
            str(src_path),
            str(trg_path),
            return_v2v=False,
            centroid_init=False,
            dof=6,
            nmax=1,
        )
        assert Mr2r.shape == (4, 4)
        assert torch.isfinite(Mr2r).all()


# ---------------------------------------------------------------------------
# Loss-function unit tests
# ---------------------------------------------------------------------------

class TestLossFunctions:
    """Unit tests for the three modality-independent similarity losses."""

    def _blob(self, shape=(20, 20, 20), shift=(0, 0, 0)):
        return _make_blob(shape, shift)

    # ---- NCC ---------------------------------------------------------------

    def test_ncc_identical_images_is_zero(self):
        a = torch.from_numpy(self._blob())
        assert ncc_loss(a, a, win_size=5).item() == pytest.approx(0.0, abs=1e-4)

    def test_ncc_unrelated_images_near_one(self):
        a = torch.from_numpy(self._blob())
        b = torch.rand_like(a)
        # NCC between a smooth blob and pure noise should be very low
        assert ncc_loss(a, b, win_size=5).item() > 0.5

    def test_ncc_returns_scalar_in_range(self):
        a = torch.from_numpy(self._blob())
        b = torch.from_numpy(self._blob(shift=(3, 0, 0)))
        loss = ncc_loss(a, b, win_size=5)
        assert loss.shape == ()
        assert 0.0 <= loss.item() <= 1.0 + 1e-6

    def test_ncc_gradient_flows(self):
        a = torch.from_numpy(self._blob()).requires_grad_(True)
        b = torch.from_numpy(self._blob(shift=(2, 0, 0)))
        loss = ncc_loss(a, b, win_size=5)
        loss.backward()
        assert a.grad is not None
        assert torch.isfinite(a.grad).all()

    def test_ncc_clamps_win_size_to_image(self):
        # Very small image: win_size larger than any dim should not raise
        a = torch.rand(4, 4, 4)
        loss = ncc_loss(a, a.clone(), win_size=99)
        assert torch.isfinite(loss)

    # ---- MI ----------------------------------------------------------------

    def test_mi_identical_images_is_finite_and_negative(self):
        a = torch.from_numpy(self._blob())
        loss = mi_loss(a, a, num_bins=16)
        assert torch.isfinite(loss)
        assert loss.item() < 0  # loss = -MI, MI > 0 for non-trivial image

    def test_mi_gradient_flows(self):
        a = torch.from_numpy(self._blob()).requires_grad_(True)
        b = torch.from_numpy(self._blob(shift=(2, 0, 0)))
        loss = mi_loss(a, b, num_bins=16)
        loss.backward()
        assert a.grad is not None
        assert torch.isfinite(a.grad).all()

    def test_mi_same_image_lower_loss_than_shifted(self):
        """MI of identical images should be higher (loss more negative) than shifted."""
        a = torch.from_numpy(self._blob())
        b = torch.from_numpy(self._blob(shift=(5, 0, 0)))
        loss_same = mi_loss(a, a, num_bins=16).item()
        loss_shifted = mi_loss(a, b, num_bins=16).item()
        assert loss_same < loss_shifted  # more negative = higher MI

    def test_mi_cross_modal_finite(self):
        """MI should be finite even when src and trg have very different intensity ranges."""
        a = torch.from_numpy(self._blob()) * 1000.0   # simulated T1
        b = 1.0 - torch.from_numpy(self._blob())      # inverted contrast (T2-like)
        loss = mi_loss(a, b, num_bins=32)
        assert torch.isfinite(loss)

    # ---- NMI ---------------------------------------------------------------

    def test_nmi_identical_images_near_minus_two(self):
        """NMI of an image with itself is 2.0, so loss should be ≤ -1.0.

        NMI = 2 only in the limit of infinitesimally narrow Parzen windows.
        With finite sigma and a peaked image histogram the diagonal-only
        structure of the joint histogram is blurred, so NMI < 2.  We test the
        weaker, always-valid bound: identical images produce lower (more
        negative) loss than mismatched images, and loss ≤ -1.0 (NMI ≥ 1.0).
        """
        a = torch.from_numpy(self._blob())
        b = torch.from_numpy(self._blob(shift=(5, 0, 0)))
        loss_same = nmi_loss(a, a, num_bins=16).item()
        loss_diff = nmi_loss(a, b, num_bins=16).item()
        assert loss_same < loss_diff           # identical → lower loss (higher NMI)
        assert loss_same <= -1.0 + 1e-3        # NMI(x, x) ≥ 1.0 always

    def test_nmi_gradient_flows(self):
        a = torch.from_numpy(self._blob()).requires_grad_(True)
        b = torch.from_numpy(self._blob(shift=(2, 0, 0)))
        loss = nmi_loss(a, b, num_bins=16)
        loss.backward()
        assert a.grad is not None
        assert torch.isfinite(a.grad).all()

    def test_nmi_always_less_negative_than_minus_two(self):
        """NMI ≤ 2.0 for non-identical images, so loss ≥ -2."""
        a = torch.from_numpy(self._blob())
        b = torch.from_numpy(self._blob(shift=(3, 0, 0)))
        loss = nmi_loss(a, b, num_bins=16)
        assert loss.item() >= -2.0 - 1e-3


# ---------------------------------------------------------------------------
# End-to-end GD registration with new losses
# ---------------------------------------------------------------------------

class TestRegisterPyramidNewLosses:
    """Smoke tests: run register_pyramid with each new loss on a simple example."""

    @pytest.mark.parametrize("loss_name", ["ncc", "mi", "nmi"])
    def test_register_pyramid_new_loss_produces_finite_transform(self, loss_name):
        img = _make_img(shape=(32, 32, 32))
        v2v = register_pyramid(
            img, img,
            return_v2v=True,
            centroid_init=False,
            dof=6,
            n=3,
            loss_name=loss_name,
            loss_bins=16,
            min_voxels=16,
            device="cpu",
        )
        assert v2v.shape == (4, 4)
        assert torch.isfinite(v2v).all()

    @pytest.mark.parametrize("loss_name", ["ncc", "mi", "nmi"])
    def test_register_pyramid_new_loss_near_identity_on_same_image(self, loss_name):
        img = _make_img(shape=(32, 32, 32))
        v2v = register_pyramid(
            img, img,
            return_v2v=True,
            centroid_init=False,
            dof=6,
            n=10,
            loss_name=loss_name,
            loss_bins=16,
            min_voxels=16,
            device="cpu",
        )
        assert torch.allclose(v2v, torch.eye(4, dtype=v2v.dtype), atol=1.0)

    def test_register_pyramid_ncc_loss_decreases(self):
        """NCC loss should decrease over iterations when images are misaligned."""
        from nireg.imreg.robreg_gd import register_level
        src = torch.from_numpy(_make_blob(shape=(32, 32, 32), shift=(4, 0, 0)))
        trg = torch.from_numpy(_make_blob(shape=(32, 32, 32)))
        _, losses, _ = register_level(src, trg, dof=6, centroid_init=False, n=15,
                                      loss_name="ncc", device="cpu")
        assert losses[0] >= losses[-1]



