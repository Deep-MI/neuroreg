from __future__ import annotations

from typing import Any, cast

import nibabel as nib
import numpy as np
import pytest
import torch

from nireg import register_pyramid as register_pyramid_public, register_sym as register_sym_public
from nireg.image import build_gaussian_pyramid, get_pyramid_limits
from nireg.imreg.robreg import register_pyramid as register_pyramid_robreg
from nireg.imreg.robreg_gd import register_pyramid, register_pyramid_sym


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


class TestPyramidHelpers:
    def test_get_pyramid_limits_basic(self):
        min_steps, max_steps = get_pyramid_limits(torch.Size([64, 64, 64]), minsize=16)
        assert int(min_steps.item()) == 0
        assert int(max_steps.item()) == 2

    def test_build_gaussian_pyramid_shapes_and_affines(self):
        image = torch.rand(32, 32, 32)
        affine = torch.eye(4)
        imgs, affines = build_gaussian_pyramid(image, affine, limits=(torch.tensor(0), torch.tensor(1)))

        assert [tuple(img.shape) for img in imgs] == [(32, 32, 32), (16, 16, 16)]
        assert len(affines) == 2
        assert torch.allclose(affines[0], affine)
        assert affines[1][0, 0].item() == pytest.approx(2.0)
        assert affines[1][1, 1].item() == pytest.approx(2.0)
        assert affines[1][2, 2].item() == pytest.approx(2.0)


class TestRegisterPyramidSynthetic:
    def test_register_pyramid_returns_v2v_on_identical_images(self):
        img = _make_img()
        v2v = register_pyramid(img, img, return_v2v=True, centroid_init=False, dof=3, n=1, device="cpu")
        assert v2v.shape == (4, 4)
        assert torch.isfinite(v2v).all()
        assert torch.allclose(v2v, torch.eye(4, dtype=v2v.dtype), atol=1.0)

    def test_register_pyramid_handles_shifted_images(self):
        src = _make_img(shift=(0.0, 0.0, 2.0))
        trg = _make_img()
        v2v = register_pyramid(src, trg, return_v2v=True, centroid_init=True, dof=3, n=1, device="cpu")
        assert v2v.shape == (4, 4)
        assert torch.isfinite(v2v).all()

    def test_register_pyramid_sym_smoke(self):
        img = _make_img()
        v2v = register_pyramid_sym(img, img, return_v2v=True, dof=6, n=1, device="cpu")
        assert v2v.shape == (4, 4)
        assert torch.isfinite(v2v).all()


class TestPublicRobregWrapper:
    def test_top_level_register_pyramid_uses_public_wrapper(self):
        img = _make_img()
        Mr2r = register_pyramid_public(img, img, return_v2v=False, centroid_init=False, dof=6, nmax=1)
        assert Mr2r.shape == (4, 4)
        assert torch.isfinite(Mr2r).all()

    def test_top_level_register_sym_uses_public_wrapper(self):
        img = _make_img()
        Mr2r = register_sym_public(img, img, return_v2v=False, centroid_init=False, dof=6, nmax=1)
        assert Mr2r.shape == (4, 4)
        assert torch.isfinite(Mr2r).all()

    def test_register_pyramid_accepts_nibabel_images(self):
        img = _make_img()
        Mr2r = register_pyramid_robreg(img, img, return_v2v=False, centroid_init=False, dof=6, nmax=1)
        assert Mr2r.shape == (4, 4)
        assert torch.isfinite(Mr2r).all()

    def test_register_pyramid_accepts_file_paths(self, tmp_path):
        src = _make_img()
        trg = _make_img()
        src_path = tmp_path / "src.nii.gz"
        trg_path = tmp_path / "trg.nii.gz"
        nib.save(src, src_path)
        nib.save(trg, trg_path)

        Mr2r = register_pyramid_robreg(str(src_path), str(trg_path), return_v2v=False, centroid_init=False, dof=6, nmax=1)
        assert Mr2r.shape == (4, 4)
        assert torch.isfinite(Mr2r).all()



