from __future__ import annotations

from typing import Any, cast

import nibabel as nib
import numpy as np
import pytest
import torch

from nireg import register_pyramid as register_pyramid_public
from nireg.image import build_gaussian_pyramid, get_pyramid_limits
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



