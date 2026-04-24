from __future__ import annotations

from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import pytest
import torch

from neuroreg import robreg
from neuroreg.imreg.robreg import _save_outlier_map
from neuroreg.transforms import LINEAR_RAS_TO_RAS, LINEAR_VOX_TO_VOX, convert_transform_type


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
    affine = np.eye(4, dtype=np.float32)
    data = _make_blob(shape, shift)
    return nib.Nifti1Image(data, affine)


def test_save_outlier_map_skips_missing_final_weights(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    outliers_path = tmp_path / "outliers.mgz"

    with caplog.at_level("WARNING"):
        _save_outlier_map(
            [
                {
                    "weights": None,
                    "valid_mask": None,
                    "image_shape": (8, 8, 8),
                    "iso_affine": np.eye(4, dtype=np.float32),
                }
            ],
            str(outliers_path),
        )

    assert not outliers_path.exists()
    assert "final IRLS level did not produce usable weights" in caplog.text


class TestPublicRobregWrapper:
    def test_top_level_robreg_defaults_to_symmetric(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, object] = {}

        def fake_register_irls_pyramid(**kwargs):
            captured.update(kwargs)
            return torch.eye(4), []

        monkeypatch.setattr("neuroreg.imreg.robreg.register_irls_pyramid", fake_register_irls_pyramid)

        img = _make_img()
        mr2r = robreg(img, img, return_v2v=False, init_type="header", dof=6, nmax=1)

        assert captured["symmetric"] is True
        assert mr2r.shape == (4, 4)
        assert torch.isfinite(mr2r).all()

    def test_robreg_accepts_file_paths(self, tmp_path: Path):
        src = _make_img()
        trg = _make_img()
        src_path = tmp_path / "src.nii.gz"
        trg_path = tmp_path / "trg.nii.gz"
        nib.save(src, src_path)
        nib.save(trg, trg_path)

        mr2r = robreg(
            str(src_path),
            str(trg_path),
            return_v2v=False,
            init_type="header",
            dof=6,
            nmax=1,
        )
        assert mr2r.shape == (4, 4)
        assert torch.isfinite(mr2r).all()

    def test_robreg_loads_init_lta_as_initial_transform(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, object] = {}
        init_r2r = np.array(
            [[1.0, 0.0, 0.0, 2.5], [0.0, 1.0, 0.0, -1.5], [0.0, 0.0, 1.0, 0.75], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        class _DummyLTA:
            def r2r(self):
                return init_r2r

        def fake_register_irls_pyramid(**kwargs):
            captured.update(kwargs)
            return torch.eye(4), []

        monkeypatch.setattr("neuroreg.imreg.robreg.LTA.read", lambda path: _DummyLTA())
        monkeypatch.setattr("neuroreg.imreg.robreg.register_irls_pyramid", fake_register_irls_pyramid)

        src = _make_img()
        trg_affine = np.array(
            [[1.5, 0.0, 0.0, -2.0], [0.0, 1.5, 0.0, 4.0], [0.0, 0.0, 1.5, 1.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        trg = nib.Nifti1Image(src.get_fdata().astype(np.float32), trg_affine)

        mr2r = robreg(src, trg, return_v2v=False, init_type="header", init_lta="init.lta", dof=6, nmax=1)

        expected_v2v = torch.from_numpy(
            convert_transform_type(
                init_r2r,
                src_affine=src.affine,
                dst_affine=trg.affine,
                from_type=LINEAR_RAS_TO_RAS,
                to_type=LINEAR_VOX_TO_VOX,
            )
        ).to(dtype=torch.float32)

        assert torch.allclose(cast(torch.Tensor, captured["initial_transform"]), expected_v2v)
        assert captured["init_type"] == "header"
        assert mr2r.shape == (4, 4)
