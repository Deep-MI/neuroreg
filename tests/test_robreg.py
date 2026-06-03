from __future__ import annotations

from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import pytest
import torch

from neuroreg import robreg
from neuroreg.image import build_gaussian_pyramid, get_pyramid_limits, load_image
from neuroreg.imreg.robreg import (
    _convert_vox_transform_between_grids,
    _save_outlier_map,
    register_irls_pyramid,
)
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
    def test_load_image_uses_ifh_affine_for_4dfp_pair(self, tmp_path: Path):
        data = _make_blob((8, 6, 4)).astype(np.float32)
        img_path = tmp_path / "example.4dfp.img"
        nib.AnalyzeImage(data, np.eye(4, dtype=np.float32)).to_filename(img_path)
        ifh_path = tmp_path / "example.4dfp.ifh"
        ifh_path.write_text(
            "\n".join(
                [
                    "INTERFILE\t:=",
                    "orientation\t\t:= 2",
                    "matrix size [1]\t:= 8",
                    "matrix size [2]\t:= 6",
                    "matrix size [3]\t:= 4",
                    "mmppix\t:=   1.000000 -2.000000 -3.000000",
                    "center\t:=   4.0000 -6.0000 -6.0000",
                ]
            )
            + "\n"
        )

        loaded = load_image(img_path)
        expected_affine = np.array(
            [
                [-1.0, 0.0, 0.0, 4.0],
                [0.0, 0.0, 3.0, -6.0],
                [0.0, -2.0, 0.0, 6.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        assert np.array_equal(np.asarray(loaded.dataobj), data)
        assert np.allclose(loaded.affine, expected_affine)

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

    def test_top_level_robreg_forwards_masks(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, object] = {}

        def fake_register_irls_pyramid(**kwargs):
            captured.update(kwargs)
            return torch.eye(4), []

        monkeypatch.setattr("neuroreg.imreg.robreg.register_irls_pyramid", fake_register_irls_pyramid)

        img = _make_img()
        src_mask = nib.Nifti1Image(np.ones((20, 20, 20), dtype=np.float32), img.affine)
        trg_mask = nib.Nifti1Image(np.ones((20, 20, 20), dtype=np.float32), img.affine)
        _ = robreg(img, img, src_mask=src_mask, trg_mask=trg_mask, return_v2v=False, init_type="header", dof=6, nmax=1)

        assert cast(torch.Tensor, captured["src_mask"]).shape == (20, 20, 20)
        assert cast(torch.Tensor, captured["trg_mask"]).shape == (20, 20, 20)

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

    def test_robreg_accepts_init_transform_matrix(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, object] = {}
        init_r2r = np.array(
            [[1.0, 0.0, 0.0, 2.5], [0.0, 1.0, 0.0, -1.5], [0.0, 0.0, 1.0, 0.75], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        def fake_register_irls_pyramid(**kwargs):
            captured.update(kwargs)
            return torch.eye(4), []

        monkeypatch.setattr("neuroreg.imreg.robreg.register_irls_pyramid", fake_register_irls_pyramid)

        src = _make_img()
        trg_affine = np.array(
            [[1.5, 0.0, 0.0, -2.0], [0.0, 1.5, 0.0, 4.0], [0.0, 0.0, 1.5, 1.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        trg = nib.Nifti1Image(src.get_fdata().astype(np.float32), trg_affine)

        mr2r = robreg(src, trg, return_v2v=False, init_type="header", init_transform=init_r2r, dof=6, nmax=1)

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

    def test_robreg_rejects_multiple_explicit_init_transforms(self):
        img = _make_img()
        with pytest.raises(ValueError, match="at most one explicit initialization transform"):
            robreg(
                img,
                img,
                return_v2v=False,
                init_type="header",
                init_transform=np.eye(4, dtype=np.float64),
                init_lta="init.lta",
                dof=6,
                nmax=1,
            )

    def test_robreg_squeezes_singleton_4d_inputs(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, object] = {}

        def fake_register_irls_pyramid(**kwargs):
            captured.update(kwargs)
            return torch.eye(4), []

        monkeypatch.setattr("neuroreg.imreg.robreg.register_irls_pyramid", fake_register_irls_pyramid)

        img4d = nib.Nifti1Image(_make_blob((16, 16, 16))[..., None], np.eye(4, dtype=np.float32))
        mr2r = robreg(img4d, img4d, return_v2v=False, init_type="header", dof=6, nmax=1)

        assert cast(torch.Tensor, captured["src"]).ndim == 3
        assert cast(torch.Tensor, captured["trg"]).ndim == 3
        assert mr2r.shape == (4, 4)


def test_register_irls_pyramid_regrids_between_level_affines(monkeypatch: pytest.MonkeyPatch):
    src = torch.zeros((80, 72, 64), dtype=torch.float32)
    trg = torch.zeros((80, 72, 64), dtype=torch.float32)
    src_affine = torch.tensor(
        [
            [1.25, 0.0, 0.0, -20.0],
            [0.0, 1.5, 0.0, 12.0],
            [0.0, 0.0, 1.75, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    trg_affine = torch.tensor(
        [
            [1.1, 0.0, 0.0, 5.0],
            [0.0, 1.4, 0.0, -8.0],
            [0.0, 0.0, 1.6, 10.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    theta = np.deg2rad(7.0)
    initial_transform = torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0.0, 4.0],
            [np.sin(theta), np.cos(theta), 0.0, -3.0],
            [0.0, 0.0, 1.0, 1.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    captured_initials: list[torch.Tensor] = []

    def fake_register_irls(
            src: torch.Tensor,
            trg: torch.Tensor,
            *,
            initial_transform: torch.Tensor | None = None,
            **_: object,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        assert initial_transform is not None
        captured_initials.append(initial_transform.clone())
        return initial_transform.clone(), {"weights": None, "valid_mask": None, "image_shape": tuple(trg.shape)}

    monkeypatch.setattr("neuroreg.imreg.robreg.register_irls", fake_register_irls)

    result, _ = register_irls_pyramid(
        src=src,
        trg=trg,
        src_affine=src_affine,
        trg_affine=trg_affine,
        initial_transform=initial_transform,
        isotropic=False,
        min_voxels=16,
        nmax=1,
        verbose=False,
    )

    limits = get_pyramid_limits(src.shape, trg.shape, minsize=16)
    _, src_level_affines = build_gaussian_pyramid(src, src_affine, limits=limits)
    _, trg_level_affines = build_gaussian_pyramid(trg, trg_affine, limits=limits)

    expected_initials: list[torch.Tensor] = []
    expected_transform = initial_transform.clone()
    expected_src_affine = src_affine
    expected_trg_affine = trg_affine
    for lvl in range(len(src_level_affines) - 1, -1, -1):
        expected_transform = _convert_vox_transform_between_grids(
            expected_transform,
            expected_src_affine,
            expected_trg_affine,
            src_level_affines[lvl],
            trg_level_affines[lvl],
        )
        expected_initials.append(expected_transform.clone())
        expected_src_affine = src_level_affines[lvl]
        expected_trg_affine = trg_level_affines[lvl]

    assert len(captured_initials) == len(expected_initials)
    for observed, expected in zip(captured_initials, expected_initials, strict=True):
        assert torch.allclose(observed, expected, atol=1e-5, rtol=1e-5)

    legacy_coarsest = initial_transform.clone()
    legacy_scale = float(2 ** (len(src_level_affines) - 1))
    legacy_coarsest[:3, 3] = legacy_coarsest[:3, 3] / legacy_scale
    assert not torch.allclose(captured_initials[0], legacy_coarsest, atol=1e-4, rtol=1e-4)
    assert torch.allclose(result, initial_transform, atol=1e-5, rtol=1e-5)
