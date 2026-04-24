import importlib
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import pytest
import torch
from torch import nn

from neuroreg import bbreg
from neuroreg.cli.bbreg import main as bbreg_main


def _write_zero_image(path: Path) -> None:
    data = torch.zeros(8, 8, 8, dtype=torch.float32).numpy()
    nib.save(nib.Nifti1Image(data, affine=torch.eye(4).numpy()), path)


def _write_uint8_image(path: Path) -> None:
    data = torch.arange(8 * 8 * 8, dtype=torch.uint8).reshape(8, 8, 8).numpy()
    nib.save(nib.Nifti1Image(data, affine=torch.eye(4).numpy()), path)


bbreg_register_module = importlib.import_module("neuroreg.bbreg.register")


class TestBbregCli:

    def test_main_runs_default_coarse_nmi_prealignment(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mov_path = tmp_path / "mov.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        out_path = tmp_path / "out.lta"

        _write_zero_image(mov_path)
        _write_zero_image(ref_path)

        captured_surface_kwargs: dict[str, object] = {}
        captured_prealign_kwargs: dict[str, object] = {}

        prealign_r2r = torch.tensor(
            [
                [1.0, 0.0, 0.0, 3.0],
                [0.0, 1.0, 0.0, -2.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )

        def fake_register_surface(**kwargs):
            captured_surface_kwargs.update(kwargs)
            return torch.eye(4), object()

        def fake_register_pyramid(*args, **kwargs):
            captured_prealign_kwargs.update(kwargs)
            return prealign_r2r

        monkeypatch.setattr(bbreg_register_module, "register_surface", fake_register_surface)
        monkeypatch.setattr("neuroreg.imreg.coreg.coreg", fake_register_pyramid)

        bbreg_main([
            "--mov", str(mov_path),
            "--ref", str(ref_path),
            "--lh_surf", str(tmp_path / "lh.white"),
            "--out", str(out_path),
        ])

        assert captured_prealign_kwargs["method"] == "powell"
        assert captured_prealign_kwargs["init_type"] == "image_center"
        assert captured_prealign_kwargs["dof"] == 6
        assert captured_prealign_kwargs["powell_sep"] == 4
        assert "level_iters" not in captured_prealign_kwargs
        assert "loss_name" not in captured_prealign_kwargs
        assert captured_surface_kwargs["n_iters"] == 200
        assert captured_surface_kwargs["init_ras"] is not None
        assert captured_surface_kwargs.get("init_type") is None
        np_init = np.asarray(cast(object, captured_surface_kwargs["init_ras"]))
        expected = prealign_r2r.numpy()
        assert np_init.shape == (4, 4)
        assert np.allclose(np_init, expected)

    def test_main_skips_prealignment_with_init_header(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mov_path = tmp_path / "mov.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        out_path = tmp_path / "out.lta"

        _write_zero_image(mov_path)
        _write_zero_image(ref_path)

        captured_surface_kwargs: dict[str, object] = {}

        def fake_register_surface(**kwargs):
            captured_surface_kwargs.update(kwargs)
            return torch.eye(4), object()

        def fail_register_pyramid(*args, **kwargs):
            raise AssertionError("prealignment should not run")

        monkeypatch.setattr(bbreg_register_module, "register_surface", fake_register_surface)
        monkeypatch.setattr("neuroreg.imreg.coreg.coreg", fail_register_pyramid)

        bbreg_main([
            "--mov", str(mov_path),
            "--ref", str(ref_path),
            "--lh_surf", str(tmp_path / "lh.white"),
            "--out", str(out_path),
            "--init-header",
        ])

        assert captured_surface_kwargs["init_type"] == "header"
        assert "init_ras" not in captured_surface_kwargs

    def test_subject_dir_prealignment_uses_aparc_aseg_mask(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mov_path = tmp_path / "mov.nii.gz"
        subject_dir = tmp_path / "subject"
        mri_dir = subject_dir / "mri"
        out_path = tmp_path / "out.lta"

        _write_zero_image(mov_path)
        mri_dir.mkdir(parents=True)

        ref_data = torch.ones(8, 8, 8, dtype=torch.float32).numpy()
        mask_data = torch.zeros(8, 8, 8, dtype=torch.float32).numpy()
        mask_data[2:6, 2:6, 2:6] = 1.0
        nib.save(nib.MGHImage(ref_data, affine=torch.eye(4).numpy()), mri_dir / "orig.mgz")
        nib.save(nib.MGHImage(mask_data, affine=torch.eye(4).numpy()), mri_dir / "aparc+aseg.mgz")

        captured_surface_kwargs: dict[str, object] = {}
        captured_ref_sum: dict[str, float] = {}

        def fake_register_surface(**kwargs):
            captured_surface_kwargs.update(kwargs)
            return torch.eye(4), object()

        def fake_register_pyramid(mov_img, ref_img, **kwargs):
            captured_ref_sum["sum"] = float(ref_img.get_fdata().sum())
            return torch.eye(4, dtype=torch.float64)

        monkeypatch.setattr(bbreg_register_module, "register_surface", fake_register_surface)
        monkeypatch.setattr("neuroreg.imreg.coreg.coreg", fake_register_pyramid)

        bbreg_main([
            "--mov", str(mov_path),
            "--subject_dir", str(subject_dir),
            "--out", str(out_path),
        ])

        assert captured_ref_sum["sum"] == pytest.approx(float(mask_data.sum()))
        assert captured_surface_kwargs["subject_dir"] == str(subject_dir)

    def test_seg_mode_uses_ref_for_prealignment_and_seg_as_mask(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mov_path = tmp_path / "mov.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        seg_path = tmp_path / "seg.nii.gz"
        out_path = tmp_path / "out.lta"

        _write_zero_image(mov_path)
        ref_data = torch.ones(8, 8, 8, dtype=torch.float32).numpy()
        seg_data = torch.zeros(8, 8, 8, dtype=torch.float32).numpy()
        seg_data[1:7, 1:7, 1:7] = 3.0
        nib.save(nib.Nifti1Image(ref_data, affine=torch.eye(4).numpy()), ref_path)
        nib.save(nib.Nifti1Image(seg_data, affine=torch.eye(4).numpy()), seg_path)

        captured_surface_kwargs: dict[str, object] = {}
        captured_ref_sum: dict[str, float] = {}

        def fake_register_surface(**kwargs):
            captured_surface_kwargs.update(kwargs)
            return torch.eye(4), object()

        def fake_register_pyramid(mov_img, ref_img, **kwargs):
            captured_ref_sum["sum"] = float(ref_img.get_fdata().sum())
            return torch.eye(4, dtype=torch.float64)

        monkeypatch.setattr(bbreg_register_module, "register_surface", fake_register_surface)
        monkeypatch.setattr("neuroreg.imreg.coreg.coreg", fake_register_pyramid)

        bbreg_main([
            "--mov", str(mov_path),
            "--seg", str(seg_path),
            "--ref", str(ref_path),
            "--out", str(out_path),
        ])

        assert captured_ref_sum["sum"] == pytest.approx(float((seg_data > 0).sum()))
        assert captured_surface_kwargs["seg"] == str(seg_path)
        assert captured_surface_kwargs["init_ras"] is not None

    def test_main_writes_mapmov_and_mapmovhdr(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mov_path = tmp_path / "mov.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        out_path = tmp_path / "out.lta"
        out_map = tmp_path / "mapped.nii.gz"
        out_hdr = tmp_path / "mapped_hdr.nii.gz"

        _write_uint8_image(mov_path)
        _write_zero_image(ref_path)

        def fake_register_surface(**kwargs):
            transform = torch.eye(4, dtype=torch.float64)
            transform[0, 3] = 4.0
            transform[1, 3] = -3.0
            return transform

        def fail_register_pyramid(*args, **kwargs):
            raise AssertionError("prealignment should not run")

        monkeypatch.setattr(bbreg_register_module, "register_surface", fake_register_surface)
        monkeypatch.setattr("neuroreg.imreg.coreg.coreg", fail_register_pyramid)

        bbreg_main([
            "--mov", str(mov_path),
            "--ref", str(ref_path),
            "--lh_surf", str(tmp_path / "lh.white"),
            "--out", str(out_path),
            "--init-header",
            "--mapmov", str(out_map),
            "--mapmovhdr", str(out_hdr),
        ])

        assert out_map.exists()
        assert out_hdr.exists()
        mapped = nib.load(str(out_map))
        mapped_hdr = nib.load(str(out_hdr))
        assert mapped.shape[:3] == nib.load(str(ref_path)).shape[:3]
        assert mapped.affine == pytest.approx(nib.load(str(ref_path)).affine)
        assert mapped.get_data_dtype() == np.dtype(np.float32)
        expected_affine = np.eye(4)
        expected_affine[0, 3] = 4.0
        expected_affine[1, 3] = -3.0
        assert mapped_hdr.affine == pytest.approx(expected_affine)


class TestBbregRegister:
    def test_register_surface_returns_best_iterate_and_stops_early(
            self,
            monkeypatch: pytest.MonkeyPatch,
            tmp_path: Path,
    ):
        mov_path = tmp_path / "mov.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        _write_zero_image(mov_path)
        _write_zero_image(ref_path)

        class FakeBBRModel(nn.Module):
            sequence = [10.0, 5.0, 5.0, 4.0, 4.0, 6.0, 6.0, 7.0]

            def __init__(self, **kwargs):
                super().__init__()
                self.transform_params = nn.Parameter(torch.zeros(6, dtype=torch.float32))
                self.call_count = 0

            def forward(self):
                value = self.sequence[self.call_count]
                with torch.no_grad():
                    self.transform_params.fill_(value)
                self.call_count += 1
                return self.transform_params.sum() * 0 + value

            def get_transform_matrix(self):
                matrix = torch.eye(4, dtype=torch.float32)
                matrix[0, 3] = self.transform_params[0]
                return matrix

        class FakeOptimizer:
            def __init__(self, params, lr):
                self.params = list(params)
                self.lr = lr
                self.step_calls = 0

            def zero_grad(self):
                return None

            def step(self):
                self.step_calls += 1
                return None

        def fake_load_surface(*args, **kwargs):
            return {
                "vertices": torch.zeros((3, 3), dtype=torch.float32),
                "faces": torch.tensor([[0, 1, 2]], dtype=torch.int64),
            }

        monkeypatch.setattr(bbreg_register_module, "load_surface", fake_load_surface)
        monkeypatch.setattr(bbreg_register_module, "get_vox2ras_tkr", lambda header: torch.eye(4).numpy())
        monkeypatch.setattr(bbreg_register_module, "BBRModel", FakeBBRModel)
        monkeypatch.setattr("torch.optim.RMSprop", FakeOptimizer)

        transform_bbreg, model = bbreg(
            mov=str(mov_path),
            ref=str(ref_path),
            lh_surf=str(tmp_path / "lh.white"),
            n_iters=10,
            early_stop_patience=2,
            return_model=True,
        )

        assert transform_bbreg[0, 3].item() == pytest.approx(-4.0)
        assert model.transform_params[0].item() == pytest.approx(4.0)
        assert model.call_count == 8

    def test_register_surface_init_ras_uses_mov_to_trg_direction(
            self,
            monkeypatch: pytest.MonkeyPatch,
            tmp_path: Path,
    ):
        mov_path = tmp_path / "mov.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        _write_zero_image(mov_path)
        _write_zero_image(ref_path)

        captured_init: dict[str, torch.Tensor] = {}

        class FakeBBRModel(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                captured_init["init_transform"] = kwargs["init_transform"].detach().clone()
                self.transform_params = nn.Parameter(torch.zeros(6, dtype=torch.float32))

            def forward(self):
                return self.transform_params.sum() * 0

            def get_transform_matrix(self):
                return torch.eye(4, dtype=torch.float32)

        class FakeOptimizer:
            def __init__(self, params, lr):
                self.params = list(params)
                self.lr = lr

            def zero_grad(self):
                return None

            def step(self):
                return None

        def fake_load_surface(*args, **kwargs):
            return {
                "vertices": torch.zeros((3, 3), dtype=torch.float32),
                "faces": torch.tensor([[0, 1, 2]], dtype=torch.int64),
            }

        init_ras = np.eye(4, dtype=np.float64)
        init_ras[0, 3] = 3.0

        monkeypatch.setattr(bbreg_register_module, "load_surface", fake_load_surface)
        monkeypatch.setattr(bbreg_register_module, "get_vox2ras_tkr", lambda header: torch.eye(4).numpy())
        monkeypatch.setattr(bbreg_register_module, "BBRModel", FakeBBRModel)
        monkeypatch.setattr("torch.optim.RMSprop", FakeOptimizer)

        transform_bbreg = bbreg(
            mov=str(mov_path),
            ref=str(ref_path),
            lh_surf=str(tmp_path / "lh.white"),
            init_ras=init_ras,
            n_iters=1,
        )

        expected_internal = torch.from_numpy(np.linalg.inv(init_ras)).float()
        assert torch.allclose(captured_init["init_transform"], expected_internal)
        assert torch.allclose(transform_bbreg, torch.eye(4, dtype=transform_bbreg.dtype))
