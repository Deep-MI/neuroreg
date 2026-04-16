from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import pytest
import torch
from torch import nn

from neuroreg.bbreg.register import register_surface
from neuroreg.cli.bbreg import main as bbreg_main


def _write_zero_image(path: Path) -> None:
    data = torch.zeros(8, 8, 8, dtype=torch.float32).numpy()
    nib.save(nib.Nifti1Image(data, affine=torch.eye(4).numpy()), path)


class TestBbregCli:
    def test_main_uses_200_iterations_by_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mov_path = tmp_path / "mov.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        out_path = tmp_path / "out.lta"

        _write_zero_image(mov_path)
        _write_zero_image(ref_path)

        captured_kwargs: dict[str, object] = {}

        def fake_register_surface(**kwargs):
            captured_kwargs.update(kwargs)
            return torch.eye(4), object()

        monkeypatch.setattr("neuroreg.bbreg.register.register_surface", fake_register_surface)
        monkeypatch.setattr(
            "neuroreg.imreg.robreg_gd.register_pyramid",
            lambda *args, **kwargs: torch.eye(4, dtype=torch.float64),
        )

        bbreg_main([
            "--mov", str(mov_path),
            "--ref", str(ref_path),
            "--lh_surf", str(tmp_path / "lh.white"),
            "--out", str(out_path),
        ])

        assert captured_kwargs["n_iters"] == 200

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

        monkeypatch.setattr("neuroreg.bbreg.register.register_surface", fake_register_surface)
        monkeypatch.setattr("neuroreg.imreg.robreg_gd.register_pyramid", fake_register_pyramid)

        bbreg_main([
            "--mov", str(mov_path),
            "--ref", str(ref_path),
            "--lh_surf", str(tmp_path / "lh.white"),
            "--out", str(out_path),
        ])

        assert captured_prealign_kwargs["loss_name"] == "nmi"
        assert captured_prealign_kwargs["centroid_init"] is False
        assert captured_prealign_kwargs["min_voxels"] == 32
        assert captured_prealign_kwargs["max_voxels"] == 64
        assert captured_prealign_kwargs["level_iters"] == [30, 10]
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

        monkeypatch.setattr("neuroreg.bbreg.register.register_surface", fake_register_surface)
        monkeypatch.setattr("neuroreg.imreg.robreg_gd.register_pyramid", fail_register_pyramid)

        bbreg_main([
            "--mov", str(mov_path),
            "--ref", str(ref_path),
            "--lh_surf", str(tmp_path / "lh.white"),
            "--out", str(out_path),
            "--init-header",
        ])

        assert captured_surface_kwargs["init_type"] == "header"
        assert "init_ras" not in captured_surface_kwargs

    def test_main_prefers_explicit_init_lta(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mov_path = tmp_path / "mov.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        out_path = tmp_path / "out.lta"
        init_lta = tmp_path / "seed.lta"

        _write_zero_image(mov_path)
        _write_zero_image(ref_path)
        init_lta.write_text("dummy")

        captured_surface_kwargs: dict[str, object] = {}

        def fake_register_surface(**kwargs):
            captured_surface_kwargs.update(kwargs)
            return torch.eye(4), object()

        def fail_register_pyramid(*args, **kwargs):
            raise AssertionError("prealignment should not run when --init-lta is provided")

        monkeypatch.setattr("neuroreg.bbreg.register.register_surface", fake_register_surface)
        monkeypatch.setattr("neuroreg.imreg.robreg_gd.register_pyramid", fail_register_pyramid)

        bbreg_main([
            "--mov", str(mov_path),
            "--ref", str(ref_path),
            "--lh_surf", str(tmp_path / "lh.white"),
            "--out", str(out_path),
            "--init-lta", str(init_lta),
        ])

        assert captured_surface_kwargs["init_type"] == "lta"
        assert captured_surface_kwargs["init_lta"] == str(init_lta)
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

        monkeypatch.setattr("neuroreg.bbreg.register.register_surface", fake_register_surface)
        monkeypatch.setattr("neuroreg.imreg.robreg_gd.register_pyramid", fake_register_pyramid)

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

        monkeypatch.setattr("neuroreg.bbreg.register.register_surface", fake_register_surface)
        monkeypatch.setattr("neuroreg.imreg.robreg_gd.register_pyramid", fake_register_pyramid)

        bbreg_main([
            "--mov", str(mov_path),
            "--seg", str(seg_path),
            "--ref", str(ref_path),
            "--out", str(out_path),
        ])

        assert captured_ref_sum["sum"] == pytest.approx(float((seg_data > 0).sum()))
        assert captured_surface_kwargs["seg"] == str(seg_path)
        assert captured_surface_kwargs["init_ras"] is not None

    def test_no_coreg_ref_mask_disables_subject_dir_masking(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
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

        captured_ref_sum: dict[str, float] = {}

        def fake_register_surface(**kwargs):
            return torch.eye(4), object()

        def fake_register_pyramid(mov_img, ref_img, **kwargs):
            captured_ref_sum["sum"] = float(ref_img.get_fdata().sum())
            return torch.eye(4, dtype=torch.float64)

        monkeypatch.setattr("neuroreg.bbreg.register.register_surface", fake_register_surface)
        monkeypatch.setattr("neuroreg.imreg.robreg_gd.register_pyramid", fake_register_pyramid)

        bbreg_main([
            "--mov", str(mov_path),
            "--subject_dir", str(subject_dir),
            "--out", str(out_path),
            "--no-coreg-ref-mask",
        ])

        assert captured_ref_sum["sum"] == pytest.approx(float(ref_data.sum()))


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

        monkeypatch.setattr("neuroreg.bbreg.register.load_surface", fake_load_surface)
        monkeypatch.setattr("neuroreg.bbreg.register.get_vox2ras_tkr", lambda header: torch.eye(4).numpy())
        monkeypatch.setattr("neuroreg.bbreg.register.BBRModel", FakeBBRModel)
        monkeypatch.setattr("torch.optim.RMSprop", FakeOptimizer)

        transform, model = register_surface(
            mov=str(mov_path),
            ref=str(ref_path),
            lh_surf=str(tmp_path / "lh.white"),
            n_iters=10,
            early_stop_patience=2,
        )

        assert transform[0, 3].item() == pytest.approx(-4.0)
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

        monkeypatch.setattr("neuroreg.bbreg.register.load_surface", fake_load_surface)
        monkeypatch.setattr("neuroreg.bbreg.register.get_vox2ras_tkr", lambda header: torch.eye(4).numpy())
        monkeypatch.setattr("neuroreg.bbreg.register.BBRModel", FakeBBRModel)
        monkeypatch.setattr("torch.optim.RMSprop", FakeOptimizer)

        transform, _ = register_surface(
            mov=str(mov_path),
            ref=str(ref_path),
            lh_surf=str(tmp_path / "lh.white"),
            init_ras=init_ras,
            n_iters=1,
        )

        expected_internal = torch.from_numpy(np.linalg.inv(init_ras)).float()
        assert torch.allclose(captured_init["init_transform"], expected_internal)
        assert torch.allclose(transform, torch.eye(4, dtype=transform.dtype))
