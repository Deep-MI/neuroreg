from pathlib import Path
from typing import Any, cast

import nibabel as nib
import pytest
import torch

from neuroreg.cli.robreg import main as robreg_main
from neuroreg.cli.coreg import main as coreg_main


def _write_zero_image(path: Path) -> None:
    data = torch.zeros(8, 8, 8, dtype=torch.float32).numpy()
    nib.save(nib.Nifti1Image(data, affine=torch.eye(4).numpy()), path)


class TestRobregCli:
    def test_main_forwards_noinit_and_symmetric_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mov_path = tmp_path / "mov.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        out_path = tmp_path / "out.lta"

        _write_zero_image(mov_path)
        _write_zero_image(ref_path)

        captured: dict[str, object] = {}

        def fake_register_pyramid(*args, **kwargs):
            captured["args"] = args
            captured.update(kwargs)
            return torch.eye(4)

        class _DummyLTA:
            def write(self, path):
                Path(path).write_text("dummy")

        monkeypatch.setattr("neuroreg.imreg.robreg.register_pyramid", fake_register_pyramid)
        monkeypatch.setattr(
            "neuroreg.transforms.LTA.from_matrix",
            lambda *args, **kwargs: _DummyLTA(),
        )

        robreg_main([
            "--mov", str(mov_path),
            "--ref", str(ref_path),
            "--out", str(out_path),
            "--noinit",
        ])

        assert captured["centroid_init"] is False
        assert captured["symmetric"] is True
        args = cast(tuple[Any, Any], captured["args"])
        assert len(args) == 2
        assert hasattr(args[0], "get_fdata") and hasattr(args[0], "affine")
        assert hasattr(args[1], "get_fdata") and hasattr(args[1], "affine")
        assert out_path.exists()

    def test_main_forwards_nosym_flag(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mov_path = tmp_path / "mov.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        out_path = tmp_path / "out.lta"

        _write_zero_image(mov_path)
        _write_zero_image(ref_path)

        captured: dict[str, object] = {}

        def fake_register_pyramid(*args, **kwargs):
            captured["args"] = args
            captured.update(kwargs)
            return torch.eye(4)

        class _DummyLTA:
            def write(self, path):
                Path(path).write_text("dummy")

        monkeypatch.setattr("neuroreg.imreg.robreg.register_pyramid", fake_register_pyramid)
        monkeypatch.setattr(
            "neuroreg.transforms.LTA.from_matrix",
            lambda *args, **kwargs: _DummyLTA(),
        )

        robreg_main([
            "--mov", str(mov_path),
            "--ref", str(ref_path),
            "--out", str(out_path),
            "--nosym",
        ])

        assert captured["symmetric"] is False
        assert out_path.exists()

    def test_parser_rejects_non_rigid_dof(self, tmp_path: Path):
        mov_path = tmp_path / "mov.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        out_path = tmp_path / "out.lta"

        _write_zero_image(mov_path)
        _write_zero_image(ref_path)

        with pytest.raises(SystemExit):
            robreg_main([
                "--mov", str(mov_path),
                "--ref", str(ref_path),
                "--out", str(out_path),
                "--dof", "12",
            ])


class TestCoregCli:
    def test_main_forwards_noinit(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mov_path = tmp_path / "mov.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        out_path = tmp_path / "out.lta"

        _write_zero_image(mov_path)
        _write_zero_image(ref_path)

        captured_args: tuple[object, ...] | None = None
        captured_kwargs: dict[str, object] = {}

        def fake_register_pyramid(*args, **kwargs):
            nonlocal captured_args
            captured_args = args
            captured_kwargs.update(kwargs)
            return torch.eye(4)

        class _DummyLTA:
            def write(self, path):
                Path(path).write_text("dummy")

        monkeypatch.setattr("neuroreg.imreg.coreg.register_pyramid", fake_register_pyramid)
        monkeypatch.setattr(
            "neuroreg.transforms.LTA.from_matrix",
            lambda *args, **kwargs: _DummyLTA(),
        )

        coreg_main([
            "--mov", str(mov_path),
            "--ref", str(ref_path),
            "--out", str(out_path),
            "--noinit",
        ])

        assert captured_args is not None
        assert len(captured_args) == 2
        assert hasattr(captured_args[0], "get_fdata") and hasattr(captured_args[0], "affine")
        assert hasattr(captured_args[1], "get_fdata") and hasattr(captured_args[1], "affine")
        assert captured_kwargs["centroid_init"] is False
        assert out_path.exists()

    def test_main_forwards_level_schedule_and_optimizer_settings(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        mov_path = tmp_path / "mov.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        out_path = tmp_path / "out.lta"

        _write_zero_image(mov_path)
        _write_zero_image(ref_path)

        captured_kwargs: dict[str, object] = {}

        def fake_register_pyramid(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return torch.eye(4)

        class _DummyLTA:
            def write(self, path):
                Path(path).write_text("dummy")

        monkeypatch.setattr("neuroreg.imreg.coreg.register_pyramid", fake_register_pyramid)
        monkeypatch.setattr(
            "neuroreg.transforms.LTA.from_matrix",
            lambda *args, **kwargs: _DummyLTA(),
        )

        coreg_main([
            "--mov", str(mov_path),
            "--ref", str(ref_path),
            "--out", str(out_path),
            "--n_iters", "25",
            "--level-iters", "20,0,5",
            "--lr", "0.002",
            "--min-voxels", "32",
            "--max-voxels", "128",
        ])

        assert captured_kwargs["n"] == 25
        assert captured_kwargs["level_iters"] == [20, 0, 5]
        assert captured_kwargs["lr"] == pytest.approx(0.002)
        assert captured_kwargs["min_voxels"] == 32
        assert captured_kwargs["max_voxels"] == 128
        assert out_path.exists()
