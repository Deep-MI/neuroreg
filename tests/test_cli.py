from pathlib import Path

import nibabel as nib
import pytest
import torch

from nireg.cmdline.robreg import main as robreg_main
from nireg.cmdline.robreg_irls import main as robreg_irls_main


def _write_zero_image(path: Path) -> None:
    data = torch.zeros(8, 8, 8, dtype=torch.float32).numpy()
    nib.save(nib.Nifti1Image(data, affine=torch.eye(4).numpy()), path)


class TestRobregIrlsCli:
    def test_main_forwards_noinit(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mov_path = tmp_path / "mov.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        out_path = tmp_path / "out.lta"

        _write_zero_image(mov_path)
        _write_zero_image(ref_path)

        captured: dict[str, object] = {}

        def fake_register_irls_pyramid(**kwargs):
            captured.update(kwargs)
            return torch.eye(4), []

        class _DummyLTA:
            def write(self, path):
                Path(path).write_text("dummy")

        monkeypatch.setattr("nireg.transforms.irls.register_irls_pyramid", fake_register_irls_pyramid)
        monkeypatch.setattr(
            "nireg.transforms.LTA.from_matrix",
            lambda *args, **kwargs: _DummyLTA(),
        )

        robreg_irls_main([
            "--mov", str(mov_path),
            "--ref", str(ref_path),
            "--out", str(out_path),
            "--noinit",
        ])

        assert captured["centroid_init"] is False
        assert out_path.exists()


class TestRobregCli:

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

        monkeypatch.setattr("nireg.register_pyramid", fake_register_pyramid)
        monkeypatch.setattr(
            "nireg.transforms.LTA.from_matrix",
            lambda *args, **kwargs: _DummyLTA(),
        )

        robreg_main([
            "--mov", str(mov_path),
            "--ref", str(ref_path),
            "--out", str(out_path),
            "--noinit",
        ])

        assert captured_args is not None
        assert captured_kwargs["centroid_init"] is False
        assert out_path.exists()

