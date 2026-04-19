from pathlib import Path
from typing import Any, cast

import nibabel as nib
import pytest
import torch

from neuroreg.cli.coreg import main as coreg_main
from neuroreg.cli.robreg import main as robreg_main


class _TensorRequiringCpu:
    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor
        self._detached = False
        self._on_cpu = False

    def detach(self):
        self._detached = True
        return self

    def cpu(self):
        self._on_cpu = True
        return self

    def numpy(self):
        if not (self._detached and self._on_cpu):
            raise RuntimeError("tensor must be detached and moved to CPU before numpy()")
        return self._tensor.numpy()


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
            return _TensorRequiringCpu(torch.eye(4))

        class _DummyLTA:
            def write(self, path):
                Path(path).write_text("dummy")

        monkeypatch.setattr("neuroreg.imreg.robreg.robreg", fake_register_pyramid)
        monkeypatch.setattr(
            "neuroreg.transforms.LTA.from_matrix",
            lambda *args, **kwargs: _DummyLTA(),
        )

        robreg_main(
            [
                "--mov",
                str(mov_path),
                "--ref",
                str(ref_path),
                "--out",
                str(out_path),
                "--init-header",
            ]
        )

        assert captured["init_type"] == "header"
        assert captured["symmetric"] is True
        args = cast(tuple[Any, Any], captured["args"])
        assert len(args) == 2
        assert hasattr(args[0], "get_fdata") and hasattr(args[0], "affine")
        assert hasattr(args[1], "get_fdata") and hasattr(args[1], "affine")
        assert out_path.exists()

    def test_main_forwards_center_init(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mov_path = tmp_path / "mov.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        out_path = tmp_path / "out.lta"

        _write_zero_image(mov_path)
        _write_zero_image(ref_path)

        captured: dict[str, object] = {}

        def fake_register_pyramid(*args, **kwargs):
            captured.update(kwargs)
            return _TensorRequiringCpu(torch.eye(4))

        class _DummyLTA:
            def write(self, path):
                Path(path).write_text("dummy")

        monkeypatch.setattr("neuroreg.imreg.robreg.robreg", fake_register_pyramid)
        monkeypatch.setattr("neuroreg.transforms.LTA.from_matrix", lambda *args, **kwargs: _DummyLTA())

        robreg_main(
            [
                "--mov",
                str(mov_path),
                "--ref",
                str(ref_path),
                "--out",
                str(out_path),
                "--init-center",
            ]
        )

        assert captured["init_type"] == "image_center"
        assert out_path.exists()

    def test_parser_rejects_non_rigid_dof(self, tmp_path: Path):
        mov_path = tmp_path / "mov.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        out_path = tmp_path / "out.lta"

        _write_zero_image(mov_path)
        _write_zero_image(ref_path)

        with pytest.raises(SystemExit):
            robreg_main(
                [
                    "--mov",
                    str(mov_path),
                    "--ref",
                    str(ref_path),
                    "--out",
                    str(out_path),
                    "--dof",
                    "12",
                ]
            )


class TestCoregCli:
    @staticmethod
    def _patch_coreg(monkeypatch: pytest.MonkeyPatch, captured_kwargs: dict[str, object]):
        def fake_coreg(*args, **kwargs):
            captured_kwargs["args"] = args
            captured_kwargs.update(kwargs)
            return _TensorRequiringCpu(torch.eye(4))

        class _DummyLTA:
            def write(self, path):
                Path(path).write_text("dummy")

        monkeypatch.setattr("neuroreg.imreg.coreg.coreg", fake_coreg)
        monkeypatch.setattr("neuroreg.transforms.LTA.from_matrix", lambda *args, **kwargs: _DummyLTA())

    @pytest.mark.parametrize(
        ("flag", "expected_init"),
        [
            ("--init-header", "header"),
            ("--init-center", "image_center"),
        ],
    )
    def test_main_forwards_init_mode(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flag: str, expected_init: str
    ):
        mov_path = tmp_path / "mov.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        out_path = tmp_path / "out.lta"
        _write_zero_image(mov_path)
        _write_zero_image(ref_path)

        captured: dict[str, object] = {}
        self._patch_coreg(monkeypatch, captured)

        coreg_main(
            [
                "--mov",
                str(mov_path),
                "--ref",
                str(ref_path),
                "--out",
                str(out_path),
                flag,
            ]
        )

        args = cast(tuple[Any, Any], captured["args"])
        assert len(args) == 2
        assert hasattr(args[0], "get_fdata") and hasattr(args[0], "affine")
        assert hasattr(args[1], "get_fdata") and hasattr(args[1], "affine")
        assert captured["init_type"] == expected_init
        assert captured["method"] == "powell"
        assert out_path.exists()

    def test_main_defaults_to_powell_method_and_forwards_powell_knobs(
            self,
            monkeypatch: pytest.MonkeyPatch,
            tmp_path: Path,
    ):
        mov_path = tmp_path / "mov.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        out_path = tmp_path / "out.lta"
        _write_zero_image(mov_path)
        _write_zero_image(ref_path)

        captured: dict[str, object] = {}
        self._patch_coreg(monkeypatch, captured)

        coreg_main(
            [
                "--mov",
                str(mov_path),
                "--ref",
                str(ref_path),
                "--out",
                str(out_path),
                "--powell-brute-limit",
                "15",
                "--powell-brute-iters",
                "2",
                "--powell-brute-samples",
                "11",
                "--powell-maxiter",
                "7",
                "--powell-sep",
                "6",
            ]
        )

        assert captured["method"] == "powell"
        assert captured["powell_brute_force_limit"] == pytest.approx(15.0)
        assert captured["powell_brute_force_iters"] == 2
        assert captured["powell_brute_force_samples"] == 11
        assert captured["powell_maxiter"] == 7
        assert captured["powell_sep"] == 6
        assert out_path.exists()

    def test_main_can_select_gd_method_and_forward_schedule(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mov_path = tmp_path / "mov.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        out_path = tmp_path / "out.lta"
        _write_zero_image(mov_path)
        _write_zero_image(ref_path)

        captured: dict[str, object] = {}
        self._patch_coreg(monkeypatch, captured)

        coreg_main(
            [
                "--mov",
                str(mov_path),
                "--ref",
                str(ref_path),
                "--out",
                str(out_path),
                "--method",
                "gd",
                "--n_iters",
                "25",
                "--level-iters",
                "20,0,5",
                "--lr",
                "0.002",
                "--min-voxels",
                "32",
                "--max-voxels",
                "128",
            ]
        )

        assert captured["method"] == "gd"
        assert captured["n"] == 25
        assert captured["level_iters"] == [20, 0, 5]
        assert captured["lr"] == pytest.approx(0.002)
        assert captured["min_voxels"] == 32
        assert captured["max_voxels"] == 128
        assert out_path.exists()
