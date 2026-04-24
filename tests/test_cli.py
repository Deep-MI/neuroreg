from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
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


def _write_uint8_image(path: Path) -> None:
    data = torch.arange(8 * 8 * 8, dtype=torch.uint8).reshape(8, 8, 8).numpy()
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

    def test_main_init_lta_overrides_other_init_flags(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
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

        init_lta = tmp_path / "init.lta"
        robreg_main(
            [
                "--mov",
                str(mov_path),
                "--ref",
                str(ref_path),
                "--out",
                str(out_path),
                "--init-header",
                "--init-lta",
                str(init_lta),
            ]
        )

        assert captured["init_lta"] == str(init_lta)
        assert "init_type" not in captured
        assert out_path.exists()

    def test_main_writes_mapmov_and_mapmovhdr(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mov_path = tmp_path / "mov.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        out_path = tmp_path / "out.lta"
        out_map = tmp_path / "mapped.nii.gz"
        out_hdr = tmp_path / "mapped_hdr.nii.gz"

        _write_uint8_image(mov_path)
        _write_zero_image(ref_path)

        class _DummyLTA:
            def write(self, path):
                Path(path).write_text("dummy")

        transform = torch.eye(4, dtype=torch.float64)
        transform[0, 3] = 1.5
        transform[2, 3] = -2.0

        monkeypatch.setattr("neuroreg.imreg.robreg.robreg", lambda *args, **kwargs: _TensorRequiringCpu(transform))
        monkeypatch.setattr("neuroreg.transforms.LTA.from_matrix", lambda *args, **kwargs: _DummyLTA())

        robreg_main(
            [
                "--mov",
                str(mov_path),
                "--ref",
                str(ref_path),
                "--out",
                str(out_path),
                "--mapmov",
                str(out_map),
                "--mapmovhdr",
                str(out_hdr),
            ]
        )

        assert out_path.exists()
        assert out_map.exists()
        assert out_hdr.exists()
        mapped = nib.load(str(out_map))
        mapped_hdr = nib.load(str(out_hdr))
        ref_img = nib.load(str(ref_path))
        assert mapped.shape[:3] == ref_img.shape[:3]
        assert mapped.affine == pytest.approx(ref_img.affine)
        assert mapped.get_data_dtype() == np.dtype(np.float32)
        expected_affine = np.eye(4)
        expected_affine[0, 3] = 1.5
        expected_affine[2, 3] = -2.0
        assert mapped_hdr.affine == pytest.approx(expected_affine)

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
            mapped_name = kwargs.get("mapped_name")
            if mapped_name is not None:
                ref_img = cast(Any, args[1])
                zeros = torch.zeros(tuple(int(v) for v in ref_img.shape[:3]), dtype=torch.float32).numpy()
                nib.save(nib.Nifti1Image(zeros, affine=ref_img.affine), str(mapped_name))
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

    def test_main_init_lta_overrides_other_init_flags(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mov_path = tmp_path / "mov.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        out_path = tmp_path / "out.lta"
        _write_zero_image(mov_path)
        _write_zero_image(ref_path)

        captured: dict[str, object] = {}
        self._patch_coreg(monkeypatch, captured)
        init_lta = tmp_path / "init.lta"

        coreg_main(
            [
                "--mov",
                str(mov_path),
                "--ref",
                str(ref_path),
                "--out",
                str(out_path),
                "--init-center",
                "--init-lta",
                str(init_lta),
            ]
        )

        assert captured["init_lta"] == str(init_lta)
        assert "init_type" not in captured
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

    def test_main_writes_mapmov_and_mapmovhdr(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mov_path = tmp_path / "mov.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        out_path = tmp_path / "out.lta"
        out_map = tmp_path / "mapped.nii.gz"
        out_hdr = tmp_path / "mapped_hdr.nii.gz"

        mov_affine = np.array(
            [[2.0, 0.0, 0.0, 10.0], [0.0, 3.0, 0.0, -5.0], [0.0, 0.0, 4.0, 2.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        ref_affine = np.array(
            [[1.5, 0.0, 0.0, -4.0], [0.0, 2.5, 0.0, 7.0], [0.0, 0.0, 3.5, -1.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        nib.save(nib.Nifti1Image(torch.zeros(8, 8, 8, dtype=torch.float32).numpy(), mov_affine), mov_path)
        nib.save(nib.Nifti1Image(torch.zeros(8, 8, 8, dtype=torch.float32).numpy(), ref_affine), ref_path)

        captured: dict[str, object] = {}

        def fake_coreg(*args, **kwargs):
            captured["args"] = args
            captured.update(kwargs)
            mapped_name = kwargs.get("mapped_name")
            if mapped_name is not None:
                ref_img = cast(Any, args[1])
                zeros = torch.zeros(tuple(int(v) for v in ref_img.shape[:3]), dtype=torch.float32).numpy()
                nib.save(nib.Nifti1Image(zeros, affine=ref_img.affine), str(mapped_name))
            transform = torch.eye(4, dtype=torch.float64)
            transform[0, 3] = 2.5
            transform[1, 3] = -1.0
            return _TensorRequiringCpu(transform)

        class _DummyLTA:
            def __init__(self, matrix):
                self.matrix = matrix

            def write(self, path):
                captured["lta_matrix"] = self.matrix
                Path(path).write_text("dummy")

        monkeypatch.setattr("neuroreg.imreg.coreg.coreg", fake_coreg)
        monkeypatch.setattr("neuroreg.transforms.LTA.from_matrix", lambda matrix, *args, **kwargs: _DummyLTA(matrix))

        coreg_main(
            [
                "--mov",
                str(mov_path),
                "--ref",
                str(ref_path),
                "--out",
                str(out_path),
                "--mapmov",
                str(out_map),
                "--mapmovhdr",
                str(out_hdr),
            ]
        )

        expected_r2r = np.eye(4)
        expected_r2r[0, 3] = 2.5
        expected_r2r[1, 3] = -1.0
        expected_v2v = np.linalg.inv(ref_affine) @ expected_r2r @ mov_affine

        assert captured["mapped_name"] == str(out_map)
        assert captured["return_v2v"] is False
        assert np.asarray(captured["lta_matrix"]) == pytest.approx(expected_v2v)
        assert out_path.exists()
        assert out_map.exists()
        assert out_hdr.exists()
        mapped_hdr = nib.load(str(out_hdr))
        assert mapped_hdr.affine == pytest.approx(expected_r2r @ mov_affine)
