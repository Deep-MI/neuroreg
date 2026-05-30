from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest

from neuroreg.cli.multireg import main as multireg_main
from neuroreg.multireg import MultiRegResult


def _write_zero_image(path: Path) -> None:
    data = np.zeros((8, 8, 8), dtype=np.float32)
    nib.save(nib.Nifti1Image(data, affine=np.eye(4, dtype=np.float32)), path)


class TestMultiregCli:
    def test_main_writes_template_ltas_and_mapmovs(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mov1 = tmp_path / "tp1.nii.gz"
        mov2 = tmp_path / "tp2.nii.gz"
        template = tmp_path / "template.nii.gz"
        lta1 = tmp_path / "tp1.lta"
        lta2 = tmp_path / "tp2.lta"
        mapmov_dir = tmp_path / "mapped"
        _write_zero_image(mov1)
        _write_zero_image(mov2)

        captured: dict[str, Any] = {}

        def fake_multireg(*args, **kwargs):
            captured["args"] = args
            captured.update(kwargs)
            template_img = nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), np.eye(4, dtype=np.float32))
            mapped_images = [
                nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.uint8), np.eye(4, dtype=np.float32)),
                nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.uint8), np.eye(4, dtype=np.float32)),
            ]
            identity = np.eye(4, dtype=np.float64)
            return MultiRegResult(
                template_image=template_img,
                transforms_r2r=[identity, identity],
                ltas=[],
                initial_target_index=1,
                seed=123,
                mapped_images=mapped_images,
                template_iterations_run=2,
                iteration_distances=[0.2, 0.01],
            )

        class _DummyLTA:
            def write(self, path):
                Path(path).write_text("dummy")

        monkeypatch.setattr("neuroreg.cli.multireg.multireg", fake_multireg)
        monkeypatch.setattr("neuroreg.cli.multireg.LTA.from_matrix", lambda *args, **kwargs: _DummyLTA())

        multireg_main(
            [
                "--mov",
                str(mov1),
                str(mov2),
                "--template",
                str(template),
                "--lta",
                str(lta1),
                str(lta2),
                "--mapmov-dir",
                str(mapmov_dir),
                "--average",
                "1",
                "--inittp",
                "2",
                "--iterate",
                "4",
                "--template-eps",
                "0.05",
                "--init-header",
                "--device",
                "gpu",
                "--keep-dtype",
            ]
        )

        assert captured["init_target_index"] == 1
        assert captured["average"] == "1"
        assert captured["init_type"] == "header"
        assert captured["device"] == "gpu"
        assert captured["template_iterations"] == 4
        assert captured["template_eps"] == pytest.approx(0.05)
        assert captured["return_mapped"] is True
        assert captured["mapped_keep_dtype"] is True
        args = captured["args"][0]
        assert len(args) == 2
        assert template.exists()
        assert lta1.exists()
        assert lta2.exists()
        assert (mapmov_dir / mov1.name).exists()
        assert (mapmov_dir / mov2.name).exists()

    def test_main_rejects_mismatched_lta_count(self, tmp_path: Path):
        mov1 = tmp_path / "tp1.nii.gz"
        mov2 = tmp_path / "tp2.nii.gz"
        template = tmp_path / "template.nii.gz"
        _write_zero_image(mov1)
        _write_zero_image(mov2)

        with pytest.raises(SystemExit):
            multireg_main(
                [
                    "--mov",
                    str(mov1),
                    str(mov2),
                    "--template",
                    str(template),
                    "--lta",
                    str(tmp_path / "only_one.lta"),
                ]
            )

    def test_main_forwards_ixforms(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mov1 = tmp_path / "tp1.nii.gz"
        mov2 = tmp_path / "tp2.nii.gz"
        template = tmp_path / "template.nii.gz"
        _write_zero_image(mov1)
        _write_zero_image(mov2)

        captured: dict[str, Any] = {}

        def fake_multireg(*args, **kwargs):
            captured["args"] = args
            captured.update(kwargs)
            template_img = nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), np.eye(4, dtype=np.float32))
            identity = np.eye(4, dtype=np.float64)
            return MultiRegResult(
                template_image=template_img,
                transforms_r2r=[identity, identity],
                ltas=[],
                initial_target_index=0,
                seed=123,
                mapped_images=None,
                template_iterations_run=0,
                iteration_distances=[],
            )

        monkeypatch.setattr("neuroreg.cli.multireg.multireg", fake_multireg)

        multireg_main(
            [
                "--mov",
                str(mov1),
                str(mov2),
                "--template",
                str(template),
                "--ixforms",
                str(tmp_path / "tp1_to_template.lta"),
                str(tmp_path / "tp2_to_template.lta"),
            ]
        )

        assert captured["init_ltas"] == [
            str(tmp_path / "tp1_to_template.lta"),
            str(tmp_path / "tp2_to_template.lta"),
        ]

    def test_main_rejects_mismatched_ixforms_count(self, tmp_path: Path):
        mov1 = tmp_path / "tp1.nii.gz"
        mov2 = tmp_path / "tp2.nii.gz"
        template = tmp_path / "template.nii.gz"
        _write_zero_image(mov1)
        _write_zero_image(mov2)

        with pytest.raises(SystemExit):
            multireg_main(
                [
                    "--mov",
                    str(mov1),
                    str(mov2),
                    "--template",
                    str(template),
                    "--ixforms",
                    str(tmp_path / "only_one.lta"),
                ]
            )
