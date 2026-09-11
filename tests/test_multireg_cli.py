from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest

from neuroreg.cli.multireg import main as multireg_main
from neuroreg.cli.segreg import main as segreg_main
from neuroreg.multireg import MultiRegResult
from neuroreg.transforms import LTA


def _write_zero_image(path: Path) -> None:
    data = np.zeros((8, 8, 8), dtype=np.float32)
    nib.save(nib.Nifti1Image(data, affine=np.eye(4, dtype=np.float32)), path)


def _write_centroid_poses(tmp_path: Path, count: int) -> list[str]:
    """Fit one pose per time point with segreg against bare centroids.

    The resulting LTAs are the real case for --template-geom: the target is a set
    of centroid coordinates with no image behind it, so each LTA carries
    valid = 0 and cannot supply a template geometry.
    """
    centroids = tmp_path / "centroids.json"
    centroids.write_text('{"1": [1.0, 1.0, 1.0], "2": [1.0, 5.0, 2.0], "3": [5.0, 2.0, 6.0], "4": [6.0, 6.0, 4.0]}\n')
    lta_paths = []
    for index in range(1, count + 1):
        seg_path = tmp_path / f"tp{index}_seg.nii.gz"
        seg = np.zeros((8, 8, 8), dtype=np.int16)
        seg[1, 1, 1] = 1
        seg[1, 5, 2] = 2
        seg[5, 2, 6] = 3
        seg[6, 6, 4] = 4
        nib.save(nib.Nifti1Image(seg, affine=np.eye(4, dtype=np.float32)), seg_path)
        lta_path = tmp_path / f"tp{index}_to_centroids.lta"
        segreg_main(["--seg", str(seg_path), "--centroids", str(centroids), "--lta", str(lta_path)])
        assert LTA.read(lta_path).dst["valid"] == 0
        lta_paths.append(str(lta_path))
    return lta_paths


class TestMultiregCli:
    def test_main_writes_template_ltas_and_mapmovs(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mov1 = tmp_path / "tp1.nii.gz"
        mov2 = tmp_path / "tp2.nii.gz"
        template = tmp_path / "template.nii.gz"
        lta1 = tmp_path / "tp1.lta"
        lta2 = tmp_path / "tp2.lta"
        mapmov1 = tmp_path / "tp1_mapped.nii.gz"
        mapmov2 = tmp_path / "tp2_mapped.nii.gz"
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
                "--mapmov",
                str(mapmov1),
                str(mapmov2),
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
        assert mapmov1.exists()
        assert mapmov2.exists()

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

    def test_main_rejects_mismatched_mapmov_count(self, tmp_path: Path):
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
                    "--mapmov",
                    str(tmp_path / "only_one.nii.gz"),
                ]
            )

    def test_main_writes_mgz_template_from_nifti_result(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mov1 = tmp_path / "tp1.nii.gz"
        mov2 = tmp_path / "tp2.nii.gz"
        template = tmp_path / "template.mgz"
        _write_zero_image(mov1)
        _write_zero_image(mov2)

        def fake_multireg(*args, **kwargs):
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
            ]
        )

        written = nib.load(str(template))
        assert template.exists()
        assert isinstance(written, nib.MGHImage)
        assert written.shape == (8, 8, 8)

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

    def test_main_rejects_ixforms_with_fixtp(self, tmp_path: Path, capsys):
        # Both choose the template space, so accepting the pair would silently
        # discard --fixtp rather than doing what was asked.
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
                    str(tmp_path / "tp1.lta"),
                    str(tmp_path / "tp2.lta"),
                    "--fixtp",
                ]
            )

        assert "pass only one" in capsys.readouterr().err

    def test_main_reports_ixforms_without_destination_geometry(self, tmp_path: Path, capsys):
        # LTAs from atlas registration with no target image (e.g. centroid-based
        # segreg) carry valid = 0, so they cannot define the template space.
        # That has to read as an error with a remedy, not as a traceback.
        mov1 = tmp_path / "tp1.nii.gz"
        mov2 = tmp_path / "tp2.nii.gz"
        _write_zero_image(mov1)
        _write_zero_image(mov2)
        mov_img = nib.load(str(mov1))
        lta_paths = []
        for index, mov_path in enumerate((mov1, mov2), start=1):
            lta_path = tmp_path / f"tp{index}_to_mni.lta"
            LTA.from_matrix(np.eye(4, dtype=np.float64), str(mov_path), mov_img, "mni.mgz", None).write(lta_path)
            lta_paths.append(str(lta_path))

        with pytest.raises(SystemExit):
            multireg_main(
                [
                    "--mov",
                    str(mov1),
                    str(mov2),
                    "--template",
                    str(tmp_path / "template.mgz"),
                    "--ixforms",
                    *lta_paths,
                    "--noit",
                ]
            )

        # LTA.read also logs its own valid = 0 warning to stderr, so assert on
        # the parts unique to the failure and its remedy.
        err = capsys.readouterr().err
        assert "Traceback" not in err
        assert "ERROR:" in err
        assert "must include valid destination geometry" in err
        assert "lta convert --dst-img" in err

    def test_main_accepts_fixtp_without_ixforms(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        # The rejection must be specific to the combination, not to --fixtp.
        mov1 = tmp_path / "tp1.nii.gz"
        mov2 = tmp_path / "tp2.nii.gz"
        template = tmp_path / "template.nii.gz"
        _write_zero_image(mov1)
        _write_zero_image(mov2)

        captured: dict[str, Any] = {}

        def fake_multireg(*args, **kwargs):
            captured.update(kwargs)
            identity = np.eye(4, dtype=np.float64)
            return MultiRegResult(
                template_image=nib.Nifti1Image(
                    np.ones((8, 8, 8), dtype=np.float32), np.eye(4, dtype=np.float32)
                ),
                transforms_r2r=[identity, identity],
                ltas=[],
                initial_target_index=0,
                seed=123,
                mapped_images=None,
                template_iterations_run=0,
                iteration_distances=[],
            )

        monkeypatch.setattr("neuroreg.cli.multireg.multireg", fake_multireg)

        multireg_main(["--mov", str(mov1), str(mov2), "--template", str(template), "--fixtp"])

        assert captured["fix_target"] is True
        assert captured["init_ltas"] is None

    def test_main_forwards_template_geom(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        mov1 = tmp_path / "tp1.nii.gz"
        mov2 = tmp_path / "tp2.nii.gz"
        geom = tmp_path / "std.nii.gz"
        template = tmp_path / "template.nii.gz"
        _write_zero_image(mov1)
        _write_zero_image(mov2)
        geom_affine = np.diag([0.8, 0.8, 0.8, 1.0]).astype(np.float32)
        geom_affine[:3, 3] = (-4.0, -4.0, -4.0)
        nib.save(nib.Nifti1Image(np.zeros((10, 11, 12), dtype=np.float32), geom_affine), geom)

        captured: dict[str, Any] = {}

        def fake_multireg(*args, **kwargs):
            captured.update(kwargs)
            identity = np.eye(4, dtype=np.float64)
            return MultiRegResult(
                template_image=nib.Nifti1Image(np.ones((10, 11, 12), dtype=np.float32), geom_affine),
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
            ["--mov", str(mov1), str(mov2), "--template", str(template), "--template-geom", str(geom)]
        )

        forwarded = captured["template_geom"]
        assert forwarded is not None
        assert forwarded.shape[:3] == (10, 11, 12)
        assert np.asarray(forwarded.affine) == pytest.approx(geom_affine)

    def test_main_accepts_ixforms_without_destination_geometry_when_template_geom_is_given(self, tmp_path: Path):
        # The FastSurfer longitudinal base case: poses fitted by segreg against
        # centroids have no destination image, so their LTAs carry valid = 0.
        # With --template-geom they no longer need one.
        mov1 = tmp_path / "tp1.nii.gz"
        mov2 = tmp_path / "tp2.nii.gz"
        geom = tmp_path / "std.nii.gz"
        template = tmp_path / "base_brainmask.mgz"
        mapmov1 = tmp_path / "tp1_mapped.mgz"
        mapmov2 = tmp_path / "tp2_mapped.mgz"
        _write_zero_image(mov1)
        _write_zero_image(mov2)
        geom_affine = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        geom_affine[:3, 3] = (-5.0, -6.0, -7.0)
        nib.save(nib.Nifti1Image(np.zeros((10, 11, 12), dtype=np.float32), geom_affine), geom)
        lta_paths = _write_centroid_poses(tmp_path, 2)

        multireg_main(
            [
                "--mov",
                str(mov1),
                str(mov2),
                "--template",
                str(template),
                "--ixforms",
                *lta_paths,
                "--template-geom",
                str(geom),
                "--mapmov",
                str(mapmov1),
                str(mapmov2),
                "--noit",
            ]
        )

        written = nib.load(str(template))
        assert written.shape[:3] == (10, 11, 12)
        assert np.asarray(written.affine) == pytest.approx(geom_affine)
        for mapmov_path in (mapmov1, mapmov2):
            mapped = nib.load(str(mapmov_path))
            assert mapped.shape[:3] == (10, 11, 12)
            assert np.asarray(mapped.affine) == pytest.approx(geom_affine)

    def test_main_rejects_template_geom_with_fixtp(self, tmp_path: Path, capsys):
        mov1 = tmp_path / "tp1.nii.gz"
        mov2 = tmp_path / "tp2.nii.gz"
        geom = tmp_path / "std.nii.gz"
        _write_zero_image(mov1)
        _write_zero_image(mov2)
        _write_zero_image(geom)

        with pytest.raises(SystemExit):
            multireg_main(
                [
                    "--mov",
                    str(mov1),
                    str(mov2),
                    "--template",
                    str(tmp_path / "template.nii.gz"),
                    "--template-geom",
                    str(geom),
                    "--fixtp",
                ]
            )

        err = capsys.readouterr().err
        assert "--template-geom and --fixtp" in err
        assert "pass only one" in err

    def test_main_rejects_template_geom_with_cras_center(self, tmp_path: Path, capsys):
        mov1 = tmp_path / "tp1.nii.gz"
        mov2 = tmp_path / "tp2.nii.gz"
        geom = tmp_path / "std.nii.gz"
        _write_zero_image(mov1)
        _write_zero_image(mov2)
        _write_zero_image(geom)

        with pytest.raises(SystemExit):
            multireg_main(
                [
                    "--mov",
                    str(mov1),
                    str(mov2),
                    "--template",
                    str(tmp_path / "template.nii.gz"),
                    "--template-geom",
                    str(geom),
                    "--cras-center",
                ]
            )

        err = capsys.readouterr().err
        assert "--cras-center has no effect" in err
        assert "pass only one" in err
