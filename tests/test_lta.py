"""Tests for the LTA API, CLI, and adjacent transform formats."""
from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
import subprocess
import sys
from pathlib import Path

from neuroreg.cli.lta import main
from neuroreg.transforms import (
    LTA,
    XFM,
    FSLMat,
    ITKTransform,
    RegisterDat,
    affine_dist,
    corner_dist,
    decompose_transform,
    rigid_dist,
    sphere_dist,
)

# ---------------------------------------------------------------------------
# Shared geometry and helpers
# ---------------------------------------------------------------------------

_GEOM = {
    "dims": [256, 256, 256],
    "delta": [1.0, 1.0, 1.0],
    "Mdc": np.eye(3),
    "Pxyz_c": np.zeros(3),
}

_IDENTITY = np.eye(4)


def _rotation_z(deg: float) -> np.ndarray:
    """Return a 4x4 rotation matrix (z-axis, degrees)."""
    r = np.radians(deg)
    M = np.eye(4)
    M[0, 0] = np.cos(r)
    M[0, 1] = -np.sin(r)
    M[1, 0] = np.sin(r)
    M[1, 1] = np.cos(r)
    return M


def _translation(tx: float, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    """Return a 4x4 pure-translation matrix (mm)."""
    M = np.eye(4)
    M[:3, 3] = [tx, ty, tz]
    return M


def _write_lta(path: Path, M: np.ndarray, lta_type: int = 1) -> str:
    """Write a minimal LTA file with full geometry; return its path as a string."""
    s = str(path)
    LTA.from_matrix(M, "src.mgz", _GEOM, "dst.mgz", _GEOM, lta_type=lta_type).write(s)
    return s


def _write_bare_lta(path: Path, M: np.ndarray, lta_type: int = 1) -> str:
    """Write a minimal LTA file with no volume-info blocks; return its path as a string."""
    rows = "\n".join(" ".join(f"{v:.15e}" for v in row) for row in M)
    path.write_text(
        f"type      = {lta_type}\n"
        f"nxforms   = 1\n"
        f"mean      = 0.0 0.0 0.0\n"
        f"sigma     = 1.0\n"
        f"1 4 4\n"
        f"{rows}\n"
    )
    return str(path)


def _make_image(path: Path, affine: np.ndarray | None = None) -> str:
    affine = np.eye(4) if affine is None else affine
    img = nib.Nifti1Image(np.zeros((16, 16, 16), dtype=np.float32), affine)
    nib.save(img, path)
    return str(path)


# ---------------------------------------------------------------------------
# rigid_dist
# ---------------------------------------------------------------------------

class TestRigidDist:
    def test_identity(self):
        assert rigid_dist(_IDENTITY) == pytest.approx(0.0, abs=1e-10)

    def test_pure_translation(self):
        assert rigid_dist(_translation(3.0, 4.0)) == pytest.approx(5.0, rel=1e-6)

    def test_pure_rotation_angle(self):
        theta = np.radians(5.0)
        assert rigid_dist(_rotation_z(5.0)) == pytest.approx(np.sqrt(2.0) * theta, rel=1e-6)


# ---------------------------------------------------------------------------
# affine_dist
# ---------------------------------------------------------------------------

class TestAffineDist:
    def test_identity(self):
        assert affine_dist(_IDENTITY) == pytest.approx(0.0, abs=1e-10)

    def test_pure_translation(self):
        assert affine_dist(_translation(3.0, 4.0)) == pytest.approx(5.0, rel=1e-6)

    def test_radius_scales_rotation_part(self):
        R = _rotation_z(5.0)
        assert affine_dist(R, radius=200.0) == pytest.approx(2.0 * affine_dist(R, radius=100.0), rel=1e-6)


# ---------------------------------------------------------------------------
# corner_dist
# ---------------------------------------------------------------------------

class TestCornerDist:
    _SHAPE = (256, 256, 256)
    _AFFINE = np.eye(4)

    def test_identity(self):
        assert corner_dist(_IDENTITY, self._SHAPE, src_affine=self._AFFINE) == pytest.approx(0.0, abs=1e-10)

    def test_pure_translation_mm(self):
        assert corner_dist(_translation(3.0, 4.0), self._SHAPE, src_affine=self._AFFINE) == pytest.approx(5.0, rel=1e-6)


# ---------------------------------------------------------------------------
# sphere_dist
# ---------------------------------------------------------------------------

class TestSphereDist:
    def test_identity(self):
        assert sphere_dist(_IDENTITY) == pytest.approx(0.0, abs=1e-10)

    def test_pure_translation(self):
        t = np.array([3.0, 4.0, 0.0])
        assert sphere_dist(_translation(*t)) == pytest.approx(float(np.linalg.norm(t)), rel=1e-6)

    def test_radius_affects_rotation_result(self):
        R = _rotation_z(5.0)
        assert sphere_dist(R, radius=100.0) == pytest.approx(2.0 * sphere_dist(R, radius=50.0), rel=1e-4)


# ---------------------------------------------------------------------------
# decompose_transform
# ---------------------------------------------------------------------------

class TestDecomposeTransform:
    def test_identity(self):
        d = decompose_transform(_IDENTITY)
        assert d["rotation"] == pytest.approx(np.eye(3), abs=1e-10)
        assert d["translation"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-10)
        assert d["scales"] == pytest.approx([1.0, 1.0, 1.0], rel=1e-6)
        assert d["rot_angle_deg"] == pytest.approx(0.0, abs=1e-6)
        assert d["determinant"] == pytest.approx(1.0, rel=1e-10)

    def test_pure_rotation(self):
        d = decompose_transform(_rotation_z(30.0))
        assert d["rot_angle_deg"] == pytest.approx(30.0, rel=1e-4)
        assert d["abs_trans"] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# LTA class and adjacent formats
# ---------------------------------------------------------------------------

class TestLTAClass:
    def test_r2r_v2v_no_volume_info(self):
        """r2r()/v2v() must not touch volume-info when no conversion is needed."""
        M = _rotation_z(5.0) @ _translation(2.0, 1.0)
        assert LTA(M, lta_type=1, src={}, dst={}).r2r() == pytest.approx(M, rel=1e-10)
        assert LTA(M, lta_type=0, src={}, dst={}).v2v() == pytest.approx(M, rel=1e-10)

    def test_read_write_roundtrip(self, tmp_path):
        M = _rotation_z(5.0) @ _translation(2.0, 1.0)
        lta = LTA.from_matrix(M, "src.mgz", _GEOM, "dst.mgz", _GEOM)
        lta.write(tmp_path / "out.lta")
        lta2 = LTA.read(tmp_path / "out.lta")
        assert lta2.matrix == pytest.approx(M, rel=1e-10)
        assert lta2.type == 1

    def test_subject_and_fscale_roundtrip(self, tmp_path: Path):
        lta = LTA(
            np.eye(4),
            1,
            {
                "filename": "src",
                "volume": [1, 1, 1],
                "voxelsize": [1, 1, 1],
                "xras": [1, 0, 0],
                "yras": [0, 1, 0],
                "zras": [0, 0, 1],
                "cras": [0, 0, 0],
            },
            {
                "filename": "dst",
                "volume": [1, 1, 1],
                "voxelsize": [1, 1, 1],
                "xras": [1, 0, 0],
                "yras": [0, 1, 0],
                "zras": [0, 0, 1],
                "cras": [0, 0, 0],
            },
            subject="bert",
            fscale=0.1,
        )
        path = tmp_path / "meta.lta"
        lta.write(path)
        reread = LTA.read(path)
        assert reread.subject == "bert"
        assert reread.fscale == pytest.approx(0.1)

    def test_r2r_and_v2v(self, tmp_path):
        p = _write_lta(tmp_path / "id.lta", _IDENTITY, lta_type=1)
        assert LTA.read(p).r2r() == pytest.approx(_IDENTITY, abs=1e-10)
        p = _write_lta(tmp_path / "id_v.lta", _IDENTITY, lta_type=0)
        assert LTA.read(p).v2v() == pytest.approx(_IDENTITY, abs=1e-10)

    def test_invert_restores_identity(self):
        M = _rotation_z(5.0) @ _translation(2.0, 1.0)
        lta = LTA.from_matrix(M, "src.mgz", _GEOM, "dst.mgz", _GEOM)
        assert lta.invert().r2r() @ lta.r2r() == pytest.approx(np.eye(4), abs=1e-10)

    def test_concat_matrix_and_geometry(self):
        geom_A = {**_GEOM, "volume": [256, 256, 256]}
        geom_B = {**_GEOM, "volume": [128, 128, 128]}
        geom_C = {**_GEOM, "volume": [64, 64, 64]}
        M_AB = _translation(1.0, 2.0, 3.0)
        M_BC = _rotation_z(5.0)
        lta_AB = LTA(M_AB, 1, geom_A, geom_B)
        lta_BC = LTA(M_BC, 1, geom_B, geom_C)
        lta_AC = lta_AB.concat(lta_BC)
        assert lta_AC.r2r() == pytest.approx(M_BC @ M_AB, abs=1e-10)
        assert lta_AC.src["volume"] == geom_A["volume"]
        assert lta_AC.dst["volume"] == geom_C["volume"]
        assert lta_AC.type == 1

    def test_concat_invert_roundtrip(self):
        M_AB = _translation(1.0, 2.0, 3.0)
        M_BC = _rotation_z(5.0)
        lta_AB = LTA.from_matrix(M_AB, "a.mgz", _GEOM, "b.mgz", _GEOM)
        lta_BC = LTA.from_matrix(M_BC, "b.mgz", _GEOM, "c.mgz", _GEOM)
        lta_AC = lta_AB.concat(lta_BC)
        assert lta_AC.invert().r2r() @ lta_AC.r2r() == pytest.approx(np.eye(4), abs=1e-10)

    def test_write_type_conversion(self, tmp_path):
        M = _rotation_z(5.0) @ _translation(2.0, 1.0)
        lta = LTA.from_matrix(M, "src.mgz", _GEOM, "dst.mgz", _GEOM, lta_type=1)
        lta.write(tmp_path / "out_v2v.lta", lta_type=0)
        lta2 = LTA.read(tmp_path / "out_v2v.lta")
        assert lta2.type == 0
        assert lta2.r2r() == pytest.approx(lta.r2r(), rel=1e-10)

    def test_invalid_dst_geometry_roundtrip_and_v2v_failure(self, tmp_path):
        M = _rotation_z(5.0) @ _translation(2.0, 1.0)
        lta = LTA.from_matrix(M, "src.mgz", _GEOM, "unknown.json", None, lta_type=1)
        lta.write(tmp_path / "out_invalid_dst.lta")
        reloaded = LTA.read(tmp_path / "out_invalid_dst.lta")

        assert reloaded.r2r() == pytest.approx(M, rel=1e-10)
        assert reloaded.dst["valid"] == 0
        assert reloaded.dst["filename"] == "unknown.json"
        with pytest.raises(ValueError, match="valid = 0"):
            reloaded.v2v()

    def test_corner_dist_method_translation(self, tmp_path):
        p = _write_lta(tmp_path / "t.lta", _translation(3.0, 4.0))
        assert LTA.read(p).corner_dist() == pytest.approx(5.0, rel=1e-5)

    def test_repr(self):
        lta = LTA.from_matrix(_IDENTITY, "src.mgz", _GEOM, "dst.mgz", _GEOM)
        assert "R2R" in repr(lta)
        assert "src.mgz" in repr(lta)


class TestXFM:
    def test_read_write_roundtrip(self, tmp_path: Path):
        M = _rotation_z(5.0) @ _translation(2.0, 1.0, -0.5)
        xfm = XFM(M, comments=["%Generated by neuroreg src mov.mgz dst ref.mgz"])
        path = tmp_path / "test.xfm"
        xfm.write(path)
        reread = XFM.read(path)
        assert reread.matrix == pytest.approx(M, rel=1e-10)
        assert reread.src_path == "mov.mgz"
        assert reread.dst_path == "ref.mgz"

    def test_to_lta_without_geometry_marks_invalid(self):
        M = _rotation_z(5.0) @ _translation(2.0, 1.0, -0.5)
        lta = XFM(M, comments=["%Generated by neuroreg src mov.mgz dst ref.mgz"]).to_lta()
        assert lta.r2r() == pytest.approx(M, rel=1e-10)
        assert lta.src["valid"] == 0
        assert lta.dst["valid"] == 0
        assert lta.src["filename"] == "mov.mgz"
        assert lta.dst["filename"] == "ref.mgz"


class TestRegisterDat:
    def test_read_write_roundtrip(self, tmp_path: Path):
        reg = RegisterDat(
            np.eye(4),
            subject="bert",
            inplane_resolution=1.0,
            between_plane_resolution=2.0,
            intensity=0.15,
            float2int="round",
        )
        path = tmp_path / "register.dat"
        reg.write(path)
        reread = RegisterDat.read(path)
        assert reread.matrix == pytest.approx(np.eye(4), rel=1e-10)
        assert reread.subject == "bert"
        assert reread.between_plane_resolution == pytest.approx(2.0)
        assert reread.float2int == "round"

    def test_lta_roundtrip_with_images(self, tmp_path: Path):
        src_img = _make_image(tmp_path / "mov.nii.gz")
        dst_img = _make_image(tmp_path / "ref.nii.gz")
        M = _rotation_z(5.0) @ _translation(2.0, 1.0, -0.5)
        lta = LTA.from_matrix(M, src_img, src_img, dst_img, dst_img, lta_type=1)
        lta.subject = "bert"
        lta.fscale = 0.1
        reg = RegisterDat.from_lta(lta, float2int="round")
        restored = reg.to_lta(src_fname=src_img, src_img=src_img, dst_fname=dst_img, dst_img=dst_img)
        assert restored.r2r() == pytest.approx(M, rel=1e-6, abs=1e-6)
        assert restored.subject == "bert"
        assert restored.fscale == pytest.approx(0.1)


class TestFSLMat:
    def test_read_write_roundtrip(self, tmp_path: Path):
        fsl = FSLMat(np.eye(4))
        path = tmp_path / "xfm.mat"
        fsl.write(path)
        reread = FSLMat.read(path)
        assert reread.matrix == pytest.approx(np.eye(4), rel=1e-10)

    def test_lta_roundtrip_with_positive_det_nifti(self, tmp_path: Path):
        src_img = _make_image(tmp_path / "mov.nii.gz", affine=np.eye(4))
        dst_img = _make_image(tmp_path / "ref.nii.gz", affine=np.eye(4))
        M = _rotation_z(5.0) @ _translation(2.0, 1.0, -0.5)
        lta = LTA.from_matrix(M, src_img, src_img, dst_img, dst_img, lta_type=1)
        fsl = FSLMat.from_lta(lta)
        restored = fsl.to_lta(src_fname=src_img, src_img=src_img, dst_fname=dst_img, dst_img=dst_img)
        assert restored.r2r() == pytest.approx(M, rel=1e-6, abs=1e-6)


class TestITKTransform:
    def test_read_write_roundtrip(self, tmp_path: Path):
        itk = ITKTransform(np.eye(4))
        path = tmp_path / "affine.tfm"
        itk.write(path)
        reread = ITKTransform.read(path)
        assert reread.matrix == pytest.approx(np.eye(4), rel=1e-10)

    def test_read_with_fixed_parameters(self, tmp_path: Path):
        path = tmp_path / "fixed.tfm"
        path.write_text(
            "#Insight Transform File V1.0\n"
            "#Transform 0\n"
            "Transform: AffineTransform_double_3_3\n"
            "Parameters: 0 -1 0 1 0 0 0 0 1 10 20 30\n"
            "FixedParameters: 5 6 7\n"
        )
        reread = ITKTransform.read(path)
        expected = np.eye(4)
        expected[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        fixed = np.array([5.0, 6.0, 7.0])
        translation = np.array([10.0, 20.0, 30.0])
        expected[:3, 3] = translation + fixed - expected[:3, :3] @ fixed
        assert reread.matrix == pytest.approx(expected, rel=1e-10)

    def test_lta_roundtrip_with_images(self, tmp_path: Path):
        src_img = _make_image(tmp_path / "mov.nii.gz")
        dst_img = _make_image(tmp_path / "ref.nii.gz")
        M = _rotation_z(5.0) @ _translation(2.0, 1.0, -0.5)
        lta = LTA.from_matrix(M, src_img, src_img, dst_img, dst_img, lta_type=1)
        itk = ITKTransform.from_lta(lta)
        restored = itk.to_lta(src_fname=src_img, src_img=src_img, dst_fname=dst_img, dst_img=dst_img)
        assert restored.r2r() == pytest.approx(M, rel=1e-6, abs=1e-6)

    def test_to_lta_without_geometry_marks_invalid(self):
        M = _rotation_z(5.0) @ _translation(2.0, 1.0, -0.5)
        lta = LTA.from_matrix(M, "mov.nii.gz", _GEOM, "ref.nii.gz", _GEOM, lta_type=1)
        restored = ITKTransform.from_lta(lta).to_lta(src_fname="mov.nii.gz", dst_fname="ref.nii.gz")
        assert restored.r2r() == pytest.approx(M, rel=1e-6, abs=1e-6)
        assert restored.src["valid"] == 0
        assert restored.dst["valid"] == 0


# ---------------------------------------------------------------------------
# CLI (end-to-end)
# ---------------------------------------------------------------------------

@pytest.fixture()
def identity_lta(tmp_path: Path) -> str:
    return _write_lta(tmp_path / "identity.lta", _IDENTITY)


@pytest.fixture()
def rigid_lta(tmp_path: Path) -> str:
    return _write_lta(tmp_path / "rigid.lta", _rotation_z(5.0) @ _translation(2.0, 1.0, 0.5))


class TestLtaDiffCLI:
    def test_module_entrypoint_runs(self, identity_lta, tmp_path):
        output_path = tmp_path / "module_diff.txt"
        proc = subprocess.run(
            [sys.executable, "-m", "neuroreg.cli.lta", "diff", identity_lta],
            check=False,
            capture_output=True,
            text=True,
        )
        output_path.write_text(proc.stdout + proc.stderr)
        assert proc.returncode == 0, output_path.read_text()
        assert float(proc.stdout.strip()) == pytest.approx(0.0, abs=1e-10)

    def test_dist2_is_default(self, identity_lta, capsys):
        main(["diff", identity_lta])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(0.0, abs=1e-10)

    def test_dist1(self, rigid_lta, capsys):
        main(["diff", rigid_lta, "--dist", "1"])
        assert float(capsys.readouterr().out.strip()) > 0.0

    def test_dist3_pure_translation(self, tmp_path, capsys):
        lta = _write_lta(tmp_path / "trans.lta", _translation(3.0, 4.0))
        main(["diff", lta, "--dist", "3"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(5.0, rel=1e-5)

    def test_dist4(self, identity_lta, capsys):
        main(["diff", identity_lta, "--dist", "4"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(0.0, abs=1e-10)

    def test_dist5_det_is_one(self, rigid_lta, capsys):
        main(["diff", rigid_lta, "--dist", "5"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(1.0, rel=1e-4)

    def test_dist7_output_fields(self, identity_lta, capsys):
        main(["diff", identity_lta, "--dist", "7"])
        out = capsys.readouterr().out
        for field in ("Rot", "Trans", "Determinant", "Scales", "RotAngle"):
            assert field in out

    def test_normdiv_halves_result(self, rigid_lta, capsys):
        main(["diff", rigid_lta, "--dist", "2"])
        d_full = float(capsys.readouterr().out.strip())
        main(["diff", rigid_lta, "--dist", "2", "--normdiv", "2"])
        assert float(capsys.readouterr().out.strip()) == pytest.approx(d_full / 2.0, rel=1e-6)

    def test_invert1_preserves_dist2_for_rigid_transform(self, rigid_lta, capsys):
        main(["diff", rigid_lta, "--dist", "2"])
        d1 = float(capsys.readouterr().out.strip())
        main(["diff", rigid_lta, "--dist", "2", "--invert1"])
        d2 = float(capsys.readouterr().out.strip())
        assert d1 == pytest.approx(d2, rel=1e-4)

    def test_invert2_requires_second_lta(self, identity_lta):
        with pytest.raises(SystemExit):
            main(["diff", identity_lta, "--invert2"])

    def test_missing_lta_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["diff", str(tmp_path / "does_not_exist.lta")])

    def test_dist3_missing_vol_info_exits(self, tmp_path):
        bare = _write_bare_lta(tmp_path / "bare.lta", _IDENTITY)
        with pytest.raises(SystemExit):
            main(["diff", bare, "--dist", "3"])


class TestLtaInvertCLI:
    def test_invert_roundtrip(self, tmp_path):
        src = _write_lta(tmp_path / "rigid.lta", _rotation_z(5.0) @ _translation(2.0, 1.0))
        inv = str(tmp_path / "rigid_inv.lta")
        inv2 = str(tmp_path / "rigid_inv2.lta")
        main(["invert", src, inv])
        main(["invert", inv, inv2])
        assert LTA.read(inv2).r2r() == pytest.approx(LTA.read(src).r2r(), abs=1e-10)

    def test_invert_matrix_is_correct(self, tmp_path):
        M = _rotation_z(15.0) @ _translation(3.0, -1.0, 2.0)
        src = _write_lta(tmp_path / "orig.lta", M)
        inv = str(tmp_path / "inv.lta")
        main(["invert", src, inv])
        assert LTA.read(inv).r2r() @ M == pytest.approx(np.eye(4), abs=1e-10)

    def test_invert_swaps_src_dst(self, tmp_path):
        src = _write_lta(tmp_path / "orig.lta", _IDENTITY)
        inv = str(tmp_path / "inv.lta")
        main(["invert", src, inv])
        orig = LTA.read(src)
        inverted = LTA.read(inv)
        assert inverted.src["filename"] == orig.dst["filename"]
        assert inverted.dst["filename"] == orig.src["filename"]

    def test_invert_missing_input_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["invert", str(tmp_path / "missing.lta"), str(tmp_path / "out.lta")])


class TestLtaConcatCLI:
    def test_concat_matrix(self, tmp_path):
        M_AB = _translation(1.0, 2.0, 3.0)
        M_BC = _rotation_z(5.0)
        lta_ab = _write_lta(tmp_path / "ab.lta", M_AB)
        lta_bc = _write_lta(tmp_path / "bc.lta", M_BC)
        out = str(tmp_path / "ac.lta")
        main(["concat", lta_ab, lta_bc, out])
        assert LTA.read(out).r2r() == pytest.approx(M_BC @ M_AB, abs=1e-10)

    def test_concat_geometry(self, tmp_path):
        lta_ab = _write_lta(tmp_path / "ab.lta", _IDENTITY)
        lta_bc = _write_lta(tmp_path / "bc.lta", _IDENTITY)
        out = str(tmp_path / "ac.lta")
        main(["concat", lta_ab, lta_bc, out])
        result = LTA.read(out)
        assert result.src["filename"] == LTA.read(lta_ab).src["filename"]
        assert result.dst["filename"] == LTA.read(lta_bc).dst["filename"]

    def test_concat_then_invert_is_identity(self, tmp_path):
        M_AB = _translation(1.0, 2.0, 3.0)
        M_BC = _rotation_z(10.0)
        lta_ab = _write_lta(tmp_path / "ab.lta", M_AB)
        lta_bc = _write_lta(tmp_path / "bc.lta", M_BC)
        out = str(tmp_path / "ac.lta")
        main(["concat", lta_ab, lta_bc, out])
        lta_AC = LTA.read(out)
        assert lta_AC.invert().r2r() @ lta_AC.r2r() == pytest.approx(np.eye(4), abs=1e-10)

    def test_concat_missing_lta1_exits(self, tmp_path):
        lta_bc = _write_lta(tmp_path / "bc.lta", _IDENTITY)
        with pytest.raises(SystemExit):
            main(["concat", str(tmp_path / "missing.lta"), lta_bc, str(tmp_path / "out.lta")])


class TestConvertCLI:
    def test_convert_xfm_to_lta_with_geometry(self, tmp_path: Path):
        src_img = _make_image(tmp_path / "mov.nii.gz")
        dst_img = _make_image(tmp_path / "ref.nii.gz")
        M = _rotation_z(5.0) @ _translation(2.0, 1.0, -0.5)
        xfm = tmp_path / "input.xfm"
        XFM(M, comments=["%Generated by neuroreg src mov.nii.gz dst ref.nii.gz"]).write(xfm)
        out = tmp_path / "out.lta"
        main(["convert", str(xfm), str(out), "--src-img", src_img, "--dst-img", dst_img])
        reread = LTA.read(out)
        assert reread.r2r() == pytest.approx(M, rel=1e-6, abs=1e-6)
        assert reread.src["valid"] == 1
        assert reread.dst["valid"] == 1

    def test_convert_lta_to_registerdat_and_back(self, tmp_path: Path):
        src_img = _make_image(tmp_path / "mov.nii.gz")
        dst_img = _make_image(tmp_path / "ref.nii.gz")
        M = _rotation_z(5.0) @ _translation(2.0, 1.0, -0.5)
        lta = LTA.from_matrix(M, src_img, src_img, dst_img, dst_img, lta_type=1)
        lta.subject = "bert"
        lta.fscale = 0.1
        lta_path = tmp_path / "input.lta"
        lta.write(lta_path)
        reg_path = tmp_path / "output.dat"
        roundtrip_lta = tmp_path / "roundtrip.lta"

        main(["convert", str(lta_path), str(reg_path), "--subject", "bert", "--fscale", "0.1"])
        main(["convert", str(reg_path), str(roundtrip_lta), "--src-img", src_img, "--dst-img", dst_img])

        reread = LTA.read(roundtrip_lta)
        assert reread.r2r() == pytest.approx(M, rel=1e-6, abs=1e-6)

    def test_convert_lta_to_fsl_and_back(self, tmp_path: Path):
        src_img = _make_image(tmp_path / "mov.nii.gz", affine=np.eye(4))
        dst_img = _make_image(tmp_path / "ref.nii.gz", affine=np.eye(4))
        M = _rotation_z(5.0) @ _translation(2.0, 1.0, -0.5)
        lta = LTA.from_matrix(M, src_img, src_img, dst_img, dst_img, lta_type=1)
        lta_path = tmp_path / "input.lta"
        lta.write(lta_path)
        fsl_path = tmp_path / "output.mat"
        roundtrip_lta = tmp_path / "roundtrip_fsl.lta"

        main(["convert", str(lta_path), str(fsl_path)])
        main(["convert", str(fsl_path), str(roundtrip_lta), "--src-img", src_img, "--dst-img", dst_img])

        reread = LTA.read(roundtrip_lta)
        assert reread.r2r() == pytest.approx(M, rel=1e-6, abs=1e-6)

    def test_convert_lta_to_itk_and_back(self, tmp_path: Path):
        src_img = _make_image(tmp_path / "mov.nii.gz")
        dst_img = _make_image(tmp_path / "ref.nii.gz")
        M = _rotation_z(5.0) @ _translation(2.0, 1.0, -0.5)
        lta = LTA.from_matrix(M, src_img, src_img, dst_img, dst_img, lta_type=1)
        lta_path = tmp_path / "input.lta"
        lta.write(lta_path)
        itk_path = tmp_path / "output.txt"
        roundtrip_lta = tmp_path / "roundtrip_itk.lta"

        main(["convert", str(lta_path), str(itk_path), "--out-format", "itk"])
        main(
            [
                "convert",
                str(itk_path),
                str(roundtrip_lta),
                "--in-format",
                "itk",
                "--src-img",
                src_img,
                "--dst-img",
                dst_img,
            ]
        )

        reread = LTA.read(roundtrip_lta)
        assert reread.r2r() == pytest.approx(M, rel=1e-6, abs=1e-6)

    def test_convert_itk_to_lta_without_geometry(self, tmp_path: Path):
        M = _rotation_z(5.0) @ _translation(2.0, 1.0, -0.5)
        lta = LTA.from_matrix(M, "mov.nii.gz", _GEOM, "ref.nii.gz", _GEOM, lta_type=1)
        itk_path = tmp_path / "input.txt"
        ITKTransform.from_lta(lta).write(itk_path)
        out = tmp_path / "out.lta"

        main(["convert", str(itk_path), str(out), "--in-format", "itk"])

        reread = LTA.read(out)
        assert reread.r2r() == pytest.approx(M, rel=1e-6, abs=1e-6)
        assert reread.src["valid"] == 0
        assert reread.dst["valid"] == 0

    def test_registerdat_requires_geometry(self, tmp_path: Path):
        reg = tmp_path / "input.dat"
        RegisterDat(np.eye(4)).write(reg)
        with pytest.raises(SystemExit):
            main(["convert", str(reg), str(tmp_path / "out.lta")])

    def test_fsl_requires_geometry(self, tmp_path: Path):
        fsl = tmp_path / "input.mat"
        FSLMat(np.eye(4)).write(fsl)
        with pytest.raises(SystemExit):
            main(["convert", str(fsl), str(tmp_path / "out.lta")])
