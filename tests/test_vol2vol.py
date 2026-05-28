from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neuroreg.cli.vol2vol import main as vol2vol_main
from neuroreg.transforms import LTA


def _write_image(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> Path:
    img = nib.Nifti1Image(data, np.eye(4) if affine is None else affine)
    nib.save(img, path)
    return path


def _geom(shape: tuple[int, int, int]) -> dict[str, object]:
    return {
        "dims": list(shape),
        "delta": [1.0, 1.0, 1.0],
        "Mdc": np.eye(3),
        "Pxyz_c": np.zeros(3),
    }


def _write_lta(
        path: Path,
        matrix: np.ndarray,
        src_shape: tuple[int, int, int],
        dst_shape: tuple[int, int, int],
) -> Path:
    LTA.from_matrix(matrix, "mov.nii.gz", _geom(src_shape), "ref.nii.gz", _geom(dst_shape)).write(path)
    return path


class TestVol2VolCli:
    def test_identity_mapping_defaults_to_float32(self, tmp_path: Path):
        mov_path = _write_image(tmp_path / "mov.nii.gz", np.arange(27, dtype=np.uint8).reshape(3, 3, 3))
        out_path = tmp_path / "out.nii.gz"

        vol2vol_main(["--mov", str(mov_path), "--out", str(out_path)])

        mapped = nib.load(str(out_path))
        assert mapped.shape == (3, 3, 3)
        assert mapped.affine == pytest.approx(np.eye(4))
        assert mapped.get_data_dtype() == np.dtype(np.float32)
        assert mapped.get_fdata(dtype=np.float32) == pytest.approx(np.arange(27, dtype=np.float32).reshape(3, 3, 3))

    def test_keep_dtype_preserves_linear_output_dtype(self, tmp_path: Path):
        mov_path = _write_image(tmp_path / "mov.nii.gz", np.arange(8, dtype=np.uint8).reshape(2, 2, 2))
        out_path = tmp_path / "out_keep_dtype.nii.gz"

        vol2vol_main(["--mov", str(mov_path), "--out", str(out_path), "--keep-dtype"])

        mapped = nib.load(str(out_path))
        assert mapped.get_data_dtype() == np.dtype(np.uint8)
        assert np.asarray(mapped.dataobj) == pytest.approx(np.arange(8, dtype=np.uint8).reshape(2, 2, 2))

    def test_nearest_with_numeric_padding_preserves_integer_dtype(self, tmp_path: Path):
        mov = np.zeros((3, 3, 3), dtype=np.uint8)
        mov[1, 1, 1] = 7
        mov_path = _write_image(tmp_path / "mov.nii.gz", mov)
        ref_path = _write_image(tmp_path / "ref.nii.gz", np.zeros((5, 5, 5), dtype=np.float32))
        out_path = tmp_path / "out_pad.nii.gz"

        vol2vol_main(
            [
                "--mov",
                str(mov_path),
                "--ref",
                str(ref_path),
                "--out",
                str(out_path),
                "--interp",
                "nearest",
                "--pad",
                "255",
            ]
        )

        mapped = nib.load(str(out_path))
        data = np.asarray(mapped.dataobj)
        assert mapped.get_data_dtype() == np.dtype(np.uint8)
        assert data.shape == (5, 5, 5)
        assert data[4, 4, 4] == 255
        assert data[1, 1, 1] == 7

    def test_transform_geometry_fallback_uses_dst_and_inverse_uses_src(self, tmp_path: Path):
        src_affine = np.array(
            [[2.0, 0.0, 0.0, 10.0], [0.0, 2.0, 0.0, 20.0], [0.0, 0.0, 2.0, 30.0], [0.0, 0.0, 0.0, 1.0]]
        )
        dst_affine = np.array(
            [[1.0, 0.0, 0.0, -1.0], [0.0, 1.5, 0.0, 5.0], [0.0, 0.0, 2.0, 7.0], [0.0, 0.0, 0.0, 1.0]]
        )
        mov_path = _write_image(tmp_path / "mov.nii.gz", np.ones((3, 3, 3), dtype=np.float32), affine=src_affine)
        ref_path = _write_image(tmp_path / "ref.nii.gz", np.zeros((5, 5, 5), dtype=np.float32), affine=dst_affine)
        lta_path = tmp_path / "geom.lta"
        LTA.from_matrix(np.eye(4), str(mov_path), str(mov_path), str(ref_path), str(ref_path)).write(lta_path)
        out_dst = tmp_path / "out_dst.nii.gz"
        out_src = tmp_path / "out_src.nii.gz"

        vol2vol_main(["--mov", str(mov_path), "--transform", str(lta_path), "--out", str(out_dst)])
        vol2vol_main(["--mov", str(mov_path), "--transform", str(lta_path), "--inverse", "--out", str(out_src)])

        mapped_dst = nib.load(str(out_dst))
        mapped_src = nib.load(str(out_src))
        assert mapped_dst.shape == (5, 5, 5)
        assert mapped_dst.affine == pytest.approx(dst_affine)
        assert mapped_src.shape == (3, 3, 3)
        assert mapped_src.affine == pytest.approx(src_affine)

    def test_header_only_updates_affine_and_preserves_payload(self, tmp_path: Path):
        data = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
        mov_path = _write_image(tmp_path / "mov.nii.gz", data)
        matrix = np.eye(4)
        matrix[0, 3] = 2.5
        matrix[2, 3] = -1.0
        lta_path = _write_lta(tmp_path / "shift.lta", matrix, (2, 2, 2), (2, 2, 2))
        out_path = tmp_path / "out_hdr.nii.gz"

        vol2vol_main(["--mov", str(mov_path), "--transform", str(lta_path), "--header-only", "--out", str(out_path)])

        mapped = nib.load(str(out_path))
        expected_affine = np.eye(4)
        expected_affine[0, 3] = 2.5
        expected_affine[2, 3] = -1.0
        assert mapped.affine == pytest.approx(expected_affine)
        assert mapped.get_data_dtype() == np.dtype(np.uint8)
        assert np.asarray(mapped.dataobj) == pytest.approx(data)

    def test_rescale_to_uint8_uses_zero_anchored_target_max(self, tmp_path: Path):
        data = np.array([[[0.0, 1.0], [2.0, 0.0]], [[1.0, 2.0], [0.0, 2.0]]], dtype=np.float32)
        mov_path = _write_image(tmp_path / "mov.nii.gz", data)
        out_path = tmp_path / "out_uint8.nii.gz"

        vol2vol_main(
            [
                "--mov",
                str(mov_path),
                "--out",
                str(out_path),
                "--out-dtype",
                "uint8",
                "--scale-mode",
                "rescale",
                "--target-max",
                "100",
            ]
        )

        mapped = nib.load(str(out_path))
        mapped_data = np.asarray(mapped.dataobj)
        expected = np.array([[[0, 50], [100, 0]], [[50, 100], [0, 100]]], dtype=np.uint8)
        assert mapped.get_data_dtype() == np.dtype(np.uint8)
        assert mapped_data == pytest.approx(expected)

    def test_brightest_padding_ignores_non_finite_source_values(self, tmp_path: Path):
        mov = np.zeros((2, 2, 2), dtype=np.float32)
        mov[0, 0, 0] = np.nan
        mov[1, 1, 1] = 7.0
        mov_path = _write_image(tmp_path / "mov_nan.nii.gz", mov)
        ref_path = _write_image(tmp_path / "ref_big.nii.gz", np.zeros((4, 4, 4), dtype=np.float32))
        out_path = tmp_path / "out_brightest.nii.gz"

        vol2vol_main(
            [
                "--mov",
                str(mov_path),
                "--ref",
                str(ref_path),
                "--out",
                str(out_path),
                "--interp",
                "nearest",
                "--pad",
                "brightest",
            ]
        )

        mapped = nib.load(str(out_path))
        mapped_data = np.asarray(mapped.dataobj, dtype=np.float32)
        assert mapped_data[3, 3, 3] == pytest.approx(7.0)
        assert np.isfinite(mapped_data[3, 3, 3])

    def test_rescale_ignores_non_finite_values_when_estimating_source_upper_bound(self, tmp_path: Path):
        data = np.array([[[np.nan, 1.0], [2.0, 0.0]], [[1.0, 2.0], [0.0, 2.0]]], dtype=np.float32)
        mov_path = _write_image(tmp_path / "mov_nan_rescale.nii.gz", data)
        out_path = tmp_path / "out_nan_rescale.nii.gz"

        vol2vol_main(
            [
                "--mov",
                str(mov_path),
                "--out",
                str(out_path),
                "--scale-mode",
                "rescale",
                "--target-max",
                "100",
            ]
        )

        mapped = nib.load(str(out_path))
        mapped_data = np.asarray(mapped.dataobj, dtype=np.float32)
        assert mapped.get_data_dtype() == np.dtype(np.float32)
        assert mapped_data[0, 0, 1] == pytest.approx(50.0)
        assert mapped_data[0, 1, 0] == pytest.approx(100.0)
