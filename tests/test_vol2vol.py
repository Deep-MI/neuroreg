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
    def test_no_flags_reads_and_writes_without_reslicing(self, tmp_path: Path):
        # With no transform/ref the image is copied as-is: dtype and voxels are
        # preserved (no float32 cast, no interpolation).
        data = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        mov_path = _write_image(tmp_path / "mov.nii.gz", data, affine=affine)
        out_path = tmp_path / "out.nii.gz"

        vol2vol_main(["--in", str(mov_path), "--out", str(out_path)])

        mapped = nib.load(str(out_path))
        assert mapped.shape == (3, 3, 3)
        assert mapped.affine == pytest.approx(affine)
        assert mapped.get_data_dtype() == np.dtype(np.uint8)
        assert np.asarray(mapped.dataobj) == pytest.approx(data)

    @pytest.mark.parametrize(
        ("in_ext", "in_class", "out_ext", "out_class"),
        [
            (".mgz", nib.MGHImage, ".nii.gz", nib.Nifti1Image),
            (".nii.gz", nib.Nifti1Image, ".mgz", nib.MGHImage),
            (".nii", nib.Nifti1Image, ".nii.gz", nib.Nifti1Image),
        ],
    )
    def test_pure_conversion_follows_output_extension(self, tmp_path: Path, in_ext, in_class, out_ext, out_class):
        data = np.arange(27, dtype=np.int16).reshape(3, 3, 3)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        mov_path = tmp_path / f"mov{in_ext}"
        in_class(data, affine).to_filename(str(mov_path))
        out_path = tmp_path / f"out{out_ext}"

        vol2vol_main(["--in", str(mov_path), "--out", str(out_path)])

        mapped = nib.load(str(out_path))
        assert isinstance(mapped, out_class)
        assert mapped.affine == pytest.approx(affine)
        assert np.asarray(mapped.dataobj) == pytest.approx(data)

    def test_unsupported_output_extension_errors(self, tmp_path: Path):
        mov_path = _write_image(tmp_path / "mov.nii.gz", np.ones((2, 2, 2), dtype=np.float32))
        with pytest.raises(SystemExit):
            vol2vol_main(["--in", str(mov_path), "--out", str(tmp_path / "out.foo")])

    def test_dtype_only_conversion_does_not_reslice(self, tmp_path: Path):
        data = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
        mov_path = _write_image(tmp_path / "mov.nii.gz", data)
        out_path = tmp_path / "out_short.nii.gz"

        vol2vol_main(["--in", str(mov_path), "--out", str(out_path), "--out-dtype", "int16"])

        mapped = nib.load(str(out_path))
        assert mapped.get_data_dtype() == np.dtype(np.int16)
        assert np.asarray(mapped.dataobj) == pytest.approx(data)

    def test_keep_dtype_preserves_linear_output_dtype(self, tmp_path: Path):
        mov_path = _write_image(tmp_path / "mov.nii.gz", np.arange(8, dtype=np.uint8).reshape(2, 2, 2))
        out_path = tmp_path / "out_keep_dtype.nii.gz"

        vol2vol_main(["--in", str(mov_path), "--out", str(out_path), "--keep-dtype"])

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
                "--in",
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
        dst_affine = np.array([[1.0, 0.0, 0.0, -1.0], [0.0, 1.5, 0.0, 5.0], [0.0, 0.0, 2.0, 7.0], [0.0, 0.0, 0.0, 1.0]])
        mov_path = _write_image(tmp_path / "mov.nii.gz", np.ones((3, 3, 3), dtype=np.float32), affine=src_affine)
        ref_path = _write_image(tmp_path / "ref.nii.gz", np.zeros((5, 5, 5), dtype=np.float32), affine=dst_affine)
        lta_path = tmp_path / "geom.lta"
        LTA.from_matrix(np.eye(4), str(mov_path), str(mov_path), str(ref_path), str(ref_path)).write(lta_path)
        out_dst = tmp_path / "out_dst.nii.gz"
        out_src = tmp_path / "out_src.nii.gz"

        vol2vol_main(["--in", str(mov_path), "--transform", str(lta_path), "--out", str(out_dst)])
        vol2vol_main(["--in", str(mov_path), "--transform", str(lta_path), "--inverse", "--out", str(out_src)])

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

        vol2vol_main(["--in", str(mov_path), "--transform", str(lta_path), "--header-only", "--out", str(out_path)])

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
                "--in",
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
                "--in",
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
                "--in",
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

    def test_mask_same_geometry_matches_mri_mask(self, tmp_path: Path, capsys):
        mov = np.arange(1, 28, dtype=np.uint8).reshape(3, 3, 3)
        mask = np.zeros((3, 3, 3), dtype=np.uint8)
        mask[1, 1, 1] = 1
        mask[0, 0, 0] = 5
        mov_path = _write_image(tmp_path / "mov.nii.gz", mov)
        mask_path = _write_image(tmp_path / "mask.nii.gz", mask)
        out_path = tmp_path / "out_masked.nii.gz"

        vol2vol_main(["--in", str(mov_path), "--mask", str(mask_path), "--out", str(out_path), "--keep-dtype"])

        mapped = nib.load(str(out_path))
        expected = np.where(mask > 0, mov, 0)
        assert mapped.get_data_dtype() == np.dtype(np.uint8)
        assert np.asarray(mapped.dataobj) == pytest.approx(expected)
        # same geometry -> no reslicing notice
        assert "geometry differs" not in capsys.readouterr().out

    def test_mask_threshold_and_fill(self, tmp_path: Path):
        mov = np.full((2, 2, 2), 9.0, dtype=np.float32)
        mask = np.array([[[0, 1], [2, 3]], [[1, 2], [3, 0]]], dtype=np.float32)
        mov_path = _write_image(tmp_path / "mov.nii.gz", mov)
        mask_path = _write_image(tmp_path / "mask.nii.gz", mask)
        out_path = tmp_path / "out_thr.nii.gz"

        vol2vol_main(
            [
                "--in",
                str(mov_path),
                "--mask",
                str(mask_path),
                "--out",
                str(out_path),
                "--mask-threshold",
                "1",
                "--mask-fill",
                "-1",
            ]
        )

        mapped = np.asarray(nib.load(str(out_path)).dataobj, dtype=np.float32)
        assert mapped == pytest.approx(np.where(mask > 1, 9.0, -1.0))

    def test_mask_fill_out_of_range_clamps_for_integer_dtype(self, tmp_path: Path):
        # fill=-1 with uint8 image: should clamp to 0, not wrap to 255.
        mov = np.full((2, 2, 2), 10, dtype=np.uint8)
        mask = np.zeros((2, 2, 2), dtype=np.uint8)
        mask[0, 0, 0] = 1  # keep only one voxel
        mov_path = _write_image(tmp_path / "mov.nii.gz", mov)
        mask_path = _write_image(tmp_path / "mask.nii.gz", mask)
        out_path = tmp_path / "out_clamp.nii.gz"

        vol2vol_main(
            [
                "--in",
                str(mov_path),
                "--mask",
                str(mask_path),
                "--out",
                str(out_path),
                "--keep-dtype",
                "--mask-fill",
                "-1",
            ]
        )

        result = np.asarray(nib.load(str(out_path)).dataobj)
        assert result[0, 0, 0] == 10  # kept voxel unchanged
        assert result[1, 1, 1] == 0  # fill=-1 clamped to uint8 min, not wrapped to 255

    def test_mask_bool_image_clamps_fill_to_false(self):
        # Nifti1Image rejects bool dtype, so we use a minimal duck-typed stub
        # that satisfies the reslice_and_apply_mask + create_image_like interface.
        # fill=-1 must clamp to False, not stay True (non-zero → True in plain cast).
        from neuroreg.image import reslice_and_apply_mask

        class _BoolImg:
            def __init__(self, data, affine, _header=None):
                self._data = np.asarray(data)
                self.affine = np.asarray(affine, dtype=np.float64)
                self.shape = self._data.shape

            def get_data_dtype(self):
                return self._data.dtype

            def get_fdata(self):
                return self._data.astype(np.float64)

            @property
            def dataobj(self):
                return self._data

            @property
            def header(self):
                return self

            def copy(self):
                return self

            def set_data_dtype(self, dt):
                pass

        data = np.array([[[True, False], [True, False]], [[True, False], [True, False]]], dtype=np.bool_)
        mask_data = np.zeros((2, 2, 2), dtype=np.uint8)
        mask_data[0, 0, 0] = 1
        img = _BoolImg(data, np.eye(4))
        mask_img = nib.Nifti1Image(mask_data, np.eye(4))

        result = reslice_and_apply_mask(img, mask_img, fill=-1)
        result_data = np.asarray(result.dataobj)

        assert result.get_data_dtype() == np.dtype(np.bool_)
        assert result_data[0, 0, 0]  # kept voxel: True stays True
        assert not result_data[1, 1, 1]  # fill=-1 clamped to 0 → False

    def test_mask_different_geometry_is_resampled(self, tmp_path: Path, capsys):
        mov = np.full((3, 3, 3), 5.0, dtype=np.float32)
        mask = np.ones((2, 2, 2), dtype=np.uint8)  # covers RAS voxels 0,1 only
        mov_path = _write_image(tmp_path / "mov.nii.gz", mov)
        mask_path = _write_image(tmp_path / "mask.nii.gz", mask)
        out_path = tmp_path / "out_diff.nii.gz"

        vol2vol_main(["--in", str(mov_path), "--mask", str(mask_path), "--out", str(out_path)])

        mapped = np.asarray(nib.load(str(out_path)).dataobj, dtype=np.float32)
        expected = np.zeros((3, 3, 3), dtype=np.float32)
        expected[:2, :2, :2] = 5.0
        assert mapped == pytest.approx(expected)
        assert "geometry differs" in capsys.readouterr().out

    def test_mask_rejected_with_header_only(self, tmp_path: Path):
        mov_path = _write_image(tmp_path / "mov.nii.gz", np.ones((2, 2, 2), dtype=np.float32))
        mask_path = _write_image(tmp_path / "mask.nii.gz", np.ones((2, 2, 2), dtype=np.uint8))
        lta_path = _write_lta(tmp_path / "id.lta", np.eye(4), (2, 2, 2), (2, 2, 2))
        with pytest.raises(SystemExit):
            vol2vol_main(
                [
                    "--in",
                    str(mov_path),
                    "--transform",
                    str(lta_path),
                    "--header-only",
                    "--mask",
                    str(mask_path),
                    "--out",
                    str(tmp_path / "o.nii.gz"),
                ]
            )

    def test_mask_threshold_requires_mask(self, tmp_path: Path):
        mov_path = _write_image(tmp_path / "mov.nii.gz", np.ones((2, 2, 2), dtype=np.float32))
        with pytest.raises(SystemExit):
            vol2vol_main(["--in", str(mov_path), "--out", str(tmp_path / "o.nii.gz"), "--mask-threshold", "1"])
