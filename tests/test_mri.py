from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neuroreg.cli.mri import main as mri_main


def _write_image(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> Path:
    img = nib.Nifti1Image(data, np.eye(4) if affine is None else affine)
    nib.save(img, path)
    return path


class TestMask:
    def test_same_geometry_matches_mri_mask(self, tmp_path: Path, capsys):
        mov = np.arange(1, 28, dtype=np.uint8).reshape(3, 3, 3)
        mask = np.zeros((3, 3, 3), dtype=np.uint8)
        mask[1, 1, 1] = 1
        mask[0, 0, 0] = 5
        in_path = _write_image(tmp_path / "in.nii.gz", mov)
        mask_path = _write_image(tmp_path / "mask.nii.gz", mask)
        out_path = tmp_path / "out_masked.nii.gz"

        mri_main(["mask", str(in_path), str(mask_path), str(out_path)])

        masked = nib.load(str(out_path))
        expected = np.where(mask > 0, mov, 0)
        # input dtype is preserved by default
        assert masked.get_data_dtype() == np.dtype(np.uint8)
        assert np.asarray(masked.dataobj) == pytest.approx(expected)
        # same geometry -> no reslicing notice
        assert "geometry differs" not in capsys.readouterr().out

    def test_threshold_and_fill(self, tmp_path: Path):
        mov = np.full((2, 2, 2), 9.0, dtype=np.float32)
        mask = np.array([[[0, 1], [2, 3]], [[1, 2], [3, 0]]], dtype=np.float32)
        in_path = _write_image(tmp_path / "in.nii.gz", mov)
        mask_path = _write_image(tmp_path / "mask.nii.gz", mask)
        out_path = tmp_path / "out_thr.nii.gz"

        mri_main(["mask", str(in_path), str(mask_path), str(out_path), "--threshold", "1", "--oval", "-1"])

        masked = np.asarray(nib.load(str(out_path)).dataobj, dtype=np.float32)
        assert masked == pytest.approx(np.where(mask > 1, 9.0, -1.0))

    def test_fill_out_of_range_clamps_for_integer_dtype(self, tmp_path: Path):
        # fill=-1 with uint8 image: should clamp to 0, not wrap to 255.
        mov = np.full((2, 2, 2), 10, dtype=np.uint8)
        mask = np.zeros((2, 2, 2), dtype=np.uint8)
        mask[0, 0, 0] = 1  # keep only one voxel
        in_path = _write_image(tmp_path / "in.nii.gz", mov)
        mask_path = _write_image(tmp_path / "mask.nii.gz", mask)
        out_path = tmp_path / "out_clamp.nii.gz"

        mri_main(["mask", str(in_path), str(mask_path), str(out_path), "--oval", "-1"])

        result = np.asarray(nib.load(str(out_path)).dataobj)
        assert result[0, 0, 0] == 10  # kept voxel unchanged
        assert result[1, 1, 1] == 0  # fill=-1 clamped to uint8 min, not wrapped to 255

    def test_bool_image_clamps_fill_to_false(self):
        # Nifti1Image rejects bool dtype, so we use a minimal duck-typed stub
        # that satisfies the reslice_and_apply_mask + create_image_like interface.
        # fill=-1 must clamp to False, not stay True (non-zero -> True in plain cast).
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
        assert not result_data[1, 1, 1]  # fill=-1 clamped to 0 -> False

    def test_different_geometry_is_resampled(self, tmp_path: Path, capsys):
        mov = np.full((3, 3, 3), 5.0, dtype=np.float32)
        mask = np.ones((2, 2, 2), dtype=np.uint8)  # covers RAS voxels 0,1 only
        in_path = _write_image(tmp_path / "in.nii.gz", mov)
        mask_path = _write_image(tmp_path / "mask.nii.gz", mask)
        out_path = tmp_path / "out_diff.nii.gz"

        mri_main(["mask", str(in_path), str(mask_path), str(out_path)])

        masked = np.asarray(nib.load(str(out_path)).dataobj, dtype=np.float32)
        expected = np.zeros((3, 3, 3), dtype=np.float32)
        expected[:2, :2, :2] = 5.0
        assert masked == pytest.approx(expected)
        assert "geometry differs" in capsys.readouterr().out


class TestInfo:
    def test_full_dump_contains_key_fields(self, tmp_path: Path, capsys):
        data = np.zeros((4, 5, 6), dtype=np.int16)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        in_path = _write_image(tmp_path / "in.nii.gz", data, affine=affine)

        mri_main(["info", str(in_path)])

        out = capsys.readouterr().out
        assert "Volume information for" in out
        assert "dimensions: 4 x 5 x 6" in out
        assert "voxel sizes: 2.000000, 2.000000, 2.000000" in out
        assert "Orientation: RAS" in out
        assert "voxel to ras transform:" in out

    def test_dim_selector(self, tmp_path: Path, capsys):
        in_path = _write_image(tmp_path / "in.nii.gz", np.zeros((4, 5, 6), dtype=np.uint8))

        mri_main(["info", str(in_path), "--dim"])

        assert capsys.readouterr().out.strip() == "4 5 6"

    def test_res_and_type_selectors(self, tmp_path: Path, capsys):
        affine = np.diag([1.5, 2.0, 2.5, 1.0])
        in_path = _write_image(tmp_path / "in.nii.gz", np.zeros((2, 2, 2), dtype=np.uint8), affine=affine)

        mri_main(["info", str(in_path), "--res", "--type"])

        lines = capsys.readouterr().out.strip().splitlines()
        assert lines[0] == "1.500000 2.000000 2.500000"
        assert lines[1] == "uint8"

    def test_orientation_alias(self, tmp_path: Path, capsys):
        in_path = _write_image(tmp_path / "in.nii.gz", np.zeros((2, 2, 2), dtype=np.uint8))

        mri_main(["info", str(in_path), "--ori"])

        assert capsys.readouterr().out.strip() == "RAS"

    def test_nframes_for_4d(self, tmp_path: Path, capsys):
        in_path = _write_image(tmp_path / "in.nii.gz", np.zeros((2, 2, 2, 3), dtype=np.uint8))

        mri_main(["info", str(in_path), "--nframes"])

        assert capsys.readouterr().out.strip() == "3"

    def test_vox2ras_matrix_selector(self, tmp_path: Path, capsys):
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        in_path = _write_image(tmp_path / "in.nii.gz", np.zeros((2, 2, 2), dtype=np.uint8), affine=affine)

        mri_main(["info", str(in_path), "--vox2ras"])

        lines = capsys.readouterr().out.strip().splitlines()
        assert len(lines) == 4
        assert np.asarray([row.split() for row in lines], dtype=np.float64) == pytest.approx(affine)

    def test_stats_selector(self, tmp_path: Path, capsys):
        data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)  # 0..7, mean 3.5
        in_path = _write_image(tmp_path / "in.nii.gz", data)

        mri_main(["info", str(in_path), "--stats"])

        mn, mx, mean = capsys.readouterr().out.strip().split()
        assert (float(mn), float(mx), float(mean)) == pytest.approx((0.0, 7.0, 3.5))


def _run_diff(args: list[str]) -> int:
    with pytest.raises(SystemExit) as exc:
        mri_main(["diff", *args])
    return 0 if exc.value.code is None else int(exc.value.code)


class TestDiff:
    def test_identical_exits_zero(self, tmp_path: Path, capsys):
        data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        a = _write_image(tmp_path / "a.nii.gz", data)
        b = _write_image(tmp_path / "b.nii.gz", data)

        assert _run_diff([str(a), str(b)]) == 0
        assert "Volumes are the same" in capsys.readouterr().out

    def test_pixel_diff_exits_106(self, tmp_path: Path):
        a_data = np.zeros((2, 2, 2), dtype=np.float32)
        b_data = a_data.copy()
        b_data[0, 0, 0] = 9.0
        a = _write_image(tmp_path / "a.nii.gz", a_data)
        b = _write_image(tmp_path / "b.nii.gz", b_data)

        assert _run_diff([str(a), str(b)]) == 106

    def test_pixel_diff_under_thresh_is_same(self, tmp_path: Path):
        a_data = np.zeros((2, 2, 2), dtype=np.float32)
        b_data = a_data.copy()
        b_data[0, 0, 0] = 3.0
        a = _write_image(tmp_path / "a.nii.gz", a_data)
        b = _write_image(tmp_path / "b.nii.gz", b_data)

        assert _run_diff([str(a), str(b), "--thresh", "5"]) == 0

    def test_dimension_mismatch_exits_101(self, tmp_path: Path):
        a = _write_image(tmp_path / "a.nii.gz", np.zeros((2, 2, 2), dtype=np.float32))
        b = _write_image(tmp_path / "b.nii.gz", np.zeros((3, 3, 3), dtype=np.float32))

        assert _run_diff([str(a), str(b)]) == 101

    def test_resolution_diff_exits_102(self, tmp_path: Path):
        z = np.zeros((2, 2, 2), dtype=np.float32)
        a = _write_image(tmp_path / "a.nii.gz", z, affine=np.eye(4))
        b = _write_image(tmp_path / "b.nii.gz", z, affine=np.diag([2.0, 2.0, 2.0, 1.0]))

        assert _run_diff([str(a), str(b)]) == 102

    def test_geometry_diff_exits_104(self, tmp_path: Path):
        affine_b = np.eye(4)
        affine_b[0, 3] = 5.0  # same voxel sizes, shifted origin
        a = _write_image(tmp_path / "a.nii.gz", np.zeros((2, 2, 2), dtype=np.float32), affine=np.eye(4))
        b = _write_image(tmp_path / "b.nii.gz", np.zeros((2, 2, 2), dtype=np.float32), affine=affine_b)

        assert _run_diff([str(a), str(b)]) == 104

    def test_dtype_diff_exits_105(self, tmp_path: Path):
        data = np.zeros((2, 2, 2))
        a = _write_image(tmp_path / "a.nii.gz", data.astype(np.uint8))
        b = _write_image(tmp_path / "b.nii.gz", data.astype(np.int16))

        assert _run_diff([str(a), str(b)]) == 105

    def test_count_prints_diffcount(self, tmp_path: Path, capsys):
        a_data = np.zeros((2, 2, 2), dtype=np.float32)
        b_data = a_data.copy()
        b_data[0, 0, 0] = 9.0
        b_data[1, 1, 1] = 9.0
        a = _write_image(tmp_path / "a.nii.gz", a_data)
        b = _write_image(tmp_path / "b.nii.gz", b_data)

        code = _run_diff([str(a), str(b), "--count"])
        assert code == 106
        assert "diffcount 2" in capsys.readouterr().out

    def test_count_thresh_suppresses_small_diff(self, tmp_path: Path):
        a_data = np.zeros((2, 2, 2), dtype=np.float32)
        b_data = a_data.copy()
        b_data[0, 0, 0] = 9.0  # only 1 differing voxel
        a = _write_image(tmp_path / "a.nii.gz", a_data)
        b = _write_image(tmp_path / "b.nii.gz", b_data)

        assert _run_diff([str(a), str(b), "--count-thresh", "5"]) == 0
