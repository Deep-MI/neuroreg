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

    def test_nan_on_one_side_counts_as_difference(self, tmp_path: Path, capsys):
        a_data = np.zeros((2, 2, 2), dtype=np.float32)
        b_data = a_data.copy()
        b_data[0, 0, 0] = float("nan")  # one side NaN, other finite → should differ
        a = _write_image(tmp_path / "a.nii.gz", a_data)
        b = _write_image(tmp_path / "b.nii.gz", b_data)

        code = _run_diff([str(a), str(b), "--count"])
        assert code == 106
        assert "diffcount 1" in capsys.readouterr().out

    def test_nan_on_both_sides_treated_as_same(self, tmp_path: Path):
        data = np.full((2, 2, 2), float("nan"), dtype=np.float32)
        a = _write_image(tmp_path / "a.nii.gz", data)
        b = _write_image(tmp_path / "b.nii.gz", data)

        assert _run_diff([str(a), str(b)]) == 0

    def test_count_thresh_suppresses_small_diff(self, tmp_path: Path):
        a_data = np.zeros((2, 2, 2), dtype=np.float32)
        b_data = a_data.copy()
        b_data[0, 0, 0] = 9.0  # only 1 differing voxel
        a = _write_image(tmp_path / "a.nii.gz", a_data)
        b = _write_image(tmp_path / "b.nii.gz", b_data)

        assert _run_diff([str(a), str(b), "--count-thresh", "5"]) == 0

    def test_skip_res_ignores_resolution_diff(self, tmp_path: Path):
        # Changing voxel sizes also changes geometry, so skipping res alone advances
        # past the 102 check but hits the 104 geometry check — not 102.
        z = np.zeros((2, 2, 2), dtype=np.float32)
        a = _write_image(tmp_path / "a.nii.gz", z, affine=np.eye(4))
        b = _write_image(tmp_path / "b.nii.gz", z, affine=np.diag([2.0, 2.0, 2.0, 1.0]))

        assert _run_diff([str(a), str(b), "--skip-res"]) == 104

    def test_notallow_res_is_alias_for_skip_res(self, tmp_path: Path):
        z = np.zeros((2, 2, 2), dtype=np.float32)
        a = _write_image(tmp_path / "a.nii.gz", z, affine=np.eye(4))
        b = _write_image(tmp_path / "b.nii.gz", z, affine=np.diag([2.0, 2.0, 2.0, 1.0]))

        assert _run_diff([str(a), str(b), "--notallow-res"]) == 104

    def test_skip_geo_ignores_geometry_diff(self, tmp_path: Path):
        affine_b = np.eye(4)
        affine_b[0, 3] = 5.0
        a = _write_image(tmp_path / "a.nii.gz", np.zeros((2, 2, 2), dtype=np.float32), affine=np.eye(4))
        b = _write_image(tmp_path / "b.nii.gz", np.zeros((2, 2, 2), dtype=np.float32), affine=affine_b)

        assert _run_diff([str(a), str(b), "--skip-geo"]) == 0

    def test_skip_prec_ignores_dtype_diff(self, tmp_path: Path):
        data = np.zeros((2, 2, 2))
        a = _write_image(tmp_path / "a.nii.gz", data.astype(np.uint8))
        b = _write_image(tmp_path / "b.nii.gz", data.astype(np.int16))

        assert _run_diff([str(a), str(b), "--skip-prec"]) == 0

    def test_skip_pix_ignores_pixel_diff(self, tmp_path: Path, capsys):
        a_data = np.zeros((2, 2, 2), dtype=np.float32)
        b_data = a_data.copy()
        b_data[0, 0, 0] = 99.0
        a = _write_image(tmp_path / "a.nii.gz", a_data)
        b = _write_image(tmp_path / "b.nii.gz", b_data)

        assert _run_diff([str(a), str(b), "--skip-pix"]) == 0

    def test_notallow_pix_notallow_geo_fastsurfer_pattern(self, tmp_path: Path):
        # Mirrors the FastSurfer long_prepare_template.sh usage:
        # mri_diff --notallow-pix --notallow-geo vol1 vol2 --res-thresh 0.000001
        # Checks only resolution (tight threshold) — pixels and geometry are skipped.
        affine_b = np.eye(4)
        affine_b[0, 3] = 5.0  # geometry differs
        a_data = np.zeros((2, 2, 2), dtype=np.float32)
        b_data = a_data.copy()
        b_data[0, 0, 0] = 99.0  # pixels differ
        a = _write_image(tmp_path / "a.nii.gz", a_data, affine=np.eye(4))
        b = _write_image(tmp_path / "b.nii.gz", b_data, affine=affine_b)

        assert _run_diff([str(a), str(b), "--notallow-pix", "--notallow-geo", "--res-thresh", "0.000001"]) == 0


class TestBinarize:
    def test_min_max_range_default_int32(self, tmp_path: Path):
        data = np.arange(27, dtype=np.int16).reshape(3, 3, 3)
        in_path = _write_image(tmp_path / "in.nii.gz", data)
        out_path = tmp_path / "out.nii.gz"

        mri_main(["binarize", "--i", str(in_path), "--o", str(out_path), "--min", "5", "--max", "10"])

        out = nib.load(str(out_path))
        expected = ((data >= 5) & (data <= 10)).astype(np.int32)
        assert out.get_data_dtype() == np.dtype(np.int32)
        assert np.asarray(out.dataobj) == pytest.approx(expected)

    def test_match_labels(self, tmp_path: Path):
        data = np.array([[[0, 2], [5, 3]], [[2, 7], [5, 0]]], dtype=np.int16)
        in_path = _write_image(tmp_path / "in.nii.gz", data)
        out_path = tmp_path / "out.nii.gz"

        mri_main(["binarize", "--i", str(in_path), "--o", str(out_path), "--match", "2", "5"])

        out = np.asarray(nib.load(str(out_path)).dataobj)
        assert out == pytest.approx(np.isin(data, [2, 5]).astype(np.int32))

    def test_inv_swaps_values(self, tmp_path: Path):
        data = np.arange(8, dtype=np.int16).reshape(2, 2, 2)
        in_path = _write_image(tmp_path / "in.nii.gz", data)
        out_path = tmp_path / "out.nii.gz"

        mri_main(["binarize", "--i", str(in_path), "--o", str(out_path), "--min", "4", "--inv"])

        out = np.asarray(nib.load(str(out_path)).dataobj)
        # default 1/0 swapped: selected (>=4) -> 0, rest -> 1
        assert out == pytest.approx(np.where(data >= 4, 0, 1))

    def test_custom_binval_binvalnot(self, tmp_path: Path):
        data = np.array([[[3, 1], [3, 0]], [[2, 3], [1, 3]]], dtype=np.int16)
        in_path = _write_image(tmp_path / "in.nii.gz", data)
        out_path = tmp_path / "out.nii.gz"

        mri_main(
            ["binarize", "--i", str(in_path), "--o", str(out_path),
             "--match", "3", "--binval", "7", "--binvalnot", "2"]
        )

        out = np.asarray(nib.load(str(out_path)).dataobj)
        assert out == pytest.approx(np.where(data == 3, 7, 2))

    def test_uchar_output(self, tmp_path: Path):
        data = np.arange(8, dtype=np.int16).reshape(2, 2, 2)
        in_path = _write_image(tmp_path / "in.nii.gz", data)
        out_path = tmp_path / "out.nii.gz"

        mri_main(["binarize", "--i", str(in_path), "--o", str(out_path), "--min", "4", "--uchar"])

        assert nib.load(str(out_path)).get_data_dtype() == np.dtype(np.uint8)

    def test_in_out_aliases(self, tmp_path: Path):
        data = np.arange(8, dtype=np.int16).reshape(2, 2, 2)
        in_path = _write_image(tmp_path / "in.nii.gz", data)
        out_path = tmp_path / "out.nii.gz"

        # the FreeSurfer-style --i/--o and our --in/--out must be equivalent
        mri_main(["binarize", "--in", str(in_path), "--out", str(out_path), "--min", "4"])

        out = np.asarray(nib.load(str(out_path)).dataobj)
        assert out == pytest.approx((data >= 4).astype(np.int32))

    def test_abs_before_threshold(self, tmp_path: Path):
        data = np.array([[[-9, -1], [2, 0]], [[-5, 1], [9, -2]]], dtype=np.int16)
        in_path = _write_image(tmp_path / "in.nii.gz", data)
        out_path = tmp_path / "out.nii.gz"

        mri_main(["binarize", "--i", str(in_path), "--o", str(out_path), "--min", "5", "--abs"])

        out = np.asarray(nib.load(str(out_path)).dataobj)
        assert out == pytest.approx((np.abs(data) >= 5).astype(np.int32))

    def test_frame_selects_single_frame(self, tmp_path: Path):
        data = np.zeros((2, 2, 2, 3), dtype=np.int16)
        data[..., 1] = 9  # only frame 1 is nonzero
        in_path = _write_image(tmp_path / "in.nii.gz", data)
        out_path = tmp_path / "out.nii.gz"

        mri_main(["binarize", "--i", str(in_path), "--o", str(out_path), "--min", "1", "--frame", "1"])

        out = np.asarray(nib.load(str(out_path)).dataobj)
        assert out.shape == (2, 2, 2)
        assert np.all(out == 1)

    def test_requires_a_selection_criterion(self, tmp_path: Path):
        in_path = _write_image(tmp_path / "in.nii.gz", np.zeros((2, 2, 2), dtype=np.int16))
        with pytest.raises(SystemExit):
            mri_main(["binarize", "--i", str(in_path), "--o", str(tmp_path / "o.nii.gz")])


def _cras(affine: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    center_vox = np.array([shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0, 1.0], dtype=np.float64)
    return (np.asarray(affine, dtype=np.float64) @ center_vox)[:3]


def _oblique_affine() -> np.ndarray:
    rot = np.array(
        [
            [0.9362934, -0.2896295, 0.1986693],
            [0.3129918, 0.9447025, -0.0978434],
            [-0.1593451, 0.1537920, 0.9751703],
        ]
    )
    affine = np.eye(4)
    affine[:3, :3] = rot @ np.diag([0.9, 1.3, 2.1])
    affine[:3, 3] = [11.0, -23.0, 7.0]
    return affine


class TestGeom:
    """``mri geom`` writes a volume carrying only a target geometry."""

    def test_writes_the_mni305_frame_at_a_native_voxel_size(self, tmp_path: Path):
        # The motivating case: the mni305.cor.mgz frame (256mm FOV, LIA,
        # cras 0) at 0.8mm, which no stock file provides.
        out_path = tmp_path / "target.mgz"

        mri_main(
            [
                "geom",
                "--out", str(out_path),
                "--fov", "256",
                "--vox-size", "0.8",
                "--orientation", "LIA",
                "--cras", "0,0,0",
            ]
        )

        written = nib.load(str(out_path))
        assert written.shape == (320, 320, 320)
        assert written.header.get_zooms()[:3] == pytest.approx((0.8, 0.8, 0.8))
        assert "".join(nib.aff2axcodes(written.affine)) == "LIA"
        # MGH stores Pxyz_c as float32, so the round-trip is exact to ~1e-5mm.
        assert _cras(written.affine, written.shape) == pytest.approx([0.0, 0.0, 0.0], abs=1e-4)

    def test_fov_and_voxel_size_give_the_expected_matrix_size(self, tmp_path: Path):
        # 256 / 0.8 happens to be exactly 320.0 in float64; the epsilon that
        # guards counts landing just above an integer is covered in
        # tests/test_geometry.py, which can construct that case directly.
        out_path = tmp_path / "target.mgz"

        mri_main(["geom", "--out", str(out_path), "--fov", "256", "--vox-size", "0.8", "--orientation", "LIA"])

        assert nib.load(str(out_path)).shape == (320, 320, 320)

    def test_cras_defaults_to_the_world_origin(self, tmp_path: Path):
        out_path = tmp_path / "target.nii.gz"

        mri_main(["geom", "--out", str(out_path), "--shape", "16", "--vox-size", "1", "--orientation", "RAS"])

        written = nib.load(str(out_path))
        assert _cras(written.affine, written.shape) == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)

    def test_like_reproduces_the_source_geometry(self, tmp_path: Path):
        source_affine = _oblique_affine()
        source = _write_image(tmp_path / "src.nii.gz", np.zeros((15, 17, 9), dtype=np.uint8), affine=source_affine)
        out_path = tmp_path / "copy.nii.gz"

        mri_main(["geom", "--out", str(out_path), "--like", str(source)])

        written = nib.load(str(out_path))
        assert written.shape == (15, 17, 9)
        assert np.asarray(written.affine) == pytest.approx(np.asarray(nib.load(str(source)).affine), abs=1e-5)

    def test_like_plus_cras_moves_only_the_placement(self, tmp_path: Path):
        source_affine = _oblique_affine()
        source = _write_image(tmp_path / "src.nii.gz", np.zeros((15, 17, 9), dtype=np.uint8), affine=source_affine)
        out_path = tmp_path / "moved.nii.gz"

        mri_main(["geom", "--out", str(out_path), "--like", str(source), "--cras", "0,0,0"])

        written = nib.load(str(out_path))
        source_block = np.asarray(nib.load(str(source)).affine)[:3, :3]
        assert written.shape == (15, 17, 9)
        assert np.asarray(written.affine)[:3, :3] == pytest.approx(source_block, abs=1e-5)
        assert _cras(written.affine, written.shape) == pytest.approx([0.0, 0.0, 0.0], abs=1e-4)

    def test_like_plus_orientation_replaces_oblique_cosines(self, tmp_path: Path):
        # --orientation is strict, so it discards the source's oblique rotation.
        source = _write_image(tmp_path / "src.nii.gz", np.zeros((8, 8, 8), dtype=np.uint8), affine=_oblique_affine())
        out_path = tmp_path / "straight.nii.gz"

        mri_main(["geom", "--out", str(out_path), "--like", str(source), "--orientation", "LIA"])

        written = nib.load(str(out_path))
        assert "".join(nib.aff2axcodes(written.affine)) == "LIA"
        # Voxel sizes still come from --like.
        assert written.header.get_zooms()[:3] == pytest.approx((0.9, 1.3, 2.1), abs=1e-5)

    def test_like_plus_voxel_size_keeps_the_extent_and_rescales_the_dimensions(self, tmp_path: Path):
        # --like preserves what the source covers. Keeping its dimensions at a
        # finer voxel size would silently crop the anatomy to half the FOV.
        source = _write_image(tmp_path / "src.nii.gz", np.zeros((16, 16, 16), dtype=np.uint8))
        out_path = tmp_path / "finer.nii.gz"

        mri_main(["geom", "--out", str(out_path), "--like", str(source), "--vox-size", "0.5"])

        written = nib.load(str(out_path))
        assert written.shape == (32, 32, 32)
        assert written.header.get_zooms()[:3] == pytest.approx((0.5, 0.5, 0.5))
        # 32 * 0.5 == 16 * 1.0: the same 16mm extent per axis.
        assert np.asarray(written.shape) * np.asarray(written.header.get_zooms()[:3]) == pytest.approx(
            [16.0, 16.0, 16.0]
        )

    def test_like_plus_coarser_voxel_size_shrinks_the_matrix(self, tmp_path: Path):
        source = _write_image(tmp_path / "src.nii.gz", np.zeros((32, 32, 32), dtype=np.uint8))
        out_path = tmp_path / "coarser.nii.gz"

        mri_main(["geom", "--out", str(out_path), "--like", str(source), "--vox-size", "2"])

        assert nib.load(str(out_path)).shape == (16, 16, 16)

    @pytest.mark.parametrize(
        ("shape", "zooms"),
        [
            ((16, 16, 16), (1.0, 1.0, 1.0)),
            ((320, 320, 320), (float(np.float32(0.8)),) * 3),  # a zoom as stored in a float32 header
            ((15, 17, 9), (0.9, 1.3, 2.1)),  # odd and anisotropic
        ],
    )
    def test_like_plus_the_sources_own_voxel_size_is_a_no_op(
            self, tmp_path: Path, shape: tuple[int, int, int], zooms: tuple[float, float, float]
    ):
        # Deriving the dimensions from the extent must round-trip: passing the
        # voxel size the source already has cannot change its dimensions.
        source = _write_image(
            tmp_path / "src.nii.gz", np.zeros(shape, dtype=np.uint8), affine=np.diag([*zooms, 1.0])
        )
        out_path = tmp_path / "same.nii.gz"

        mri_main(["geom", "--out", str(out_path), "--like", str(source), "--vox-size", ",".join(map(repr, zooms))])

        assert nib.load(str(out_path)).shape == shape

    def test_like_plus_shape_fixes_the_dimensions_instead(self, tmp_path: Path):
        # The escape hatch for wanting the source dimensions at a new voxel size.
        source = _write_image(tmp_path / "src.nii.gz", np.zeros((16, 16, 16), dtype=np.uint8))
        out_path = tmp_path / "cropped.nii.gz"

        mri_main(
            ["geom", "--out", str(out_path), "--like", str(source), "--vox-size", "0.5", "--shape", "16"]
        )

        assert nib.load(str(out_path)).shape == (16, 16, 16)

    def test_accepts_negative_cras_in_the_bare_form(self, tmp_path: Path):
        out_path = tmp_path / "target.nii.gz"

        mri_main(
            [
                "geom",
                "--out", str(out_path),
                "--shape", "8",
                "--vox-size", "1",
                "--orientation", "LIA",
                "--cras", "-4.5,3,-2",
            ]
        )

        written = nib.load(str(out_path))
        assert _cras(written.affine, written.shape) == pytest.approx([-4.5, 3.0, -2.0], abs=1e-5)

    def test_anisotropic_shape_and_voxel_size(self, tmp_path: Path):
        out_path = tmp_path / "target.nii.gz"

        mri_main(
            [
                "geom",
                "--out", str(out_path),
                "--shape", "8,9,10",
                "--vox-size", "0.9,1.3,2.1",
                "--orientation", "PSR",
            ]
        )

        written = nib.load(str(out_path))
        assert written.shape == (8, 9, 10)
        assert written.header.get_zooms()[:3] == pytest.approx((0.9, 1.3, 2.1))
        assert "".join(nib.aff2axcodes(written.affine)) == "PSR"

    def test_output_is_empty_uint8(self, tmp_path: Path):
        out_path = tmp_path / "target.mgz"

        mri_main(["geom", "--out", str(out_path), "--shape", "8", "--vox-size", "1", "--orientation", "LIA"])

        written = nib.load(str(out_path))
        assert written.get_data_dtype() == np.dtype(np.uint8)
        assert not np.any(np.asanyarray(written.dataobj))

    def test_prints_the_geometry_as_stored(self, tmp_path: Path, capsys):
        out_path = tmp_path / "target.mgz"

        mri_main(["geom", "--out", str(out_path), "--fov", "256", "--vox-size", "0.8", "--orientation", "LIA"])

        out = capsys.readouterr().out
        assert "dimensions: 320 x 320 x 320" in out
        assert "voxel sizes: 0.800000, 0.800000, 0.800000" in out
        assert "Orientation: LIA" in out

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            (["--shape", "8", "--orientation", "LIA"], "--vox-size is required"),
            (["--vox-size", "1", "--orientation", "LIA"], "--shape or --fov is required"),
            (["--shape", "8", "--vox-size", "1"], "--orientation is required"),
        ],
    )
    def test_missing_components_without_like_are_usage_errors(
            self, tmp_path: Path, capsys, args: list[str], message: str
    ):
        with pytest.raises(SystemExit):
            mri_main(["geom", "--out", str(tmp_path / "o.nii.gz"), *args])

        assert message in capsys.readouterr().err

    def test_shape_and_fov_are_mutually_exclusive(self, tmp_path: Path, capsys):
        with pytest.raises(SystemExit):
            mri_main(
                [
                    "geom",
                    "--out", str(tmp_path / "o.nii.gz"),
                    "--shape", "8",
                    "--fov", "256",
                    "--vox-size", "1",
                    "--orientation", "LIA",
                ]
            )

        assert "not allowed with argument" in capsys.readouterr().err

    @pytest.mark.parametrize("code", ["LI", "XYZ", "LRA"])
    def test_invalid_orientation_is_a_usage_error(self, tmp_path: Path, capsys, code: str):
        with pytest.raises(SystemExit):
            mri_main(
                [
                    "geom",
                    "--out", str(tmp_path / "o.nii.gz"),
                    "--shape", "8",
                    "--vox-size", "1",
                    "--orientation", code,
                ]
            )

        assert "Orientation must" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("flag", "value"),
        [("--vox-size", "0"), ("--vox-size", "-1"), ("--fov", "0"), ("--shape", "0"), ("--shape", "-8")],
    )
    def test_non_positive_sizes_are_rejected(self, tmp_path: Path, capsys, flag: str, value: str):
        # Keep the invalid value the only occurrence of its flag. argparse does
        # reject at the first occurrence, so a trailing valid duplicate would
        # still fail the run, but it reads as though it might mask the value.
        args = ["geom", "--out", str(tmp_path / "o.nii.gz"), "--orientation", "LIA", flag, value]
        if flag != "--vox-size":
            args += ["--vox-size", "1"]
        if flag not in ("--shape", "--fov"):
            args += ["--shape", "8"]

        with pytest.raises(SystemExit):
            mri_main(args)

        assert f"{flag} must be positive" in capsys.readouterr().err

    def test_integer_beyond_int64_is_a_usage_error(self, tmp_path: Path, capsys):
        # Python integers are unbounded, so int() accepts 2**63 and the overflow
        # only surfaces at array conversion. argparse does not translate
        # OverflowError, so it has to be caught before it escapes as a traceback.
        with pytest.raises(SystemExit):
            mri_main(
                [
                    "geom",
                    "--out", str(tmp_path / "o.mgz"),
                    "--shape", str(2 ** 63),
                    "--vox-size", "1",
                    "--orientation", "LIA",
                ]
            )

        err = capsys.readouterr().err
        assert "Traceback" not in err
        assert "--shape is out of range" in err

    @pytest.mark.parametrize("flag", ["--vox-size", "--fov", "--cras"])
    def test_non_finite_floats_are_usage_errors(self, tmp_path: Path, capsys, flag: str):
        # float() accepts these; only the finiteness check rejects them, and an
        # inf or nan would otherwise poison the whole output affine.
        args = ["geom", "--out", str(tmp_path / "o.mgz"), "--orientation", "LIA"]
        if flag != "--vox-size":
            args += ["--vox-size", "1"]
        if flag != "--fov":
            args += ["--shape", "8"]
        args += [flag, "1e400" if flag != "--cras" else "1e400,0,0"]

        with pytest.raises(SystemExit):
            mri_main(args)

        err = capsys.readouterr().err
        assert "Traceback" not in err
        assert f"{flag} must be finite" in err

    def test_unreadable_like_reports_like_the_other_subcommands(self, tmp_path: Path, capsys):
        # Not a usage error, so it must surface as "ERROR: ..." and exit 1
        # rather than a Python traceback.
        with pytest.raises(SystemExit):
            mri_main(["geom", "--out", str(tmp_path / "o.mgz"), "--like", str(tmp_path / "missing.mgz")])

        err = capsys.readouterr().err
        assert err.startswith("ERROR:")
        assert "Traceback" not in err

    def test_unsupported_output_extension_is_rejected(self, tmp_path: Path, capsys):
        with pytest.raises(SystemExit):
            mri_main(
                ["geom", "--out", str(tmp_path / "o.txt"), "--shape", "8", "--vox-size", "1", "--orientation", "LIA"]
            )

        assert "recognized image extension" in capsys.readouterr().err

    def test_output_makes_a_geometry_less_transform_usable(self, tmp_path: Path):
        # An FSL .mat carries no geometry, so vol2vol cannot interpret it
        # without a destination *image*. Component flags cannot supply that;
        # a materialised grid can. This is the capability geom exists for.
        from neuroreg.cli.vol2vol import main as vol2vol_main

        mov = _write_image(
            tmp_path / "mov.nii.gz", np.ones((8, 8, 8), dtype=np.uint8), affine=np.diag([0.8, 0.8, 0.8, 1.0])
        )
        mat = tmp_path / "x.mat"
        mat.write_text("\n".join("  ".join(f"{v:.8f}" for v in row) for row in np.eye(4)) + "\n")

        with pytest.raises(SystemExit):
            vol2vol_main(
                [
                    "--in", str(mov),
                    "--transform", str(mat),
                    "--transform-format", "fsl",
                    "--out", str(tmp_path / "no_ref.mgz"),
                ]
            )

        ref = tmp_path / "ref.mgz"
        out_path = tmp_path / "with_ref.mgz"
        mri_main(["geom", "--out", str(ref), "--fov", "8", "--vox-size", "0.8", "--orientation", "LIA"])
        vol2vol_main(
            [
                "--in", str(mov),
                "--transform", str(mat),
                "--transform-format", "fsl",
                "--ref", str(ref),
                "--out", str(out_path),
            ]
        )

        assert nib.load(str(out_path)).shape == nib.load(str(ref)).shape

    def test_output_is_usable_as_a_vol2vol_reference(self, tmp_path: Path):
        # The whole point: a grid no stock file provides, consumed by a tool
        # that only accepts a reference image.
        from neuroreg.cli.vol2vol import main as vol2vol_main

        mov_affine = np.diag([0.8, 0.8, 0.8, 1.0])
        mov_affine[:3, 3] = [-0.115, 23.431, -25.661]
        mov = _write_image(tmp_path / "mov.nii.gz", np.ones((8, 8, 8), dtype=np.uint8), affine=mov_affine)
        ref = tmp_path / "ref.mgz"
        out_path = tmp_path / "mapped.mgz"

        mri_main(["geom", "--out", str(ref), "--fov", "8", "--vox-size", "0.8", "--orientation", "LIA"])
        vol2vol_main(["--in", str(mov), "--ref", str(ref), "--out", str(out_path), "--interp", "nearest"])

        mapped = nib.load(str(out_path))
        reference = nib.load(str(ref))
        assert mapped.shape == reference.shape
        assert np.asarray(mapped.affine) == pytest.approx(np.asarray(reference.affine), abs=1e-4)
