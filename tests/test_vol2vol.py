import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neuroreg.cli.vol2vol import main as vol2vol_main
from neuroreg.transforms import LTA


def _write_image(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> Path:
    # Pass dtype explicitly so wide integer types (int64) round-trip instead of
    # tripping nibabel's "may cause incompatibilities" guard.
    img = nib.Nifti1Image(data, np.eye(4) if affine is None else affine, dtype=data.dtype)
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


def _cras(affine: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """World coordinate of a grid's centre, using MGH's ``shape / 2`` convention."""
    center_vox = np.array([shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0, 1.0], dtype=np.float64)
    return (np.asarray(affine, dtype=np.float64) @ center_vox)[:3]


def _oblique_affine() -> np.ndarray:
    """An anisotropic, oblique voxel-to-RAS affine with an off-centre origin."""
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


def _center_of_mass(data: np.ndarray) -> np.ndarray:
    weights = np.asarray(data, dtype=np.float64)
    grids = np.indices(weights.shape, dtype=np.float64)
    return np.array([float((g * weights).sum() / weights.sum()) for g in grids])


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

    @pytest.mark.parametrize("dtype", ["float64", "int64", "uint16"])
    def test_explicit_out_dtype_mgh_cannot_store_is_refused(self, tmp_path: Path, dtype: str, capsys):
        # Silently writing a different dtype than requested would hand back a
        # file that does not match the request, so the write is refused.
        mov_path = _write_image(tmp_path / "mov.nii.gz", np.ones((2, 2, 2), dtype=np.float32))
        out_path = tmp_path / "out.mgz"

        with pytest.raises(SystemExit):
            vol2vol_main(["--in", str(mov_path), "--out", str(out_path), "--out-dtype", dtype])

        assert "MGH/MGZ cannot store" in capsys.readouterr().err
        assert not out_path.exists()

    def test_keep_dtype_is_refused_when_mgh_cannot_store_the_input_dtype(self, tmp_path: Path, capsys):
        mov_path = _write_image(tmp_path / "mov.nii.gz", np.ones((2, 2, 2), dtype=np.float64))
        out_path = tmp_path / "out.mgz"

        with pytest.raises(SystemExit):
            vol2vol_main(["--in", str(mov_path), "--out", str(out_path), "--keep-dtype"])

        assert "MGH/MGZ cannot store" in capsys.readouterr().err

    @pytest.mark.parametrize("dtype", ["float64", "int64", "uint16"])
    def test_explicit_out_dtype_is_honoured_for_nifti(self, tmp_path: Path, dtype: str):
        mov_path = _write_image(tmp_path / "mov.nii.gz", np.ones((2, 2, 2), dtype=np.float32))
        out_path = tmp_path / f"out_{dtype}.nii.gz"

        vol2vol_main(["--in", str(mov_path), "--out", str(out_path), "--out-dtype", dtype])

        assert nib.load(str(out_path)).get_data_dtype() == np.dtype(dtype)

    def test_float64_input_to_mgz_narrows_to_float32_with_a_warning(self, tmp_path: Path, caplog):
        # MGH has no 64-bit float, so this conversion is allowed but must not be silent.
        data = np.array([[[1 / 3, 2 / 3]], [[0.1, 0.2]]], dtype=np.float64)
        mov_path = _write_image(tmp_path / "mov.nii.gz", data)
        out_path = tmp_path / "out.mgz"

        with caplog.at_level(logging.WARNING, logger="neuroreg.image.io"):
            vol2vol_main(["--in", str(mov_path), "--out", str(out_path)])

        assert nib.load(str(out_path)).get_data_dtype().newbyteorder("=") == np.dtype(np.float32)
        assert "Narrowing float64 to float32" in caplog.text

    def test_int64_beyond_int32_range_to_mgz_is_refused(self, tmp_path: Path, capsys):
        data = np.array([[[0, 2 ** 40 + 1]], [[5, 7]]], dtype=np.int64)
        mov_path = _write_image(tmp_path / "mov.nii.gz", data)
        out_path = tmp_path / "out.mgz"

        with pytest.raises(SystemExit):
            vol2vol_main(["--in", str(mov_path), "--out", str(out_path)])

        assert "exceeds int32" in capsys.readouterr().err
        assert not out_path.exists()

    def test_dtype_only_conversion_does_not_reslice(self, tmp_path: Path):
        data = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
        mov_path = _write_image(tmp_path / "mov.nii.gz", data)
        out_path = tmp_path / "out_short.nii.gz"

        vol2vol_main(["--in", str(mov_path), "--out", str(out_path), "--out-dtype", "int16"])

        mapped = nib.load(str(out_path))
        assert mapped.get_data_dtype() == np.dtype(np.int16)
        assert np.asarray(mapped.dataobj) == pytest.approx(data)

    @pytest.mark.parametrize("interp", ["linear", "cubic"])
    @pytest.mark.parametrize("out_ext", [".mgz", ".nii.gz"])
    def test_interpolated_output_stays_float_for_uint8_input(self, tmp_path: Path, interp: str, out_ext: str):
        # Interpolation produces fractional values, so the result must not be
        # quantized back into the uint8 input dtype unless the caller asks for
        # it via --keep-dtype/--out-dtype.
        mov = np.zeros((3, 3, 3), dtype=np.uint8)
        mov[1, 1, 1] = 255
        mov_path = _write_image(tmp_path / "mov.nii.gz", mov)
        ref_path = _write_image(tmp_path / "ref.nii.gz", np.zeros((3, 3, 3), dtype=np.float32))
        shift = np.eye(4)
        shift[0, 3] = 0.5
        lta_path = _write_lta(tmp_path / "shift.lta", shift, (3, 3, 3), (3, 3, 3))
        out_path = tmp_path / f"out{out_ext}"

        vol2vol_main(
            [
                "--in",
                str(mov_path),
                "--ref",
                str(ref_path),
                "--out",
                str(out_path),
                "--transform",
                str(lta_path),
                "--interp",
                interp,
            ]
        )

        mapped = nib.load(str(out_path))
        data = np.asarray(mapped.dataobj)
        assert mapped.get_data_dtype().newbyteorder("=") == np.dtype(np.float32)
        assert data.dtype.newbyteorder("=") == np.dtype(np.float32)
        # Half-voxel shift: interpolation must actually produce fractional values,
        # not just a float-typed copy of the integer input.
        assert np.any(data != np.rint(data))

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

    def test_transform_without_destination_geometry_falls_back_to_the_input_grid(self, tmp_path: Path):
        # An ITK affine carries no geometry of its own, so read without --ref
        # its LTA destination is marked invalid with a zero volume. The
        # RAS-to-RAS matrix is still usable, so the documented fallback to the
        # input grid applies rather than refusing the transform.
        data = np.zeros((6, 6, 6), dtype=np.uint8)
        data[1, 2, 3] = 200
        mov_affine = np.diag([1.0, 1.0, 1.0, 1.0])
        mov_affine[:3, 3] = [5.0, -7.0, 2.0]
        mov_path = _write_image(tmp_path / "mov.nii.gz", data, affine=mov_affine)
        itk_path = tmp_path / "shift.txt"
        itk_path.write_text(
            "#Insight Transform File V1.0\n"
            "#Transform 0\n"
            "Transform: MatrixOffsetTransformBase_double_3_3\n"
            "Parameters: 1 0 0 0 1 0 0 0 1 1 0 0\n"
            "FixedParameters: 0 0 0\n"
        )
        out_path = tmp_path / "out.nii.gz"

        vol2vol_main(
            [
                "--in", str(mov_path),
                "--transform", str(itk_path),
                "--transform-format", "itk",
                "--interp", "nearest",
                "--out", str(out_path),
            ]
        )

        mapped = nib.load(str(out_path))
        assert mapped.shape == (6, 6, 6)
        assert mapped.affine == pytest.approx(mov_affine)

    def test_transform_without_destination_geometry_composes_with_ref_cras(self, tmp_path: Path):
        mov_affine = np.diag([1.0, 1.0, 1.0, 1.0])
        mov_affine[:3, 3] = [5.0, -7.0, 2.0]
        mov_path = _write_image(tmp_path / "mov.nii.gz", np.ones((6, 6, 6), dtype=np.uint8), affine=mov_affine)
        itk_path = tmp_path / "shift.txt"
        itk_path.write_text(
            "#Insight Transform File V1.0\n"
            "#Transform 0\n"
            "Transform: MatrixOffsetTransformBase_double_3_3\n"
            "Parameters: 1 0 0 0 1 0 0 0 1 1 0 0\n"
            "FixedParameters: 0 0 0\n"
        )
        out_path = tmp_path / "out.nii.gz"

        vol2vol_main(
            [
                "--in", str(mov_path),
                "--transform", str(itk_path),
                "--transform-format", "itk",
                "--ref-cras", "0,0,0",
                "--out", str(out_path),
            ]
        )

        mapped = nib.load(str(out_path))
        assert mapped.shape == (6, 6, 6)
        assert _cras(mapped.affine, mapped.shape) == pytest.approx([0.0, 0.0, 0.0], abs=1e-4)

    def test_malformed_transform_geometry_does_not_defeat_an_explicit_ref(self, tmp_path: Path):
        # --ref supplies the whole target geometry and the RAS-to-RAS matrix
        # needs none of it, so unusable destination metadata must not be
        # inspected at all, let alone fail the run.
        from neuroreg.cli.vol2vol import _resolve_target_geometry

        class _ShortVoxelSizeLTA:
            dst = {
                "valid": 1,
                "volume": [4, 4, 4],
                "voxelsize": [1.0],  # too short: affine_from_volume_info raises IndexError
                "xras": [1.0, 0.0, 0.0],
                "yras": [0.0, 1.0, 0.0],
                "zras": [0.0, 0.0, 1.0],
                "cras": [0.0, 0.0, 0.0],
            }

        mov_img = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4), dtype=np.uint8)
        ref_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        ref_img = nib.Nifti1Image(np.zeros((7, 8, 9), dtype=np.uint8), ref_affine, dtype=np.uint8)

        affine, shape = _resolve_target_geometry(mov_img, ref_img, _ShortVoxelSizeLTA())

        assert shape == (7, 8, 9)
        assert affine == pytest.approx(ref_affine)

    @pytest.mark.parametrize(
        ("field", "bad"),
        [
            ("voxelsize", [1.0]),  # too short: affine_from_volume_info raises IndexError
            ("volume", [4, 4]),  # too short to be a 3-D shape
            ("volume", [0, 4, 4]),  # zero extent would ask for an empty grid
            ("volume", [4.7, 4, 4]),  # fractional voxel counts, previously truncated
            ("volume", [float("inf"), 4, 4]),  # previously raised OverflowError from int()
            # These three do not raise; they quietly poison the affine.
            ("voxelsize", [float("nan"), 1.0, 1.0]),
            ("cras", [float("inf"), 0.0, 0.0]),
            ("xras", [float("nan"), 0.0, 0.0]),
        ],
    )
    def test_unusable_transform_geometry_falls_back_to_the_input_grid(
        self, tmp_path: Path, field: str, bad: list
    ):
        from neuroreg.cli.vol2vol import _resolve_target_geometry

        class _BadLTA:
            dst = {
                "valid": 1,
                "volume": [4, 4, 4],
                "voxelsize": [1.0, 1.0, 1.0],
                "xras": [1.0, 0.0, 0.0],
                "yras": [0.0, 1.0, 0.0],
                "zras": [0.0, 0.0, 1.0],
                "cras": [0.0, 0.0, 0.0],
                field: bad,
            }

        mov_affine = np.diag([1.5, 1.5, 1.5, 1.0])
        mov_img = nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.uint8), mov_affine, dtype=np.uint8)

        affine, shape = _resolve_target_geometry(mov_img, None, _BadLTA())

        assert shape == (5, 5, 5)
        assert affine == pytest.approx(mov_affine)
        assert np.all(np.isfinite(affine))

    def test_usable_transform_geometry_is_still_used(self, tmp_path: Path):
        # The guards must not reject a healthy destination block.
        from neuroreg.cli.vol2vol import _resolve_target_geometry

        class _GoodLTA:
            dst = {
                "valid": 1,
                "volume": [7, 8, 9],
                "voxelsize": [2.0, 2.0, 2.0],
                "xras": [1.0, 0.0, 0.0],
                "yras": [0.0, 1.0, 0.0],
                "zras": [0.0, 0.0, 1.0],
                "cras": [0.0, 0.0, 0.0],
            }

        mov_img = nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.uint8), np.eye(4), dtype=np.uint8)

        affine, shape = _resolve_target_geometry(mov_img, None, _GoodLTA())

        assert shape == (7, 8, 9)
        assert np.linalg.norm(affine[:3, :3], axis=0) == pytest.approx([2.0, 2.0, 2.0])

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


class TestVol2VolRefCras:
    """``--ref-cras`` overrides where the target grid sits in world space."""

    def test_recentres_grid_on_requested_world_point(self, tmp_path: Path):
        # The motivating case: reslice into a standard-space pose while keeping
        # the reference dimensions and voxel size, which no stock file provides.
        ref_affine = np.diag([0.8, 0.8, 0.8, 1.0])
        ref_affine[:3, 3] = [-40.0, 17.0, -9.0]
        mov_path = _write_image(tmp_path / "mov.nii.gz", np.ones((8, 8, 8), dtype=np.float32))
        ref_path = _write_image(tmp_path / "ref.nii.gz", np.zeros((10, 10, 10), dtype=np.float32), affine=ref_affine)
        out_path = tmp_path / "out.nii.gz"

        vol2vol_main(
            ["--in", str(mov_path), "--ref", str(ref_path), "--ref-cras", "0,0,0", "--out", str(out_path)]
        )

        mapped = nib.load(str(out_path))
        assert _cras(mapped.affine, mapped.shape) == pytest.approx([0.0, 0.0, 0.0], abs=1e-4)

    def test_applies_without_a_ref_image(self, tmp_path: Path):
        # With no --transform and no --ref the base geometry is the input's own
        # grid, so --ref-cras alone must still take effect (and must not fall
        # into the no-resample fast path).
        affine = np.diag([1.5, 1.5, 1.5, 1.0])
        affine[:3, 3] = [3.0, -8.0, 21.0]
        mov_path = _write_image(tmp_path / "mov.nii.gz", np.ones((6, 6, 6), dtype=np.float32), affine=affine)
        out_path = tmp_path / "out.nii.gz"

        vol2vol_main(["--in", str(mov_path), "--ref-cras", "5,-5,2.5", "--out", str(out_path)])

        mapped = nib.load(str(out_path))
        assert mapped.shape == (6, 6, 6)
        assert _cras(mapped.affine, mapped.shape) == pytest.approx([5.0, -5.0, 2.5], abs=1e-4)

    def test_changes_only_the_translation_column(self, tmp_path: Path):
        ref_affine = _oblique_affine()
        mov_path = _write_image(tmp_path / "mov.nii.gz", np.ones((5, 5, 5), dtype=np.float32))
        ref_path = _write_image(tmp_path / "ref.nii.gz", np.zeros((7, 9, 11), dtype=np.float32), affine=ref_affine)
        out_path = tmp_path / "out.nii.gz"

        vol2vol_main(
            ["--in", str(mov_path), "--ref", str(ref_path), "--ref-cras", "1,2,3", "--out", str(out_path)]
        )

        mapped = nib.load(str(out_path))
        # Direction cosines and voxel sizes live in the 3x3 block; it must be
        # untouched. Compare against the reference as stored on disk, since a
        # NIfTI header keeps the affine in float32 and both files pay that cost.
        expected = np.asarray(nib.load(str(ref_path)).affine)[:3, :3]
        assert np.array_equal(np.asarray(mapped.affine)[:3, :3], expected)
        assert mapped.shape == (7, 9, 11)

    def test_handles_odd_anisotropic_oblique_grids(self, tmp_path: Path):
        # Odd dimensions are what distinguish shape/2 from (shape-1)/2, and
        # the oblique 3x3 block is what distinguishes the full-matrix formula
        # from a per-axis shortcut.
        ref_affine = _oblique_affine()
        shape = (15, 17, 9)
        mov_path = _write_image(tmp_path / "mov.nii.gz", np.ones((5, 5, 5), dtype=np.float32))
        ref_path = _write_image(tmp_path / "ref.nii.gz", np.zeros(shape, dtype=np.float32), affine=ref_affine)
        out_path = tmp_path / "out.nii.gz"

        vol2vol_main(
            ["--in", str(mov_path), "--ref", str(ref_path), "--ref-cras", "-4.25,6.5,0", "--out", str(out_path)]
        )

        mapped = nib.load(str(out_path))
        assert mapped.shape == shape
        assert _cras(mapped.affine, mapped.shape) == pytest.approx([-4.25, 6.5, 0.0], abs=1e-4)

    def test_setting_the_existing_cras_preserves_voxels(self, tmp_path: Path):
        # Asking for the centre the grid already has must be a no-op. A sign
        # error in the placement maths would displace the content instead.
        data = np.arange(6 * 6 * 6, dtype=np.uint8).reshape(6, 6, 6)
        affine = np.diag([1.25, 1.25, 1.25, 1.0])
        affine[:3, 3] = [-7.0, 13.0, 2.0]
        mov_path = _write_image(tmp_path / "mov.nii.gz", data, affine=affine)
        existing = _cras(affine, data.shape)
        out_path = tmp_path / "out.nii.gz"

        vol2vol_main(
            [
                "--in", str(mov_path),
                "--ref", str(mov_path),
                "--ref-cras", ",".join(repr(float(v)) for v in existing),
                "--interp", "nearest",
                "--keep-dtype",
                "--out", str(out_path),
            ]
        )

        mapped = nib.load(str(out_path))
        assert mapped.affine == pytest.approx(affine)
        assert np.array_equal(np.asarray(mapped.dataobj), data)

    def test_shifting_cras_moves_content_the_opposite_way_in_voxels(self, tmp_path: Path):
        # Content keeps its world position while the grid moves under it, so
        # moving the grid centre +4mm in x moves the anatomy -4 voxels in i.
        data = np.zeros((16, 16, 16), dtype=np.uint8)
        data[9:12, 7:10, 7:10] = 100
        mov_path = _write_image(tmp_path / "mov.nii.gz", data)
        before = _center_of_mass(data)
        shifted = _cras(np.eye(4), data.shape) + np.array([4.0, 0.0, 0.0])
        out_path = tmp_path / "out.nii.gz"

        vol2vol_main(
            [
                "--in", str(mov_path),
                "--ref-cras", ",".join(repr(float(v)) for v in shifted),
                "--interp", "nearest",
                "--out", str(out_path),
            ]
        )

        after = _center_of_mass(np.asarray(nib.load(str(out_path)).dataobj))
        assert after == pytest.approx(before + np.array([-4.0, 0.0, 0.0]), abs=1e-4)

    def test_composes_with_inverse(self, tmp_path: Path):
        # --inverse swaps which geometry is the target; the override applies to
        # the post-inversion target, so the src grid is what gets recentred.
        src_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        src_affine[:3, 3] = [10.0, 20.0, 30.0]
        dst_affine = np.diag([1.0, 1.5, 2.0, 1.0])
        mov_path = _write_image(tmp_path / "mov.nii.gz", np.ones((3, 3, 3), dtype=np.float32), affine=src_affine)
        ref_path = _write_image(tmp_path / "ref.nii.gz", np.zeros((5, 5, 5), dtype=np.float32), affine=dst_affine)
        lta_path = tmp_path / "geom.lta"
        LTA.from_matrix(np.eye(4), str(mov_path), str(mov_path), str(ref_path), str(ref_path)).write(lta_path)
        out_path = tmp_path / "out.nii.gz"

        vol2vol_main(
            [
                "--in", str(mov_path),
                "--transform", str(lta_path),
                "--inverse",
                "--ref-cras", "0,0,0",
                "--out", str(out_path),
            ]
        )

        mapped = nib.load(str(out_path))
        assert mapped.shape == (3, 3, 3)
        assert np.array_equal(np.asarray(mapped.affine)[:3, :3], src_affine[:3, :3])
        assert _cras(mapped.affine, mapped.shape) == pytest.approx([0.0, 0.0, 0.0], abs=1e-4)

    def test_composes_with_keep_dtype(self, tmp_path: Path):
        data = np.arange(4 * 4 * 4, dtype=np.uint8).reshape(4, 4, 4)
        mov_path = _write_image(tmp_path / "mov.nii.gz", data)
        out_path = tmp_path / "out.nii.gz"

        vol2vol_main(
            [
                "--in", str(mov_path),
                "--ref-cras", "0,0,0",
                "--interp", "nearest",
                "--keep-dtype",
                "--out", str(out_path),
            ]
        )

        mapped = nib.load(str(out_path))
        assert mapped.get_data_dtype() == np.dtype(np.uint8)
        assert _cras(mapped.affine, mapped.shape) == pytest.approx([0.0, 0.0, 0.0], abs=1e-4)

    @pytest.mark.parametrize("out_ext", [".nii.gz", ".mgz"])
    def test_written_header_describes_the_moved_grid(self, tmp_path: Path, out_ext: str):
        # NIfTI has no c_ras field, but the concept is just the affine
        # translation, so the round-tripped header must still agree.
        affine = np.diag([0.8, 0.8, 0.8, 1.0])
        affine[:3, 3] = [-0.115, 23.431, -25.661]
        mov_path = _write_image(tmp_path / "mov.nii.gz", np.ones((8, 8, 8), dtype=np.float32), affine=affine)
        out_path = tmp_path / f"out{out_ext}"

        vol2vol_main(["--in", str(mov_path), "--ref", str(mov_path), "--ref-cras", "0,0,0", "--out", str(out_path)])

        mapped = nib.load(str(out_path))
        assert _cras(mapped.affine, mapped.shape) == pytest.approx([0.0, 0.0, 0.0], abs=1e-4)
        assert np.asarray(mapped.affine)[:3, :3] == pytest.approx(affine[:3, :3], abs=1e-5)

    def test_rejected_with_header_only(self, tmp_path: Path, capsys):
        mov_path = _write_image(tmp_path / "mov.nii.gz", np.ones((2, 2, 2), dtype=np.float32))

        with pytest.raises(SystemExit):
            vol2vol_main(
                [
                    "--in", str(mov_path),
                    "--header-only",
                    "--ref-cras", "0,0,0",
                    "--out", str(tmp_path / "out.nii.gz"),
                ]
            )

        assert "--header-only cannot be combined with --ref-cras" in capsys.readouterr().err

    @pytest.mark.parametrize("value", ["1,2", "1,2,3,4", "a,b,c", "0,0,x", "", "1 2 3", "nan,0,0"])
    def test_rejects_malformed_values(self, tmp_path: Path, capsys, value: str):
        mov_path = _write_image(tmp_path / "mov.nii.gz", np.ones((2, 2, 2), dtype=np.float32))

        with pytest.raises(SystemExit):
            vol2vol_main(
                ["--in", str(mov_path), "--ref-cras", value, "--out", str(tmp_path / "out.nii.gz")]
            )

        assert "--ref-cras" in capsys.readouterr().err

    @pytest.mark.parametrize("spelling", ["bare", "equals"])
    def test_accepts_negative_coordinates(self, tmp_path: Path, spelling: str):
        # A scanner c_ras is usually negative in at least one axis, so the bare
        # form must work and not be mistaken for an option by argparse. This
        # also pins behaviour that depends on an argparse implementation detail.
        mov_path = _write_image(tmp_path / "mov.nii.gz", np.ones((4, 4, 4), dtype=np.float32))
        out_path = tmp_path / "out.nii.gz"
        value = "-4.25,-6.5,-0.75"
        flag = ["--ref-cras", value] if spelling == "bare" else [f"--ref-cras={value}"]

        vol2vol_main(["--in", str(mov_path), *flag, "--out", str(out_path)])

        mapped = nib.load(str(out_path))
        assert _cras(mapped.affine, mapped.shape) == pytest.approx([-4.25, -6.5, -0.75], abs=1e-4)

    def test_accepts_whitespace_around_components(self, tmp_path: Path):
        mov_path = _write_image(tmp_path / "mov.nii.gz", np.ones((4, 4, 4), dtype=np.float32))
        out_path = tmp_path / "out.nii.gz"

        vol2vol_main(["--in", str(mov_path), "--ref-cras", " 1.5 , -2 , 0 ", "--out", str(out_path)])

        mapped = nib.load(str(out_path))
        assert _cras(mapped.affine, mapped.shape) == pytest.approx([1.5, -2.0, 0.0], abs=1e-4)
