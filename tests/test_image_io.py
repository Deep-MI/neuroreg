"""Tests for neuroreg.image.io: format-aware save/load and MGH fov metadata."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neuroreg.image import as_mgh_image, save_image


def test_as_mgh_image_sets_fov_to_largest_physical_extent():
    shape = (100, 50, 180)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    data = np.zeros(shape, dtype=np.float32)

    image = as_mgh_image(data, affine)

    assert float(image.header["fov"]) == pytest.approx(180.0)


def test_as_mgh_image_recomputes_fov_instead_of_inheriting_a_stale_value():
    stale_header = as_mgh_image(np.zeros((256, 256, 256), dtype=np.float32), np.eye(4)).header
    assert float(stale_header["fov"]) == pytest.approx(256.0)

    small_data = np.zeros((7, 100, 100), dtype=np.uint8)
    affine = np.diag([0.8, 0.8, 0.8, 1.0])

    image = as_mgh_image(small_data, affine, stale_header)

    assert float(image.header["fov"]) == pytest.approx(80.0)


def test_save_image_nifti_to_mgz_writes_nonzero_fov(tmp_path: Path):
    # Anisotropic voxels where "largest dimension" and "largest extent" disagree:
    # shape (40, 200, 80) * zooms (2.0, 0.5, 1.5) -> extents (80, 100, 120).
    shape = (40, 200, 80)
    affine = np.diag([2.0, 0.5, 1.5, 1.0])
    data = np.zeros(shape, dtype=np.uint8)
    src = nib.Nifti1Image(data, affine)
    out_path = tmp_path / "out.mgz"

    save_image(src, out_path)

    loaded = nib.load(str(out_path))
    zooms = loaded.header.get_zooms()[:3]
    expected = max(n * z for n, z in zip(loaded.shape[:3], zooms, strict=True))

    assert float(loaded.header["fov"]) == pytest.approx(expected)
    assert float(loaded.header["fov"]) == pytest.approx(120.0)


def test_as_mgh_image_preserves_dtype_from_a_foreign_header():
    # MGHHeader.from_header(nifti_header) defaults the data dtype to float32
    # instead of carrying over the array's actual (already MGH-supported) dtype.
    data = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
    affine = np.eye(4)
    nifti_header = nib.Nifti1Image(data, affine).header

    image = as_mgh_image(data, affine, nifti_header)

    assert image.get_data_dtype() == np.dtype(np.uint8)
    assert np.asanyarray(image.dataobj).dtype == np.dtype(np.uint8)


def test_save_image_nifti_to_mgz_preserves_uint8_dtype(tmp_path: Path):
    data = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
    src = nib.Nifti1Image(data, np.eye(4))
    out_path = tmp_path / "labels.mgz"

    save_image(src, out_path)

    loaded = nib.load(str(out_path))
    assert loaded.get_data_dtype() == np.dtype(np.uint8)
    assert np.array_equal(np.asanyarray(loaded.dataobj), data)


@pytest.mark.parametrize(
    ("dtype", "values", "expected"),
    [
        # Integer data lands in the narrowest MGH integer type holding its range,
        # so an aseg-style label volume stays integral instead of becoming float32.
        (np.uint16, [0, 41, 1000, 2035], np.int16),
        (np.uint16, [0, 40000, 1, 2], np.int32),  # exceeds int16, still exact in int32
        (np.int64, [0, 41, 1000, 2035], np.int16),
        (np.bool_, [0, 1, 1, 0], np.uint8),
        (np.float64, [0.5, 1.25, 2.0, 3.5], np.float32),  # no exact integer target exists
    ],
)
def test_as_mgh_image_maps_unsupported_dtypes_without_losing_values(dtype, values, expected):
    data = np.array(values, dtype=dtype).reshape(2, 2, 1)

    image = as_mgh_image(data, np.eye(4))

    assert image.get_data_dtype().newbyteorder("=") == np.dtype(expected)
    assert np.array_equal(np.asanyarray(image.dataobj), data)


def test_as_mgh_image_keeps_large_integers_exact():
    # float32 has a 24-bit mantissa, so 2**24 + 1 would round down to 2**24.
    data = np.array([[[2**24 + 1]]], dtype=np.int64)

    image = as_mgh_image(data, np.eye(4))

    assert image.get_data_dtype().newbyteorder("=") == np.dtype(np.int32)
    assert int(np.asanyarray(image.dataobj)[0, 0, 0]) == 2**24 + 1


def test_save_image_rejects_a_read_only_format(tmp_path: Path):
    # MINC is readable but not writable by nibabel; it raises NotImplementedError,
    # which save_image must translate into its documented ValueError.
    src = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))

    with pytest.raises(ValueError, match="Unsupported output image format"):
        save_image(src, tmp_path / "out.mnc")


def test_save_image_writes_analyze_pair(tmp_path: Path):
    data = np.arange(8, dtype=np.int16).reshape(2, 2, 2)
    src = nib.Nifti1Image(data, np.eye(4))
    out_path = tmp_path / "out.img"

    save_image(src, out_path)

    assert out_path.exists()
    assert np.array_equal(np.asanyarray(nib.load(str(out_path)).dataobj), data)


def test_save_image_preserves_int16_dtype_on_mgh_to_mgh_round_trip(tmp_path: Path):
    # MGH is always big-endian on disk, so a loaded proxy exposes ">i2", not
    # the native "<i2"/"=i2" that plain dtype-set membership checks assume.
    data = np.arange(8, dtype=np.int16).reshape(2, 2, 2)
    src_path = tmp_path / "src.mgz"
    nib.save(nib.MGHImage(data, np.eye(4)), str(src_path))
    loaded = nib.load(str(src_path))
    assert np.asanyarray(loaded.dataobj).dtype.newbyteorder("=") == np.dtype(np.int16)

    out_path = tmp_path / "out.mgz"
    save_image(loaded, out_path)

    result = nib.load(str(out_path))
    assert result.get_data_dtype().newbyteorder("=") == np.dtype(np.int16)
    assert np.array_equal(np.asanyarray(result.dataobj), data)
