"""Tests for neuroreg.image.io: format-aware save/load and MGH fov metadata."""

from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neuroreg.image import IMAGE_SUFFIXES, as_mgh_image, check_dtype_storable, save_image


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


def test_as_mgh_image_refuses_integers_too_wide_for_int32():
    # float32 would round these, so refuse rather than corrupt the values.
    data = np.array([[[0, 2**40 + 1]]], dtype=np.int64)

    with pytest.raises(ValueError, match="exceeds int32"):
        as_mgh_image(data, np.eye(4))


def test_as_mgh_image_refuses_complex_data():
    data = np.array([[[1 + 2j]]], dtype=np.complex128)

    with pytest.raises(ValueError, match="cannot store complex128"):
        as_mgh_image(data, np.eye(4))


def test_as_mgh_image_warns_when_narrowing_float64(caplog: pytest.LogCaptureFixture):
    data = np.array([[[1 / 3, 2 / 3]]], dtype=np.float64)

    with caplog.at_level(logging.WARNING, logger="neuroreg.image.io"):
        image = as_mgh_image(data, np.eye(4))

    assert image.get_data_dtype().newbyteorder("=") == np.dtype(np.float32)
    assert "Narrowing float64 to float32" in caplog.text


def test_as_mgh_image_does_not_warn_for_storable_dtypes(caplog: pytest.LogCaptureFixture):
    data = np.array([[[1.5, 2.5]]], dtype=np.float32)

    with caplog.at_level(logging.WARNING, logger="neuroreg.image.io"):
        as_mgh_image(data, np.eye(4))

    assert caplog.text == ""


@pytest.mark.parametrize("suffix", [".mgz", ".mgh"])
@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.uint16])
def test_check_dtype_storable_refuses_dtypes_mgh_cannot_hold(suffix: str, dtype, tmp_path: Path):
    with pytest.raises(ValueError, match="MGH/MGZ cannot store"):
        check_dtype_storable(dtype, tmp_path / f"out{suffix}")


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int32, np.float32])
def test_check_dtype_storable_accepts_mgh_dtypes(dtype, tmp_path: Path):
    check_dtype_storable(dtype, tmp_path / "out.mgz")


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.uint16, np.uint8])
def test_check_dtype_storable_allows_anything_for_nifti(dtype, tmp_path: Path):
    check_dtype_storable(dtype, tmp_path / "out.nii.gz")


def test_save_image_rejects_a_read_only_format(tmp_path: Path):
    # MINC is readable but not writable by nibabel; it raises NotImplementedError,
    # which save_image must translate into its documented ValueError.
    src = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))

    with pytest.raises(ValueError, match="Unsupported output image format"):
        save_image(src, tmp_path / "out.mnc")


def test_image_suffixes_matches_what_nibabel_can_actually_write(tmp_path: Path):
    """The CLI gate must accept exactly the formats ``save_image`` can write.

    Accepting less would reject usable outputs; accepting more would restore the
    late failure the up-front check exists to prevent. Probing nibabel directly
    means this fails loudly if a future nibabel gains or loses a writable format.
    """
    src = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))
    declared = {ext for klass in nib.imageclasses.all_image_classes for ext in klass.valid_exts}
    declared.add(".nii.gz")  # handled by nibabel's transparent compression, not valid_exts

    writable = set()
    for ext in sorted(declared):
        try:
            nib.save(src, str(tmp_path / f"probe{ext}"))
        except Exception:  # noqa: BLE001 - capability probe: any failure means "not writable"
            continue
        writable.add(ext)

    assert writable == set(IMAGE_SUFFIXES)


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
