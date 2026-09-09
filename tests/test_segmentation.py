"""Tests for neuroreg.image.segmentation label simplification and its output dtype."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neuroreg.image.segmentation import simplify_segmentation


def _aseg_like() -> nib.Nifti1Image:
    """Build a small parcellation covering each simplification branch."""
    data = np.zeros((6, 6, 6), dtype=np.int32)
    data[0:3, :, :] = 2  # LH white matter, kept as-is
    data[3:6, :, :] = 41  # RH white matter, kept as-is
    data[0:3, 0:2, 0:2] = 1001  # LH cortical parcel -> LH GM (3)
    data[3:6, 0:2, 0:2] = 2001  # RH cortical parcel -> RH GM (42)
    data[0:3, 4:6, 4:6] = 10  # LH thalamus -> LH WM (2)
    data[3:6, 4:6, 4:6] = 17  # hippocampus -> background (0)
    return nib.Nifti1Image(data, np.eye(4), dtype=np.int32)


@pytest.mark.parametrize("suffix", [".mgz", ".nii.gz"])
def test_simplify_segmentation_writes_labels_as_uint8(tmp_path: Path, suffix: str):
    # A five-value label map has no reason to occupy 4 bytes per voxel; writing
    # it as float32/int32 also misrepresents discrete labels as continuous data.
    out_path = tmp_path / f"simplified{suffix}"

    returned = simplify_segmentation(_aseg_like(), out_path)

    loaded = nib.load(str(out_path))
    assert loaded.get_data_dtype() == np.dtype(np.uint8)
    assert np.array_equal(np.asanyarray(loaded.dataobj), returned.astype(np.uint8))


def test_simplify_segmentation_keeps_an_int32_return_value(tmp_path: Path):
    # Callers such as extract_wm_surface keep computing on the returned array,
    # so only the on-disk dtype narrows.
    returned = simplify_segmentation(_aseg_like(), tmp_path / "simplified.mgz")

    assert returned.dtype == np.dtype(np.int32)


def test_simplify_segmentation_maps_labels_to_the_four_class_scheme():
    returned = simplify_segmentation(_aseg_like())

    assert set(np.unique(returned).tolist()) <= {0, 2, 3, 41, 42}
    assert returned[0, 0, 0] == 3  # LH cortical parcel -> LH GM
    assert returned[5, 0, 0] == 42  # RH cortical parcel -> RH GM
    assert returned[0, 4, 4] == 2  # LH thalamus -> LH WM
    assert returned[5, 5, 5] == 0  # hippocampus -> background
