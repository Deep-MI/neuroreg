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
