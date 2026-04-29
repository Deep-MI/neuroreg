"""Bundled centroid target resources for segmentation-based registration."""

from __future__ import annotations

from importlib import resources

import numpy as np
import numpy.typing as npt

from .io import CentroidDict, GeometryDict, TargetFile, read_target_json

_ATLAS_FILES = {
    "fsaverage": "fsaverage.json",
    "mni_icbm152_t1_tal_nlin_asym_09c": "mni_icbm152_t1_tal_nlin_asym_09c.json",
}


def _resource(name: str) -> resources.abc.Traversable:
    """Return a handle to a bundled target resource."""
    return resources.files("neuroreg.segreg").joinpath("data", name)


def available_atlases() -> tuple[str, ...]:
    """Return the names of bundled centroid targets."""
    return tuple(sorted(_ATLAS_FILES))


def load_atlas_target(name: str) -> TargetFile:
    """Load a supported bundled centroid target."""
    resource_name = _ATLAS_FILES.get(name)
    if resource_name is None:
        raise ValueError(f"Unknown atlas '{name}'. Available atlases: {', '.join(available_atlases())}.")
    with resources.as_file(_resource(resource_name)) as path:
        return read_target_json(path)


def load_fsaverage_centroids() -> CentroidDict:
    """Load bundled fsaverage centroid coordinates."""
    return load_atlas_centroids("fsaverage")


def load_fsaverage_data() -> tuple[npt.NDArray[np.float64], GeometryDict]:
    """Load bundled fsaverage geometry metadata."""
    return load_atlas_data("fsaverage")


def load_atlas_centroids(name: str) -> CentroidDict:
    """Load centroid coordinates for a supported bundled target."""
    return load_atlas_target(name).centroids


def load_atlas_data(name: str) -> tuple[npt.NDArray[np.float64], GeometryDict]:
    """Load geometry metadata for a supported bundled target."""
    target = load_atlas_target(name)
    if target.geometry is None:
        raise ValueError(f"Atlas '{name}' does not include bundled geometry metadata.")
    return affine_from_header(target.geometry), target.geometry


def affine_from_header(header: GeometryDict) -> npt.NDArray[np.float64]:
    """Reconstruct a voxel-to-RAS affine from header-like geometry metadata."""
    dims = np.asarray(header["dims"], dtype=np.float64)
    delta = np.asarray(header["delta"], dtype=np.float64)
    mdc = np.asarray(header["Mdc"], dtype=np.float64)
    pxyz_c = np.asarray(header["Pxyz_c"], dtype=np.float64)

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = mdc * delta
    affine[:3, 3] = pxyz_c - affine[:3, :3] @ (dims / 2.0)
    return affine
