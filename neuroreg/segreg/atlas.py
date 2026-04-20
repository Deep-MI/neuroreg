"""Atlas resources for segmentation-based registration."""

from __future__ import annotations

import json
from importlib import resources
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from .io import CentroidDict, read_centroids_json


class AtlasHeaderDict(TypedDict):
    """Minimal geometry metadata needed for LTA writing and atlas reslicing."""

    dims: list[int]
    delta: list[float]
    Mdc: npt.NDArray[np.float64]
    Pxyz_c: npt.NDArray[np.float64]


_ATLAS_NAMES = {"fsaverage"}


def _resource(name: str) -> resources.abc.Traversable:
    return resources.files("neuroreg.segreg").joinpath("data", name)


def available_atlases() -> tuple[str, ...]:
    """Return the atlas names bundled with the package."""
    return tuple(sorted(_ATLAS_NAMES))


def load_fsaverage_centroids() -> CentroidDict:
    """Load the bundled fsaverage centroid resource."""
    with resources.as_file(_resource("fsaverage_centroids.json")) as path:
        return read_centroids_json(path)


def load_fsaverage_data() -> tuple[npt.NDArray[np.float64], AtlasHeaderDict]:
    """Load the bundled fsaverage affine and header metadata."""
    with _resource("fsaverage_data.json").open() as f:
        data = json.load(f)

    affine = np.asarray(data["affine"], dtype=np.float64)
    header = AtlasHeaderDict(
        dims=[int(v) for v in data["header"]["dims"]],
        delta=[float(v) for v in data["header"]["delta"]],
        Mdc=np.asarray(data["header"]["Mdc"], dtype=np.float64),
        Pxyz_c=np.asarray(data["header"]["Pxyz_c"], dtype=np.float64),
    )
    return affine, header


def load_atlas_centroids(name: str) -> CentroidDict:
    """Load centroid resources for a supported atlas."""
    if name != "fsaverage":
        raise ValueError(f"Unknown atlas '{name}'. Available atlases: {', '.join(available_atlases())}.")
    return load_fsaverage_centroids()


def load_atlas_data(name: str) -> tuple[npt.NDArray[np.float64], AtlasHeaderDict]:
    """Load geometry resources for a supported atlas."""
    if name != "fsaverage":
        raise ValueError(f"Unknown atlas '{name}'. Available atlases: {', '.join(available_atlases())}.")
    return load_fsaverage_data()


def affine_from_header(header: AtlasHeaderDict) -> npt.NDArray[np.float64]:
    """Reconstruct a voxel-to-RAS affine from ``dims/delta/Mdc/Pxyz_c`` metadata."""
    dims = np.asarray(header["dims"], dtype=np.float64)
    delta = np.asarray(header["delta"], dtype=np.float64)
    mdc = np.asarray(header["Mdc"], dtype=np.float64)
    pxyz_c = np.asarray(header["Pxyz_c"], dtype=np.float64)

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = mdc * delta
    affine[:3, 3] = pxyz_c - affine[:3, :3] @ (dims / 2.0)
    return affine
