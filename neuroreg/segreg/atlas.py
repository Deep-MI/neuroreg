"""Bundled atlas resources for segmentation-based registration.

This module ships lightweight centroid and geometry metadata for atlas-backed
``segreg`` modes so atlas registration can run without external downloads.
"""

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
    """Return a handle to a bundled atlas resource.

    Parameters
    ----------
    name : str
        Resource filename inside ``neuroreg/segreg/data``.

    Returns
    -------
    importlib.resources.abc.Traversable
        Traversable handle that can be opened directly or materialized on disk.
    """
    return resources.files("neuroreg.segreg").joinpath("data", name)


def available_atlases() -> tuple[str, ...]:
    """Return the names of bundled atlas resources.

    Returns
    -------
    tuple[str, ...]
        Supported atlas names in sorted order.
    """
    return tuple(sorted(_ATLAS_NAMES))


def load_fsaverage_centroids() -> CentroidDict:
    """Load bundled fsaverage centroid coordinates.

    Returns
    -------
    dict[int, np.ndarray]
        Label-to-centroid mapping in scanner-RAS coordinates.
    """
    with resources.as_file(_resource("fsaverage_centroids.json")) as path:
        return read_centroids_json(path)


def load_fsaverage_data() -> tuple[npt.NDArray[np.float64], AtlasHeaderDict]:
    """Load bundled fsaverage geometry metadata.

    Returns
    -------
    affine : np.ndarray
        Voxel-to-RAS affine stored for the bundled fsaverage geometry.
    header : AtlasHeaderDict
        Minimal header-like metadata needed for LTA writing and output geometry
        reconstruction.
    """
    with _resource("fsaverage_data.json").open() as f:
        data = json.load(f)

    affine = np.asarray(data["affine"], dtype=np.float64)
    header: AtlasHeaderDict = {
        "dims": [int(v) for v in data["header"]["dims"]],
        "delta": [float(v) for v in data["header"]["delta"]],
        "Mdc": np.asarray(data["header"]["Mdc"], dtype=np.float64),
        "Pxyz_c": np.asarray(data["header"]["Pxyz_c"], dtype=np.float64),
    }
    return affine, header


def load_atlas_centroids(name: str) -> CentroidDict:
    """Load centroid resources for a supported bundled atlas.

    Parameters
    ----------
    name : str
        Bundled atlas name.

    Returns
    -------
    dict[int, np.ndarray]
        Label-to-centroid mapping for the requested atlas.

    Raises
    ------
    ValueError
        If ``name`` does not match a bundled atlas.
    """
    if name != "fsaverage":
        raise ValueError(f"Unknown atlas '{name}'. Available atlases: {', '.join(available_atlases())}.")
    return load_fsaverage_centroids()


def load_atlas_data(name: str) -> tuple[npt.NDArray[np.float64], AtlasHeaderDict]:
    """Load geometry metadata for a supported bundled atlas.

    Parameters
    ----------
    name : str
        Bundled atlas name.

    Returns
    -------
    affine : np.ndarray
        Voxel-to-RAS affine for the atlas geometry.
    header : AtlasHeaderDict
        Minimal atlas header metadata.

    Raises
    ------
    ValueError
        If ``name`` does not match a bundled atlas.
    """
    if name != "fsaverage":
        raise ValueError(f"Unknown atlas '{name}'. Available atlases: {', '.join(available_atlases())}.")
    return load_fsaverage_data()


def affine_from_header(header: AtlasHeaderDict) -> npt.NDArray[np.float64]:
    """Reconstruct a voxel-to-RAS affine from atlas header metadata.

    Parameters
    ----------
    header : AtlasHeaderDict
        Dictionary containing ``dims``, ``delta``, ``Mdc``, and ``Pxyz_c`` as
        stored in bundled atlas metadata.

    Returns
    -------
    np.ndarray
        Reconstructed ``(4, 4)`` voxel-to-RAS affine.
    """
    dims = np.asarray(header["dims"], dtype=np.float64)
    delta = np.asarray(header["delta"], dtype=np.float64)
    mdc = np.asarray(header["Mdc"], dtype=np.float64)
    pxyz_c = np.asarray(header["Pxyz_c"], dtype=np.float64)

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = mdc * delta
    affine[:3, 3] = pxyz_c - affine[:3, :3] @ (dims / 2.0)
    return affine
