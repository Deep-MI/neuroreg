"""JSON I/O helpers for centroid target files.

The on-disk target format stores required label centroids plus optional geometry
metadata used for LTA destination volume info. These helpers also accept the
legacy centroid-only JSON shape where the top-level object is the label-to-point
mapping itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt

from ..transforms.lta import _header_info

CentroidDict = dict[int, npt.NDArray[np.float64]]


class GeometryDict(TypedDict):
    """Minimal geometry metadata needed for LTA destination volume info."""

    dims: list[int]
    delta: list[float]
    Mdc: npt.NDArray[np.float64]
    Pxyz_c: npt.NDArray[np.float64]


@dataclass(frozen=True)
class TargetFile:
    """Centroid target payload loaded from or written to JSON."""

    centroids: CentroidDict
    geometry: GeometryDict | None = None


def convert_numpy_to_json_serializable(obj: object) -> object:
    """Convert nested NumPy-backed objects into JSON-safe Python values.

    Parameters
    ----------
    obj : object
        Object tree that may contain dictionaries, lists, tuples, NumPy arrays,
        or NumPy scalar types.

    Returns
    -------
    object
        Equivalent structure composed only of JSON-serializable Python values.
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_to_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_to_json_serializable(item) for item in obj]
    if isinstance(obj, tuple):
        return [convert_numpy_to_json_serializable(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def _coerce_centroids(payload: dict[Any, Any]) -> CentroidDict:
    """Normalize a decoded centroid payload to float64 NumPy arrays."""
    return {int(label): np.asarray(point, dtype=np.float64) for label, point in payload.items()}


def _coerce_geometry(payload: dict[str, Any]) -> GeometryDict:
    """Normalize decoded geometry metadata to the internal header-like shape."""
    return {
        "dims": [int(v) for v in payload["dims"]],
        "delta": [float(v) for v in payload["delta"]],
        "Mdc": np.asarray(payload["Mdc"], dtype=np.float64),
        "Pxyz_c": np.asarray(payload["Pxyz_c"], dtype=np.float64),
    }


def geometry_from_image(image: Any) -> GeometryDict:
    """Extract LTA-relevant geometry metadata from an image or path.

    Parameters
    ----------
    image : Any
        Image-like object or path accepted by ``neuroreg.transforms.lta._header_info``.

    Returns
    -------
    GeometryDict
        Geometry metadata compatible with bundled atlas targets and LTA writing.
    """
    return _coerce_geometry(_header_info(image))


def read_target_json(path: str | Path) -> TargetFile:
    """Read a centroid target JSON file.

    Parameters
    ----------
    path : str or Path
        Path to a target JSON file.

    Returns
    -------
    TargetFile
        Target payload with required centroids and optional geometry.
    """
    target_path = Path(path)
    with target_path.open() as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Target JSON at '{target_path}' must decode to an object.")

    if "centroids" in data:
        raw_centroids = data["centroids"]
        raw_geometry = data.get("geometry")
    else:
        raw_centroids = data
        raw_geometry = None

    if not isinstance(raw_centroids, dict):
        raise ValueError(f"Target JSON at '{target_path}' is missing a valid 'centroids' object.")

    geometry = None
    if raw_geometry is not None:
        if not isinstance(raw_geometry, dict):
            raise ValueError(f"Target JSON at '{target_path}' has a non-object 'geometry' section.")
        geometry = _coerce_geometry(raw_geometry)

    return TargetFile(centroids=_coerce_centroids(raw_centroids), geometry=geometry)


def write_target_json(
        path: str | Path,
        centroids: dict[int, npt.ArrayLike | None],
        *,
        geometry: GeometryDict | None = None,
) -> None:
    """Write a centroid target JSON file.

    Parameters
    ----------
    path : str or Path
        Output JSON path.
    centroids : dict[int, array-like or None]
        Label-to-centroid mapping. Entries with value ``None`` are skipped.
    geometry : GeometryDict or None, optional
        Optional geometry metadata to embed alongside the centroid coordinates.

    Returns
    -------
    None
        Writes the target payload to ``path``.
    """
    payload_centroids: dict[str, Any] = {}
    for label, point in centroids.items():
        if point is None:
            continue
        payload_centroids[str(int(label))] = np.asarray(point, dtype=np.float64)

    payload: dict[str, Any] = {"centroids": payload_centroids}
    if geometry is not None:
        payload["geometry"] = {
            "dims": [int(v) for v in geometry["dims"]],
            "delta": [float(v) for v in geometry["delta"]],
            "Mdc": np.asarray(geometry["Mdc"], dtype=np.float64),
            "Pxyz_c": np.asarray(geometry["Pxyz_c"], dtype=np.float64),
        }

    out_path = Path(path)
    with out_path.open("w") as f:
        json.dump(convert_numpy_to_json_serializable(payload), f, indent=2)
        f.write("\n")


def read_centroids_json(path: str | Path) -> CentroidDict:
    """Read centroid coordinates from either rich or legacy target JSON."""
    return read_target_json(path).centroids


def write_centroids_json(path: str | Path, centroids: dict[int, npt.ArrayLike | None]) -> None:
    """Write centroid coordinates using the rich target JSON envelope."""
    write_target_json(path, centroids)
