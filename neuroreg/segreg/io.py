"""JSON I/O helpers for centroid-based registration.

The centroid JSON format is intentionally simple: label IDs map to 3-vector
coordinates. These helpers normalize NumPy-heavy in-memory objects to that
portable representation and back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

CentroidDict = dict[int, npt.NDArray[np.float64]]


def convert_numpy_to_json_serializable(obj: object) -> object:
    """Recursively convert NumPy scalars, arrays, and tuples into JSON-safe values."""
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


def read_centroids_json(path: str | Path) -> CentroidDict:
    """Read label centroids from a JSON file into float64 NumPy arrays."""
    centroid_path = Path(path)
    with centroid_path.open() as f:
        data = json.load(f)
    return {int(label): np.asarray(point, dtype=np.float64) for label, point in data.items()}


def write_centroids_json(path: str | Path, centroids: dict[int, npt.ArrayLike | None]) -> None:
    """Write centroid coordinates to JSON, skipping labels whose points are missing."""
    payload: dict[str, Any] = {}
    for label, point in centroids.items():
        if point is None:
            continue
        payload[str(int(label))] = np.asarray(point, dtype=np.float64)
    out_path = Path(path)
    with out_path.open("w") as f:
        json.dump(convert_numpy_to_json_serializable(payload), f, indent=2)
        f.write("\n")
