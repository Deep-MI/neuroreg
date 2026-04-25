"""Shared image-loading helpers with optional 4dfp sidecar support."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

_4DFP_TRANSVERSE_BASIS = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class _IfhMetadata:
    """Parsed subset of the 4dfp ``.ifh`` sidecar metadata."""

    orientation: int
    shape: tuple[int, int, int]
    mmppix: tuple[float, float, float]
    center: tuple[float, float, float] | None


def _find_ifh_sidecar(path: Path) -> Path | None:
    """Return the matching ``.ifh`` sidecar for an Analyze-style image path."""

    if path.suffix.lower() not in {".img", ".hdr"}:
        return None
    sidecar = path.with_suffix(".ifh")
    return sidecar if sidecar.exists() else None


def _is_analyze_like_image(image: Any) -> bool:
    """Return whether nibabel decoded the file as an Analyze/SPM image."""

    module = type(image).__module__
    return module.startswith("nibabel.analyze") or module.startswith("nibabel.spm")


def _parse_triplet(value: str, *, field: str, value_type: type[float] | type[int]) -> tuple[Any, Any, Any]:
    """Parse a whitespace-separated three-value IFH field."""

    parts = value.split()
    if len(parts) != 3:
        raise ValueError(f"IFH field '{field}' must contain exactly three values, got {value!r}.")
    return tuple(value_type(part) for part in parts)


def _read_ifh_metadata(path: Path) -> _IfhMetadata:
    """Parse the 4dfp metadata needed to reconstruct a FreeSurfer-compatible affine.

    Parameters
    ----------
    path : Path
        Path to the ``.ifh`` sidecar.

    Returns
    -------
    _IfhMetadata
        Parsed orientation code, spatial dimensions, signed voxel sizes, and
        optional center values from the sidecar.

    Raises
    ------
    ValueError
        If required metadata are missing or malformed.
    """
    fields: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        if ":=" not in raw_line:
            continue
        key, value = raw_line.split(":=", 1)
        fields[key.strip().lower()] = value.strip()

    try:
        orientation = int(fields["orientation"])
        shape = (
            int(fields["matrix size [1]"]),
            int(fields["matrix size [2]"]),
            int(fields["matrix size [3]"]),
        )
        mmppix = _parse_triplet(fields["mmppix"], field="mmppix", value_type=float)
    except KeyError as exc:
        raise ValueError(f"IFH sidecar {path} is missing required field {exc.args[0]!r}.") from exc
    except ValueError as exc:
        raise ValueError(f"IFH sidecar {path} contains invalid geometry metadata.") from exc

    center_value = fields.get("center")
    center = _parse_triplet(center_value, field="center", value_type=float) if center_value is not None else None
    return _IfhMetadata(orientation=orientation, shape=shape, mmppix=mmppix, center=center)


def _build_4dfp_affine(metadata: _IfhMetadata) -> np.ndarray:
    """Construct a FreeSurfer-compatible affine from parsed 4dfp sidecar metadata.

    Parameters
    ----------
    metadata : _IfhMetadata
        Parsed ``.ifh`` metadata.

    Returns
    -------
    np.ndarray
        A 4 × 4 voxel-to-RAS affine.

    Raises
    ------
    ValueError
        If the 4dfp orientation code is not currently supported.
    """
    if metadata.orientation != 2:
        raise ValueError(
            f"Unsupported 4dfp orientation code {metadata.orientation}; only transverse orientation 2 is supported."
        )

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = _4DFP_TRANSVERSE_BASIS @ np.diag(np.asarray(metadata.mmppix, dtype=np.float64))
    affine[:3, 3] = -(affine[:3, :3] @ (np.asarray(metadata.shape, dtype=np.float64) / 2.0))
    return affine


def _with_affine(image: Any, affine: np.ndarray) -> Any:
    """Return a copy of ``image`` with the supplied voxel-to-RAS affine."""

    header = image.header.copy()
    return image.__class__(image.dataobj, affine, header, extra=getattr(image, "extra", None))


def load_image(image: str | Path | Any) -> Any:
    """Load an image path, applying 4dfp ``.ifh`` geometry when available.

    Parameters
    ----------
    image : str or Path or Any
        Filesystem path to load, or an already-loaded nibabel-like image object.

    Returns
    -------
    Any
        Loaded image object. When a matching 4dfp ``.ifh`` sidecar exists for an
        Analyze/SPM-style ``.img`` or ``.hdr`` file, the returned image carries a
        FreeSurfer-compatible affine reconstructed from the sidecar metadata.

    Raises
    ------
    ValueError
        If a matching ``.ifh`` exists but contains malformed geometry metadata or
        an unsupported orientation code.
    """
    if not isinstance(image, (str, Path)):
        return image

    path = Path(image)
    loaded = nib.load(str(path))
    sidecar = _find_ifh_sidecar(path)
    if sidecar is None or not _is_analyze_like_image(loaded):
        return loaded

    metadata = _read_ifh_metadata(sidecar)
    shape3 = tuple(int(dim) for dim in loaded.shape[:3])
    if shape3 != metadata.shape:
        raise ValueError(
            f"IFH sidecar {sidecar} reports shape {metadata.shape}, but image data shape is {shape3}."
        )

    if metadata.center is not None:
        logger.debug("Applying 4dfp affine override from %s (center=%s).", sidecar, metadata.center)
    else:
        logger.debug("Applying 4dfp affine override from %s.", sidecar)
    return _with_affine(loaded, _build_4dfp_affine(metadata))
