"""Shared image-loading helpers with optional 4dfp sidecar support."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.filebasedimages import ImageFileError

from .mgh_tags import MGHTags, write_mgh_image

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
        raise ValueError(f"IFH sidecar {sidecar} reports shape {metadata.shape}, but image data shape is {shape3}.")

    if metadata.center is not None:
        logger.debug("Applying 4dfp affine override from %s (center=%s).", sidecar, metadata.center)
    else:
        logger.debug("Applying 4dfp affine override from %s.", sidecar)
    return _with_affine(loaded, _build_4dfp_affine(metadata))


#: Suffix that selects the geometry JSON reader instead of the image reader.
GEOMETRY_JSON_SUFFIX = ".json"


def is_geometry_json(source: Any) -> bool:
    """Return whether a geometry source is a path to a geometry JSON rather than an image."""
    return isinstance(source, (str, Path)) and Path(source).suffix.lower() == GEOMETRY_JSON_SUFFIX


def load_geometry_source(source: str | Path | Any, *, label: str = "geometry source") -> Any:
    """Load a grid from an image or from a JSON carrying a ``geometry`` block.

    Several options and arguments read only the shape and affine of what they
    are given, so requiring an image forces callers to keep a file around
    purely for its header. A JSON with a ``geometry`` block (``dims``,
    ``delta``, ``Mdc``, ``Pxyz_c``) names the same grid without one. Any other
    keys are ignored, so both a bare grid file and a bundled centroid target
    that also carries geometry are accepted.

    The JSON case is returned as an image whose voxels are a zero-strided view
    rather than an allocation, because no caller reads them: this keeps every
    consumer on the one code path it already has.

    Parameters
    ----------
    source : str or Path or Any
        Path to an image or geometry JSON, or an already-loaded image, which is
        returned unchanged. A bundled atlas name is not accepted; pass a path.
    label : str, default="geometry source"
        Name to quote in error messages, e.g. ``"--ref"`` or
        ``"template_geometry"``.

    Returns
    -------
    Any
        Loaded nibabel image, or a geometry-only image built from the JSON.

    Raises
    ------
    ValueError
        If the JSON carries no ``geometry`` block, and so names no grid, or the
        block is malformed.
    """
    if not is_geometry_json(source):
        return load_image(source)

    # Imported lazily: segreg depends on transforms, which this module must not.
    from ..segreg.atlas import affine_from_header
    from ..segreg.io import read_geometry_json

    try:
        geometry = read_geometry_json(source)
    except ValueError as exc:
        raise ValueError(f"{label}: {exc}") from exc
    if geometry is None:
        raise ValueError(
            f"{label} was given '{source}', which has no 'geometry' block, so it does not describe "
            "a grid. Pass an image, or a JSON with a geometry block such as the bundled atlas targets."
        )
    dims = tuple(int(v) for v in geometry["dims"])
    voxels = np.broadcast_to(np.uint8(0), dims)
    # No filename is set: nibabel validates it against the image type and would
    # reject a .json path. Consumers that record one fall back to empty, which
    # is what an LTA already carries for a geometry with no image behind it.
    return nib.MGHImage(voxels, affine_from_header(geometry))


_NIFTI_SUFFIXES = (".nii.gz", ".nii")
_MGH_SUFFIXES = (".mgz", ".mgh")
_ANALYZE_SUFFIXES = (".img", ".hdr")
IMAGE_SUFFIXES = (*_NIFTI_SUFFIXES, *_MGH_SUFFIXES, *_ANALYZE_SUFFIXES)

_MGH_DTYPES = {np.dtype(np.uint8), np.dtype(np.int16), np.dtype(np.int32), np.dtype(np.float32)}
# Ordered smallest first, so integer data lands in the narrowest MGH type that holds it.
_MGH_INT_DTYPES = (np.dtype(np.uint8), np.dtype(np.int16), np.dtype(np.int32))


def recognized_image_suffix(path: str | Path) -> str | None:
    """Return the recognized image suffix of ``path``, or ``None``.

    Matching is case-insensitive and handles the compound ``.nii.gz`` suffix,
    which :attr:`pathlib.PurePath.suffix` alone does not.

    Parameters
    ----------
    path : str or Path
        Filename to inspect.

    Returns
    -------
    str or None
        The matching entry of :data:`IMAGE_SUFFIXES` (e.g. ``".nii.gz"``), or
        ``None`` when ``path`` names no format this package writes.
    """
    lowered = str(path).lower()
    for suffix in IMAGE_SUFFIXES:
        if lowered.endswith(suffix):
            return suffix
    return None


def check_dtype_storable(dtype: np.dtype | str, path: str | Path) -> None:
    """Raise if the format selected by ``path`` cannot store ``dtype``.

    Use this to validate a dtype the caller asked for *explicitly* (for example
    ``vol2vol --out-dtype``). Silently substituting a different type would hand
    back a file that does not match the request, so an unhonourable request is
    refused instead.

    Parameters
    ----------
    dtype : np.dtype or str
        Requested output dtype.
    path : str or Path
        Output filename. Its extension selects the format.

    Returns
    -------
    None
        Returns if the format can store ``dtype``.

    Raises
    ------
    ValueError
        If the format cannot store ``dtype``.
    """
    dtype = np.dtype(dtype)
    if recognized_image_suffix(path) in _MGH_SUFFIXES and dtype.newbyteorder("=") not in _MGH_DTYPES:
        supported = ", ".join(sorted(candidate.name for candidate in _MGH_DTYPES))
        raise ValueError(
            f"MGH/MGZ cannot store {dtype.name} data (it supports {supported}). "
            f"Request a supported dtype, or write NIfTI (.nii/.nii.gz) instead."
        )


def _mgh_dtype_for(data: np.ndarray) -> np.dtype:
    """Return the MGH dtype that stores ``data`` without corrupting values.

    MGH stores only uint8, int16, int32 and float32. Integer data is mapped to
    the narrowest of those that holds its value range exactly, so a uint16 or
    int64 label volume stays integral instead of being cast to float32, which
    would double its size and silently round values above 2**24. Integers too
    wide for int32 are refused rather than rounded through float32, and float64
    is narrowed to float32 with a warning, since MGH has no 64-bit float.

    Parameters
    ----------
    data : np.ndarray
        Voxel data to be stored.

    Returns
    -------
    np.dtype
        A dtype MGH can store. ``data.dtype`` is returned unchanged when it is
        already supported, preserving its byte order.

    Raises
    ------
    ValueError
        If no MGH dtype can represent ``data`` without corrupting values.
    """
    native = data.dtype.newbyteorder("=")
    if native in _MGH_DTYPES:
        return data.dtype
    if np.issubdtype(native, np.bool_):
        return np.dtype(np.uint8)
    if np.issubdtype(native, np.integer):
        if not data.size:
            return np.dtype(np.int32)
        low, high = data.min(), data.max()
        for candidate in _MGH_INT_DTYPES:
            info = np.iinfo(candidate)
            if low >= info.min and high <= info.max:
                return candidate
        raise ValueError(
            f"MGH/MGZ cannot store {native.name} values in [{low}, {high}]: the range exceeds "
            f"int32, and float32 would round them. Write NIfTI (.nii/.nii.gz) instead, or "
            f"rescale the data to fit int32."
        )
    if not np.issubdtype(native, np.floating):
        raise ValueError(
            f"MGH/MGZ cannot store {native.name} data. Write NIfTI (.nii/.nii.gz) instead, or convert the data first."
        )
    if native.itemsize > np.dtype(np.float32).itemsize:
        logger.warning(
            "Narrowing %s to float32: MGH/MGZ has no wider float type. "
            "Write NIfTI (.nii/.nii.gz) to keep full precision.",
            native.name,
        )
    return np.dtype(np.float32)


def as_mgh_image(data: np.ndarray, affine: np.ndarray, header: Any | None = None) -> nib.MGHImage:
    """Build an ``MGHImage`` with a correctly populated field-of-view and dtype.

    ``MGHHeader.from_header`` does not carry two fields over from a non-MGH
    source header (e.g. NIfTI):

    * ``fov`` defaults to 0, since NIfTI has no equivalent field to inherit
      it from.
    * The data dtype is not transferred at all, so the header keeps
      ``MGHHeader``'s float32 default. This silently quadruples storage for
      e.g. a uint8 segmentation and can change how downstream tools interpret
      the file.

    Both are recomputed here from ``data`` and ``affine`` rather than
    inherited from ``header``. Data in a dtype MGH cannot store is converted
    by :func:`_mgh_dtype_for`.

    Parameters
    ----------
    data : np.ndarray
        Voxel data.
    affine : np.ndarray
        4 x 4 voxel-to-RAS affine.
    header : Any, optional
        Source header to carry compatible fields over from (e.g. an existing
        ``MGHHeader``). May describe a different shape or dtype than
        ``data``; ``fov`` and the data dtype are always recomputed from
        ``data`` and ``affine``, not inherited.

    Returns
    -------
    nib.MGHImage
        Image whose header has ``fov`` and the data dtype set to match
        ``data``.

    Raises
    ------
    ValueError
        If no MGH dtype can represent ``data`` without corrupting values.
    """
    data = data.astype(_mgh_dtype_for(data), copy=False)
    image = nib.MGHImage(data, affine, header)
    image.header.set_data_dtype(data.dtype)
    zooms = image.header.get_zooms()[:3]
    image.header["fov"] = max(n * z for n, z in zip(image.shape[:3], zooms, strict=True))
    return image


def save_image(image: Any, path: str | Path, tags: MGHTags | None = None) -> None:
    """Write an image, choosing the on-disk format from the output extension.

    Unlike calling ``image.to_filename`` directly, the output format follows the
    extension of ``path`` rather than the in-memory image class. When the
    requested format differs from the image's native class, nibabel converts the
    image (via ``from_image``), for example ``.mgz`` -> ``.nii.gz``. Geometry and
    voxel values are preserved; the stored dtype may be coerced to one the target
    format supports (for example MGH only stores uint8, int16, int32, and
    float32).

    Writable formats are the volume formats nibabel can construct from a foreign
    header, namely those in :data:`IMAGE_SUFFIXES`: NIfTI (``.nii``, ``.nii.gz``),
    MGH (``.mgz``, ``.mgh``) and Analyze/NIfTI pairs (``.img``, ``.hdr``). Other
    formats nibabel *reads* cannot be written from a converted image: MINC and
    GIFTI raise ``NotImplementedError``, while AFNI and PAR/REC only accept their
    own header type.

    When the destination is MGH/MGZ, the image is routed through
    :func:`as_mgh_image` instead of nibabel's generic conversion, since the
    latter both leaves the header's ``fov`` field at 0 and silently drops the
    data dtype to float32 when converting from a non-MGH source (see
    :func:`as_mgh_image`).

    nibabel neither reads nor writes the tags FreeSurfer stores after the
    voxels of an MGH volume, so ``tags`` are written explicitly. Which tags an
    output should carry depends on what produced it, so the caller decides,
    typically with :func:`~neuroreg.image.mgh_tags.inherit_tags`.

    Parameters
    ----------
    image : Any
        Nibabel-like image to write.
    path : str or Path
        Output filename. Its extension selects the output format.
    tags : MGHTags or None, optional
        Tags to store after the voxels. Only MGH/MGZ outputs carry them; for
        any other format they are not written, which is logged.

    Returns
    -------
    None
        The image is written for its side effect.

    Raises
    ------
    ValueError
        If ``path`` has no extension nibabel recognizes as a writable image
        format, or if the target format cannot store the image's data without
        corrupting values (see :func:`as_mgh_image`).
    """
    destination = Path(path)
    has_tags = tags is not None and bool(tags.tags)
    if recognized_image_suffix(destination) in _MGH_SUFFIXES:
        header = image.header if isinstance(image, nib.MGHImage) else None
        image = as_mgh_image(np.asanyarray(image.dataobj), np.asarray(image.affine, dtype=np.float64), header)
        if has_tags:
            write_mgh_image(image, destination, tags)
            return
    elif has_tags:
        logger.info("Not storing %d MGH tag(s) in %s: only MGH/MGZ outputs carry them.", len(tags.tags), path)
    try:
        nib.save(image, str(destination))
    except (ImageFileError, NotImplementedError) as exc:
        # NotImplementedError covers formats nibabel reads but cannot write (e.g. MINC).
        raise ValueError(f"Unsupported output image format for {str(path)!r}: {exc}") from exc
