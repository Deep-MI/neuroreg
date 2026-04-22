"""High-level segmentation-based registration APIs.

This layer ties together centroid extraction, atlas resources, label presets,
and point-set solvers to expose one public ``segreg`` workflow that returns a
transform plus the metadata needed for downstream mapping and LTA export.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..transforms import matrix_sqrt_schur
from .atlas import load_atlas_centroids, load_atlas_data
from .centroids import (
    ImageLike,
    build_flipped_centroid_targets,
    collect_joint_centroids,
    compute_ras_centroids_from_seg,
    compute_voxel_centroids_from_seg,
    load_spatial_image,
)
from .io import CentroidDict, read_centroids_json, write_centroids_json
from .labels import LabelSetName, get_cortex_lr_labels, get_cortex_lr_pairs
from .points import register_points


@dataclass(frozen=True)
class RegistrationResult:
    """Result returned by :func:`segreg`.

    Attributes
    ----------
    r2r : np.ndarray
        Recovered 4×4 RAS-to-RAS transform mapping moving space into the chosen
        target space.
    labels : list[int]
        Label IDs that participated in the final fit.
    target_name : str
        Human-readable identifier for the target geometry written into output
        LTAs.
    target_geometry : Any or None
        Geometry object describing the target space. This may be a nibabel image,
        a header-like dictionary for atlas resources, or ``None`` when the
        target geometry is unknown.
    target_affine : np.ndarray or None
        Target voxel-to-RAS affine when explicit target geometry is available.
    target_shape : tuple[int, int, int] or None
        Target spatial shape when explicit target geometry is available.
    """

    r2r: np.ndarray
    labels: list[int]
    target_name: str
    target_geometry: Any
    target_affine: np.ndarray | None
    target_shape: tuple[int, int, int] | None


@dataclass(frozen=True)
class _GeometryInfo:
    """Internal representation of a target geometry."""

    name: str
    geometry: Any | None
    affine: np.ndarray | None
    shape: tuple[int, int, int] | None


def _default_min_common_labels(dof: int) -> int:
    """Return the default correspondence count for a given transform family.

    Parameters
    ----------
    dof : int
        Requested registration degrees of freedom.

    Returns
    -------
    int
        Minimum number of matched labels required for the default solver setup.
    """
    if dof == 3:
        return 1
    return 4 if dof in {9, 12} else 3


def _infer_label_ids(
        *,
        mode: str,
        explicit_labels: list[int] | None,
        label_set: LabelSetName | None,
        target_centroids: CentroidDict | None,
) -> list[int] | None:
    """Resolve the label IDs to evaluate for the selected registration mode.

    Parameters
    ----------
    mode : str
        Registration target mode: ``"ref"``, ``"ref_centroids"``, or
        ``"atlas"``.
    explicit_labels : list[int] or None
        Explicit label subset requested by the caller.
    label_set : {'all_shared', 'fsaverage_centroids', 'cortex_lr_pairs'} or None
        Named label preset requested by the caller.
    target_centroids : dict[int, np.ndarray] or None
        Target centroid dictionary, when applicable.

    Returns
    -------
    list[int] or None
        Label IDs that should be extracted from the moving segmentation. ``None``
        means "use all shared non-zero labels".
    """
    if explicit_labels is not None:
        return [int(label) for label in explicit_labels]
    if label_set == "cortex_lr_pairs":
        return get_cortex_lr_labels()
    if label_set == "fsaverage_centroids":
        if target_centroids is None:
            raise ValueError("label_set='fsaverage_centroids' requires atlas centroid targets.")
        return sorted(target_centroids.keys())
    if mode in {"atlas", "ref_centroids"} and target_centroids is not None:
        return sorted(target_centroids.keys())
    return None


def _geometry_from_image(image: Any, *, fallback_name: str) -> _GeometryInfo:
    """Build an internal geometry descriptor from an image object.

    Parameters
    ----------
    image : Any
        Nibabel-like image exposing ``get_filename()``, ``affine``, and
        ``shape``.
    fallback_name : str
        Name to use when the image object does not report a filename.

    Returns
    -------
    _GeometryInfo
        Geometry wrapper containing the image object, its affine, and its
        spatial shape.
    """
    return _GeometryInfo(
        name=image.get_filename() or fallback_name,
        geometry=image,
        affine=np.asarray(image.affine, dtype=np.float64),
        shape=tuple(int(v) for v in image.shape[:3]),
    )


def _geometry_from_atlas(name: str) -> _GeometryInfo:
    """Build an internal geometry descriptor from bundled atlas metadata.

    Parameters
    ----------
    name : str
        Bundled atlas name.

    Returns
    -------
    _GeometryInfo
        Geometry wrapper containing the atlas header metadata, affine, and
        spatial shape.
    """
    atlas_affine, atlas_header = load_atlas_data(name)
    return _GeometryInfo(
        name=name,
        geometry=atlas_header,
        affine=np.asarray(atlas_affine, dtype=np.float64),
        shape=tuple(int(v) for v in atlas_header["dims"]),
    )


def _resolve_target_centroids_and_geometry(
        *,
        ref: ImageLike | None,
        ref_centroids: str | Path | None,
        ref_geom: ImageLike | None,
        atlas: str | None,
) -> tuple[str, CentroidDict, _GeometryInfo | None]:
    """Resolve target centroids and optional target geometry for registration.

    Parameters
    ----------
    ref : ImageLike or None
        Reference segmentation image.
    ref_centroids : str or Path or None
        Path to a reference centroid JSON file.
    ref_geom : ImageLike or None
        Explicit geometry image to pair with ``ref_centroids``.
    atlas : str or None
        Name of a bundled atlas resource.

    Returns
    -------
    mode : str
        Selected target mode.
    centroids : dict[int, np.ndarray]
        Target centroid dictionary.
    geometry : _GeometryInfo or None
        Target geometry when available.
    """
    modes = sum(value is not None for value in (ref, ref_centroids, atlas))
    if modes != 1:
        raise ValueError("Choose exactly one registration target: ref image, ref_centroids, or atlas.")

    if ref is not None:
        ref_img = load_spatial_image(ref)
        ref_centroid_dict = {
            label: point
            for label, point in compute_ras_centroids_from_seg(ref_img).items()
            if point is not None
        }
        return "ref", ref_centroid_dict, _geometry_from_image(ref_img, fallback_name="reference.mgz")

    if ref_centroids is not None:
        centroid_dict = read_centroids_json(ref_centroids)
        geometry = None
        if ref_geom is not None:
            geometry = _geometry_from_image(load_spatial_image(ref_geom), fallback_name="reference_geom.mgz")
        return "ref_centroids", centroid_dict, geometry

    assert atlas is not None
    return "atlas", load_atlas_centroids(atlas), _geometry_from_atlas(atlas)


def segreg(
        mov: ImageLike,
        ref: ImageLike | None = None,
        *,
        ref_centroids: str | Path | None = None,
        ref_geom: ImageLike | None = None,
        atlas: str | None = None,
        dof: int = 6,
        labels: list[int] | None = None,
        label_set: LabelSetName | None = None,
        min_common_labels: int | None = None,
        flipped: bool = False,
        midslice: float | None = None,
) -> RegistrationResult:
    """Register a moving segmentation to another segmentation, an atlas, or its LR-flipped self.

    Parameters
    ----------
    mov : ImageLike
        Moving segmentation image. This may be a path or a nibabel-like image.
    ref : ImageLike or None, optional
        Reference segmentation image for segmentation-to-segmentation
        registration.
    ref_centroids : str or Path or None, optional
        Path to a centroid JSON file used as the registration target.
    ref_geom : ImageLike or None, optional
        Optional geometry image paired with ``ref_centroids``. This lets callers
        use centroid JSON for fitting while still defining a concrete output
        grid for mapped images or LTAs.
    atlas : str or None, optional
        Name of a bundled atlas resource such as ``"fsaverage"``.
    dof : {3, 6, 7, 9, 12}, default=6
        Degrees of freedom for the closed-form fit. ``3`` selects
        translation-only, ``6`` rigid, ``7`` rigid plus global scale, ``9``
        rigid plus anisotropic scaling without shear, and ``12`` affine
        registration.
    labels : list[int] or None, optional
        Explicit label subset override.
    label_set : {'all_shared', 'fsaverage_centroids', 'cortex_lr_pairs'} or None, optional
        Named label preset. Mode-specific defaults are used when omitted.
    min_common_labels : int or None, optional
        Minimum number of matched labels required to proceed. When omitted, the
        default is ``1`` for translation-only, ``3`` for rigid/similarity, and
        ``4`` for anisotropic-scale or affine registration.
    flipped : bool, default=False
        If ``True``, ignore external targets and register the moving
        segmentation to a left-right flipped self target for upright/midspace
        use cases.
    midslice : float or None, optional
        Explicit sagittal mid-slice used only with ``flipped=True``. When
        omitted, the geometric center of the moving image is used.

    Returns
    -------
    RegistrationResult
        Result object containing the recovered RAS transform, participating
        labels, and target geometry metadata.

    Raises
    ------
    ValueError
        If the arguments define no valid target, define multiple targets, or do
        not provide enough matched labels for the requested fit.
    """
    mov_img = load_spatial_image(mov)
    mov_name = mov_img.get_filename() or (str(mov) if isinstance(mov, (str, Path)) else "moving.mgz")
    mov_affine = np.asarray(mov_img.affine, dtype=np.float64)

    if dof not in {3, 6, 7, 9, 12}:
        raise ValueError(
            f"Unsupported dof={dof}. Segmentation registration supports "
            "3 (translation-only), 6 (rigid), 7 (similarity), "
            "9 (anisotropic scale without shear), or 12 (affine)."
        )

    if min_common_labels is None:
        min_common_labels = _default_min_common_labels(dof)

    if flipped:
        if any(value is not None for value in (ref, ref_centroids, atlas)):
            raise ValueError("--flipped cannot be combined with ref, ref_centroids, or atlas targets.")
        if dof != 6:
            raise ValueError("--flipped currently supports rigid registration only.")

        voxel_centroids = compute_voxel_centroids_from_seg(mov_img, label_ids=labels or get_cortex_lr_labels())
        resolved_midslice = 0.5 * (mov_img.shape[0] - 1.0) if midslice is None else float(midslice)
        mov_points, target_points, used_labels = build_flipped_centroid_targets(
            voxel_centroids,
            get_cortex_lr_pairs(),
            mid_slice=resolved_midslice,
            min_common_labels=min_common_labels,
        )
        flip_vox = register_points(mov_points, target_points, dof=6)
        flip_half, _ = matrix_sqrt_schur(torch.from_numpy(flip_vox).double())
        r2r = mov_affine @ flip_half.detach().cpu().numpy() @ np.linalg.inv(mov_affine)
        return RegistrationResult(
            r2r=np.asarray(r2r, dtype=np.float64),
            labels=used_labels,
            target_name=mov_name,
            target_geometry=mov_img,
            target_affine=mov_affine,
            target_shape=tuple(int(v) for v in mov_img.shape[:3]),
        )

    mode, target_centroids, geometry = _resolve_target_centroids_and_geometry(
        ref=ref,
        ref_centroids=ref_centroids,
        ref_geom=ref_geom,
        atlas=atlas,
    )
    resolved_labels = _infer_label_ids(
        mode=mode,
        explicit_labels=labels,
        label_set=label_set,
        target_centroids=target_centroids,
    )
    mov_centroids = compute_ras_centroids_from_seg(mov_img, label_ids=resolved_labels)
    mov_points, target_points, used_labels = collect_joint_centroids(
        mov_centroids,
        target_centroids,
        min_common_labels=min_common_labels,
    )
    r2r = register_points(mov_points, target_points, dof=dof)

    if geometry is None:
        geometry = _GeometryInfo(
            name=str(ref_centroids),
            geometry=None,
            affine=None,
            shape=None,
        )

    return RegistrationResult(
        r2r=np.asarray(r2r, dtype=np.float64),
        labels=used_labels,
        target_name=geometry.name,
        target_geometry=geometry.geometry,
        target_affine=geometry.affine,
        target_shape=geometry.shape,
    )


def export_segmentation_centroids(seg: ImageLike, out_path: str | Path, *, labels: list[int] | None = None) -> None:
    """Compute segmentation centroids and write them to FastSurfer-style JSON.

    Parameters
    ----------
    seg : ImageLike
        Segmentation image or path.
    out_path : str or Path
        Output JSON path.
    labels : list[int] or None, optional
        Optional label subset to export. When omitted, all non-zero labels are
        written.

    Returns
    -------
    None
        Writes the selected scanner-RAS centroids to ``out_path``.
    """
    centroids = compute_ras_centroids_from_seg(seg, label_ids=labels)
    write_centroids_json(out_path, centroids)


def resolve_output_geometry(
        result: RegistrationResult,
        *,
        keep_geom: str,
        mov_img: Any,
        ref_img: Any | None,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Resolve the geometry used for a resliced mapped output.

    Parameters
    ----------
    result : RegistrationResult
        Registration result returned by :func:`segreg`.
    keep_geom : {'mov', 'ref', 'atlas'}
        Output-geometry selection policy.
    mov_img : Any
        Moving image that will be mapped.
    ref_img : Any or None
        Reference geometry image used when ``keep_geom='ref'``.

    Returns
    -------
    target_affine : np.ndarray
        Voxel-to-RAS affine of the output grid.
    target_shape : tuple[int, int, int]
        Spatial shape of the output grid.

    Raises
    ------
    ValueError
        If the requested geometry cannot be satisfied.
    """
    if keep_geom == "mov":
        return (
            np.asarray(result.r2r, dtype=np.float64) @ np.asarray(mov_img.affine, dtype=np.float64),
            tuple(int(v) for v in mov_img.shape[:3]),
        )
    if keep_geom == "ref":
        if ref_img is None:
            raise ValueError("keep_geom='ref' requires a reference geometry image.")
        return np.asarray(ref_img.affine, dtype=np.float64), tuple(int(v) for v in ref_img.shape[:3])
    if keep_geom == "atlas":
        if result.target_affine is None or result.target_shape is None:
            raise ValueError("keep_geom='atlas' requires atlas target geometry.")
        return result.target_affine, result.target_shape
    raise ValueError(f"Unknown keep_geom '{keep_geom}'. Choose from: mov, ref, atlas.")
