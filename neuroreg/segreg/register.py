"""High-level segmentation-based registration APIs.

This layer ties together centroid extraction, bundled target resources, label
presets, and point-set solvers to expose one public ``segreg`` workflow that
returns a transform plus the metadata needed for LTA writing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..image import load_image
from ..transforms import matrix_sqrt_schur
from .atlas import affine_from_header, available_atlases, load_atlas_target
from .centroids import (
    ImageLike,
    build_flipped_centroid_targets,
    collect_joint_centroids,
    compute_ras_centroids_from_seg,
    compute_voxel_centroids_from_seg,
)
from .io import CentroidDict, GeometryDict, TargetFile, geometry_from_image, read_target_json, write_target_json
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
        a header-like dictionary loaded from a centroid target file, or ``None``
        when the target geometry is unknown.
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
    """Internal representation of target geometry metadata."""

    name: str
    geometry: Any | None
    affine: np.ndarray | None
    shape: tuple[int, int, int] | None


def _default_min_common_labels(dof: int) -> int:
    """Return the default correspondence count for a given transform family."""
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
    """Resolve the label IDs to evaluate for the selected registration mode."""
    if explicit_labels is not None:
        return [int(label) for label in explicit_labels]
    if label_set == "cortex_lr_pairs":
        return get_cortex_lr_labels()
    if label_set == "target_centroids":
        if target_centroids is None:
            raise ValueError("label_set='target_centroids' requires centroid targets.")
        return sorted(target_centroids.keys())
    if mode == "centroids" and target_centroids is not None:
        return sorted(target_centroids.keys())
    return None


def _geometry_from_image(image: Any, *, fallback_name: str) -> _GeometryInfo:
    """Build an internal geometry descriptor from an image object."""
    return _GeometryInfo(
        name=image.get_filename() or fallback_name,
        geometry=image,
        affine=np.asarray(image.affine, dtype=np.float64),
        shape=tuple(int(v) for v in image.shape[:3]),
    )


def _geometry_from_header(header: GeometryDict, *, name: str) -> _GeometryInfo:
    """Build an internal geometry descriptor from target JSON geometry metadata."""
    affine = np.asarray(affine_from_header(header), dtype=np.float64)
    return _GeometryInfo(
        name=name,
        geometry=header,
        affine=affine,
        shape=tuple(int(v) for v in header["dims"]),
    )


def _load_centroid_target(source: str | Path) -> tuple[str, TargetFile]:
    """Load a centroid target from either a JSON path or a bundled atlas name."""
    source_name = str(source)
    source_path = Path(source_name)
    if source_path.exists():
        return source_name, read_target_json(source_path)
    if source_name in available_atlases():
        return source_name, load_atlas_target(source_name)
    raise ValueError(
        f"Unknown centroid target '{source_name}'. Provide a JSON path or one of: {', '.join(available_atlases())}."
    )


def _resolve_target_centroids_and_geometry(
        *,
        target_seg: ImageLike | None,
        centroids: str | Path | None,
) -> tuple[str, CentroidDict, _GeometryInfo | None]:
    """Resolve target centroids and optional target geometry for registration."""
    modes = sum(value is not None for value in (target_seg, centroids))
    if modes != 1:
        raise ValueError("Choose exactly one registration target: target_seg or centroids.")

    if target_seg is not None:
        target_img = load_image(target_seg)
        target_centroid_dict = {
            label: point
            for label, point in compute_ras_centroids_from_seg(target_img).items()
            if point is not None
        }
        return "target_seg", target_centroid_dict, _geometry_from_image(target_img, fallback_name="target_seg.mgz")

    assert centroids is not None
    source_name, target = _load_centroid_target(centroids)
    geometry = None if target.geometry is None else _geometry_from_header(target.geometry, name=source_name)
    return "centroids", target.centroids, geometry


def segreg(
        seg: ImageLike,
        target_seg: ImageLike | None = None,
        *,
        centroids: str | Path | None = None,
        dof: int = 6,
        labels: list[int] | None = None,
        label_set: LabelSetName | None = None,
        min_common_labels: int | None = None,
        flipped: bool = False,
        midslice: float | None = None,
) -> RegistrationResult:
    """Register a moving segmentation to another target via label centroids.

    Parameters
    ----------
    seg : ImageLike
        Moving segmentation image. This may be a path or a nibabel-like image.
    target_seg : ImageLike or None, optional
        Target segmentation image for segmentation-to-segmentation registration.
    centroids : str or Path or None, optional
        Path to a centroid target JSON file or the name of a bundled centroid
        target such as ``"fsaverage"``.
    dof : {3, 6, 7, 9, 12}, default=6
        Degrees of freedom for the closed-form fit. ``3`` selects
        translation-only, ``6`` rigid, ``7`` rigid plus global scale, ``9``
        rigid plus anisotropic scaling without shear, and ``12`` affine
        registration.
    labels : list[int] or None, optional
        Explicit label subset override.
    label_set : {'all_shared', 'target_centroids', 'cortex_lr_pairs'} or None, optional
        Named label preset. Mode-specific defaults are used when omitted.
    min_common_labels : int or None, optional
        Minimum number of matched labels required to proceed. When omitted, the
        default is ``1`` for translation-only, ``3`` for rigid/similarity, and
        ``4`` for anisotropic-scale or affine registration.
    flipped : bool, default=False
        If ``True``, ignore external targets and register the moving
        segmentation to a left-right flipped self target for upright or midspace
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
    mov_img = load_image(seg)
    mov_name = mov_img.get_filename() or (str(seg) if isinstance(seg, (str, Path)) else "moving.mgz")
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
        if any(value is not None for value in (target_seg, centroids)):
            raise ValueError("--flipped cannot be combined with target_seg or centroids targets.")
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
        target_seg=target_seg,
        centroids=centroids,
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
            name=str(centroids),
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


def export_segmentation_target(
        seg: ImageLike,
        out_path: str | Path,
        *,
        geometry: ImageLike | None = None,
        labels: list[int] | None = None,
) -> None:
    """Compute segmentation centroids and write a centroid target JSON file.

    Parameters
    ----------
    seg : ImageLike
        Segmentation image or path used to compute scanner-RAS centroids.
    out_path : str or Path
        Output JSON path.
    geometry : ImageLike or None, optional
        Optional image or path whose geometry metadata should be embedded in the
        target file. When omitted, the segmentation geometry is embedded.
    labels : list[int] or None, optional
        Optional label subset to export. When omitted, all non-zero labels are
        written.

    Returns
    -------
    None
        Writes the selected centroid target payload to ``out_path``.
    """
    centroids_payload = compute_ras_centroids_from_seg(seg, label_ids=labels)
    geometry_source = seg if geometry is None else geometry
    write_target_json(out_path, centroids_payload, geometry=geometry_from_image(geometry_source))
