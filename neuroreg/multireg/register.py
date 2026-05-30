"""Public orchestration for multi-timepoint registration."""

from __future__ import annotations

import ctypes
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..image import load_image
from ..image.map import coerce_image_data_3d
from ..imreg.init import InitType, resolve_init_type
from ..imreg.robreg import robreg
from ..transforms import LTA, affine_dist
from .geometry import (
    create_template_geometry,
    mean_mapped_centroid,
    project_to_rotation,
    ras_center,
    template_geometry_from_lta,
    validate_input_geometries,
)
from .template import build_ltas, build_template, resolve_average_mode

logger = logging.getLogger(__name__)

ImageLike = str | Path | Any
TransformLike = str | Path | LTA


@dataclass(slots=True)
class MultiRegResult:
    """Outputs of the FreeSurfer-style multi-timepoint registration pipeline."""

    template_image: Any
    transforms_r2r: list[np.ndarray]
    ltas: list[LTA]
    initial_target_index: int
    seed: int
    mapped_images: list[Any] | None = None
    template_iterations_run: int = 0
    iteration_distances: list[float] = field(default_factory=list)

def _c_rand_choice(start: int, end: int, seed: int) -> int:
    """Match FreeSurfer's libc-backed pseudo-random choice helper.

    Parameters
    ----------
    start : int
        Inclusive lower bound for the sampled integer.
    end : int
        Inclusive upper bound for the sampled integer.
    seed : int
        Seed passed to ``srand`` before sampling with ``rand``.

    Returns
    -------
    int
        Pseudo-random integer in the inclusive range ``[start, end]``.

    Raises
    ------
    OSError
        If the platform C runtime cannot be loaded.
    AttributeError
        If the loaded C runtime does not expose ``srand`` or ``rand``.
    """
    if os.name == "nt":
        libc = ctypes.CDLL("msvcrt")
    else:
        libc = ctypes.CDLL(None)
    libc.srand.argtypes = [ctypes.c_uint]
    libc.rand.restype = ctypes.c_int
    libc.srand(ctypes.c_uint(seed))
    return int(libc.rand() % (end - start + 1) + start)


def _resolve_init_ltas(
    init_ltas: Sequence[TransformLike],
) -> tuple[
    tuple[int, int, int],
    np.ndarray,
    list[np.ndarray],
]:
    """Resolve precomputed LTAs into template geometry and RAS transforms.

    Parameters
    ----------
    init_ltas : sequence of TransformLike
        Precomputed timepoint-to-template LTAs or paths to them.

    Returns
    -------
    template_shape : tuple of int
        Template image shape extracted from the destination geometry.
    template_affine : numpy.ndarray
        Template voxel-to-RAS affine extracted from the destination geometry.
    transforms_r2r : list of numpy.ndarray
        RAS-to-RAS matrices for each input time point.

    Raises
    ------
    ValueError
        If ``init_ltas`` is empty or the LTAs do not share identical destination
        geometry.
    """
    loaded_ltas = [LTA.read(transform) if isinstance(transform, str | Path) else transform for transform in init_ltas]
    if not loaded_ltas:
        raise ValueError("init_ltas must not be empty when provided.")
    template_shape, template_affine = template_geometry_from_lta(loaded_ltas[0])
    transforms_r2r = [loaded_ltas[0].r2r()]
    for transform in loaded_ltas[1:]:
        current_shape, current_affine = template_geometry_from_lta(transform)
        if current_shape != template_shape or not np.allclose(current_affine, template_affine, atol=1e-6):
            raise ValueError("All init_ltas must share identical destination geometry.")
        transforms_r2r.append(transform.r2r())
    return template_shape, template_affine, transforms_r2r


def _resolve_iterations(template_iterations: int | None, n_images: int) -> int:
    """Resolve the requested number of global template-refinement iterations.

    Parameters
    ----------
    template_iterations : int or None
        User-requested maximum iteration count. ``None`` applies the FreeSurfer-
        style defaults for two versus three-or-more time points.
    n_images : int
        Number of input time points.

    Returns
    -------
    int
        Effective number of refinement iterations to run.

    Raises
    ------
    ValueError
        If ``template_iterations`` is negative.
    """
    if template_iterations is None:
        return 0 if n_images <= 2 else 6
    resolved = int(template_iterations)
    if resolved < 0:
        raise ValueError("template_iterations must be >= 0.")
    if n_images <= 2 and resolved > 0:
        logger.info("Skipping iterative template refinement because only two time points were provided.")
        return 0
    return resolved


def _build_initial_space(
    images: Sequence[Any],
    masks: Sequence[Any | None],
    *,
    target_index: int,
    init_type: InitType,
    nmax: int,
    sat: float,
    symmetric: bool,
    device: str,
    use_cras_center: bool,
    fix_target: bool,
    verbose: bool,
) -> tuple[tuple[int, int, int], np.ndarray, list[np.ndarray]]:
    """Build the initial common space from pairwise registrations.

    Parameters
    ----------
    images : sequence of Any
        Loaded input images.
    masks : sequence of Any or None
        Optional per-timepoint masks aligned with ``images``.
    target_index : int
        Index of the initial target time point.
    init_type : InitType
        Pairwise initialization mode forwarded to ``robreg``.
    nmax : int
        Maximum number of outer IRLS iterations per pairwise call.
    sat : float
        Tukey biweight saturation threshold.
    symmetric : bool
        Whether to use symmetric halfway-space pairwise registration.
    device : str
        Torch device string forwarded to ``robreg``.
    use_cras_center : bool
        If ``True``, center the template on the average image CRAS instead of the
        average mapped intensity centroid.
    fix_target : bool
        If ``True``, keep the initial target geometry instead of constructing an
        unbiased mean-space grid.
    verbose : bool
        Whether to request verbose logging from the pairwise kernel.

    Returns
    -------
    template_shape : tuple of int
        Output template grid shape.
    template_affine : numpy.ndarray
        Output template voxel-to-RAS affine.
    transforms_r2r : list of numpy.ndarray
        Initial timepoint-to-template RAS transforms.

    Raises
    ------
    ValueError
        If the composed mean-space transform is measurably non-rigid.
    """
    pairwise_r2r: list[np.ndarray] = [np.eye(4, dtype=np.float64) for _ in images]
    target_image = images[target_index]
    target_mask = masks[target_index]
    for index, (image, mask) in enumerate(zip(images, masks, strict=False)):
        if index == target_index:
            continue
        logger.info("Registering TP %d to initial target TP %d.", index + 1, target_index + 1)
        pairwise_result = robreg(
            image,
            target_image,
            return_v2v=False,
            src_mask=mask,
            trg_mask=target_mask,
            init_type=init_type,
            nmax=nmax,
            sat=sat,
            symmetric=symmetric,
            isotropic=True,
            device=device,
            verbose=verbose,
        )
        if hasattr(pairwise_result, "detach"):
            pairwise_r2r[index] = np.asarray(pairwise_result.detach().cpu().numpy(), dtype=np.float64)
        else:
            pairwise_r2r[index] = np.asarray(pairwise_result, dtype=np.float64)
    if fix_target:
        return (
            tuple(int(v) for v in target_image.shape[:3]),
            np.asarray(target_image.affine, dtype=np.float64),
            pairwise_r2r,
        )

    target_to_each = [np.linalg.inv(matrix) for matrix in pairwise_r2r]
    mean_translation = np.mean(
        np.stack([matrix[:3, 3] for matrix in target_to_each], axis=0),
        axis=0,
    )
    mean_rotation = project_to_rotation(
        np.mean(
            np.stack([matrix[:3, :3] for matrix in target_to_each], axis=0),
            axis=0,
        )
    )
    mean_space_from_target_r2r = np.eye(4, dtype=np.float64)
    mean_space_from_target_r2r[:3, :3] = mean_rotation
    mean_space_from_target_r2r[:3, 3] = mean_translation
    center_ras = (
        np.mean(np.stack([ras_center(image) for image in images], axis=0), axis=0)
        if use_cras_center
        else mean_mapped_centroid(
            images,
            pairwise_r2r,
            target_index=target_index,
            mean_space_from_target_r2r=mean_space_from_target_r2r,
        )
    )
    template_shape, template_affine = create_template_geometry(images, center_ras)
    final_r2r = []
    for matrix in pairwise_r2r:
        combined = mean_space_from_target_r2r @ matrix
        projected = project_to_rotation(combined[:3, :3])
        error = np.linalg.norm(projected - combined[:3, :3], ord="fro")
        if error > 1e-4:
            raise ValueError(
                "Computed mean-space transform is not rigid after composition. "
                "Make sure all inputs share identical voxel sizes."
            )
        combined[:3, :3] = projected
        final_r2r.append(combined)
    return template_shape, template_affine, final_r2r


def compute_seed(movables: Sequence[ImageLike]) -> int:
    """Compute the deterministic FreeSurfer-style seed from input intensities.

    Parameters
    ----------
    movables : sequence of ImageLike
        Input time-point images or paths to them.

    Returns
    -------
    int
        Integer seed computed from center-line intensity samples.

    Raises
    ------
    ValueError
        If fewer than two input images are provided.
    """
    if len(movables) < 2:
        raise ValueError("multireg requires at least two input images.")
    images = [load_image(image) if isinstance(image, str | Path) else image for image in movables]
    dseed = 0.0
    for image in images:
        data = coerce_image_data_3d(np.asarray(image.dataobj), name="movable image")
        x = data.shape[0] // 2
        y = data.shape[1] // 2
        z = data.shape[2] // 2
        for p in range(20):
            xup = x + p
            xdown = x - p
            yup = y + p
            ydown = y - p
            zup = z + p
            zdown = z - p
            if xdown >= 0:
                dseed += abs(float(data[xdown, y, z]))
            if ydown >= 0:
                dseed += abs(float(data[x, ydown, z]))
            if zdown >= 0:
                dseed += abs(float(data[x, y, zdown]))
            if xup < data.shape[0]:
                dseed += abs(float(data[xup, y, z]))
            if yup < data.shape[1]:
                dseed += abs(float(data[x, yup, z]))
            if zup < data.shape[2]:
                dseed += abs(float(data[x, y, zup]))
    while dseed != 0.0 and dseed < 10.0:
        dseed *= 10.0
    return int(dseed)


def choose_initial_target(movables: Sequence[ImageLike], seed: int | None = None) -> tuple[int, int]:
    """Choose the deterministic initial target time point.

    Parameters
    ----------
    movables : sequence of ImageLike
        Input time-point images or paths to them.
    seed : int or None, optional
        Seed override. ``None`` and ``0`` recompute the seed from image content.

    Returns
    -------
    target_index : int
        Zero-based index of the chosen initial target.
    resolved_seed : int
        Effective seed used for the pseudo-random choice.

    Raises
    ------
    ValueError
        If fewer than two input images are provided.
    """
    if len(movables) < 2:
        raise ValueError("multireg requires at least two input images.")
    resolved_seed = compute_seed(movables) if seed in (None, 0) else int(seed)
    try:
        target = _c_rand_choice(1, len(movables), resolved_seed) - 1
    except Exception:
        target = int(np.random.RandomState(resolved_seed).randint(0, len(movables)))
    return target, resolved_seed


def multireg(
    movables: Sequence[ImageLike],
    *,
    masks: Sequence[ImageLike | None] | None = None,
    init_ltas: Sequence[TransformLike] | None = None,
    average: str | int = "median",
    init_target_index: int | None = None,
    seed: int | None = None,
    fix_target: bool = False,
    init_type: InitType | None = None,
    nmax: int = 5,
    sat: float = 6.0,
    symmetric: bool = True,
    device: str = "gpu",
    use_cras_center: bool = False,
    template_iterations: int | None = None,
    template_eps: float = 0.03,
    return_mapped: bool = False,
    mapped_keep_dtype: bool = False,
    verbose: bool = False,
) -> MultiRegResult:
    """Run the FreeSurfer-style multi-timepoint registration pipeline.

    Parameters
    ----------
    movables : sequence of ImageLike
        Input time-point images or paths to them.
    masks : sequence of ImageLike or None, optional
        Optional per-timepoint masks aligned with ``movables``.
    init_ltas : sequence of TransformLike or None, optional
        Optional precomputed LTAs defining the initial timepoint-to-template
        mappings and template geometry.
    average : {"mean", "median", 0, 1}, default="median"
        Template aggregation mode. Integer aliases match the FreeSurfer CLI.
    init_target_index : int or None, optional
        Zero-based initial target index. If omitted, select one deterministically
        from image content.
    seed : int or None, optional
        Seed override for initial target selection. ``None`` and ``0`` recompute
        the seed from image content.
    fix_target : bool, default=False
        If ``True``, keep the initial target geometry instead of constructing an
        unbiased mean-space template grid.
    init_type : InitType or None, optional
        Pairwise registration initialization mode.
    nmax : int, default=5
        Maximum number of outer IRLS iterations per pairwise registration.
    sat : float, default=6.0
        Tukey biweight saturation threshold for robust pairwise registration.
    symmetric : bool, default=True
        Whether pairwise registrations should use symmetric halfway-space updates.
    device : str, default="gpu"
        Torch device string forwarded to the pairwise kernel.
    use_cras_center : bool, default=False
        If ``True``, center the template geometry on the average CRAS instead of
        the average mapped intensity centroid.
    template_iterations : int or None, optional
        Maximum number of global template-refinement passes. ``None`` uses the
        built-in defaults for two versus three-or-more time points.
    template_eps : float, default=0.03
        Convergence threshold in millimeters for the maximum per-iteration
        transform change.
    return_mapped : bool, default=False
        If ``True``, include mapped images in the returned result.
    mapped_keep_dtype : bool, default=False
        If ``True``, preserve source dtypes for returned mapped images.
    verbose : bool, default=False
        Whether to enable verbose pairwise-registration logging.

    Returns
    -------
    MultiRegResult
        Template image, final transforms, LTAs, iteration metadata, and optional
        mapped images.

    Raises
    ------
    ValueError
        If the inputs are invalid, incompatible in geometry, or the requested
        iteration settings are inconsistent.
    """
    if len(movables) < 2:
        raise ValueError("multireg requires at least two input images.")
    if template_eps <= 0.0:
        raise ValueError("template_eps must be > 0.")

    images = [load_image(image) if isinstance(image, str | Path) else image for image in movables]
    source_names = [str(image) if isinstance(image, str | Path) else "" for image in movables]
    resolved_average = resolve_average_mode(average)

    if masks is None:
        loaded_masks: list[Any | None] = [None] * len(images)
    else:
        if len(masks) != len(images):
            raise ValueError("Number of masks must match the number of input images.")
        loaded_masks = [
            (load_image(mask) if isinstance(mask, str | Path) else mask) if mask is not None else None
            for mask in masks
        ]

    if init_ltas is not None and len(init_ltas) != len(images):
        raise ValueError("Number of init_ltas must match the number of input images.")

    validate_input_geometries(images)
    resolved_init_type = resolve_init_type(init_type, default_init_type="centroid")
    resolved_template_iterations = _resolve_iterations(template_iterations, len(images))

    if init_target_index is None:
        init_target_index, resolved_seed = choose_initial_target(images, seed=seed)
    else:
        if not 0 <= init_target_index < len(images):
            raise ValueError(f"init_target_index must be in [0, {len(images) - 1}], got {init_target_index}.")
        resolved_seed = compute_seed(images) if seed in (None, 0) else int(seed)
    logger.info("Using TP %d as the initial target (seed=%d).", init_target_index + 1, resolved_seed)

    if init_ltas is None:
        template_shape, template_affine, current_transforms = _build_initial_space(
            images,
            loaded_masks,
            target_index=init_target_index,
            init_type=resolved_init_type,
            nmax=nmax,
            sat=sat,
            symmetric=symmetric,
            device=device,
            use_cras_center=use_cras_center,
            fix_target=fix_target,
            verbose=verbose,
        )
    else:
        if fix_target:
            logger.info("Ignoring fix_target because init_ltas already define the template geometry.")
        template_shape, template_affine, current_transforms = _resolve_init_ltas(init_ltas)

    target_image = images[init_target_index]
    current_template, mapped_images = build_template(
        images,
        current_transforms,
        average=resolved_average,
        template_affine=template_affine,
        template_shape=template_shape,
        template_like=target_image,
        return_mapped=return_mapped and resolved_template_iterations == 0,
        mapped_keep_dtype=mapped_keep_dtype,
    )

    iteration_distances: list[float] = []
    for iteration in range(1, resolved_template_iterations + 1):
        logger.info("Refining multireg template: iteration %d/%d.", iteration, resolved_template_iterations)
        next_transforms: list[np.ndarray] = []
        distances: list[float] = []
        for index, (image, mask, previous_r2r) in enumerate(
            zip(images, loaded_masks, current_transforms, strict=False)
        ):
            logger.info("Registering TP %d to current template.", index + 1)
            updated_result = robreg(
                image,
                current_template,
                return_v2v=False,
                src_mask=mask,
                trg_mask=None,
                init_type=resolved_init_type,
                nmax=nmax,
                sat=sat,
                symmetric=symmetric,
                isotropic=True,
                device=device,
                verbose=verbose,
                initial_r2r=previous_r2r,
            )
            if hasattr(updated_result, "detach"):
                updated_r2r = np.asarray(updated_result.detach().cpu().numpy(), dtype=np.float64)
            else:
                updated_r2r = np.asarray(updated_result, dtype=np.float64)
            next_transforms.append(updated_r2r)
            distances.append(float(affine_dist(previous_r2r, updated_r2r, radius=100.0)))
        current_transforms = next_transforms
        current_template, _ = build_template(
            images,
            current_transforms,
            average=resolved_average,
            template_affine=np.asarray(current_template.affine, dtype=np.float64),
            template_shape=tuple(int(v) for v in current_template.shape[:3]),
            template_like=current_template,
            return_mapped=False,
            mapped_keep_dtype=False,
        )
        max_change = max(distances, default=0.0)
        iteration_distances.append(max_change)
        logger.info("Template iteration %d max change: %.6f mm", iteration, max_change)
        if max_change <= template_eps:
            break

    if return_mapped and resolved_template_iterations > 0:
        current_template, mapped_images = build_template(
            images,
            current_transforms,
            average=resolved_average,
            template_affine=np.asarray(current_template.affine, dtype=np.float64),
            template_shape=tuple(int(v) for v in current_template.shape[:3]),
            template_like=current_template,
            return_mapped=True,
            mapped_keep_dtype=mapped_keep_dtype,
        )

    ltas = build_ltas(current_transforms, source_names, images, current_template)
    return MultiRegResult(
        template_image=current_template,
        transforms_r2r=current_transforms,
        ltas=ltas,
        initial_target_index=init_target_index,
        seed=resolved_seed,
        mapped_images=mapped_images,
        template_iterations_run=len(iteration_distances),
        iteration_distances=iteration_distances,
    )


__all__ = [
    "ImageLike",
    "MultiRegResult",
    "TransformLike",
    "choose_initial_target",
    "compute_seed",
    "multireg",
]
