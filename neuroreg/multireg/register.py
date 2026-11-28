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

from ..image import cast_image_dtype, load_geometry_source, load_image
from ..image.map import coerce_image_data_3d
from ..imreg.init import InitType, resolve_init_type
from ..imreg.robreg import robreg
from ..transforms import LTA, affine_dist
from .geometry import (
    check_template_grid_covers_inputs,
    create_template_geometry,
    init_transform_template_geometry,
    mean_mapped_centroid,
    project_to_rotation,
    ras_center,
    rotation_mean,
    template_geometry_from_lta,
    validate_input_geometries,
)
from .template import build_ltas, build_template, resolve_average_mode

logger = logging.getLogger(__name__)

ImageLike = str | Path | Any
TransformLike = str | Path | LTA


@dataclass(slots=True)
class MultiRegResult:
    """Outputs of the FreeSurfer-style multi-timepoint registration pipeline.

    ``initial_target_index`` and ``seed`` are ``None`` when the pre-alignment,
    the only step that registers to a target time point, was skipped.
    """

    template_image: Any
    transforms_r2r: list[np.ndarray]
    ltas: list[LTA]
    initial_target_index: int | None
    seed: int | None
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


def _load_init_ltas(
    init_ltas: Sequence[TransformLike],
    *,
    read_destination: bool,
) -> tuple[list[np.ndarray], tuple[tuple[int, int, int], np.ndarray] | None]:
    """Load precomputed LTAs as RAS transforms, with their shared destination grid.

    Parameters
    ----------
    init_ltas : sequence of TransformLike
        Precomputed timepoint-to-template LTAs or paths to them.
    read_destination : bool
        Whether the destination grid is needed. It is not when a template
        geometry is supplied outright, and then the destination blocks are never
        read, so the transforms neither have to agree on a destination nor carry
        one at all. That is what makes a pose fitted by ``segreg`` against a
        centroid target without geometry usable, since such a transform has its
        destination block marked ``valid = 0``. The bundled targets do carry
        one, so those poses already name a destination.

    Returns
    -------
    transforms_r2r : list of numpy.ndarray
        RAS-to-RAS matrices for each input time point.
    destination : tuple or None
        ``(shape, affine)`` of the shared destination grid, or ``None`` when
        ``read_destination`` is ``False``.

    Raises
    ------
    ValueError
        If ``init_ltas`` is empty, or the destination is needed and the LTAs do
        not share an identical, valid one.
    """
    loaded_ltas = [LTA.read(transform) if isinstance(transform, str | Path) else transform for transform in init_ltas]
    if not loaded_ltas:
        raise ValueError("init_ltas must not be empty when provided.")
    transforms_r2r = [np.asarray(transform.r2r(), dtype=np.float64) for transform in loaded_ltas]
    if not read_destination:
        return transforms_r2r, None
    template_shape, template_affine = template_geometry_from_lta(loaded_ltas[0])
    for transform in loaded_ltas[1:]:
        current_shape, current_affine = template_geometry_from_lta(transform)
        if current_shape != template_shape or not np.allclose(current_affine, template_affine, atol=1e-6):
            raise ValueError("All init_ltas must share identical destination geometry.")
    return transforms_r2r, (template_shape, template_affine)


def _is_rigid(matrix: np.ndarray) -> bool:
    """Return whether a 4 x 4 RAS transform is a rotation plus translation."""
    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    return bool(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) and np.linalg.det(rotation) > 0)


def _resolve_iterations(
    template_iterations: int | None,
    n_images: int,
    *,
    prealign: bool,
) -> int:
    """Resolve the requested number of global template-refinement iterations.

    With two time points the pre-alignment is the registration: registering
    one to the other is symmetric, and the mean of two images is also their
    median, so both are represented equally and there is nothing for a global
    loop to improve. Two time points therefore never run one, and a request for
    it, or for registration without the pre-alignment, is an error rather than
    being quietly changed.

    Parameters
    ----------
    template_iterations : int or None
        User-requested maximum iteration count. ``None`` applies the defaults:
        6 for three or more time points, and none for two.
    n_images : int
        Number of input time points.
    prealign : bool
        Whether the pre-alignment runs.

    Returns
    -------
    int
        Effective number of refinement iterations to run.

    Raises
    ------
    ValueError
        If ``template_iterations`` is negative, or with two time points if it
        asks for iterations, or if the pre-alignment is skipped without asking
        for none (``0``), which only maps the inputs.
    """
    if template_iterations is not None and int(template_iterations) < 0:
        raise ValueError("template_iterations must be >= 0.")
    if n_images <= 2:
        if template_iterations:
            raise ValueError(
                "With two time points the pre-alignment is the registration: it is symmetric, and the mean "
                "of two images is their median, so there is no global loop to iterate. Leave "
                "template_iterations unset, or pass 0 together with prealign=False to only map the inputs."
            )
        if not prealign and template_iterations is None:
            raise ValueError(
                "With two time points the pre-alignment is the registration, so skipping it leaves nothing "
                "to register. Pass template_iterations=0 to only map the inputs (--map-only)."
            )
        return 0
    return 6 if template_iterations is None else int(template_iterations)


def _prealign(
    images: Sequence[Any],
    masks: Sequence[Any | None],
    *,
    target_index: int,
    init_r2r: Sequence[np.ndarray] | None,
    init_type: InitType,
    nmax: int,
    sat: float,
    symmetric: bool,
    device: str,
    fix_target: bool,
    verbose: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Register every time point to the target and build the initial common space.

    Without ``init_r2r`` this is FreeSurfer's opening round: each time point is
    registered to the target, and the transforms are composed with the mean of
    the target-to-each transforms, which places the template in the unbiased
    middle space (or, with ``fix_target``, they are kept in the target's space).

    With ``init_r2r`` (timepoint-to-template transforms ``X``) each registration
    starts from the relative pose ``X_target^-1 X_i``, and the result is
    expressed in the space the transforms define: time point ``i`` maps through
    the target's given transform, ``X_target P_i``. The average correction
    relative to the given transforms is then removed, so the template stays
    where they put it on average whichever time point was the target. With
    ``fix_target`` it is not removed, and the template sits where the target's
    own transform puts it. With identity transforms the composition reduces
    exactly to the case without ``init_r2r``; the registrations then differ only
    in starting from those poses rather than as ``init_type`` says.

    Parameters
    ----------
    images : sequence of Any
        Loaded input images.
    masks : sequence of Any or None
        Optional per-timepoint masks aligned with ``images``.
    target_index : int
        Index of the target time point.
    init_r2r : sequence of numpy.ndarray or None
        Optional initial timepoint-to-template RAS transforms.
    init_type : InitType
        Pairwise initialization mode forwarded to ``robreg`` when no initial
        transforms are given.
    nmax : int
        Maximum number of outer IRLS iterations per pairwise call.
    sat : float
        Tukey biweight saturation threshold.
    symmetric : bool
        Whether to use symmetric halfway-space pairwise registration.
    device : str
        Torch device string forwarded to ``robreg``.
    fix_target : bool
        If ``True``, keep the target's space instead of the average one.
    verbose : bool
        Whether to request verbose logging from the pairwise kernel.

    Returns
    -------
    transforms_r2r : list of numpy.ndarray
        Timepoint-to-template RAS transforms.
    pairwise_r2r : list of numpy.ndarray
        Timepoint-to-target RAS transforms from the pairwise registrations.
    mean_space_from_target_r2r : numpy.ndarray
        RAS transform from the target into the template space, as used to place
        a derived grid; the identity with ``fix_target``.

    Raises
    ------
    ValueError
        If a composed transform is measurably non-rigid.
    """
    identity = np.eye(4, dtype=np.float64)
    given = [identity] * len(images) if init_r2r is None else [np.asarray(m, dtype=np.float64) for m in init_r2r]
    target_given = given[target_index]
    target_given_inv = np.linalg.inv(target_given)
    pairwise_r2r: list[np.ndarray] = [identity.copy() for _ in images]
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
            init_transform=None if init_r2r is None else target_given_inv @ given[index],
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
        return [target_given @ matrix for matrix in pairwise_r2r], pairwise_r2r, identity

    # Build the unbiased space as FreeSurfer's MultiRegistration does: average
    # the translation parts arithmetically and the rotation parts via their
    # axis-angle rotation vectors (log-Euclidean mean on SO(3)). Averaging the
    # rotation matrices directly and SVD-projecting biases the template pose by
    # a small fixed rotation (~0.1 deg). Each term maps the template back to
    # itself through the target and time point i; without initial transforms it
    # is the target-to-each transform, as in FreeSurfer.
    corrections = [given[i] @ np.linalg.inv(pairwise_r2r[i]) @ target_given_inv for i in range(len(images))]
    mean_space_from_target_r2r = identity.copy()
    mean_space_from_target_r2r[:3, :3] = rotation_mean([matrix[:3, :3] for matrix in corrections])
    mean_space_from_target_r2r[:3, 3] = np.mean(np.stack([matrix[:3, 3] for matrix in corrections], axis=0), axis=0)
    final_r2r = []
    for matrix in pairwise_r2r:
        combined = mean_space_from_target_r2r @ (target_given @ matrix)
        projected = project_to_rotation(combined[:3, :3])
        error = np.linalg.norm(projected - combined[:3, :3], ord="fro")
        if error > 1e-4:
            raise ValueError(
                "Computed mean-space transform is not rigid after composition. "
                "Make sure all inputs share identical voxel sizes."
            )
        combined[:3, :3] = projected
        final_r2r.append(combined)
    return final_r2r, pairwise_r2r, mean_space_from_target_r2r


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
    template_geometry: ImageLike | None = None,
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
    prealign: bool | None = None,
    allow_diff_vox_size: bool = False,
    return_mapped: bool = False,
    mapped_keep_dtype: bool = False,
    verbose: bool = False,
) -> MultiRegResult:
    """Run the FreeSurfer-style multi-timepoint registration pipeline.

    The registration has two phases, as in FreeSurfer's ``mri_robust_template``:
    a pre-alignment, which registers every time point to one target time point
    and builds an unbiased initial space from the results, and a global loop,
    which registers every time point to the current template and rebuilds it.
    With two time points the pre-alignment alone is the registration.

    Parameters
    ----------
    movables : sequence of ImageLike
        Input time-point images or paths to them.
    masks : sequence of ImageLike or None, optional
        Optional per-timepoint masks aligned with ``movables``.
    init_ltas : sequence of TransformLike or None, optional
        Optional precomputed timepoint-to-template LTAs, for example poses to
        an atlas. The pre-alignment starts from them, and its result is
        expressed in the space they define, so the template keeps their pose.
        The template grid takes their shared destination grid, or only its
        orientation when its voxel size differs from the inputs' (see
        :func:`~neuroreg.multireg.geometry.init_transform_template_geometry`).
        With ``template_iterations=0`` and ``prealign`` unset they are used
        exactly as given, as FreeSurfer's ``--ixforms --noit``; only then may
        they scale or shear, since every registration here is rigid. When
        ``template_geometry`` is given the destination geometry is ignored and
        need not be present. Mutually exclusive with ``use_cras_center``.
    template_geometry : ImageLike or None, optional
        Optional image, path to one, or path to a JSON with a ``geometry``
        block (see :func:`neuroreg.image.load_geometry_source`) supplying the
        output template geometry, replacing the grid that would otherwise be
        taken from ``init_ltas``, the target (``fix_target``) or derived from the
        inputs. Only the shape and affine are used, not the voxel data. It only
        chooses the grid: the space the transforms land in is still set by the
        registration, ``init_ltas`` and ``fix_target``. Mutually exclusive with
        ``use_cras_center``, which only centers a derived grid.
    average : {"mean", "median", 0, 1}, default="median"
        Template aggregation mode. Integer aliases match the FreeSurfer CLI.
    init_target_index : int or None, optional
        Zero-based pre-alignment target index. If omitted, select one
        deterministically from image content. Needs the pre-alignment.
    seed : int or None, optional
        Seed override for initial target selection. ``None`` and ``0`` recompute
        the seed from image content.
    fix_target : bool, default=False
        If ``True``, keep the pre-alignment target's space instead of the
        unbiased average one: without ``init_ltas`` the template takes the
        target's own grid, with them it sits where the target's transform puts
        it. With ``template_geometry`` that space is sampled on the given grid
        instead. Needs the pre-alignment. Mutually exclusive with
        ``use_cras_center``, since no geometry is then derived.
    init_type : InitType or None, optional
        How each pre-alignment registration starts (default ``"centroid"``).
        Only applies without ``init_ltas`` and with the pre-alignment: the
        global loop always starts from the current transforms.
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
        the average mapped intensity centroid. Only selects how a derived
        geometry is centered, so it is mutually exclusive with ``init_ltas``,
        ``template_geometry`` and ``fix_target``, none of which derives one.
    template_iterations : int or None, optional
        Maximum number of global loop passes. ``None`` runs 6 for three or more
        time points. Two time points never run the global loop, because the
        pre-alignment already registers them symmetrically; asking for passes
        then is an error.
    template_eps : float, default=0.03
        Convergence threshold in millimeters for the maximum per-iteration
        transform change.
    prealign : bool or None, optional
        Whether to run the pre-alignment. ``None`` runs it, except with
        ``init_ltas`` and ``template_iterations=0``, which use the transforms as
        given. Skipping it registers only in the global loop, starting from
        ``init_ltas`` or from the image headers, which avoids choosing a target
        time point. Two time points have no global loop, so they can skip it
        only to map without registering (``template_iterations=0``).
    allow_diff_vox_size : bool, default=False
        Accept inputs with different voxel sizes, with a warning. By default
        they are rejected, as in FreeSurfer: the template takes the coarsest
        size, so finer time points would be downsampled and differ from the
        rest in noise and detail.
    return_mapped : bool, default=False
        If ``True``, include mapped images in the returned result.
    mapped_keep_dtype : bool, default=False
        If ``True``, preserve source dtypes for returned mapped images and cast
        the output template to the inputs' dtype (clipping cubic-interpolation
        over/undershoot instead of rescaling). All inputs must then share one
        dtype.
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
        If the inputs are invalid, incompatible in geometry, the requested
        iteration settings are inconsistent, mutually exclusive template-space
        options are combined, or a time point maps entirely outside the template
        grid.
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
            (load_image(mask) if isinstance(mask, str | Path) else mask) if mask is not None else None for mask in masks
        ]

    if init_ltas is not None and len(init_ltas) != len(images):
        raise ValueError("Number of init_ltas must match the number of input images.")
    # FreeSurfer's --ixforms --noit: transforms given and no iterations asked
    # for means they are final, so there is nothing to pre-align either.
    resolved_prealign = prealign if prealign is not None else not (init_ltas is not None and template_iterations == 0)
    if fix_target and not resolved_prealign:
        raise ValueError(
            "fix_target keeps the space of the pre-alignment target, so it needs the pre-alignment, which is "
            "skipped here."
        )
    if init_target_index is not None and not resolved_prealign:
        raise ValueError("init_target_index chooses the pre-alignment target, and the pre-alignment is skipped here.")
    if init_type is not None and (init_ltas is not None or not resolved_prealign):
        # Every global-loop registration starts from the previous transform, so
        # init_type only ever chooses how a pre-alignment registration starts.
        raise ValueError(
            "init_type chooses how each pre-alignment registration starts without init_ltas; with init_ltas "
            "they start from those poses, and without a pre-alignment there is nothing to start."
        )
    if mapped_keep_dtype:
        dtypes = sorted({np.dtype(image.get_data_dtype()).name for image in images})
        if len(dtypes) > 1:
            raise ValueError(
                f"mapped_keep_dtype keeps the inputs' dtype for the template, but they differ ({', '.join(dtypes)}); "
                "convert the time points to one dtype first."
            )
    # The template geometry is derived from the inputs only when nothing else
    # supplies one. Both the centering choice and the orientation-averaging
    # warning depend on that single fact, so it is settled once here.
    derives_geometry = init_ltas is None and template_geometry is None and not fix_target
    if use_cras_center and not derives_geometry:
        # use_cras_center picks between two ways of centering a grid that is
        # being derived. Each alternative supplies the grid, or with init_ltas
        # its orientation and placement, so there is nothing left to center.
        source = (
            "init_ltas"
            if init_ltas is not None
            else "template_geometry"
            if template_geometry is not None
            else "fix_target"
        )
        raise ValueError(
            f"use_cras_center only selects how a derived template geometry is centered, and {source} "
            "supplies the geometry instead of deriving one; pass only one."
        )

    external_geometry: tuple[tuple[int, int, int], np.ndarray] | None = None
    if template_geometry is not None:
        geom_image = load_geometry_source(template_geometry, label="template_geometry")
        if len(geom_image.shape) < 3:
            raise ValueError(
                "template_geometry must describe a 3-D grid, but the given image has shape "
                f"{tuple(geom_image.shape)}."
            )
        external_geometry = (
            tuple(int(v) for v in geom_image.shape[:3]),
            np.asarray(geom_image.affine, dtype=np.float64),
        )

    validate_input_geometries(images, derives_geometry=derives_geometry, allow_diff_vox_size=allow_diff_vox_size)
    resolved_init_type = resolve_init_type(init_type, default_init_type="centroid")
    resolved_template_iterations = _resolve_iterations(template_iterations, len(images), prealign=resolved_prealign)

    init_r2r: list[np.ndarray] | None = None
    destination: tuple[tuple[int, int, int], np.ndarray] | None = None
    if init_ltas is not None:
        init_r2r, destination = _load_init_ltas(init_ltas, read_destination=external_geometry is None)
        non_rigid = [str(index + 1) for index, matrix in enumerate(init_r2r) if not _is_rigid(matrix)]
        if non_rigid and (resolved_prealign or resolved_template_iterations > 0):
            # As FreeSurfer's mri_robust_template without --affine: the
            # registration is rigid, so it cannot start from, or keep, a
            # scaling or shear. Mapping alone applies them as given.
            raise ValueError(
                f"init_ltas for TP {', '.join(non_rigid)} are not rigid, but multireg registers rigidly. Use "
                "rigid (6 DOF) transforms, or only map with them (template_iterations=0 and prealign unset)."
            )

    # Only the pre-alignment registers to a target time point, so only then is
    # one chosen; choosing it reads every input's voxels.
    resolved_seed: int | None = None
    if resolved_prealign and init_target_index is None:
        init_target_index, resolved_seed = choose_initial_target(images, seed=seed)
    elif resolved_prealign:
        if not 0 <= init_target_index < len(images):
            raise ValueError(f"init_target_index must be in [0, {len(images) - 1}], got {init_target_index}.")
        resolved_seed = compute_seed(images) if seed in (None, 0) else int(seed)

    identity = np.eye(4, dtype=np.float64)
    if resolved_prealign:
        logger.info("Using TP %d as the initial target (seed=%d).", init_target_index + 1, resolved_seed)
        current_transforms, pairwise_r2r, mean_space_from_target_r2r = _prealign(
            images,
            loaded_masks,
            target_index=init_target_index,
            init_r2r=init_r2r,
            init_type=resolved_init_type,
            nmax=nmax,
            sat=sat,
            symmetric=symmetric,
            device=device,
            fix_target=fix_target,
            verbose=verbose,
        )
    else:
        logger.info("Skipping the pre-alignment; starting from %s.", "init_ltas" if init_r2r else "the image headers")
        current_transforms = [identity.copy() for _ in images] if init_r2r is None else init_r2r
        pairwise_r2r = [identity.copy() for _ in images]
        mean_space_from_target_r2r = identity

    if external_geometry is not None:
        # Deriving a grid would read every input volume to find the mapped
        # centroid and would average the input orientations, both of which can
        # fail on inputs that an explicit grid is meant to accommodate.
        template_shape, template_affine = external_geometry
    elif destination is not None:
        template_shape, template_affine = init_transform_template_geometry(images, current_transforms, destination)
    elif fix_target:
        target = images[init_target_index]
        template_shape = tuple(int(v) for v in target.shape[:3])
        template_affine = np.asarray(target.affine, dtype=np.float64)
    else:
        center_ras = (
            np.mean(np.stack([ras_center(image) for image in images], axis=0), axis=0)
            if use_cras_center
            else mean_mapped_centroid(
                images,
                pairwise_r2r,
                # Without a pre-alignment the transforms are identities, and any
                # image serves as the intermediate space.
                target_index=0 if init_target_index is None else init_target_index,
                mean_space_from_target_r2r=mean_space_from_target_r2r,
            )
        )
        template_shape, template_affine = create_template_geometry(images, center_ras)

    check_template_grid_covers_inputs(
        images,
        current_transforms,
        template_shape=template_shape,
        template_affine=template_affine,
    )

    # The target only lends its header type to the template; any input will do.
    target_image = images[0 if init_target_index is None else init_target_index]
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
                init_transform=previous_r2r,
            )
            if hasattr(updated_result, "detach"):
                updated_r2r = np.asarray(updated_result.detach().cpu().numpy(), dtype=np.float64)
            else:
                updated_r2r = np.asarray(updated_result, dtype=np.float64)
            projected = project_to_rotation(updated_r2r[:3, :3])
            projection_error = np.linalg.norm(projected - updated_r2r[:3, :3], ord="fro")
            if projection_error > 0.0:
                logger.debug(
                    "Projecting TP %d iterative update back to rigid rotation (error=%.6g).",
                    index + 1,
                    projection_error,
                )
            updated_r2r[:3, :3] = projected
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

    if mapped_keep_dtype:
        # All inputs share this dtype; that was checked above.
        current_template = cast_image_dtype(current_template, np.dtype(images[0].get_data_dtype()))

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
