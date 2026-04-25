"""Public IRLS-backed robust image-registration API."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import torch
from torch import Tensor

import logging
from .device import resolve_torch_device
from .init import InitType, get_init_vox2vox, resolve_init_type
from .irls import move_tensor, register_irls
from ..image import build_gaussian_pyramid, get_pyramid_limits, load_image
from ..image.map import coerce_image_data_3d, resample_isotropic_tensor
from ..image.masking import as_mask_tensor_and_affine, build_binary_mask_pyramid
from ..transforms import LINEAR_RAS_TO_RAS, LINEAR_VOX_TO_VOX, LTA, convert_transform_type

ImageLike = str | Path | Any | Tensor

logger = logging.getLogger(__name__)


def _resolve_robreg_device(device: str | torch.device) -> torch.device:
    """Resolve the requested robreg device, warning on unsupported MPS."""
    resolved = resolve_torch_device(device)
    if resolved.type == "mps":
        warnings.warn(
            "IRLS robreg does not support MPS due to lack of float64; falling back to CPU. "
            "Keep using the device argument for CPU/CUDA selection.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    return resolved


def _as_tensor_and_affine(
        image: ImageLike,
        affine: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Convert a supported image input into tensor data and a voxel-to-RAS affine.

    Parameters
    ----------
    image : ImageLike
        Input image specification. This may be a filesystem path, a nibabel-like
        image object exposing ``get_fdata()`` and ``affine``, or a pre-loaded
        ``torch.Tensor`` volume.
    affine : Tensor, optional
        Explicit affine to use when ``image`` is already a tensor. If omitted,
        an identity affine is assumed.

    Returns
    -------
    data : Tensor
        Image data as a float32 tensor.
    image_affine : Tensor
        Corresponding voxel-to-RAS affine as a float32 tensor.

    Raises
    ------
    TypeError
        If ``image`` is not one of the supported input types.
    """
    if isinstance(image, str | Path):
        img = cast(Any, load_image(image))
        data = torch.from_numpy(coerce_image_data_3d(img.get_fdata(), name="moving image")).float()
        return data, torch.from_numpy(img.affine).float()

    if hasattr(image, "get_fdata") and hasattr(image, "affine"):
        data = torch.from_numpy(coerce_image_data_3d(image.get_fdata(), name="image")).float()
        return data, torch.from_numpy(image.affine).float()

    if isinstance(image, torch.Tensor):
        return image.float(), (affine.float() if affine is not None else torch.eye(4, dtype=torch.float32))

    raise TypeError(f"Unsupported image type: {type(image)!r}")


def _save_outlier_map(all_info: list[dict[str, Any]], outliers_name: str, verbose: bool = False) -> None:
    """Write the final IRLS outlier volume to disk.

    Parameters
    ----------
    all_info : list of dict
        Per-level information dictionaries returned by the pyramid registration
        loop. The final entry is used.
    outliers_name : str
        Output filename. ``.nii`` and ``.nii.gz`` produce NIfTI output;
        other suffixes default to MGH/MGZ.
    verbose : bool, default=False
        If ``True``, emit logging about the saved outlier statistics.
    """
    if not all_info:
        logger.warning("Cannot save outlier map: no pyramid levels were executed")
        return

    final_info = all_info[-1]
    if "weights" not in final_info or "valid_mask" not in final_info:
        logger.warning("Cannot save outlier map: no weights in final level")
        return

    weights_sqrt = final_info["weights"]
    valid_mask = final_info["valid_mask"]
    if weights_sqrt is None or valid_mask is None:
        logger.warning("Cannot save outlier map: final IRLS level did not produce usable weights")
        return

    reg_affine = final_info.get("iso_affine")
    if reg_affine is None:
        logger.warning("Cannot save outlier map: no affine available")
        return

    reg_shape = final_info["image_shape"]
    if torch.is_tensor(reg_affine):
        reg_affine = reg_affine.detach().cpu().numpy()

    weights = weights_sqrt ** 2
    weight_volume = torch.zeros(reg_shape, dtype=torch.float32, device=weights.device)
    weight_volume.view(-1)[valid_mask] = weights
    outlier_volume = (1.0 - weight_volume).detach().cpu()

    if outliers_name.endswith(".nii") or outliers_name.endswith(".nii.gz"):
        outlier_img = nib.Nifti1Image(outlier_volume.numpy(), reg_affine)
    else:
        outlier_img = nib.MGHImage(outlier_volume.numpy(), reg_affine)

    outlier_img.to_filename(outliers_name)

    if verbose:
        outlier_pct = (outlier_volume > 0.5).sum().item() / outlier_volume.numel() * 100
        logger.info("Saved outlier map: %s (%.1f%% high outliers)", outliers_name, outlier_pct)


def register_irls_pyramid(
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor | None = None,
        trg_mask: Tensor | None = None,
        src_affine: Tensor | None = None,
        trg_affine: Tensor | None = None,
        initial_transform: Tensor | None = None,
        init_type: InitType = "centroid",
        min_voxels: int = 16,
        max_voxels: int | None = None,
        nmax: int = 5,
        sat: float = 6.0,
        epsit: float = 0.01,
        max_irls: int = 20,
        isotropic: bool = True,
        symmetric: bool = True,
        adaptive_sat: bool = False,
        target_outlier_pct: float = 5.0,
        outliers_name: str | None = None,
        verbose: bool = False,
) -> tuple[Tensor, list[dict[str, Any]]]:
    """Run the tensor-level IRLS pyramid registration pipeline.

    This is the high-level multiresolution orchestration used by the public
    :func:`robreg` API. It keeps the solver logic in
    :func:`neuroreg.imreg.irls.register_irls` but owns the product-level concerns:
    isotropic preprocessing, pyramid scheduling, level-to-level transform
    propagation, and optional outlier-map writing.

    Parameters
    ----------
    src, trg : Tensor
        Full-resolution source and target image tensors in ``(D, H, W)`` order.
    src_mask, trg_mask : Tensor, optional
        Optional binary masks in source and target space. Masked-out voxels are
        excluded from the IRLS system instead of being treated as zero-valued data.
    src_affine, trg_affine : Tensor, optional
        Voxel-to-RAS affines. Required when ``isotropic=True``.
    initial_transform : Tensor, optional
        Initial voxel-to-voxel transform. If provided, it takes precedence over
        the requested initialization mode.
    init_type : {"header", "centroid", "image_center"}, default="centroid"
        Explicit initialization mode used when ``initial_transform`` is not
        provided. ``"image_center"`` matches FreeSurfer's cras0-style center start.
    min_voxels : int, default=16
        Minimum size constraint passed to the shared pyramid builder.
    max_voxels : int, optional
        Maximum allowed size of the finest pyramid level to process. When
        ``None`` (default), include the original/full-resolution level.
    nmax : int, default=5
        Maximum number of outer IRLS iterations per pyramid level.
    sat : float, default=6.0
        Tukey biweight saturation threshold.
    epsit : float, default=0.01
        Convergence threshold for the per-level affine update distance.
    max_irls : int, default=20
        Maximum number of inner IRLS iterations per outer step.
    isotropic : bool, default=True
        If ``True``, resample both images to a shared isotropic grid before
        registration.
    symmetric : bool, default=True
        If ``True``, use symmetric (midspace) mode.
    adaptive_sat : bool, default=False
        Whether to adapt the Tukey saturation threshold based on the outlier
        fraction.
    target_outlier_pct : float, default=5.0
        Target outlier fraction used when ``adaptive_sat`` is enabled.
    outliers_name : str, optional
        Output filename for the final outlier map.
    verbose : bool, default=False
        If ``True``, emit progress logging.

    Returns
    -------
    T : Tensor
        Final voxel-to-voxel transform in the original image space.
    all_info : list of dict
        Per-level information dictionaries, finest level last.

    Raises
    ------
    ValueError
        If isotropic registration is requested without both affines, or if no
        pyramid level satisfies ``min_voxels``.
    """
    if src.device.type == "mps" or trg.device.type == "mps":
        warnings.warn(
            "IRLS robreg does not support MPS due to lack of float64; falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        src = src.to(device="cpu")
        trg = trg.to(device="cpu")
        if src_mask is not None:
            src_mask = src_mask.to(device="cpu")
        if trg_mask is not None:
            trg_mask = trg_mask.to(device="cpu")
        if src_affine is not None:
            src_affine = src_affine.to(device="cpu")
        if trg_affine is not None:
            trg_affine = trg_affine.to(device="cpu")
        if initial_transform is not None:
            initial_transform = initial_transform.to(device="cpu")

    resolved_init_type = resolve_init_type(init_type=init_type, default_init_type="centroid")

    if isotropic:
        if src_affine is None or trg_affine is None:
            raise ValueError("src_affine and trg_affine required when isotropic=True")

        src_affine_np = src_affine.detach().cpu().numpy()
        trg_affine_np = trg_affine.detach().cpu().numpy()
        src_zooms = np.linalg.norm(src_affine_np[:3, :3], axis=0)
        trg_zooms = np.linalg.norm(trg_affine_np[:3, :3], axis=0)
        isosize = float(max(src_zooms.min(), trg_zooms.min()))

        if verbose:
            logger.info("Isotropic resampling: isosize=%.4f mm", isosize)

        src_iso, src_iso_aff, Rsrc = resample_isotropic_tensor(src, src_affine_np, isosize, mode="linear")
        trg_iso, trg_iso_aff, Rtrg = resample_isotropic_tensor(trg, trg_affine_np, isosize, mode="linear")
        src_mask_iso = None
        trg_mask_iso = None
        if src_mask is not None:
            src_mask_iso, _, _ = resample_isotropic_tensor(
                (src_mask > 0).float(),
                src_affine_np,
                isosize,
                out_shape=tuple(int(v) for v in src_iso.shape),
                mode="nearest",
            )
            src_mask_iso = (src_mask_iso > 0.5).float()
        if trg_mask is not None:
            trg_mask_iso, _, _ = resample_isotropic_tensor(
                (trg_mask > 0).float(),
                trg_affine_np,
                isosize,
                out_shape=tuple(int(v) for v in trg_iso.shape),
                mode="nearest",
            )
            trg_mask_iso = (trg_mask_iso > 0.5).float()

        if verbose:
            logger.info("  Src resampled: %s → %s", src.shape, src_iso.shape)
            logger.info("  Trg resampled: %s → %s", trg.shape, trg_iso.shape)

        if initial_transform is not None:
            T_iso = (
                    move_tensor(Rtrg, device=src.device, dtype=src.dtype)
                    @ move_tensor(initial_transform, device=src.device, dtype=src.dtype)
                    @ torch.inverse(move_tensor(Rsrc, device=src.device, dtype=src.dtype))
            )
        else:
            T_iso = move_tensor(
                get_init_vox2vox(
                    src_iso.float(),
                    trg_iso.float(),
                    saffine=src_iso_aff,
                    taffine=trg_iso_aff,
                    init_type=resolved_init_type,
                ),
                device=src.device,
                dtype=src.dtype,
            )
            if verbose:
                t = T_iso[:3, 3].tolist()
                logger.info(
                    "%s initialization (isotropic space): [%.6f, %.6f, %.6f]",
                    resolved_init_type,
                    t[0],
                    t[1],
                    t[2],
                )

        shared_limits = get_pyramid_limits(src_iso.shape, trg_iso.shape, minsize=min_voxels, maxsize=max_voxels)
        pyramid_src, _ = build_gaussian_pyramid(src_iso, src_iso_aff, limits=shared_limits)
        pyramid_trg, _ = build_gaussian_pyramid(trg_iso, trg_iso_aff, limits=shared_limits)
        src_mask_levels = (
            build_binary_mask_pyramid(src_mask_iso, [tuple(int(v) for v in level.shape) for level in pyramid_src])
            if src_mask_iso is not None
            else None
        )
        trg_mask_levels = (
            build_binary_mask_pyramid(trg_mask_iso, [tuple(int(v) for v in level.shape) for level in pyramid_trg])
            if trg_mask_iso is not None
            else None
        )
        iso_affine = trg_iso_aff
    else:
        src_affine_for_pyramid = (
            src_affine if src_affine is not None else torch.eye(4, dtype=src.dtype, device=src.device)
        )
        trg_affine_for_pyramid = (
            trg_affine if trg_affine is not None else torch.eye(4, dtype=trg.dtype, device=trg.device)
        )
        shared_limits = get_pyramid_limits(src.shape, trg.shape, minsize=min_voxels, maxsize=max_voxels)
        pyramid_src, _ = build_gaussian_pyramid(src, src_affine_for_pyramid, limits=shared_limits)
        pyramid_trg, _ = build_gaussian_pyramid(trg, trg_affine_for_pyramid, limits=shared_limits)
        src_mask_levels = (
            build_binary_mask_pyramid(
                (src_mask > 0).float(),
                [tuple(int(v) for v in level.shape) for level in pyramid_src],
            )
            if src_mask is not None
            else None
        )
        trg_mask_levels = (
            build_binary_mask_pyramid(
                (trg_mask > 0).float(),
                [tuple(int(v) for v in level.shape) for level in pyramid_trg],
            )
            if trg_mask is not None
            else None
        )
        if initial_transform is not None:
            T_iso = move_tensor(initial_transform, device=src.device, dtype=src.dtype)
        else:
            T_iso = move_tensor(
                get_init_vox2vox(
                    src.float(),
                    trg.float(),
                    saffine=src_affine_for_pyramid,
                    taffine=trg_affine_for_pyramid,
                    init_type=resolved_init_type,
                ),
                device=src.device,
                dtype=src.dtype,
            )
            if verbose:
                t = T_iso[:3, 3].tolist()
                logger.info("%s initialization: [%.6f, %.6f, %.6f]", resolved_init_type, t[0], t[1], t[2])
        Rsrc = torch.eye(4, dtype=torch.float32)
        Rtrg = torch.eye(4, dtype=torch.float32)
        iso_affine = trg_affine.detach().cpu().numpy() if trg_affine is not None else None

    if not pyramid_src or not pyramid_trg:
        raise ValueError(
            "Pyramid construction returned no levels. "
            f"Source levels: {[tuple(level.shape) for level in pyramid_src]}; "
            f"Target levels: {[tuple(level.shape) for level in pyramid_trg]}"
        )

    T = T_iso if T_iso is not None else torch.eye(4, dtype=src.dtype, device=src.device)
    all_info: list[dict[str, Any]] = []

    for lvl in range(len(pyramid_src) - 1, -1, -1):
        s = pyramid_src[lvl].float()
        t = pyramid_trg[lvl].float()
        sm = src_mask_levels[lvl].float() if src_mask_levels is not None else None
        tm = trg_mask_levels[lvl].float() if trg_mask_levels is not None else None
        scale = float(2 ** lvl)

        T_lvl = T.clone()
        T_lvl[:3, 3] = T[:3, 3] / scale

        if verbose:
            logger.info("Pyramid level %d  shape=%s  (scale ×1/%d)", lvl, list(s.shape), int(scale))

        T_lvl, info = register_irls(
            s,
            t,
            src_mask=sm,
            trg_mask=tm,
            initial_transform=T_lvl,
            nmax=nmax,
            sat=sat,
            epsit=epsit,
            max_irls=max_irls,
            symmetric=symmetric,
            adaptive_sat=adaptive_sat,
            target_outlier_pct=target_outlier_pct,
            verbose=verbose,
        )

        T_lvl[:3, 3] = T_lvl[:3, 3] * scale
        T = T_lvl
        info["iso_affine"] = iso_affine
        all_info.append(info)

    if isotropic:
        T = (
                Rtrg.to(device=T.device, dtype=T.dtype)
                @ T
                @ torch.inverse(Rsrc.to(device=T.device, dtype=T.dtype))
        )

    if outliers_name is not None:
        _save_outlier_map(all_info, outliers_name, verbose=verbose)

    return T, all_info


def robreg(
        src: ImageLike,
        trg: ImageLike,
        *,
        src_affine: Tensor | None = None,
        trg_affine: Tensor | None = None,
        src_mask: ImageLike | None = None,
        trg_mask: ImageLike | None = None,
        return_v2v: bool = False,
        init_type: InitType = "centroid",
        init_lta: str | None = None,
        dof: int = 6,
        nmax: int = 5,
        sat: float = 6.0,
        symmetric: bool = True,
        isotropic: bool = True,
        adaptive_sat: bool = False,
        target_outlier_pct: float = 5.0,
        outliers_name: str | None = None,
        verbose: bool = False,
        device: str = "cpu",
) -> Tensor:
    """Register two images with the public IRLS robust-registration path.

    Parameters are intentionally close to the tensor-level IRLS pyramid
    implementation, but this wrapper also accepts filenames and nibabel images.

    Parameters
    ----------
    src, trg : ImageLike
        Moving/source and fixed/target images. Each input may be a path, a
        nibabel-like image object, or a ``torch.Tensor`` volume.
    src_affine, trg_affine : Tensor, optional
        Explicit voxel-to-RAS affines to use when ``src`` or ``trg`` are passed
        as tensors.
    src_mask, trg_mask : ImageLike, optional
        Optional source and target masks. Voxels outside these masks are
        ignored during IRLS fitting.
    return_v2v : bool, default=False
        If ``True``, return the estimated transform in voxel coordinates. If
        ``False``, return the corresponding RAS-to-RAS transform.
    init_type : {"header", "centroid", "image_center"}, default="centroid"
        Explicit initialization mode used when no ``init_lta`` is supplied.
        ``"image_center"`` matches FreeSurfer's cras0-style center start.
    init_lta : str, optional
        Existing LTA used for initialization. When provided, it overrides the
        requested ``init_type``.
    dof : int, default=6
        Degrees of freedom. The public IRLS path currently supports rigid
        registration only, so this must remain ``6``.
    nmax : int, default=5
        Maximum number of IRLS outer iterations per pyramid level.
    sat : float, default=6.0
        Tukey biweight saturation threshold.
    symmetric : bool, default=True
        If ``True``, run symmetric halfway-space registration. This is the
        default/public robreg behavior.
    isotropic : bool, default=True
        If ``True``, resample to isotropic voxels before building the pyramid.
    adaptive_sat : bool, default=False
        Whether to adapt the Tukey saturation threshold based on the observed
        outlier fraction.
    target_outlier_pct : float, default=5.0
        Target outlier percentage used when ``adaptive_sat`` is enabled.
    outliers_name : str, optional
        Output filename for the final outlier map.
    verbose : bool, default=False
        If ``True``, emit progress logging from the IRLS implementation.
    device : str, default="cpu"
        Torch device on which to place the image tensors before registration.

    Returns
    -------
    Tensor
        Estimated transform matrix. This is voxel-to-voxel when
        ``return_v2v=True`` and RAS-to-RAS otherwise.

    Raises
    ------
    ValueError
        If ``dof`` is anything other than ``6``.
    """
    if dof != 6:
        raise ValueError("IRLS robreg currently supports rigid registration only (dof=6).")

    src_data, src_aff = _as_tensor_and_affine(src, src_affine)
    trg_data, trg_aff = _as_tensor_and_affine(trg, trg_affine)
    src_mask_data, _ = as_mask_tensor_and_affine(src_mask, affine=src_affine, name="moving mask")
    trg_mask_data, _ = as_mask_tensor_and_affine(trg_mask, affine=trg_affine, name="reference mask")
    initial_transform = None
    if init_lta is not None:
        logger.info("Loading init transform from LTA: %s", init_lta)
        initial_transform = torch.from_numpy(
            convert_transform_type(
                LTA.read(init_lta).r2r(),
                src_affine=src_aff.detach().cpu().numpy(),
                dst_affine=trg_aff.detach().cpu().numpy(),
                from_type=LINEAR_RAS_TO_RAS,
                to_type=LINEAR_VOX_TO_VOX,
            )
        ).to(dtype=src_data.dtype)

    run_device = _resolve_robreg_device(device)
    src_data = src_data.to(run_device)
    trg_data = trg_data.to(run_device)
    if src_mask_data is not None:
        src_mask_data = src_mask_data.to(run_device)
    if trg_mask_data is not None:
        trg_mask_data = trg_mask_data.to(run_device)
    src_aff = src_aff.to(run_device)
    trg_aff = trg_aff.to(run_device)

    T_v2v, _ = register_irls_pyramid(
        src=src_data,
        trg=trg_data,
        src_mask=src_mask_data,
        trg_mask=trg_mask_data,
        src_affine=src_aff,
        trg_affine=trg_aff,
        initial_transform=initial_transform,
        init_type=init_type,
        nmax=nmax,
        sat=sat,
        symmetric=symmetric,
        isotropic=isotropic,
        adaptive_sat=adaptive_sat,
        target_outlier_pct=target_outlier_pct,
        outliers_name=outliers_name,
        verbose=verbose,
    )

    if return_v2v:
        return T_v2v

    work_device = T_v2v.device
    work_dtype = T_v2v.dtype
    return (
            move_tensor(trg_aff, device=work_device, dtype=work_dtype)
            @ move_tensor(T_v2v, device=work_device, dtype=work_dtype)
            @ torch.inverse(move_tensor(src_aff, device=work_device, dtype=work_dtype))
    )


__all__ = ["register_irls_pyramid", "robreg"]
