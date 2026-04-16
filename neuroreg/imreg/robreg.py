"""Public IRLS-backed robust image-registration API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import torch
from torch import Tensor

from ..image import build_gaussian_pyramid, get_pyramid_limits
from ..image.map import resample_isotropic_tensor
from .init import get_ixform_centroids
from .irls import register_irls

ImageLike = str | Path | Any | Tensor

logger = logging.getLogger(__name__)


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
        img = cast(Any, nib.load(str(image)))
        return torch.from_numpy(img.get_fdata()).float(), torch.from_numpy(img.affine).float()

    if hasattr(image, "get_fdata") and hasattr(image, "affine"):
        return torch.from_numpy(image.get_fdata()).float(), torch.from_numpy(image.affine).float()

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

    reg_affine = final_info.get("iso_affine")
    if reg_affine is None:
        logger.warning("Cannot save outlier map: no affine available")
        return

    weights_sqrt = final_info["weights"]
    valid_mask = final_info["valid_mask"]
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
    src_affine: Tensor | None = None,
    trg_affine: Tensor | None = None,
    initial_transform: Tensor | None = None,
    centroid_init: bool = True,
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
    src_affine, trg_affine : Tensor, optional
        Voxel-to-RAS affines. Required when ``isotropic=True``.
    initial_transform : Tensor, optional
        Initial voxel-to-voxel transform. If provided, it takes precedence over
        centroid initialization.
    centroid_init : bool, default=True
        Whether to initialize with centroid alignment when no explicit initial
        transform is supplied.
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

        src_iso, src_iso_aff, Rsrc = resample_isotropic_tensor(src, src_affine_np, isosize, mode="bilinear")
        trg_iso, trg_iso_aff, Rtrg = resample_isotropic_tensor(trg, trg_affine_np, isosize, mode="bilinear")

        if verbose:
            logger.info("  Src resampled: %s → %s", src.shape, src_iso.shape)
            logger.info("  Trg resampled: %s → %s", trg.shape, trg_iso.shape)

        if initial_transform is not None:
            T_iso = Rtrg.double() @ initial_transform.double() @ torch.inverse(Rsrc.double())
        elif centroid_init:
            T_iso = get_ixform_centroids(src_iso.float(), trg_iso.float()).float()
            if verbose:
                t = T_iso[:3, 3].tolist()
                logger.info(
                    "Centroid initialization (isotropic space): [%.6f, %.6f, %.6f]",
                    t[0],
                    t[1],
                    t[2],
                )
        else:
            T_iso = torch.eye(4, dtype=torch.float32)
            if verbose:
                logger.info("Centroid initialization disabled; starting from identity in isotropic space")

        shared_limits = get_pyramid_limits(src_iso.shape, trg_iso.shape, minsize=min_voxels, maxsize=max_voxels)
        pyramid_src, _ = build_gaussian_pyramid(src_iso, src_iso_aff, limits=shared_limits)
        pyramid_trg, _ = build_gaussian_pyramid(trg_iso, trg_iso_aff, limits=shared_limits)
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
        if initial_transform is not None:
            T_iso = initial_transform.float()
        elif centroid_init:
            T_iso = get_ixform_centroids(src.float(), trg.float()).float()
            if verbose:
                t = T_iso[:3, 3].tolist()
                logger.info("Centroid initialization: [%.6f, %.6f, %.6f]", t[0], t[1], t[2])
        else:
            T_iso = torch.eye(4, dtype=torch.float32)
            if verbose:
                logger.info("Centroid initialization disabled; starting from identity")
        Rsrc = torch.eye(4, dtype=torch.float32)
        Rtrg = torch.eye(4, dtype=torch.float32)
        iso_affine = trg_affine.detach().cpu().numpy() if trg_affine is not None else None

    if not pyramid_src or not pyramid_trg:
        raise ValueError(
            "Pyramid construction returned no levels. "
            f"Source levels: {[tuple(level.shape) for level in pyramid_src]}; "
            f"Target levels: {[tuple(level.shape) for level in pyramid_trg]}"
        )

    T = T_iso if T_iso is not None else torch.eye(4, dtype=torch.float32)
    all_info: list[dict[str, Any]] = []

    for lvl in range(len(pyramid_src) - 1, -1, -1):
        s = pyramid_src[lvl].float()
        t = pyramid_trg[lvl].float()
        scale = float(2 ** lvl)

        T_lvl = T.clone()
        T_lvl[:3, 3] = T[:3, 3] / scale

        if verbose:
            logger.info("Pyramid level %d  shape=%s  (scale ×1/%d)", lvl, list(s.shape), int(scale))

        T_lvl, info = register_irls(
            s,
            t,
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
        T = Rtrg.double() @ T.double() @ torch.inverse(Rsrc.double())
        T = T.float()

    if outliers_name is not None:
        _save_outlier_map(all_info, outliers_name, verbose=verbose)

    return T, all_info


def robreg(
    src: ImageLike,
    trg: ImageLike,
    *,
    src_affine: Tensor | None = None,
    trg_affine: Tensor | None = None,
    return_v2v: bool = False,
    centroid_init: bool = True,
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
    return_v2v : bool, default=False
        If ``True``, return the estimated transform in voxel coordinates. If
        ``False``, return the corresponding RAS-to-RAS transform.
    centroid_init : bool, default=True
        Whether to initialize the registration with centroid alignment.
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

    src_data = src_data.to(device)
    trg_data = trg_data.to(device)
    src_aff = src_aff.to(device)
    trg_aff = trg_aff.to(device)

    T_v2v, _ = register_irls_pyramid(
        src=src_data,
        trg=trg_data,
        src_affine=src_aff,
        trg_affine=trg_aff,
        centroid_init=centroid_init,
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

    return trg_aff.double() @ T_v2v.double() @ torch.inverse(src_aff.double())


__all__ = ["register_irls_pyramid", "robreg"]


