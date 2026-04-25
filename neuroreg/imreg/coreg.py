"""Public dispatcher for image-to-image registration backends."""

import nibabel as nib
from torch import Tensor

from .gd import RegModel, register_gd_pyramid, register_level
from .init import InitType
from .powell import register_powell_coreg


def coreg(
        src: str | nib.Nifti1Image,
        trg: str | nib.Nifti1Image,
        src_mask: str | nib.spatialimages.SpatialImage | Tensor | None = None,
        trg_mask: str | nib.spatialimages.SpatialImage | Tensor | None = None,
        lta_name: str | None = None,
        mapped_name: str | None = None,
        return_v2v: bool = False,
        init_type: InitType = "image_center",
        init_lta: str | None = None,
        method: str = "powell",
        symmetric: bool = True,
        dof: int = 6,
        n: int = 30,
        level_iters: list[int] | tuple[int, ...] | None = None,
        loss_name: str = "mse",
        loss_beta: float | None = None,
        loss_bins: int = 32,
        optimizer: str = "adam",
        lr: float | None = None,
        translation_weight_scale: float = 1.0,
        rotation_weight_scale: float = 4.0,
        scale_weight_scale: float = 1.0,
        shear_weight_scale: float = 1.0,
        min_voxels: int = 16,
        max_voxels: int | None = None,
        isotropic: bool = False,
        device: str = "cpu",
        powell_brute_force_limit: float = 30.0,
        powell_brute_force_iters: int = 1,
        powell_brute_force_samples: int = 30,
        powell_maxiter: int = 4,
        powell_sep: int = 4,
        trace_fn=None,
) -> Tensor:
    """Run public image-to-image registration.

    Parameters
    ----------
    src, trg : str or nibabel image
        Moving and reference images.
    src_mask, trg_mask : optional
        Optional masks in moving/source and reference/target space. Voxels
        outside these masks are excluded from the similarity objective.
    lta_name, mapped_name : str or None, optional
        Optional output paths for the final transform and mapped moving image.
    return_v2v : bool, default=False
        Return the final transform in voxel coordinates instead of RAS.
    init_type : {"header", "centroid", "image_center"}, default="image_center"
        Initialization strategy for the selected backend when ``init_lta`` is
        not provided.
    init_lta : str, optional
        Existing LTA used for initialization. When provided, it overrides the
        requested ``init_type``.
    method : {"powell", "gd"}, default="powell"
        Registration backend. ``"powell"`` uses the MRI_coreg-style brute-force
        plus Powell path; ``"gd"`` runs the legacy PyTorch gradient-descent
        pyramid.
    symmetric, dof, n, level_iters, loss_name, loss_beta, loss_bins, optimizer, lr
        Gradient-descent backend options.
    *_weight_scale : float
        Parameter-block scaling forwarded to the GD registration model.
    min_voxels, max_voxels, isotropic, device
        Pyramid and device settings.
    powell_brute_force_limit, powell_brute_force_iters, powell_brute_force_samples
        Coarse search settings for the Powell backend.
    powell_maxiter : int, default=4
        Maximum Powell refinement iterations.
    powell_sep : int, default=4
        Sampling spacing for the Powell evaluator.
    trace_fn : callable, optional
        Optional callback receiving backend-specific progress events.

    Returns
    -------
    Tensor
        Final RAS-to-RAS transform by default, or voxel-to-voxel when
        ``return_v2v=True``.
    """
    resolved_method = method.lower()
    if resolved_method == "powell":
        return register_powell_coreg(
            src=src,
            trg=trg,
            src_mask=src_mask,
            trg_mask=trg_mask,
            lta_name=lta_name,
            mapped_name=mapped_name,
            return_v2v=return_v2v,
            init_type=init_type,
            init_lta=init_lta,
            dof=dof,
            brute_force_limit=powell_brute_force_limit,
            brute_force_iters=powell_brute_force_iters,
            brute_force_samples=powell_brute_force_samples,
            powell_maxiter=powell_maxiter,
            sep=powell_sep,
            device=device,
            trace_fn=trace_fn,
        )
    if resolved_method != "gd":
        raise ValueError("method must be 'powell' or 'gd'")
    return register_gd_pyramid(
        src=src,
        trg=trg,
        src_mask=src_mask,
        trg_mask=trg_mask,
        lta_name=lta_name,
        mapped_name=mapped_name,
        return_v2v=return_v2v,
        init_type=init_type,
        init_lta=init_lta,
        symmetric=symmetric,
        dof=dof,
        n=n,
        level_iters=level_iters,
        loss_name=loss_name,
        loss_beta=loss_beta,
        loss_bins=loss_bins,
        optimizer=optimizer,
        lr=lr,
        translation_weight_scale=translation_weight_scale,
        rotation_weight_scale=rotation_weight_scale,
        scale_weight_scale=scale_weight_scale,
        shear_weight_scale=shear_weight_scale,
        min_voxels=min_voxels,
        max_voxels=max_voxels,
        isotropic=isotropic,
        device=device,
        trace_fn=trace_fn,
    )


__all__ = ["register_level", "register_gd_pyramid", "register_powell_coreg", "coreg", "RegModel"]
