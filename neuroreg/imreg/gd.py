"""Legacy gradient-descent image-registration implementation."""

import logging
import time

import nibabel as nib
import numpy as np
import torch
from torch import Tensor

from ..image import build_gaussian_pyramid, save_resliced_r2r_image
from ..image.pyramid import _PYRAMID_FILTER, _smooth3d, get_pyramid_limits
from ..transforms import LTA
from ..transforms.matrices import matrix_sqrt_schur
from .device import resolve_torch_device
from .init import InitType, get_init_vox2vox, resolve_init_type
from .optimize import training_loop
from .reg_model import RegModel

logger = logging.getLogger(__name__)


def _shape3(shape: torch.Size | tuple[int, ...]) -> tuple[int, int, int]:
    """Return the leading three spatial dimensions as an explicit 3-tuple."""
    return int(shape[0]), int(shape[1]), int(shape[2])


def _resolve_level_iterations(
    n_levels: int,
    n: int,
    level_iters: list[int] | tuple[int, ...] | None,
) -> list[int]:
    """Return the optimizer iteration budget for each pyramid level.

    The returned list follows the execution order used by the pyramid loop:
    coarse -> fine.
    """
    if level_iters is None:
        return [int(n)] * int(n_levels)
    resolved = [int(v) for v in level_iters]
    if len(resolved) != int(n_levels):
        raise ValueError(
            "level_iters must provide exactly one iteration count per executed pyramid level "
            f"(expected {n_levels}, got {len(resolved)})."
        )
    if any(v < 0 for v in resolved):
        raise ValueError("level_iters must contain non-negative iteration counts only.")
    return resolved


def _smooth_finest_pyramid_level(levels: list[torch.Tensor]) -> list[torch.Tensor]:
    """Smooth only the finest pyramid level to preserve legacy GD behavior."""
    if not levels:
        return levels
    smoothed = list(levels)
    smoothed[0] = _smooth3d(smoothed[0], _PYRAMID_FILTER, padding_mode="replicate")
    return smoothed


def register_level(
    simg: Tensor,
    timg: Tensor,
    dof: int = 6,
    v2v_init: Tensor | None = None,
    init_type: InitType = "image_center",
    src_affine: Tensor | None = None,
    trg_affine: Tensor | None = None,
    n: int = 30,
    loss_name: str = "mse",
    loss_beta: float | None = None,
    loss_bins: int = 32,
    optimizer: str = "adam",
    lr: float | None = None,
    verbose: bool = False,
    device: str = "cpu",
    translation_weight_scale: float = 1.0,
    rotation_weight_scale: float = 4.0,
    scale_weight_scale: float = 1.0,
    shear_weight_scale: float = 1.0,
    trace_fn=None,
) -> tuple[Tensor, list[float], RegModel]:
    """Run legacy gradient-descent registration on a single pyramid level.

    Parameters
    ----------
    simg, timg : Tensor
        Source and target tensors for the current pyramid level.
    dof : int, default=6
        Degrees of freedom for the registration model.
    v2v_init : Tensor or None, optional
        Explicit voxel-to-voxel initialization. When provided it takes
        precedence over ``init_type``.
    init_type : {"header", "centroid", "image_center"}, default="image_center"
        Initialization strategy used when ``v2v_init`` is not supplied.
    src_affine, trg_affine : Tensor or None, optional
        Level-specific voxel-to-RAS affines used to derive initialization.
    n : int, default=30
        Number of optimizer iterations.
    loss_name, loss_beta, loss_bins : optional
        Loss configuration forwarded to :func:`training_loop`.
    optimizer : {"adam", "lbfgs"}, default="adam"
        Optimizer used for this level.
    lr : float or None, optional
        Explicit learning rate. Backend-specific defaults are used when omitted.
    verbose : bool, default=False
        Enable per-iteration logging inside the training loop.
    device : str, default="cpu"
        Torch device string.
    *_weight_scale : float
        Relative scaling factors for translation, rotation, scale, and shear
        weights inside :class:`RegModel`.
    trace_fn : callable, optional
        Optional callback receiving iteration events from ``training_loop``.

    Returns
    -------
    tuple[Tensor, list[float], RegModel]
        The estimated voxel-to-voxel transform for this level, the recorded
        scalar loss history, and the fitted registration model.
    """
    resolved_init_type = resolve_init_type(init_type=init_type, default_init_type="image_center")
    run_device = resolve_torch_device(device)
    if v2v_init is not None and resolved_init_type != "header":
        logger.warning(
            "register_level: cannot pass v2v_init and init_type=%r, will use v2v_init",
            resolved_init_type,
        )
    elif v2v_init is None:
        v2v_init = get_init_vox2vox(
            simg,
            timg,
            saffine=src_affine,
            taffine=trg_affine,
            init_type=resolved_init_type,
        )
        logger.debug("v2v_init from %s alignment: %s", resolved_init_type, v2v_init)
    source_shape = _shape3(simg.shape)
    target_shape = _shape3(timg.shape)
    model = RegModel(
        dof=dof,
        v2v_init=v2v_init,
        source_shape=source_shape,
        target_shape=target_shape,
        device=run_device,
        translation_weight_scale=translation_weight_scale,
        rotation_weight_scale=rotation_weight_scale,
        scale_weight_scale=scale_weight_scale,
        shear_weight_scale=shear_weight_scale,
    )

    if optimizer.lower() == "lbfgs":
        opt = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0 if lr is None else float(lr),
            max_iter=20,
            line_search_fn="strong_wolfe",
        )
    elif optimizer.lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=0.001 if lr is None else float(lr))
    else:
        raise ValueError(f"Unknown optimizer '{optimizer}'. Choose from: 'adam', 'lbfgs'.")

    losses = training_loop(
        model,
        opt,
        simg.to(run_device),
        timg.to(run_device),
        n=n,
        loss_name=loss_name,
        loss_beta=loss_beta,
        loss_bins=loss_bins,
        optimizer_name=optimizer,
        verbose=verbose,
        trace_fn=trace_fn,
    )
    v2v = model.get_v2v_from_weights(source_shape, target_shape)
    return v2v, losses, model


def register_gd_pyramid(
    src: str | nib.Nifti1Image,
    trg: str | nib.Nifti1Image,
    lta_name: str | None = None,
    mapped_name: str | None = None,
    return_v2v: bool = False,
    init_type: InitType = "image_center",
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
    trace_fn=None,
) -> Tensor:
    """Run the legacy gradient-descent multiresolution registration path.

    This is the original PyTorch optimizer-based image-registration backend.
    It builds matching source/target pyramids, optionally resamples both images
    to a shared isotropic grid, and then optimizes from coarse to fine while
    propagating each level's solution to the next.

    Parameters
    ----------
    src, trg : str or nibabel image
        Moving and reference images.
    lta_name, mapped_name : str or None, optional
        Optional output paths for the final LTA and resampled moving image.
    return_v2v : bool, default=False
        Return voxel-to-voxel instead of RAS-to-RAS when ``True``.
    init_type : {"header", "centroid", "image_center"}, default="image_center"
        Initialization strategy used on the coarsest level.
    symmetric : bool, default=True
        Run symmetric halfway-space registration when ``True``.
    dof, n, level_iters, loss_name, loss_beta, loss_bins, optimizer, lr
        Registration and optimization settings for the GD path.
    *_weight_scale : float
        Relative scaling factors applied to the model parameter blocks.
    min_voxels, max_voxels : optional
        Shared pyramid size limits.
    isotropic : bool, default=False
        Resample both inputs to a shared isotropic grid before building the
        pyramid.
    device : str, default="cpu"
        Torch device string used by the GD optimizer.
    trace_fn : callable, optional
        Optional callback receiving run, level, and iteration events.

    Returns
    -------
    Tensor
        The final transform as RAS-to-RAS by default, or voxel-to-voxel when
        ``return_v2v=True``.
    """
    start = time.perf_counter()
    run_device = resolve_torch_device(device)
    resolved_init_type = resolve_init_type(init_type=init_type, default_init_type="image_center")
    if isinstance(src, str):
        src = nib.load(src)
    if isinstance(trg, str):
        trg = nib.load(trg)

    src_affine_t = torch.from_numpy(src.affine).double()
    trg_affine_t = torch.from_numpy(trg.affine).double()
    src_iso_aff = None
    trg_iso_aff = None
    Rsrc = torch.eye(4, dtype=torch.float32)
    Rtrg = torch.eye(4, dtype=torch.float32)

    if isotropic:
        from ..image.map import resample_isotropic

        src_zooms = np.linalg.norm(src.affine[:3, :3], axis=0)
        trg_zooms = np.linalg.norm(trg.affine[:3, :3], axis=0)
        isosize = float(max(src_zooms.min(), trg_zooms.min()))
        logger.info("%s registration: isosize=%.4f mm", "Symmetric" if symmetric else "Directed", isosize)

        if symmetric:

            def _find_out_shape(img: nib.Nifti1Image, iso: float) -> tuple[int, int, int]:
                zooms = np.linalg.norm(img.affine[:3, :3], axis=0)
                shape = np.array(img.shape[:3])
                return (
                    int(max(1, int(np.ceil(shape[0] * zooms[0] / iso)))),
                    int(max(1, int(np.ceil(shape[1] * zooms[1] / iso)))),
                    int(max(1, int(np.ceil(shape[2] * zooms[2] / iso)))),
                )

            s_dim = _find_out_shape(src, isosize)
            t_dim = _find_out_shape(trg, isosize)
            mid_dim = (
                int(max(s_dim[0], t_dim[0])),
                int(max(s_dim[1], t_dim[1])),
                int(max(s_dim[2], t_dim[2])),
            )
            logger.info("Isotropic grid: src_dim=%s  trg_dim=%s  mid_dim=%s", s_dim, t_dim, mid_dim)
            sdata, src_iso_aff, Rsrc = resample_isotropic(src, isosize, out_shape=mid_dim, mode="linear")
            tdata, trg_iso_aff, Rtrg = resample_isotropic(trg, isosize, out_shape=mid_dim, mode="linear")
        else:
            sdata, src_iso_aff, Rsrc = resample_isotropic(src, isosize, mode="linear")
            tdata, trg_iso_aff, Rtrg = resample_isotropic(trg, isosize, mode="linear")
            logger.info("  Src resampled: %s -> %s", src.shape[:3], sdata.shape)
            logger.info("  Trg resampled: %s -> %s", trg.shape[:3], tdata.shape)

        src_affine_for_pyramid = src_iso_aff
        trg_affine_for_pyramid = trg_iso_aff
    else:
        sdata = torch.from_numpy(src.get_fdata()).float()
        tdata = torch.from_numpy(trg.get_fdata()).float()
        src_affine_for_pyramid = src_affine_t.float()
        trg_affine_for_pyramid = trg_affine_t.float()

    shared_limits = get_pyramid_limits(sdata.shape, tdata.shape, minsize=min_voxels, maxsize=max_voxels)
    simgs, saffines = build_gaussian_pyramid(sdata, src_affine_for_pyramid, limits=shared_limits)
    timgs, taffines = build_gaussian_pyramid(tdata, trg_affine_for_pyramid, limits=shared_limits)
    simgs = _smooth_finest_pyramid_level(simgs)
    timgs = _smooth_finest_pyramid_level(timgs)

    if not simgs:
        raise ValueError(f"build_gaussian_pyramid returned no levels for the source image (shape {list(sdata.shape)}).")
    if not timgs:
        raise ValueError(f"build_gaussian_pyramid returned no levels for the target image (shape {list(tdata.shape)}).")

    n_levels = len(simgs)
    iterations_per_level = _resolve_level_iterations(n_levels, n=n, level_iters=level_iters)
    Mr2r = torch.eye(4, dtype=torch.float64)

    if trace_fn is not None:
        trace_fn(
            event="run_start",
            Mr2r=Mr2r.detach().clone(),
            Mv2v=torch.eye(4, dtype=torch.float64),
            n_levels=n_levels,
        )

    if symmetric:
        from ..image.map import map as _map_img

        for level_idx, (si, sa, ti, ta) in enumerate(
            zip(reversed(simgs), reversed(saffines), reversed(timgs), reversed(taffines), strict=True)
        ):
            pyramid_level = n_levels - 1 - level_idx
            logger.info("Sym level %d (pyramid %d): shape=%s", level_idx, pyramid_level, list(si.shape))
            n_level = iterations_per_level[level_idx]
            if trace_fn is not None:
                trace_fn(
                    event="level_start",
                    level_index=level_idx,
                    pyramid_shape=tuple(int(v) for v in si.shape),
                    Mr2r=Mr2r.detach().clone(),
                    n_iterations=n_level,
                    optimizer=optimizer,
                    lr=lr,
                )

            M = torch.inverse(ta.double()) @ Mr2r @ sa.double()
            midspace_shape = (
                int(max(si.shape[0], ti.shape[0])),
                int(max(si.shape[1], ti.shape[1])),
                int(max(si.shape[2], ti.shape[2])),
            )
            mh, mhi = matrix_sqrt_schur(M.float())

            src_mid = _map_img(si.float(), mh.float(), is_torch_mat=False, target_shape=midspace_shape)
            trg_mid = _map_img(ti.float(), mhi.float(), is_torch_mat=False, target_shape=midspace_shape)

            def _level_trace(_mhi=mhi, _mh=mh, _ta=ta, _sa=sa, _level_idx=level_idx, _si=si, **payload):
                if trace_fn is None:
                    return
                event = payload.pop("event")
                if event == "iter_end":
                    delta_iter = payload["v2v"].double()
                    m_new_iter = torch.inverse(_mhi.double()) @ delta_iter @ _mh.double()
                    mr2r_iter = _ta.double() @ m_new_iter @ torch.inverse(_sa.double())
                    trace_fn(
                        event="iter_end",
                        level_index=_level_idx,
                        pyramid_shape=tuple(int(v) for v in _si.shape),
                        Mr2r=mr2r_iter.detach().clone(),
                        **payload,
                    )
                else:
                    trace_fn(
                        event=event,
                        level_index=_level_idx,
                        pyramid_shape=tuple(int(v) for v in _si.shape),
                        **payload,
                    )

            delta_v2v, losses, _ = register_level(
                src_mid,
                trg_mid,
                dof=dof,
                v2v_init=None,
                init_type=resolved_init_type if level_idx == 0 else "header",
                src_affine=torch.eye(4, dtype=src_mid.dtype, device=src_mid.device),
                trg_affine=torch.eye(4, dtype=trg_mid.dtype, device=trg_mid.device),
                n=n_level,
                loss_name=loss_name,
                loss_beta=loss_beta,
                loss_bins=loss_bins,
                optimizer=optimizer,
                lr=lr,
                device=run_device,
                translation_weight_scale=translation_weight_scale,
                rotation_weight_scale=rotation_weight_scale,
                scale_weight_scale=scale_weight_scale,
                shear_weight_scale=shear_weight_scale,
                trace_fn=_level_trace,
            )
            Mv2v_level = torch.inverse(mhi.double()) @ delta_v2v.double() @ mh.double()
            Mr2r = ta.double() @ Mv2v_level @ torch.inverse(sa.double())
            if trace_fn is not None:
                trace_fn(
                    event="level_end",
                    level_index=level_idx,
                    pyramid_shape=tuple(int(v) for v in si.shape),
                    Mr2r=Mr2r.detach().clone(),
                    Mv2v=Mv2v_level.detach().clone(),
                    n_iterations=n_level,
                    optimizer=optimizer,
                    lr=lr,
                    losses=list(losses),
                )
    else:
        for level_idx, (si, sa, ti, ta) in enumerate(
            zip(reversed(simgs), reversed(saffines), reversed(timgs), reversed(taffines), strict=True)
        ):
            n_level = iterations_per_level[level_idx]
            logger.info("Resolution level %d: %s", level_idx, list(si.size()))
            if trace_fn is not None:
                trace_fn(
                    event="level_start",
                    level_index=level_idx,
                    pyramid_shape=tuple(int(v) for v in si.shape),
                    Mr2r=Mr2r.detach().clone(),
                    n_iterations=n_level,
                    optimizer=optimizer,
                    lr=lr,
                )

            def _level_trace(_level_idx=level_idx, _si=si, _sa=sa, _ta=ta, **payload):
                if trace_fn is None:
                    return
                event = payload.pop("event")
                if event == "iter_end":
                    v2v_iter = payload["v2v"].double()
                    mr2r_iter = _ta.double() @ v2v_iter @ torch.inverse(_sa.double())
                    trace_fn(
                        event="iter_end",
                        level_index=_level_idx,
                        pyramid_shape=tuple(int(v) for v in _si.shape),
                        Mr2r=mr2r_iter.detach().clone(),
                        **payload,
                    )
                else:
                    trace_fn(
                        event=event,
                        level_index=_level_idx,
                        pyramid_shape=tuple(int(v) for v in _si.shape),
                        **payload,
                    )

            if level_idx == 0:
                Mv2v_level, losses, _ = register_level(
                    si,
                    ti,
                    dof=dof,
                    init_type=resolved_init_type,
                    src_affine=sa.float(),
                    trg_affine=ta.float(),
                    n=n_level,
                    loss_name=loss_name,
                    loss_beta=loss_beta,
                    loss_bins=loss_bins,
                    optimizer=optimizer,
                    lr=lr,
                    device=run_device,
                    translation_weight_scale=translation_weight_scale,
                    rotation_weight_scale=rotation_weight_scale,
                    scale_weight_scale=scale_weight_scale,
                    shear_weight_scale=shear_weight_scale,
                    trace_fn=_level_trace,
                )
            else:
                Mv2v_init = torch.inverse(ta.double()) @ Mr2r @ sa.double()
                Mv2v_level, losses, _ = register_level(
                    si,
                    ti,
                    dof=dof,
                    v2v_init=Mv2v_init,
                    init_type="header",
                    src_affine=sa.float(),
                    trg_affine=ta.float(),
                    n=n_level,
                    loss_name=loss_name,
                    loss_beta=loss_beta,
                    loss_bins=loss_bins,
                    optimizer=optimizer,
                    lr=lr,
                    device=run_device,
                    translation_weight_scale=translation_weight_scale,
                    rotation_weight_scale=rotation_weight_scale,
                    scale_weight_scale=scale_weight_scale,
                    shear_weight_scale=shear_weight_scale,
                    trace_fn=_level_trace,
                )
            Mv2v_level = Mv2v_level.double()
            Mr2r = ta.double() @ Mv2v_level @ torch.inverse(sa.double())
            if trace_fn is not None:
                trace_fn(
                    event="level_end",
                    level_index=level_idx,
                    pyramid_shape=tuple(int(v) for v in si.shape),
                    Mr2r=Mr2r.detach().clone(),
                    Mv2v=Mv2v_level.detach().clone(),
                    n_iterations=n_level,
                    optimizer=optimizer,
                    lr=lr,
                    losses=list(losses),
                )

    if isotropic and src_iso_aff is not None and trg_iso_aff is not None:
        Mv2v_iso = torch.inverse(trg_iso_aff.double()) @ Mr2r @ src_iso_aff.double()
        Mv2v_orig = Rtrg.double() @ Mv2v_iso @ torch.inverse(Rsrc.double())
        Mr2r = trg_affine_t @ Mv2v_orig @ torch.inverse(src_affine_t)
    else:
        Mv2v_orig = torch.inverse(trg_affine_t) @ Mr2r @ src_affine_t

    if lta_name is not None:
        logger.info("Writing final LTA file: %s", lta_name)
        LTA.from_matrix(Mr2r.numpy(), src.get_filename(), src, trg.get_filename(), trg).write(lta_name)
    if mapped_name is not None:
        logger.info("Writing mapped image: %s", mapped_name)
        save_resliced_r2r_image(
            src,
            Mr2r.numpy(),
            mapped_name,
            target_affine=trg.affine,
            target_shape=_shape3(trg.shape),
            mode="linear",
        )

    logger.info("register_gd_pyramid total time: %.2f s", time.perf_counter() - start)
    if return_v2v:
        return Mv2v_orig
    return Mr2r


__all__ = ["register_level", "register_gd_pyramid", "RegModel"]
