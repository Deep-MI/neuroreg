import logging
import time

import nibabel as nib
import numpy as np
import torch
from torch import Tensor

from ..image import build_gaussian_pyramid
from ..image.pyramid import _PYRAMID_FILTER, _smooth3d, get_pyramid_limits
from ..transforms import LTA
from ..transforms.matrices import matrix_sqrt_schur
from .init import get_ixform_centroids
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
    """Return pyramid levels with only the finest level smoothed for legacy GD registration."""
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
    centroid_init: bool = True,
    n: int = 30,
    loss_name: str = "mse",
    loss_beta: float | None = None,
    loss_bins: int = 32,
    optimizer: str = "adam",
    lr: float | None = None,
    verbose: bool = False,
    device: str = "cpu",
    trace_fn=None,
) -> tuple[Tensor, list[float], "RegModel"]:
    """Run legacy gradient-descent registration on a single pyramid level.

    Parameters
    ----------
    simg, timg : Tensor
        Source and target images for one pyramid level.
    dof : int, default=6
        Transformation degrees of freedom passed to :class:`RegModel`.
    v2v_init : Tensor, optional
        Optional initial voxel-to-voxel transform for this level.
    centroid_init : bool, default=True
        If ``True`` and ``v2v_init`` is not provided, initialize from centroid
        alignment.
    n : int, default=30
        Number of optimizer iterations.
    loss_name : str, default="mse"
        Similarity metric forwarded to :func:`training_loop`. Accepted values:

        * ``"mse"``       - mean squared error (same-modality)
        * ``"huber"``     - Huber loss; ``loss_beta`` sets the delta threshold
        * ``"smooth_l1"`` - smooth L1; ``loss_beta`` sets the beta threshold
        * ``"l1"``        - mean absolute error
        * ``"ncc"``       - local normalized cross-correlation; ``loss_beta``
          sets the window size in voxels (default 9)
        * ``"mi"``        - mutual information (cross-modal); ``loss_beta``
          sets the Parzen-window sigma (default 0.1)
        * ``"nmi"``       - normalized mutual information (cross-modal);
          same hyper-parameters as ``"mi"``
    loss_beta : float, optional
        Primary hyper-parameter for the chosen loss (see ``loss_name``).
    loss_bins : int, default=32
        Number of intensity histogram bins used by ``"mi"`` and ``"nmi"``.
        Ignored for other loss functions.
    optimizer : {"adam", "lbfgs"}, default="adam"
        Optimizer used to fit the registration model.
    lr : float, optional
        Optimizer learning rate / step size. When ``None``, preserve the legacy
        defaults: ``1e-3`` for Adam and ``1.0`` for LBFGS.
    verbose : bool, default=False
        If ``True``, emit progress information during optimization.
    device : str, default="cpu"
        Torch device on which to run the optimization.
    trace_fn : callable, optional
        Optional callback invoked during optimization. Forwarded to
        :func:`training_loop`.

    Returns
    -------
    v2v : Tensor
        Estimated voxel-to-voxel transform for this level.
    losses : list of float
        Loss history returned by :func:`training_loop`.
    model : RegModel
        Fitted legacy registration model for this level.

    Raises
    ------
    ValueError
        If ``optimizer`` is not one of the supported optimizer names.
    """
    if v2v_init is not None and centroid_init:
        logger.warning("register_level: cannot pass v2v_init and centroid_init=True, will use v2v_init")
        centroid_init = False
    if centroid_init:
        v2v_init = get_ixform_centroids(simg, timg)
        logger.debug("v2v_init from centroid alignment: %s", v2v_init)
    source_shape = _shape3(simg.shape)
    target_shape = _shape3(timg.shape)
    model = RegModel(dof=dof, v2v_init=v2v_init, source_shape=source_shape, target_shape=target_shape, device=device)

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
        simg.to(device),
        timg.to(device),
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


def register_pyramid(
    src: str | nib.Nifti1Image,
    trg: str | nib.Nifti1Image,
    lta_name: str | None = None,
    mapped_name: str | None = None,
    return_v2v: bool = False,
    centroid_init: bool = True,
    symmetric: bool = True,
    dof: int = 6,
    n: int = 30,
    level_iters: list[int] | tuple[int, ...] | None = None,
    loss_name: str = "mse",
    loss_beta: float | None = None,
    loss_bins: int = 32,
    optimizer: str = "adam",
    lr: float | None = None,
    min_voxels: int = 16,
    max_voxels: int | None = None,
    isotropic: bool = False,
    device: str = "cpu",
    trace_fn=None,
) -> Tensor:
    """Run the legacy gradient-descent multiresolution registration path.

    Parameters
    ----------
    src, trg : str or nib.Nifti1Image
        Moving/source and fixed/target images, either as file paths or loaded
        nibabel images.
    lta_name : str, optional
        If provided, write the final transform to this LTA file.
    mapped_name : str, optional
        If provided, write the final mapped moving image to this file.
    return_v2v : bool, default=False
        If ``True``, return the final voxel-to-voxel transform. Otherwise
        return the final RAS-to-RAS transform.
    centroid_init : bool, default=True
        Whether to use centroid alignment for initialization on the coarsest
        pyramid level.
    symmetric : bool, default=True
        If ``True``, run the legacy halfway-space updates. If ``False``, run
        the directed source-to-target optimization path.
    dof : int, default=6
        Transformation degrees of freedom used by the legacy model.
    n : int, default=30
        Number of optimizer iterations per pyramid level.
    level_iters : sequence of int, optional
        Per-level iteration budget in pyramid execution order (coarse -> fine).
        When provided, overrides the uniform ``n`` value.
    loss_name : str, default="mse"
        Similarity metric - see :func:`register_level` for accepted values and
        the meaning of ``loss_beta`` for each choice.
    loss_beta : float, optional
        Primary hyper-parameter for the chosen loss (see ``loss_name``).
    loss_bins : int, default=32
        Number of intensity histogram bins used by ``"mi"`` and ``"nmi"``.
        Ignored for other loss functions.
    optimizer : {"adam", "lbfgs"}, default="adam"
        Optimizer used at each pyramid level.
    lr : float, optional
        Optimizer learning rate / step size forwarded to
        :func:`register_level`. When ``None``, preserve the legacy defaults.
    min_voxels : int, default=16
        Minimum size constraint passed to the shared pyramid builder.
    max_voxels : int, optional
        Maximum allowed size of the finest pyramid level to process. When
        ``None`` (default), include the original/full-resolution level.
    isotropic : bool, default=False
        If ``True``, resample both images to a common isotropic grid before
        building the pyramid.
    device : str, default="cpu"
        Torch device on which to run the legacy optimization.
    trace_fn : callable, optional
        Optional callback invoked as ``trace_fn(event=..., **payload)``.
        Emits ``run_start``, ``level_start``, ``iter_end``, and ``level_end``
        events containing the current transform estimate on each pyramid level.

    Returns
    -------
    Tensor
        Final transform matrix. This is voxel-to-voxel when
        ``return_v2v=True`` and RAS-to-RAS otherwise.

    Raises
    ------
    ValueError
        If pyramid construction yields no usable levels for either image.
    """
    start = time.perf_counter()
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
            sdata, src_iso_aff, Rsrc = resample_isotropic(src, isosize, out_shape=mid_dim, mode="bilinear")
            tdata, trg_iso_aff, Rtrg = resample_isotropic(trg, isosize, out_shape=mid_dim, mode="bilinear")
        else:
            sdata, src_iso_aff, Rsrc = resample_isotropic(src, isosize, mode="bilinear")
            tdata, trg_iso_aff, Rtrg = resample_isotropic(trg, isosize, mode="bilinear")
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
                centroid_init=centroid_init and level_idx == 0,
                n=n_level,
                loss_name=loss_name,
                loss_beta=loss_beta,
                loss_bins=loss_bins,
                optimizer=optimizer,
                lr=lr,
                device=device,
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
                    centroid_init=centroid_init,
                    n=n_level,
                    loss_name=loss_name,
                    loss_beta=loss_beta,
                    loss_bins=loss_bins,
                    optimizer=optimizer,
                    lr=lr,
                    device=device,
                    trace_fn=_level_trace,
                )
            else:
                Mv2v_init = torch.inverse(ta.double()) @ Mr2r @ sa.double()
                Mv2v_level, losses, _ = register_level(
                    si,
                    ti,
                    dof=dof,
                    v2v_init=Mv2v_init,
                    centroid_init=False,
                    n=n_level,
                    loss_name=loss_name,
                    loss_beta=loss_beta,
                    loss_bins=loss_bins,
                    optimizer=optimizer,
                    lr=lr,
                    device=device,
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
        from ..image.map import map_r2r as _map_r2r

        sdata_full = torch.from_numpy(src.get_fdata()).float()
        mapped = _map_r2r(
            sdata_full,
            Mr2r.float(),
            source_affine=src_affine_t.float(),
            target_affine=trg_affine_t.float(),
            target_shape=_shape3(trg.shape),
            mode="bilinear",
        ).detach()
        header = trg.header.copy()
        header.set_data_dtype(np.float32)
        mapped_img = nib.MGHImage(mapped.squeeze().numpy().astype(np.float32), trg.affine, header)
        mapped_img.to_filename(mapped_name)

    logger.info("register_pyramid total time: %.2f s", time.perf_counter() - start)
    if return_v2v:
        return Mv2v_orig
    return Mr2r


__all__ = ["register_level", "register_pyramid", "RegModel"]
