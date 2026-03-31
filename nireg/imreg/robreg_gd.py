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
    optimizer: str = "adam",
    verbose: bool = False,
    device: str = "cpu",
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
        Loss function name forwarded to :func:`training_loop`.
    loss_beta : float, optional
        Optional loss hyperparameter for robust losses that require one.
    optimizer : {"adam", "lbfgs"}, default="adam"
        Optimizer used to fit the registration model.
    verbose : bool, default=False
        If ``True``, emit progress information during optimization.
    device : str, default="cpu"
        Torch device on which to run the optimization.

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
    m = RegModel(dof=dof, v2v_init=v2v_init, source_shape=source_shape, target_shape=target_shape, device=device)

    if optimizer.lower() == "lbfgs":
        opt = torch.optim.LBFGS(m.parameters(), lr=1.0, max_iter=20, line_search_fn="strong_wolfe")
    elif optimizer.lower() == "adam":
        opt = torch.optim.Adam(m.parameters(), lr=0.001)
    else:
        raise ValueError(f"Unknown optimizer '{optimizer}'. Choose from: 'adam', 'lbfgs'.")

    losses = training_loop(
        m,
        opt,
        simg.to(device),
        timg.to(device),
        n=n,
        loss_name=loss_name,
        loss_beta=loss_beta,
        optimizer_name=optimizer,
        verbose=verbose,
    )
    v2v = m.get_v2v_from_weights(source_shape, target_shape)
    return v2v, losses, m


def register_pyramid(
    src: str | nib.Nifti1Image,
    trg: str | nib.Nifti1Image,
    lta_name: str | None = None,
    mapped_name: str | None = None,
    return_v2v: bool = False,
    centroid_init: bool = True,
    dof: int = 6,
    n: int = 30,
    loss_name: str = "mse",
    loss_beta: float | None = None,
    optimizer: str = "adam",
    min_voxels: int = 16,
    max_voxels: int | None = None,
    isotropic: bool = False,
    device: str = "cpu",
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
        level.
    dof : int, default=6
        Transformation degrees of freedom used by the legacy model.
    n : int, default=30
        Number of optimizer iterations per pyramid level.
    loss_name : str, default="mse"
        Loss function name forwarded to :func:`training_loop`.
    loss_beta : float, optional
        Optional loss hyperparameter for robust losses that require one.
    optimizer : {"adam", "lbfgs"}, default="adam"
        Optimizer used at each pyramid level.
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

    src_iso_aff = None
    trg_iso_aff = None

    if isotropic:
        from ..image.map import resample_isotropic

        src_zooms = np.linalg.norm(src.affine[:3, :3], axis=0)
        trg_zooms = np.linalg.norm(trg.affine[:3, :3], axis=0)
        isosize = float(max(src_zooms.min(), trg_zooms.min()))
        logger.info("Isotropic resampling: isosize=%.4f mm", isosize)

        sdata, src_iso_aff, Rsrc = resample_isotropic(src, isosize, mode="bilinear")
        tdata, trg_iso_aff, Rtrg = resample_isotropic(trg, isosize, mode="bilinear")
        logger.info("  Src resampled: %s → %s", src.shape[:3], sdata.shape)
        logger.info("  Trg resampled: %s → %s", trg.shape[:3], tdata.shape)
        src_affine_for_pyramid = src_iso_aff
        trg_affine_for_pyramid = trg_iso_aff
    else:
        sdata = torch.from_numpy(src.get_fdata()).float()
        tdata = torch.from_numpy(trg.get_fdata()).float()
        Rsrc = torch.eye(4, dtype=torch.float32)
        Rtrg = torch.eye(4, dtype=torch.float32)
        src_affine_for_pyramid = src.affine
        trg_affine_for_pyramid = trg.affine

    shared_limits = get_pyramid_limits(sdata.shape, tdata.shape, minsize=min_voxels, maxsize=max_voxels)
    simgs, saffines = build_gaussian_pyramid(sdata, src_affine_for_pyramid, limits=shared_limits)
    timgs, taffines = build_gaussian_pyramid(tdata, trg_affine_for_pyramid, limits=shared_limits)
    simgs = _smooth_finest_pyramid_level(simgs)
    timgs = _smooth_finest_pyramid_level(timgs)

    if not simgs:
        raise ValueError(
            f"build_gaussian_pyramid returned no levels for the source image "
            f"(shape {list(sdata.shape)}). The image may be too small — "
            f"the default minimum dimension is 32 voxels."
        )
    if not timgs:
        raise ValueError(
            f"build_gaussian_pyramid returned no levels for the target image "
            f"(shape {list(tdata.shape)}). The image may be too small — "
            f"the default minimum dimension is 32 voxels."
        )

    Mr2r = torch.eye(4, 4, dtype=saffines[0].dtype)
    Mv2v = torch.eye(4, 4, dtype=saffines[0].dtype)
    count = 0
    debug = False
    m = None
    for si, sa, ti, ta in zip(reversed(simgs), reversed(saffines), reversed(timgs), reversed(taffines), strict=True):
        logger.info("Resolution level %d: %s", count, list(si.size()))
        if count == 0:
            Mv2v, losses, m = register_level(
                si,
                ti,
                dof=dof,
                centroid_init=centroid_init,
                n=n,
                loss_name=loss_name,
                loss_beta=loss_beta,
                optimizer=optimizer,
                device=device,
            )
        else:
            Mv2v_init = torch.inverse(ta.double()) @ Mr2r @ sa.double()
            logger.debug("Mv2v_init:\n%s", Mv2v_init.numpy())
            Mv2v, losses, m = register_level(
                si,
                ti,
                dof=dof,
                v2v_init=Mv2v_init,
                centroid_init=False,
                n=n,
                loss_name=loss_name,
                loss_beta=loss_beta,
                optimizer=optimizer,
                device=device,
            )
        Mv2v = Mv2v.double()
        logger.debug("Mv2v:\n%s", Mv2v.numpy())
        Mr2r = ta.double() @ Mv2v @ torch.inverse(sa.double())
        logger.debug("Mr2r:\n%s", Mr2r.numpy())
        if debug:
            sname = "pyramidS-rr" + str(count) + ".mgz"
            tname = "pyramidT-rr" + str(count) + ".mgz"
            ltaname = "pyramid_S2T_rr" + str(count) + ".lta"
            smgh = nib.MGHImage(si.squeeze().numpy(), sa.numpy(), src.header)
            tmgh = nib.MGHImage(ti.squeeze().numpy(), ta.numpy(), trg.header)
            smgh.to_filename(sname)
            tmgh.to_filename(tname)
            LTA.from_matrix(Mr2r.numpy(), sname, smgh, tname, tmgh).write(ltaname)
        count = count + 1

    if isotropic and src_iso_aff is not None and trg_iso_aff is not None:
        src_aff_orig = torch.from_numpy(src.affine).double()
        trg_aff_orig = torch.from_numpy(trg.affine).double()
        Mv2v_iso = torch.inverse(trg_iso_aff.double()) @ Mr2r.double() @ src_iso_aff.double()
        Mv2v_orig = Rtrg.double() @ Mv2v_iso @ torch.inverse(Rsrc.double())
        Mr2r = trg_aff_orig @ Mv2v_orig @ torch.inverse(src_aff_orig)
    else:
        Mr2r = Mr2r.double()

    if lta_name is not None:
        logger.info("Writing final LTA file: %s", lta_name)
        LTA.from_matrix(Mr2r.numpy(), src.get_filename(), src, trg.get_filename(), trg).write(lta_name)
    if mapped_name is not None:
        if m is None:
            logger.warning(
                "Skipping mapped image output ('%s'): no registration model was produced "
                "(pyramid loop did not execute).",
                mapped_name,
            )
        else:
            logger.info("Writing mapped image: %s", mapped_name)
            mapped = m.map_image(sdata, mode="bilinear").detach()
            mapped_img = nib.MGHImage(mapped.squeeze().numpy(), src.affine, src.header)
            mapped_img.to_filename(mapped_name)
    logger.info("register_pyramid total time: %.2f s", time.perf_counter() - start)
    if return_v2v:
        return Mv2v
    return Mr2r


def register_pyramid_sym(
    src: str | nib.Nifti1Image,
    trg: str | nib.Nifti1Image,
    lta_name: str | None = None,
    mapped_name: str | None = None,
    return_v2v: bool = False,
    dof: int = 6,
    n: int = 30,
    loss_name: str = "mse",
    loss_beta: float | None = None,
    optimizer: str = "adam",
    min_voxels: int = 16,
    max_voxels: int | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Run legacy symmetric gradient-descent registration in halfway space.

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
    dof : int, default=6
        Transformation degrees of freedom used by the legacy model.
    n : int, default=30
        Number of optimizer iterations per pyramid level.
    loss_name : str, default="mse"
        Loss function name forwarded to :func:`training_loop`.
    loss_beta : float, optional
        Optional loss hyperparameter for robust losses that require one.
    optimizer : {"adam", "lbfgs"}, default="adam"
        Optimizer used at each symmetric pyramid level.
    min_voxels : int, default=16
        Minimum size constraint passed to the shared pyramid builder.
    max_voxels : int, optional
        Maximum allowed size of the finest pyramid level to process. When
        ``None`` (default), include the original/full-resolution level.
    device : str, default="cpu"
        Torch device on which to run the legacy optimization.

    Returns
    -------
    Tensor
        Final transform matrix. This is voxel-to-voxel when
        ``return_v2v=True`` and RAS-to-RAS otherwise.

    Raises
    ------
    ValueError
        If pyramid construction yields no usable levels after isotropic
        resampling.
    """
    start = time.perf_counter()
    if isinstance(src, str):
        src = nib.load(src)
    if isinstance(trg, str):
        trg = nib.load(trg)

    from ..image.map import resample_isotropic

    src_zooms = np.linalg.norm(src.affine[:3, :3], axis=0)
    trg_zooms = np.linalg.norm(trg.affine[:3, :3], axis=0)
    isosize = float(max(src_zooms.min(), trg_zooms.min()))
    logger.info("Symmetric registration: isosize=%.4f mm", isosize)

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

    src_iso, src_iso_aff, Rsrc = resample_isotropic(src, isosize, out_shape=mid_dim, mode="bilinear")
    trg_iso, trg_iso_aff, Rtrg = resample_isotropic(trg, isosize, out_shape=mid_dim, mode="bilinear")

    shared_limits = get_pyramid_limits(src_iso.shape, trg_iso.shape, minsize=min_voxels, maxsize=max_voxels)
    simgs, saffines = build_gaussian_pyramid(src_iso, src_iso_aff, limits=shared_limits)
    timgs, taffines = build_gaussian_pyramid(trg_iso, trg_iso_aff, limits=shared_limits)
    simgs = _smooth_finest_pyramid_level(simgs)
    timgs = _smooth_finest_pyramid_level(timgs)

    if not simgs:
        raise ValueError(f"build_gaussian_pyramid returned no levels (isotropic shape {list(src_iso.shape)}).")

    n_levels = len(simgs)
    M = get_ixform_centroids(simgs[-1], timgs[-1]).double()

    from ..image.map import map as _map_img

    for level_idx, (si, _sa, ti, _ta) in enumerate(
        zip(reversed(simgs), reversed(saffines), reversed(timgs), reversed(taffines), strict=True)
    ):
        pyramid_level = n_levels - 1 - level_idx
        logger.info("Sym level %d (pyramid %d): shape=%s", level_idx, pyramid_level, list(si.shape))

        midspace_shape = _shape3(si.shape)
        mh, mhi = matrix_sqrt_schur(M.float())

        src_mid = _map_img(si.float(), mh.float(), is_torch_mat=False, target_shape=midspace_shape)
        trg_mid = _map_img(ti.float(), mhi.float(), is_torch_mat=False, target_shape=midspace_shape)

        delta_v2v, _, _ = register_level(
            src_mid,
            trg_mid,
            dof=dof,
            v2v_init=None,
            centroid_init=False,
            n=n,
            loss_name=loss_name,
            loss_beta=loss_beta,
            optimizer=optimizer,
            device=device,
        )
        delta_v2v = delta_v2v.double()

        mh2 = torch.inverse(mhi.double())
        M = mh2 @ delta_v2v @ mh.double()

        if level_idx < n_levels - 1:
            M[:3, 3] *= 2.0

    Mv2v_orig = Rtrg.double() @ M @ torch.inverse(Rsrc.double())
    src_affine_t = torch.from_numpy(src.affine).double()
    trg_affine_t = torch.from_numpy(trg.affine).double()
    Mr2r = trg_affine_t @ Mv2v_orig @ torch.inverse(src_affine_t)

    if lta_name is not None:
        logger.info("Writing final LTA file: %s", lta_name)
        LTA.from_matrix(Mr2r.numpy(), src.get_filename(), src, trg.get_filename(), trg).write(lta_name)

    if mapped_name is not None:
        logger.info("Writing mapped image: %s", mapped_name)
        from ..image.map import map_r2r as _map_r2r2

        sdata_full = torch.from_numpy(src.get_fdata()).float()
        mapped = _map_r2r2(
            sdata_full,
            Mr2r.float(),
            source_affine=src_affine_t.float(),
            target_affine=trg_affine_t.float(),
            target_shape=_shape3(trg.shape),
        ).detach()
        mapped_img = nib.MGHImage(mapped.squeeze().numpy(), trg.affine, trg.header)
        mapped_img.to_filename(mapped_name)

    logger.info("register_pyramid_sym total time: %.2f s", time.perf_counter() - start)

    if return_v2v:
        return Mv2v_orig.float()
    return Mr2r.float()


__all__ = ["register_level", "register_pyramid", "register_pyramid_sym", "RegModel"]


