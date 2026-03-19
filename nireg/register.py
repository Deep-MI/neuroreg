import logging
import time
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import torch
from torch import Tensor

from .image import build_gaussian_pyramid
from .image.pyramid import get_pyramid_limits
from .nn import RegModel, training_loop
from .surface.io import get_vox2ras_tkr, load_surface, load_surface_from_subject
from .surface.optimize import BBRModel
from .transforms import (
    LINEAR_RAS_TO_RAS,
    LINEAR_VOX_TO_VOX,
    LTA,
    convert_transform_type,
    get_ixform_centroids,
    matrix_sqrt_schur,
)

logger = logging.getLogger(__name__)


def register(
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
    """
    Register (align) two images using a transformation optimization model (RegModel).

    This function performs an alignment of two images by optimizing a transformation
    model with a specified degree of freedom (dof). The alignment process can be initialized
    either by directly providing a voxel-to-voxel transformation matrix (`v2v_init`) or by
    automatically computing an initial translation based on centroid alignment
    (`centroid_init`). The optimization iterates through a training loop, minimizing the
    difference between the source image (`simg`) and target image (`timg`).

    Parameters
    ----------
    simg : Tensor
        The source image represented as a PyTorch tensor.
    timg : Tensor
        The target image represented as a PyTorch tensor.
    dof : int, optional
        The degree of freedom for the transformation model (default is 6).
        e.g., 6 represents rigid body alignment (translation + rotation).
    v2v_init : Tensor, optional
        Initial voxel-to-voxel transformation matrix, if provided.
        Overrides centroid initialization if both are supplied.
    centroid_init : bool, optional
        Whether to initialize the transformation based on centroid alignment
        (default is True). Ignored if `v2v_init` is provided.
    n : int, optional
        The number of optimization iterations in the training loop (default is 30).
    loss_name : {'mse', 'huber', 'smooth_l1', 'l1'}, optional
        Loss function name (default is 'mse').
    loss_beta : float, optional
        Transition point for robust losses (default is None).
    optimizer : {'adam', 'lbfgs'}, optional
        Optimizer to use (default is 'adam'). L-BFGS can converge faster but
        requires more memory and may not work well with all loss functions.
    verbose : bool, optional
        If True, prints progress information including the loss during optimization.
    device : str, optional
        The device to run the optimization on, such as 'cpu' or 'cuda' (default is 'cpu').

    Returns
    -------
    tuple[Tensor, list[float], RegModel]
        A tuple containing:
        - v2v (Tensor): The resulting voxel-to-voxel transformation matrix after optimization.
        - losses (list): A list of loss values observed during the optimization process.
        - m (RegModel): The trained registration model instance containing the optimized parameters.

    Notes
    -----
    - If both `v2v_init` and `centroid_init` are specified, `v2v_init` takes precedence
      and a warning is printed.
    - This function utilizes a PyTorch optimization loop with RMSProp as the optimizer.
    """
    if v2v_init is not None and centroid_init:
        logger.warning("register: cannot pass v2v_init and centroid_init=True, will use v2v_init")
        centroid_init = False
    if centroid_init:
        v2v_init = get_ixform_centroids(simg, timg)
        logger.debug("v2v_init from centroid alignment: %s", v2v_init)
    m = RegModel(dof=dof, v2v_init=v2v_init, source_shape=simg.shape, target_shape=timg.shape, device=device)

    # Create optimizer based on parameter
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
    v2v = m.get_v2v_from_weights(simg.shape, timg.shape)
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
    isotropic: bool = False,
    device: str = "cpu",
) -> Tensor:
    """
    Perform multi-resolution image registration using an iterative coarse-to-fine alignment.

    This function aligns two 3D images (`src` and `trg`) by building Gaussian pyramids for the
    images and progressively registering lower-resolution versions of the images first. The
    resulting transformation is then refined for higher-resolution levels in the pyramid to
    compute a final voxel-to-voxel transformation matrix that aligns the source image to the
    target image.

    Parameters
    ----------
    src : Union[str, nibabel.Nifti1Image]
        The source image, provided either as a file path to a `.mgz` file (string) or as a nibabel image object.
        The image will be moved to align with the target.
    trg : Union[str, nibabel.Nifti1Image]
        The target image, provided either as a file path to a `.mgz` file (string) or as a nibabel image object.
        The source image will be aligned to this image.
    lta_name : Optional[str], optional
        The output filename for saving the final transformation matrix in LTA format (default is None).
    mapped_name : Optional[str], optional
        The output filename for saving the mapped version of the source image after registration (default is None).
    return_v2v : bool, optional
        Return vox-to-vox transformation matrix after registration (default is False, returns ras-to-ras instead).
    centroid_init : bool, optional
        Whether to initialize the transformation based on centroid alignment (default is True).
    dof : int, optional
        Degrees of freedom passed to each ``register`` call: 6=rigid, 9=rigid+scale,
        12=affine (default is 6).
    n : int, optional
        Number of optimisation iterations per pyramid level (default is 30).
    device : str, optional
        The device to use for computation, such as 'cpu' or 'cuda' (default is 'cpu').

    Returns
    -------
    torch.Tensor
        The final rigid-body transformation matrix (`Mr2r`) between the reference frames of the source
        and target images, representing their alignment in 3D space (RAS-to-RAS).

    Notes
    -----
    - If `src` and `trg` are provided as strings, they are loaded using `nibabel`.
    - The images must have the same voxel size for accurate registration.
    - The function prints periodic log messages indicating the progress at each resolution level within the pyramid.
    - This function assumes that the input images are preloaded or provided as filenames referencing valid `.mgz` files.

    Examples
    --------
    >>> register_pyramid("src.mgz", "trg.mgz", lta_name="alignment.lta", mapped_name="mapped_image.mgz", device="cuda")
    >>> src_image = nib.load("src.mgz")
    >>> trg_image = nib.load("trg.mgz")
    >>> register_pyramid(src_image, trg_image, device="cpu")
    """
    start = time.perf_counter()
    if isinstance(src, str):
        src = nib.load(src)
    if isinstance(trg, str):
        trg = nib.load(trg)

    # Optional isotropic pre-processing (like FreeSurfer's setSourceAndTarget)
    if isotropic:
        from .image.map import resample_isotropic

        # Compute isotropic voxel size = max of finest resolutions
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

    # Compute shared pyramid limits from both image shapes so both pyramids
    # have the same number of levels and correspond to the same scale factors.
    shared_limits = get_pyramid_limits(sdata.shape, tdata.shape, minsize=16)
    simgs, saffines = build_gaussian_pyramid(sdata, src_affine_for_pyramid, limits=shared_limits)
    timgs, taffines = build_gaussian_pyramid(tdata, trg_affine_for_pyramid, limits=shared_limits)

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
    count = 0
    debug = False
    m = None  # initialise so mapped_name block is safe if the loop never executes
    for si, sa, ti, ta in zip(reversed(simgs), reversed(saffines), reversed(timgs), reversed(taffines), strict=True):
        logger.info("Resolution level %d: %s", count, list(si.size()))
        if count == 0:
            Mv2v, losses, m = register(
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
            Mv2v, losses, m = register(
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

    # When isotropic=True, Mr2r is on the isotropic grid → convert back to original
    if isotropic:
        src_aff_orig = torch.from_numpy(src.affine).double()
        trg_aff_orig = torch.from_numpy(trg.affine).double()
        # Mr2r_iso is RAS→RAS on isotropic grids
        # Convert to v2v on iso, then to v2v on original, then to r2r original
        Mv2v_iso = torch.inverse(trg_iso_aff.double()) @ Mr2r.double() @ src_iso_aff.double()
        Mv2v_orig = Rtrg.double() @ Mv2v_iso @ torch.inverse(Rsrc.double())
        Mr2r = trg_aff_orig @ Mv2v_orig @ torch.inverse(src_aff_orig)
    else:
        # Ensure Mr2r is double for consistency
        Mr2r = Mr2r.double()

    if lta_name is not None:
        logger.info("Writing final LTA file: %s", lta_name)
        LTA.from_matrix(Mr2r.numpy(), src.get_filename(), src, trg.get_filename(), trg).write(lta_name)
    if mapped_name is not None:
        if m is None:
            logger.warning(
                "Skipping mapped image output ('%s'): no registration model "
                "was produced (pyramid loop did not execute).",
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
    device: str = "cpu",
) -> torch.Tensor:
    """Symmetric (midspace) multi-resolution image registration.

    Implements the FreeSurfer ``mri_robust_register`` symmetric mode:
    at every pyramid level and every inner iteration both images are mapped
    to the geometric midspace (half-way between source and target) before
    computing the registration cost.  The cumulative transform ``M``
    (vox-to-vox on the isotropic resampled grid) is updated by composing
    the incremental update ``δ`` with the midspace maps:

    .. code-block:: none

        mh, mhi = matrix_sqrt_schur(M)   # mh: src→mid, mhi: trg→mid
        src_mid = map(src_level, mh)
        trg_mid = map(trg_level, mhi)
        δ       = register(src_mid, trg_mid)   # unchanged RegModel
        M_new   = inv(mhi) @ δ @ mh

    Pre-processing resamples both images to a common isotropic grid
    (isotropic voxel size = max of the two finest per-image resolutions;
    per-axis dimension = max of the two padded dims) so that the midspace
    is well-defined.

    Parameters
    ----------
    src : str or nibabel image
        Source (moving) image.
    trg : str or nibabel image
        Target (fixed) image.
    lta_name : str, optional
        Save the final RAS-to-RAS transform to this LTA file.
    mapped_name : str, optional
        Save the source image warped into the target space to this file.
    return_v2v : bool, optional
        Return vox-to-vox (on the original source/target voxel grid) instead
        of the default RAS-to-RAS matrix.
    dof : int, optional
        Degrees of freedom: 6 (rigid), 9 (rigid+scale), 12 (affine).
    n : int, optional
        Number of inner optimisation iterations per pyramid level.
    device : str, optional
        PyTorch device string (e.g. ``'cpu'`` or ``'cuda'``).

    Returns
    -------
    torch.Tensor, shape (4, 4)
        Final transform.  RAS-to-RAS by default; vox-to-vox on the original
        images when *return_v2v* is ``True``.
    """
    start = time.perf_counter()
    if isinstance(src, str):
        src = nib.load(src)
    if isinstance(trg, str):
        trg = nib.load(trg)

    # ------------------------------------------------------------------
    # Pre-processing: resample both images to a common isotropic grid
    # ------------------------------------------------------------------
    from .image.map import resample_isotropic

    # Use column norms of the 3×3 block — correct for any orientation,
    # including images whose affine has been modified by a rotation (e.g.
    # the synthetic displacement used in tests).
    src_zooms = np.linalg.norm(src.affine[:3, :3], axis=0)
    trg_zooms = np.linalg.norm(trg.affine[:3, :3], axis=0)
    isosize = float(max(src_zooms.min(), trg_zooms.min()))
    logger.info("Symmetric registration: isosize=%.4f mm", isosize)

    def _find_out_shape(img: nib.Nifti1Image, iso: float) -> tuple:
        """Smallest integer box that covers the image FOV at voxel size *iso*.

        Uses column norms of the 3×3 affine block so the result is correct
        regardless of image orientation (including rotated/displaced affines).
        """
        zooms = np.linalg.norm(img.affine[:3, :3], axis=0)  # column norms = voxel sizes
        shape = np.array(img.shape[:3])
        out = tuple(max(1, int(np.ceil(s * z / iso))) for s, z in zip(shape, zooms, strict=False))
        return out

    s_dim = _find_out_shape(src, isosize)
    t_dim = _find_out_shape(trg, isosize)
    mid_dim = tuple(max(s, t) for s, t in zip(s_dim, t_dim, strict=False))
    logger.info("Isotropic grid: src_dim=%s  trg_dim=%s  mid_dim=%s", s_dim, t_dim, mid_dim)

    src_iso, src_iso_aff, Rsrc = resample_isotropic(src, isosize, out_shape=mid_dim, mode="bilinear")
    trg_iso, trg_iso_aff, Rtrg = resample_isotropic(trg, isosize, out_shape=mid_dim, mode="bilinear")

    # ------------------------------------------------------------------
    # Build shared-limits pyramids on the isotropic resampled images
    # ------------------------------------------------------------------
    shared_limits = get_pyramid_limits(src_iso.shape, trg_iso.shape, minsize=16)
    simgs, saffines = build_gaussian_pyramid(src_iso, src_iso_aff, limits=shared_limits)
    timgs, taffines = build_gaussian_pyramid(trg_iso, trg_iso_aff, limits=shared_limits)

    if not simgs:
        raise ValueError(f"build_gaussian_pyramid returned no levels (isotropic shape {list(src_iso.shape)}).")

    # ------------------------------------------------------------------
    # Initialise cumulative transform on the coarsest level
    # ------------------------------------------------------------------
    # get_ixform_centroids operates on the coarsest pyramid images directly,
    # so M is already in coarsest-level voxel coordinates — no further
    # scaling is needed.
    n_levels = len(simgs)
    M = get_ixform_centroids(simgs[-1], timgs[-1]).double()

    # ------------------------------------------------------------------
    # Multiscale loop: coarse → fine
    # ------------------------------------------------------------------
    from .image.map import map as _map_img  # local import once

    for level_idx, (si, _sa, ti, _ta) in enumerate(
        zip(reversed(simgs), reversed(saffines), reversed(timgs), reversed(taffines), strict=True)
    ):
        pyramid_level = n_levels - 1 - level_idx
        logger.info("Sym level %d (pyramid %d): shape=%s", level_idx, pyramid_level, list(si.shape))

        midspace_shape = tuple(si.shape)
        mh, mhi = matrix_sqrt_schur(M.float())

        # Map both images to midspace
        src_mid = _map_img(si.float(), mh.float(), is_torch_mat=False, target_shape=midspace_shape)
        trg_mid = _map_img(ti.float(), mhi.float(), is_torch_mat=False, target_shape=midspace_shape)

        # Full registration on the midspace pair (n gradient steps)
        delta_v2v, _, _ = register(
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

        # Accumulate: M_new = inv(mhi) @ δ @ mh
        # inv(mhi) = inv(mh @ inv(M)) = M @ inv(mh) = mh  (since mh² = M)
        mh2 = torch.inverse(mhi.double())
        M = mh2 @ delta_v2v @ mh.double()

        # Scale translation ×2 before next finer level
        if level_idx < n_levels - 1:
            M[:3, 3] *= 2.0

    # ------------------------------------------------------------------
    # Final transform back to original (non-isotropic) voxel space
    # ------------------------------------------------------------------
    # M is v2v on the isotropic grid: iso_src_vox → iso_trg_vox
    # Rvox maps iso_vox → original_vox for each image
    Mv2v_orig = Rtrg.double() @ M @ torch.inverse(Rsrc.double())
    src_affine_t = torch.from_numpy(src.affine).double()
    trg_affine_t = torch.from_numpy(trg.affine).double()
    Mr2r = trg_affine_t @ Mv2v_orig @ torch.inverse(src_affine_t)

    # ------------------------------------------------------------------
    # Optional outputs
    # ------------------------------------------------------------------
    if lta_name is not None:
        logger.info("Writing final LTA file: %s", lta_name)
        LTA.from_matrix(Mr2r.numpy(), src.get_filename(), src, trg.get_filename(), trg).write(lta_name)

    if mapped_name is not None:
        logger.info("Writing mapped image: %s", mapped_name)
        from .image.map import map_r2r as _map_r2r2

        sdata_full = torch.from_numpy(src.get_fdata()).float()
        mapped = _map_r2r2(
            sdata_full,
            Mr2r.float(),
            source_affine=src_affine_t.float(),
            target_affine=trg_affine_t.float(),
            target_shape=tuple(trg.shape[:3]),
        ).detach()
        mapped_img = nib.MGHImage(mapped.squeeze().numpy(), trg.affine, trg.header)
        mapped_img.to_filename(mapped_name)

    logger.info("register_pyramid_sym total time: %.2f s", time.perf_counter() - start)

    if return_v2v:
        return Mv2v_orig.float()
    return Mr2r.float()


def register_surface(
    mov: str | nib.Nifti1Image,
    lh_surf: str | None = None,
    rh_surf: str | None = None,
    lh_thickness: str | None = None,
    rh_thickness: str | None = None,
    ref: str | nib.Nifti1Image | None = None,
    subject_dir: str | None = None,
    seg: str | None = None,
    lta_name: str | None = None,
    dof: int = 6,
    contrast: Literal["t1", "t2"] | None = None,
    init_type: Literal["header", "lta"] = "header",
    init_lta: str | None = None,
    init_ras: np.ndarray | None = None,
    cost_type: Literal["contrast", "gradient", "both"] = "contrast",
    wm_proj_abs: float = 1.4,
    gm_proj_frac: float = 0.5,
    gm_proj_abs: float | None = None,
    lh_cortex_label: str | None = None,
    rh_cortex_label: str | None = None,
    slope: float = 0.5,
    gradient_weight: float = 0.0,
    subsample: int = 1,
    n_iters: int = 100,
    lr: float = 0.01,
    device: str = "cpu",
) -> tuple[torch.Tensor, BBRModel]:
    """
    Register moving image to target anatomy using cortical surface boundaries (BBR).

    Implements boundary-based registration (BBR) analogous to FreeSurfer's
    bbregister / mri_segreg.  Alignment is optimised by sampling intensities
    from the moving image at cortical surface vertices that are projected
    inward (WM sample) and outward (GM sample) along surface normals.

    Progress and debug information are emitted via the standard :mod:`logging`
    module (logger ``nireg.register``).  To see iteration-level output, set
    that logger to ``INFO``; for matrix dumps use ``DEBUG``.

    Parameters
    ----------
    mov : str or nibabel.Nifti1Image
        Moving image to register (e.g., functional or T2 scan).
    lh_surf : str, optional
        Path to left hemisphere white surface (e.g. ``surf/lh.white``).
        Used when *subject_dir* is not provided.
    rh_surf : str, optional
        Path to right hemisphere white surface.
        Used when *subject_dir* is not provided.
    lh_thickness : str, optional
        Path to left-hemisphere cortical thickness file.  When omitted,
        a file named ``lh.thickness`` is looked up next to *lh_surf*.
    rh_thickness : str, optional
        Path to right-hemisphere cortical thickness file.  When omitted,
        a file named ``rh.thickness`` is looked up next to *rh_surf*.
    ref : str or nibabel image, optional
        Reference (T1) image against which the surfaces were built.
        Required for Mode B (explicit surfaces) when the surface coordinate
        space differs from the moving image.  Ignored when *subject_dir*
        is provided (``mri/orig.mgz`` is used automatically in that case).
    subject_dir : str, optional
        FreeSurfer subject directory.  If given, surfaces are loaded from
        ``{subject_dir}/surf/lh.white`` and ``rh.white``, and the
        registration target is set to ``{subject_dir}/mri/orig.mgz``.
    seg : str, optional
        Path to a FreeSurfer parcellation file (``aparc+aseg.mgz``,
        ``aseg.mgz``, or equivalent).  When provided, white-matter surfaces
        are extracted automatically via marching cubes — no pre-computed
        surface files are needed.  The segmentation header is used as the
        target reference, so *ref* is not required.  Cortical thickness is
        unavailable in this mode; a fixed outward projection distance is used
        instead (controlled by *gm_proj_frac* via ``wm_proj_abs``).
    lta_name : str, optional
        Output LTA filename.  Written as vox-to-vox (type 0).
    dof : int
        Degrees of freedom: 6 (rigid), 9 (rigid + scale), 12 (affine).
    contrast : {'t1', 't2'} or None, optional
        Expected tissue contrast: ``'t1'`` (WM > GM) or ``'t2'`` (GM > WM).
        If ``None`` (default), the contrast direction is auto-detected from
        the image by sampling WM and GM intensities at the surface vertices
        and checking which direction the majority of vertices support.
    init_type : {'header', 'lta'}
        Initialisation strategy when *init_ras* is not provided.
        ``'header'`` uses identity (relies on image headers being aligned).
        ``'lta'`` loads an existing transform from *init_lta*.
    init_lta : str, optional
        Path to an LTA file used when ``init_type='lta'``.  The stored
        mov_RAS → trg_RAS matrix is read and inverted to obtain the
        trg_RAS → mov_RAS initialisation transform.
    init_ras : np.ndarray, shape (4, 4), optional
        Explicit trg_RAS → mov_RAS initialisation matrix, e.g. from a prior
        image-based registration step.  Overrides *init_type* when supplied.
    cost_type : {'contrast', 'gradient', 'both'}
        BBR cost term to minimise.
    wm_proj_abs : float
        Absolute projection depth into white matter (mm). Default 1.4 mm.
    gm_proj_frac : float
        Fractional projection into grey matter relative to cortical thickness.
        Used when thickness is available and *gm_proj_abs* is not set.
    gm_proj_abs : float, optional
        Absolute projection depth into grey matter (mm).  When set, overrides
        *gm_proj_frac* regardless of whether thickness is available.  When
        ``None`` (default), *gm_proj_frac* × thickness is used if thickness
        is present; otherwise falls back to a fixed 1.4 mm absolute projection
        (matching ``wm_proj_abs``).
    lh_cortex_label : str, optional
        Path to left-hemisphere cortex label file (e.g. ``label/lh.cortex.label``).
        Only used when surfaces are supplied via *lh_surf* / *rh_surf* (explicit
        mode).  When *subject_dir* is provided, the cortex label is discovered
        automatically from ``{subject_dir}/label/lh.cortex.label``.
    rh_cortex_label : str, optional
        Same as *lh_cortex_label* for the right hemisphere.
    slope : float
        Slope of the BBR sigmoid cost function.
    gradient_weight : float
        Weight for the gradient cost term when ``cost_type='both'``.
    subsample : int
        Use every *n*-th surface vertex (1 = all vertices).
    n_iters : int
        Number of RMSprop optimisation iterations.
    lr : float
        Optimiser learning rate.
    device : str
        PyTorch device, e.g. ``'cpu'`` or ``'cuda'``.

    Returns
    -------
    tuple[torch.Tensor, BBRModel]
        A tuple of:

        * **transform** (*torch.Tensor*, shape (4, 4)) –
          Final trg_RAS → mov_RAS transformation matrix.
        * **model** (*BBRModel*) –
          The fitted model, which can be used to evaluate the cost at
          arbitrary transforms via :meth:`~BBRModel.eval_cost_at_ras2ras`.
    """
    start = time.perf_counter()

    # Load moving image
    if isinstance(mov, str):
        mov_img = nib.load(mov)
        mov_path = mov
    else:
        mov_img = mov
        mov_path = mov.get_filename() if hasattr(mov, "get_filename") else None

    mov_data = torch.from_numpy(mov_img.get_fdata()).float()

    # Load surfaces
    if subject_dir is not None:
        logger.info("Loading surfaces from subject directory: %s", subject_dir)
        lh_data = load_surface_from_subject(
            subject_dir, hemi="lh", surf_name="white", load_thickness=True, device=device
        )
        rh_data = load_surface_from_subject(
            subject_dir, hemi="rh", surf_name="white", load_thickness=True, device=device
        )

        orig_path = Path(subject_dir) / "mri" / "orig.mgz"
        if orig_path.exists():
            trg_header = nib.load(str(orig_path)).header
            trg_path = str(orig_path)
            logger.info(
                "Target reference: %s  shape=%s  voxel size=%s",
                orig_path,
                trg_header.get_data_shape()[:3],
                trg_header.get_zooms()[:3],
            )
        else:
            logger.warning("orig.mgz not found at %s — using moving image as target reference", orig_path)
            trg_header = mov_img.header
            trg_path = mov_path

    elif seg is not None:
        # ---- Mode C: extract surfaces from segmentation ----
        from .image.segmentation import surfaces_from_segmentation

        logger.info("Extracting WM surfaces from segmentation: %s", seg)
        seg_img = nib.load(seg)
        lh_data, rh_data = surfaces_from_segmentation(seg_img, hemispheres=("lh", "rh"), device=device)
        # The segmentation header serves as the target reference
        trg_header = seg_img.header
        trg_path = seg
        logger.info(
            "Target reference from segmentation header: shape=%s  voxel size=%s",
            trg_header.get_data_shape()[:3],
            trg_header.get_zooms()[:3],
        )

    else:
        # Load from explicit paths
        if lh_surf is None and rh_surf is None:
            raise ValueError("Must provide either subject_dir or lh_surf/rh_surf")

        lh_data = None
        rh_data = None

        if lh_surf is not None:
            logger.info("Loading left hemisphere surface: %s", lh_surf)
            lh_data = load_surface(lh_surf, lh_thickness, lh_cortex_label, device=device)

        if rh_surf is not None:
            logger.info("Loading right hemisphere surface: %s", rh_surf)
            rh_data = load_surface(rh_surf, rh_thickness, rh_cortex_label, device=device)

        # Use explicit ref image when provided, otherwise fall back to moving image
        if ref is not None:
            trg_header = (nib.load(ref) if isinstance(ref, str) else ref).header
            trg_path = (
                ref
                if isinstance(ref, str)
                else (trg_header.get_filename() if hasattr(trg_header, "get_filename") else None)
            )
            logger.info("Target reference: %s", trg_path)
        else:
            logger.warning(
                "No --ref provided for Mode B; using moving image as target reference. "
                "This is only correct if surfaces were built on the moving image."
            )
            trg_header = mov_img.header
            trg_path = mov_path

    # Surfaces live in target tkRAS space.
    # trg_tkras2ras maps target tkRAS → scanner RAS.
    trg_vox2tkras = get_vox2ras_tkr(trg_header)  # vox → tkRAS
    trg_tkras2ras = trg_header.get_best_affine() @ np.linalg.inv(trg_vox2tkras)  # tkRAS → RAS
    trg_tkras2ras_t = torch.from_numpy(trg_tkras2ras).float()

    # Moving image affine: RAS → moving vox  (inv is applied inside BBRModel)
    mov_affine_t = torch.from_numpy(mov_img.affine).float()

    logger.info("Moving image shape: %s", list(mov_data.shape))
    logger.debug("Target tkRAS→RAS:\n%s", trg_tkras2ras)
    logger.debug("Moving affine:\n%s", mov_img.affine)
    if lh_data is not None:
        logger.info("LH surface: %d vertices", lh_data["vertices"].shape[0])
    if rh_data is not None:
        logger.info("RH surface: %d vertices", rh_data["vertices"].shape[0])

    # Handle initialization
    init_transform = None
    if init_ras is not None:
        init_transform = torch.from_numpy(init_ras).float()
        logger.info("Using provided RAS-to-RAS init:\n%s", init_ras)
    elif init_type == "lta":
        if init_lta is None:
            raise ValueError("init_lta must be provided when init_type='lta'")
        logger.info("Loading init transform from LTA: %s", init_lta)
        # The stored convention is mov_RAS → trg_RAS; invert to get trg_RAS → mov_RAS.
        ras_mov_to_trg = LTA.read(init_lta).r2r()
        init_transform = torch.from_numpy(np.linalg.inv(ras_mov_to_trg)).float()
        logger.info("Loaded init transform (trg_RAS→mov_RAS) from %s", init_lta)
    elif init_type == "centroid":
        raise ValueError(
            "init_type='centroid' is not supported for surface-based registration: "
            "centroid alignment requires a pair of image volumes, but register_surface "
            "operates on surface meshes without a target image tensor. "
            "Use init_type='header' (identity) or supply an existing transform via "
            "init_type='lta' / init_ras."
        )
    else:  # header
        init_transform = torch.eye(4, dtype=torch.float32)

    logger.info(
        "Initializing BBR model  dof=%d  contrast=%s  cost=%s  subsample=%d", dof, contrast, cost_type, subsample
    )

    model = BBRModel(
        moving_volume=mov_data,
        lh_white_vertices=lh_data["vertices"] if lh_data is not None else None,
        lh_faces=lh_data["faces"] if lh_data is not None else None,
        rh_white_vertices=rh_data["vertices"] if rh_data is not None else None,
        rh_faces=rh_data["faces"] if rh_data is not None else None,
        lh_thickness=lh_data.get("thickness") if lh_data is not None else None,
        rh_thickness=rh_data.get("thickness") if rh_data is not None else None,
        lh_cortex_mask=lh_data.get("cortex_mask") if lh_data is not None else None,
        rh_cortex_mask=rh_data.get("cortex_mask") if rh_data is not None else None,
        trg_tkras2ras=trg_tkras2ras_t,  # target tkRAS → scanner RAS
        mov_affine=mov_affine_t,  # moving image vox-to-RAS affine
        dof=dof,
        init_transform=init_transform,
        contrast=contrast,
        wm_proj_abs=wm_proj_abs,
        gm_proj_frac=gm_proj_frac,
        gm_proj_abs=gm_proj_abs,
        slope=slope,
        cost_type=cost_type,
        gradient_weight=gradient_weight,
        subsample=subsample,
        device=device,
    ).to(device)

    logger.info("Optimizing: %d iterations  lr=%.4f", n_iters, lr)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    losses = []
    for iteration in range(n_iters):
        optimizer.zero_grad()
        cost = model()
        cost.backward()
        optimizer.step()
        losses.append(cost.item())
        if iteration % 10 == 0 or iteration == n_iters - 1:
            logger.info("  iter %4d  cost = %.6f", iteration, cost.item())

    final_transform = model.get_transform_matrix()
    elapsed = time.perf_counter() - start
    logger.info("Registration finished in %.2f s", elapsed)
    logger.debug("Final transform (trg_RAS→mov_RAS):\n%s", final_transform)

    if lta_name is not None:
        logger.info("Writing LTA file: %s", lta_name)
        # BBRModel returns trg_RAS → mov_RAS; LTA vox-to-vox needs mov_vox → trg_vox.
        ras_transform_np = final_transform.detach().cpu().numpy()
        ras_mov_to_trg = np.linalg.inv(ras_transform_np)
        vox_transform = convert_transform_type(
            ras_mov_to_trg,
            mov_img.affine,
            trg_header.get_best_affine(),
            from_type=LINEAR_RAS_TO_RAS,
            to_type=LINEAR_VOX_TO_VOX,
        )
        logger.debug("Vox-to-vox transform (src→target):\n%s", vox_transform)
        LTA.from_matrix(
            vox_transform,
            mov_path if mov_path else "moving.mgz",
            mov_img,
            trg_path if trg_path else "target.mgz",
            trg_header,
            lta_type=0,  # VOX_TO_VOX
        ).write(lta_name)

    return final_transform.detach(), model
