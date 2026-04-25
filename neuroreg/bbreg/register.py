import logging
import time
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import torch

from ..image import load_image
from ..transforms import LINEAR_RAS_TO_RAS, LINEAR_VOX_TO_VOX, LTA, convert_transform_type
from .io import get_vox2ras_tkr, load_surface, load_surface_from_subject
from .optimize import BBRModel

logger = logging.getLogger(__name__)


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
        n_iters: int = 200,
        lr: float = 0.01,
        early_stop_patience: int = 20,
        device: str = "cpu",
        return_model: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, BBRModel]:
    """Register a moving image to cortical surface boundaries using BBR.

    This is the main Python API for the boundary-based registration path. The
    moving image is aligned to a target anatomical space defined either by a
    FreeSurfer/FastSurfer subject directory, explicit white-matter surface
    files plus a reference image, or a segmentation from which surfaces are
    extracted on the fly.

    Public transform direction is always ``moving/source -> target/reference``.
    That convention applies to ``init_ras``, ``init_lta``, the returned tensor,
    and any written LTA. Internally the BBR model optimizes the inverse
    transform because that is the natural parameterization for sampling the
    moving volume at target-surface locations, but that internal detail is
    hidden at the API boundary.

    Parameters
    ----------
    mov : str or nib.Nifti1Image
        Moving/source image to align into the target/reference space.
    lh_surf, rh_surf : str, optional
        Explicit left/right white-matter surface files for surface-input mode.
    lh_thickness, rh_thickness : str, optional
        Optional cortical thickness files paired with ``lh_surf`` and
        ``rh_surf``.
    ref : str or nib.Nifti1Image, optional
        Reference anatomical image used with explicit-surface mode.
    subject_dir : str, optional
        FreeSurfer/FastSurfer subject directory providing surfaces and
        ``mri/orig.mgz``.
    seg : str, optional
        Segmentation volume used to extract white-matter surfaces on the fly.
    lta_name : str, optional
        Output path for a written LTA in public ``moving -> target`` direction.
    dof : int, default=6
        Transformation degrees of freedom.
    contrast : {"t1", "t2"}, optional
        Expected image contrast for the BBR intensity model. When ``None``, the
        model auto-detects the polarity.
    init_type : {"header", "lta"}, default="header"
        Initialization source when ``init_ras`` is not supplied.
    init_lta : str, optional
        Existing LTA used for initialization. It must encode a
        ``moving/source -> target/reference`` transform.
    init_ras : ndarray, optional
        Initial 4x4 RAS-to-RAS transform in public ``moving/source ->
        target/reference`` direction.
    cost_type : {"contrast", "gradient", "both"}, default="contrast"
        Cost terms included in the BBR objective.
    wm_proj_abs : float, default=1.4
        White-matter sampling depth in millimetres.
    gm_proj_frac : float, default=0.5
        Gray-matter sampling depth as a fraction of cortical thickness.
    gm_proj_abs : float, optional
        Absolute gray-matter projection depth overriding ``gm_proj_frac``.
    lh_cortex_label, rh_cortex_label : str, optional
        Optional cortex label files restricting sampled vertices.
    slope : float, default=0.5
        Slope of the sigmoid used in the contrast cost.
    gradient_weight : float, default=0.0
        Relative weight of the gradient term when ``cost_type='both'``.
    subsample : int, default=1
        Use every ``subsample``-th surface vertex during optimization.
    n_iters : int, default=200
        Maximum number of RMSprop iterations.
    lr : float, default=0.01
        RMSprop learning rate.
    early_stop_patience : int, default=20
        Stop after this many non-improving iterations. Set ``0`` to disable
        early stopping.
    device : str, default="cpu"
        Torch device on which to run the optimization.
    return_model : bool, default=False
        If ``True``, also return the fitted ``BBRModel`` for debugging or
        inspection.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, BBRModel]
        By default, returns the best-found RAS-to-RAS transform in public
        ``moving/source -> target/reference`` direction. When
        ``return_model=True``, also returns the fitted ``BBRModel``.

    Raises
    ------
    ValueError
        If the requested input mode is incomplete or unsupported.
    RuntimeError
        If optimization fails to produce any valid iterate.
    """
    start = time.perf_counter()

    if isinstance(mov, str):
        mov_img = load_image(mov)
        mov_path = mov
    else:
        mov_img = mov
        mov_path = mov.get_filename() if hasattr(mov, "get_filename") else None

    mov_data = torch.from_numpy(mov_img.get_fdata()).float()

    if subject_dir is not None:
        logger.info("Loading surfaces from subject directory: %s", subject_dir)
        lh_data = load_surface_from_subject(
            subject_dir,
            hemi="lh",
            surf_name="white",
            load_thickness=True,
            device=device,
        )
        rh_data = load_surface_from_subject(
            subject_dir,
            hemi="rh",
            surf_name="white",
            load_thickness=True,
            device=device,
        )

        orig_path = Path(subject_dir) / "mri" / "orig.mgz"
        if orig_path.exists():
            trg_img = load_image(orig_path)
            trg_header = trg_img.header
            trg_path = str(orig_path)
            logger.info(
                "Target reference: %s  shape=%s  voxel size=%s",
                orig_path,
                trg_header.get_data_shape()[:3],
                trg_header.get_zooms()[:3],
            )
        else:
            logger.warning("orig.mgz not found at %s — using moving image as target reference", orig_path)
            trg_img = mov_img
            trg_header = mov_img.header
            trg_path = mov_path

    elif seg is not None:
        from ..image.segmentation import surfaces_from_segmentation

        logger.info("Extracting WM surfaces from segmentation: %s", seg)
        seg_img = load_image(seg)
        trg_img = seg_img
        lh_data, rh_data = surfaces_from_segmentation(seg_img, hemispheres=("lh", "rh"), device=device)
        trg_header = seg_img.header
        trg_path = seg
        logger.info(
            "Target reference from segmentation header: shape=%s  voxel size=%s",
            trg_header.get_data_shape()[:3],
            trg_header.get_zooms()[:3],
        )

    else:
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

        if ref is not None:
            trg_img = load_image(ref) if isinstance(ref, str) else ref
            trg_header = trg_img.header
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
            trg_img = mov_img
            trg_header = mov_img.header
            trg_path = mov_path

    trg_vox2tkras = get_vox2ras_tkr(trg_header)
    trg_tkras2ras = trg_img.affine @ np.linalg.inv(trg_vox2tkras)
    trg_tkras2ras_t = torch.from_numpy(trg_tkras2ras).float()
    mov_affine_t = torch.from_numpy(mov_img.affine).float()

    logger.info("Moving image shape: %s", list(mov_data.shape))
    logger.debug("Target tkRAS→RAS:\n%s", trg_tkras2ras)
    logger.debug("Moving affine:\n%s", mov_img.affine)
    if lh_data is not None:
        logger.info("LH surface: %d vertices", lh_data["vertices"].shape[0])
    if rh_data is not None:
        logger.info("RH surface: %d vertices", rh_data["vertices"].shape[0])

    if init_ras is not None:
        ras_mov_to_trg = np.asarray(init_ras, dtype=np.float64)
        init_transform = torch.from_numpy(np.linalg.inv(ras_mov_to_trg)).float()
        logger.info("Using provided RAS-to-RAS init (mov_RAS→trg_RAS):\n%s", ras_mov_to_trg)
    elif init_type == "lta":
        if init_lta is None:
            raise ValueError("init_lta must be provided when init_type='lta'")
        logger.info("Loading init transform from LTA: %s", init_lta)
        ras_mov_to_trg = LTA.read(init_lta).r2r()
        init_transform = torch.from_numpy(np.linalg.inv(ras_mov_to_trg)).float()
        logger.info("Loaded init transform (mov_RAS→trg_RAS) from %s", init_lta)
    elif init_type == "centroid":
        raise ValueError(
            "init_type='centroid' is not supported for surface-based registration: "
            "centroid alignment requires a pair of image volumes, but register_surface "
            "operates on surface meshes without a target image tensor. "
            "Use init_type='header' (identity) or supply an existing transform via "
            "init_type='lta' / init_ras."
        )
    else:
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
        trg_tkras2ras=trg_tkras2ras_t,
        mov_affine=mov_affine_t,
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

    logger.info(
        "Optimizing: %d iterations  lr=%.4f  early_stop_patience=%d",
        n_iters,
        lr,
        early_stop_patience,
    )
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    losses = []
    best_cost = float("inf")
    best_iteration = -1
    iters_since_best = 0
    best_transform: torch.Tensor | None = None
    best_params: torch.Tensor | None = None

    for iteration in range(n_iters):
        optimizer.zero_grad()
        cost = model()
        cost.backward()
        optimizer.step()

        with torch.no_grad():
            current_cost = model().item()
            losses.append(current_cost)
            if current_cost < best_cost:
                best_cost = current_cost
                best_iteration = iteration
                iters_since_best = 0
                best_transform = model.get_transform_matrix().detach().clone()
                best_params = model.transform_params.detach().clone()
            else:
                iters_since_best += 1

        if iteration % 10 == 0 or iteration == n_iters - 1:
            logger.info("  iter %4d  cost = %.6f  best = %.6f @ %d", iteration, current_cost, best_cost, best_iteration)

        if early_stop_patience > 0 and iters_since_best >= early_stop_patience:
            logger.info(
                "Early stopping at iter %d after %d iterations without improvement; best cost %.6f at iter %d",
                iteration,
                iters_since_best,
                best_cost,
                best_iteration,
            )
            break

    if best_transform is None or best_params is None:
        raise RuntimeError("BBR optimization did not produce any iterate.")

    with torch.no_grad():
        model.transform_params.copy_(best_params)

    final_transform = best_transform
    ras_transform_np = final_transform.detach().cpu().numpy()
    ras_mov_to_trg = np.linalg.inv(ras_transform_np)
    elapsed = time.perf_counter() - start
    logger.info("Registration finished in %.2f s", elapsed)
    logger.info("Using best iterate: cost = %.6f at iter %d", best_cost, best_iteration)
    logger.debug("Final transform (mov_RAS→trg_RAS):\n%s", ras_mov_to_trg)

    if lta_name is not None:
        logger.info("Writing LTA file: %s", lta_name)
        vox_transform = convert_transform_type(
            ras_mov_to_trg,
            mov_img.affine,
            trg_img.affine,
            from_type=LINEAR_RAS_TO_RAS,
            to_type=LINEAR_VOX_TO_VOX,
        )
        logger.debug("Vox-to-vox transform (src→target):\n%s", vox_transform)
        LTA.from_matrix(
            vox_transform,
            mov_path if mov_path else "moving.mgz",
            mov_img,
            trg_path if trg_path else "target.mgz",
            trg_img,
            lta_type=0,
        ).write(lta_name)

    transform_bbreg = torch.from_numpy(ras_mov_to_trg).to(best_transform)
    if return_model:
        return transform_bbreg, model
    return transform_bbreg


__all__ = ["register_surface"]
