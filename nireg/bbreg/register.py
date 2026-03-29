import logging
import time
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import torch

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
    n_iters: int = 100,
    lr: float = 0.01,
    device: str = "cpu",
) -> tuple[torch.Tensor, BBRModel]:
    """Register a moving image to cortical surface boundaries using BBR."""
    start = time.perf_counter()

    if isinstance(mov, str):
        mov_img = nib.load(mov)
        mov_path = mov
    else:
        mov_img = mov
        mov_path = mov.get_filename() if hasattr(mov, "get_filename") else None

    mov_data = torch.from_numpy(mov_img.get_fdata()).float()

    if subject_dir is not None:
        logger.info("Loading surfaces from subject directory: %s", subject_dir)
        lh_data = load_surface_from_subject(subject_dir, hemi="lh", surf_name="white", load_thickness=True, device=device)
        rh_data = load_surface_from_subject(subject_dir, hemi="rh", surf_name="white", load_thickness=True, device=device)

        orig_path = Path(subject_dir) / "mri" / "orig.mgz"
        if orig_path.exists():
            trg_img = nib.load(str(orig_path))
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
        seg_img = nib.load(seg)
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
            trg_img = nib.load(ref) if isinstance(ref, str) else ref
            trg_header = trg_img.header
            trg_path = ref if isinstance(ref, str) else (trg_header.get_filename() if hasattr(trg_header, "get_filename") else None)
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
        init_transform = torch.from_numpy(init_ras).float()
        logger.info("Using provided RAS-to-RAS init:\n%s", init_ras)
    elif init_type == "lta":
        if init_lta is None:
            raise ValueError("init_lta must be provided when init_type='lta'")
        logger.info("Loading init transform from LTA: %s", init_lta)
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
        ras_transform_np = final_transform.detach().cpu().numpy()
        ras_mov_to_trg = np.linalg.inv(ras_transform_np)
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

    return final_transform.detach(), model


__all__ = ["register_surface"]


