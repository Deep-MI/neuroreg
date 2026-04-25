#!/usr/bin/env python3
"""Command-line interface for boundary-based registration (bbreg)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from neuroreg.image import load_image


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bbreg",
        description=(
            "Boundary-based registration of a moving image to a T1 anatomy\n"
            "using cortical surface meshes.  Analogous to FreeSurfer's bbregister.\n"
            "\n"
            "Surface input — choose ONE of the following modes:\n"
            "  A) --subject_dir  : FreeSurfer / FastSurfer subject directory\n"
            "                      (surfaces and T1 reference loaded automatically)\n"
            "  B) --lh_surf / --rh_surf  : explicit surface file(s) + --ref\n"
            "  C) --seg          : parcellation / aseg file; surfaces are extracted\n"
            "                      automatically via marching cubes (no pre-built\n"
            "                      surface files needed)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--mov", required=True, metavar="FILE", help="Moving image to register (e.g. fMRI or T2, NIfTI or MGZ)."
    )
    p.add_argument("--out", required=True, metavar="LTA", help="Output LTA file for the recovered transformation.")

    grp_a = p.add_argument_group("Mode A – FreeSurfer subject directory")
    grp_a.add_argument(
        "--subject_dir",
        metavar="DIR",
        help="Subject directory containing surf/lh.white, surf/rh.white, and mri/orig.mgz.",
    )

    grp_b = p.add_argument_group("Mode B – explicit surface files")
    grp_b.add_argument(
        "--ref",
        metavar="FILE",
        help=(
            "Reference anatomical intensity image. Required for Mode B (explicit surfaces). "
            "Optional for Mode C (--seg) to drive coarse NMI prealignment."
        ),
    )
    grp_b.add_argument("--lh_surf", metavar="FILE", help="Left-hemisphere white surface (e.g. surf/lh.white).")
    grp_b.add_argument("--rh_surf", metavar="FILE", help="Right-hemisphere white surface (e.g. surf/rh.white).")
    grp_b.add_argument("--lh_thickness", metavar="FILE", help="Left-hemisphere cortical thickness file.")
    grp_b.add_argument("--rh_thickness", metavar="FILE", help="Right-hemisphere cortical thickness file.")

    grp_c = p.add_argument_group(
        "Mode C – segmentation (aparc+aseg / aseg)",
        "White surfaces are extracted on-the-fly via marching cubes. "
        "No pre-built surface files are needed and the segmentation header "
        "provides the target reference geometry.",
    )
    grp_c.add_argument("--seg", metavar="FILE", help="Parcellation file (aparc+aseg.mgz, aseg.mgz, or NIfTI).")
    grp_c.add_argument(
        "--seg_smooth_sigma",
        type=float,
        default=0.5,
        metavar="SIGMA",
        help="Gaussian pre-blur sigma (voxels) before marching cubes. Default: 0.5.",
    )
    grp_c.add_argument(
        "--seg_mc_level", type=float, default=0.45, metavar="LEVEL", help="Marching-cubes iso-level. Default: 0.45."
    )
    grp_c.add_argument(
        "--seg_smooth_iters",
        type=int,
        default=50,
        metavar="N",
        help="Taubin smoothing iterations after marching cubes. Default: 50.",
    )

    p.add_argument(
        "--dof",
        type=int,
        default=6,
        choices=[6, 9, 12],
        metavar="{6,9,12}",
        help="Degrees of freedom: 6=rigid, 9=rigid+scale, 12=affine.",
    )
    p.add_argument(
        "--contrast",
        default=None,
        choices=["t1", "t2"],
        help="Tissue contrast: 't1' (WM>GM) or 't2' (GM>WM). Auto-detected from the image when not specified.",
    )

    p.add_argument("--cost", default="contrast", choices=["contrast", "gradient", "both"], help="BBR cost term.")
    p.add_argument("--wm_proj_abs", type=float, default=1.4, metavar="MM", help="Absolute WM projection depth (mm).")
    p.add_argument(
        "--gm_proj_frac", type=float, default=0.5, metavar="FRAC", help="GM projection fraction of cortical thickness."
    )
    p.add_argument("--slope", type=float, default=0.5, help="Slope of the BBR sigmoid cost function.")
    p.add_argument("--gradient_weight", type=float, default=0.0, help="Weight for gradient cost term when --cost=both.")

    p.add_argument("--n_iters", type=int, default=200, metavar="N", help="Number of RMSprop optimisation iterations.")
    p.add_argument("--lr", type=float, default=0.005, help="Optimiser learning rate.")
    p.add_argument("--subsample", type=int, default=2, metavar="N", help="Use every N-th surface vertex (1 = all).")

    p.add_argument(
        "--init-lta",
        dest="init_lta",
        metavar="FILE",
        help=(
            "Initialise registration from an existing LTA file "
            "(e.g. from a prior robreg run, or a previous bbreg pass)."
        ),
    )
    p.add_argument(
        "--init-header",
        action="store_true",
        help=(
            "Assume that the geometry information in the cross-modal and anatomical are sufficient to get a close "
            "voxel-to-voxel registration. This usually is only the case if they were acquired in the same session."
        ),
    )
    p.add_argument(
        "--no-coreg-ref-mask",
        action="store_true",
        help=(
            "Do not use aparc+aseg/aseg (or the provided segmentation) as a reference mask for the coarse "
            "NMI prealignment stage."
        ),
    )

    p.add_argument(
        "--device",
        default="cpu",
        metavar="DEVICE",
        help="Torch device string, e.g. 'cpu', 'cuda', 'mps', or 'gpu'.",
    )
    p.add_argument("--mapmov", metavar="FILE", help="Save the mapped moving image resliced into target geometry.")
    p.add_argument(
        "--mapmovhdr",
        metavar="FILE",
        help="Save a header-only mapped moving image with no interpolation.",
    )
    p.add_argument("--verbose", action="store_true", help="Enable INFO-level logging.")
    p.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging.")

    return p


def _validate_args(ns: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    """Validate mutually exclusive surface-input modes.

    Returns one of ``"subject_dir"``, ``"explicit"``, or ``"seg"`` so later
    CLI stages can dispatch loading and prealignment logic without repeating the
    same checks.
    """
    has_sdir = ns.subject_dir is not None
    has_explicit = ns.lh_surf is not None or ns.rh_surf is not None
    has_seg = ns.seg is not None

    n_modes = sum([has_sdir, has_explicit, has_seg])
    if n_modes > 1:
        parser.error(
            "--subject_dir, --lh_surf/--rh_surf, and --seg are mutually exclusive. "
            "Choose exactly one surface input mode."
        )

    if n_modes == 0:
        parser.error(
            "Surface input is required.  Provide one of:\n"
            "  --subject_dir DIR   (Mode A)\n"
            "  --lh_surf / --rh_surf FILE  (Mode B, also needs --ref)\n"
            "  --seg FILE          (Mode C)"
        )

    if has_explicit and ns.ref is None:
        parser.error("--ref is required when using Mode B (explicit surface files).")

    if ns.init_header and ns.init_lta is not None:
        parser.error("--init-header and --init-lta are mutually exclusive.")

    if has_sdir:
        return "subject_dir"
    if has_seg:
        return "seg"
    return "explicit"


def _load_reference_image_for_mode(ns: argparse.Namespace, mode: str) -> Any | None:
    """Load the anatomical intensity image used for optional NMI prealignment.

    The returned image is the fixed/target image for the coarse image-to-image
    prealignment stage that seeds the surface-based BBR optimisation. Some modes
    do not always have such an image available, in which case ``None`` is
    returned and the CLI falls back to header initialization.
    """
    if mode == "subject_dir":
        orig_path = Path(ns.subject_dir) / "mri" / "orig.mgz"
        return load_image(orig_path)
    if mode == "explicit":
        return load_image(ns.ref)
    if mode == "seg" and ns.ref is not None:
        return load_image(ns.ref)
    return None


def _load_target_geometry_image(ns: argparse.Namespace, mode: str) -> Any:
    """Load the final target geometry used for mapped output export.

    Parameters
    ----------
    ns : argparse.Namespace
        Parsed CLI arguments.
    mode : {"subject_dir", "explicit", "seg"}
        Resolved surface-input mode from :func:`_validate_args`.

    Returns
    -------
    Any
        Nibabel-like image defining the target/reference geometry of the final
        BBR solution.
    """
    if mode == "subject_dir":
        return load_image(Path(ns.subject_dir) / "mri" / "orig.mgz")
    if mode == "seg":
        return load_image(ns.seg)
    return load_image(ns.ref)


def _load_prealign_mask_image(ns: argparse.Namespace, mode: str) -> Any | None:
    """Load the mask used to focus coarse NMI prealignment on relevant anatomy.

    In ``subject_dir`` mode this prefers ``aparc+aseg.mgz`` and falls back to
    ``aseg.mgz``. In ``seg`` mode the segmentation itself acts as the mask.
    Explicit-surface mode does not define a default mask.
    """
    if mode == "subject_dir":
        mri_dir = Path(ns.subject_dir) / "mri"
        for name in ("aparc+aseg.mgz", "aseg.mgz"):
            path = mri_dir / name
            if path.exists():
                return load_image(path)
        return None
    if mode == "seg":
        return load_image(ns.seg)
    return None


def _mask_reference_image(
        ref_img: Any,
        mask_img: Any | None,
) -> Any:
    """Apply a binary mask to the fixed/reference image for NMI prealignment.

    Voxels outside the mask are set to zero so the coarse image registration is
    driven more by intracranial anatomy and less by unrelated head/background
    content.
    """
    if mask_img is None:
        return ref_img

    ref_data = np.asarray(ref_img.get_fdata(), dtype=np.float32)
    mask_data = np.asarray(mask_img.get_fdata()) > 0
    if mask_data.shape != ref_data.shape:
        raise ValueError(f"Prealignment mask shape {mask_data.shape} does not match reference shape {ref_data.shape}.")

    masked = ref_data * mask_data.astype(np.float32)
    return nib.Nifti1Image(masked, ref_img.affine)


def _run_default_nmi_prealign(
        mov_img: Any,
        ref_img: Any,
        mask_img: Any | None,
        logger: logging.Logger,
        device: str,
) -> np.ndarray:
    """Run the default coarse image-based prealignment for ``bbreg``.

    This uses the Powell-based image-registration path with the standard
    MRI_coreg-style NMI evaluator and an image-center initialization. The
    returned transform is always a RAS-to-RAS matrix in public
    ``moving/source -> target/reference`` direction so it can be passed
    directly to :func:`neuroreg.bbreg.register.register_surface` as
    ``init_ras``.
    """
    from neuroreg.imreg.coreg import coreg

    prealign_ref = _mask_reference_image(ref_img, mask_img)
    logger.info(
        "Running default Powell NMI prealignment (image-center start, sep=4)%s",
        " with aparc+aseg/aseg mask" if mask_img is not None else "",
    )
    Mr2r = coreg(
        mov_img,
        prealign_ref,
        return_v2v=False,
        method="powell",
        init_type="image_center",
        dof=6,
        powell_sep=4,
        device=device,
    )
    Mr2r_np = Mr2r.detach().cpu().numpy()
    logger.info("Finished NMI prealignment (mov_RAS->trg_RAS):\n%s", Mr2r_np)
    return Mr2r_np


def main(args=None) -> None:
    """Entry point for the ``bbreg`` command-line interface.

    The CLI normalizes the different input modes, optionally runs a coarse NMI
    prealignment to obtain a ``moving -> target`` initialization, and then calls
    :func:`neuroreg.bbreg.register.register_surface` with a consistent public
    transform direction. When requested, it also exports resliced or header-only
    mapped versions of the moving image using the same shared mapping helpers as
    the other registration CLIs.
    """
    from neuroreg.bbreg.register import register_surface
    from neuroreg.image import save_header_mapped_image, save_resliced_r2r_image

    parser = _build_parser()
    ns = parser.parse_args(args)
    mode = _validate_args(ns, parser)

    level = logging.DEBUG if ns.debug else (logging.INFO if ns.verbose else logging.WARNING)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("neuroreg.cli.bbreg")

    kwargs: dict[str, Any] = dict(
        mov=ns.mov,
        lta_name=ns.out,
        dof=ns.dof,
        contrast=ns.contrast,
        cost_type=ns.cost,
        wm_proj_abs=ns.wm_proj_abs,
        gm_proj_frac=ns.gm_proj_frac,
        slope=ns.slope,
        gradient_weight=ns.gradient_weight,
        n_iters=ns.n_iters,
        lr=ns.lr,
        subsample=ns.subsample,
        device=ns.device,
    )

    mov_img = load_image(ns.mov)
    ref_img = _load_reference_image_for_mode(ns, mode)
    mask_img = None
    if ref_img is not None and not ns.no_coreg_ref_mask:
        mask_img = _load_prealign_mask_image(ns, mode)

    if ns.init_lta is not None:
        logger.info("Using explicit LTA initialization: %s", ns.init_lta)
        kwargs["init_lta"] = ns.init_lta
        kwargs["init_type"] = "lta"
    elif ns.init_header:
        logger.info("Using header-only initialization (--init-header)")
        kwargs["init_type"] = "header"
    elif ref_img is not None:
        kwargs["init_ras"] = _run_default_nmi_prealign(mov_img, ref_img, mask_img, logger, ns.device)
    else:
        logger.info("No anatomical intensity reference available for NMI prealignment; falling back to header init")
        kwargs["init_type"] = "header"

    if mode == "subject_dir":
        logger.info("Mode A: subject directory %s", ns.subject_dir)
        kwargs["subject_dir"] = ns.subject_dir
    elif mode == "seg":
        logger.info("Mode C: segmentation file %s", ns.seg)
        kwargs["seg"] = ns.seg
    else:
        logger.info("Mode B: explicit surface file(s)")
        if ns.lh_surf is not None:
            kwargs["lh_surf"] = ns.lh_surf
        if ns.rh_surf is not None:
            kwargs["rh_surf"] = ns.rh_surf
        if ns.lh_thickness is not None:
            kwargs["lh_thickness"] = ns.lh_thickness
        if ns.rh_thickness is not None:
            kwargs["rh_thickness"] = ns.rh_thickness
        kwargs["ref"] = ns.ref

    try:
        mr2r = register_surface(**kwargs)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if ns.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    print(f"Output: {ns.out}")
    if ns.mapmov or ns.mapmovhdr:
        target_img = _load_target_geometry_image(ns, mode)
        mr2r_np = mr2r.detach().cpu().numpy()
        target_shape = tuple(int(v) for v in target_img.shape[:3])
        if ns.mapmov:
            save_resliced_r2r_image(
                mov_img,
                mr2r_np,
                ns.mapmov,
                target_affine=target_img.affine,
                target_shape=target_shape,
                mode="linear",
            )
            logger.info("Wrote resliced mapped image: %s", ns.mapmov)
            print(f"MapMov:    {ns.mapmov}")
        if ns.mapmovhdr:
            save_header_mapped_image(mov_img, mr2r_np, ns.mapmovhdr)
            logger.info("Wrote header-mapped image: %s", ns.mapmovhdr)
            print(f"MapMovHdr: {ns.mapmovhdr}")


if __name__ == "__main__":
    main()
