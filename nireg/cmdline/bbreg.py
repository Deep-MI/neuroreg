#!/usr/bin/env python3
"""Command-line interface for boundary-based registration (bbreg)."""

import argparse
import logging
import sys


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

    # ── required ────────────────────────────────────────────────────────────
    p.add_argument(
        "--mov", required=True, metavar="FILE", help="Moving image to register (e.g. fMRI or T2, NIfTI or MGZ)."
    )
    p.add_argument("--out", required=True, metavar="LTA", help="Output LTA file for the recovered transformation.")

    # ── surface input: mode A ────────────────────────────────────────────────
    grp_a = p.add_argument_group("Mode A – FreeSurfer subject directory")
    grp_a.add_argument(
        "--subject_dir",
        metavar="DIR",
        help="Subject directory containing surf/lh.white, surf/rh.white, and mri/orig.mgz.",
    )

    # ── surface input: mode B ────────────────────────────────────────────────
    grp_b = p.add_argument_group("Mode B – explicit surface files")
    grp_b.add_argument(
        "--ref", metavar="FILE", help="Reference (T1) image against which surfaces were built (required for Mode B)."
    )
    grp_b.add_argument("--lh_surf", metavar="FILE", help="Left-hemisphere white surface (e.g. surf/lh.white).")
    grp_b.add_argument("--rh_surf", metavar="FILE", help="Right-hemisphere white surface (e.g. surf/rh.white).")
    grp_b.add_argument("--lh_thickness", metavar="FILE", help="Left-hemisphere cortical thickness file.")
    grp_b.add_argument("--rh_thickness", metavar="FILE", help="Right-hemisphere cortical thickness file.")

    # ── surface input: mode C ────────────────────────────────────────────────
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

    # ── transform ───────────────────────────────────────────────────────────
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

    # ── cost function ────────────────────────────────────────────────────────
    p.add_argument("--cost", default="contrast", choices=["contrast", "gradient", "both"], help="BBR cost term.")
    p.add_argument("--wm_proj_abs", type=float, default=1.4, metavar="MM", help="Absolute WM projection depth (mm).")
    p.add_argument(
        "--gm_proj_frac", type=float, default=0.5, metavar="FRAC", help="GM projection fraction of cortical thickness."
    )
    p.add_argument("--slope", type=float, default=0.5, help="Slope of the BBR sigmoid cost function.")
    p.add_argument("--gradient_weight", type=float, default=0.0, help="Weight for gradient cost term when --cost=both.")

    # ── optimisation ─────────────────────────────────────────────────────────
    p.add_argument("--n_iters", type=int, default=500, metavar="N", help="Number of RMSprop optimisation iterations.")
    p.add_argument("--lr", type=float, default=0.005, help="Optimiser learning rate.")
    p.add_argument("--subsample", type=int, default=2, metavar="N", help="Use every N-th surface vertex (1 = all).")

    # ── initialisation ───────────────────────────────────────────────────────
    p.add_argument(
        "--init_lta",
        metavar="FILE",
        help="Initialise registration from an existing LTA file "
        "(e.g. from a prior robreg run, or a previous bbreg pass).",
    )

    # ── misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--device", default="cpu", metavar="DEVICE", help="PyTorch device, e.g. 'cpu' or 'cuda'.")
    p.add_argument("--verbose", action="store_true", help="Enable INFO-level logging.")
    p.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging.")

    return p


def _validate_args(ns: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    """Validate surface-input mode and return 'subject_dir', 'explicit', or 'seg'."""
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

    if has_seg and ns.ref is not None:
        parser.error(
            "--ref is not needed with --seg (Mode C): the segmentation header provides the target reference geometry."
        )

    if has_sdir:
        return "subject_dir"
    if has_seg:
        return "seg"
    return "explicit"


def main(args=None) -> None:
    """Entry point for the ``bbreg`` command."""
    from nireg import register_surface

    parser = _build_parser()
    ns = parser.parse_args(args)
    mode = _validate_args(ns, parser)

    # ── logging ──────────────────────────────────────────────────────────────
    level = logging.DEBUG if ns.debug else (logging.INFO if ns.verbose else logging.WARNING)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("nireg.cli.bbreg")

    # ── build kwargs for register_surface ───────────────────────────────────
    kwargs: dict = dict(
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

    if ns.init_lta is not None:
        kwargs["init_lta"] = ns.init_lta
        kwargs["init_type"] = "lta"

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

    # ── run ──────────────────────────────────────────────────────────────────
    try:
        register_surface(**kwargs)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if ns.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    print(f"Output: {ns.out}")

if __name__ == "__main__":
    main()
