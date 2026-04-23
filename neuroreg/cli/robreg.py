"""Command-line interface for IRLS-backed robust registration (robreg)."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, cast


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="robreg",
        description=(
            "IRLS robust 3-D image-to-image registration.\n"
            "Uses Iteratively Reweighted Least Squares with Tukey biweights,\n"
            "closely matching FreeSurfer's mri_robust_register algorithm."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── required ────────────────────────────────────────────────────────────
    p.add_argument("--mov", required=True, metavar="FILE",
                   help="Moving (source) image (NIfTI or MGZ).")
    p.add_argument("--ref", required=True, metavar="FILE",
                   help="Reference (target/fixed) image (NIfTI or MGZ).")
    p.add_argument("--out", required=True, metavar="LTA",
                   help="Output LTA file for the recovered transformation.")

    # ── transform ───────────────────────────────────────────────────────────
    p.add_argument(
        "--dof",
        type=int,
        default=6,
        choices=[6],
        metavar="{6}",
        help="Degrees of freedom: 6=rigid. IRLS robreg currently supports rigid registration only.",
    )

    # ── IRLS parameters ─────────────────────────────────────────────────────
    p.add_argument(
        "--nmax",
        type=int,
        default=5,
        metavar="N",
        help="Maximum number of outer IRLS iterations per pyramid level.",
    )
    p.add_argument(
        "--sat",
        type=float,
        default=6.0,
        metavar="FLOAT",
        help="Tukey biweight saturation threshold (higher = less robust).",
    )
    p.add_argument(
        "--nosym",
        dest="symmetric",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Disable symmetric halfway-space registration and run directed registration.",
    )
    init_group = p.add_mutually_exclusive_group()
    init_group.add_argument(
        "--init-header",
        dest="init_type",
        action="store_const",
        const="header",
        help="Use header alignment only.",
    )
    init_group.add_argument(
        "--init-centroid",
        dest="init_type",
        action="store_const",
        const="centroid",
        help="Initialize by aligning intensity centroids in RAS.",
    )
    init_group.add_argument(
        "--init-center",
        dest="init_type",
        action="store_const",
        const="image_center",
        help="Initialize by aligning geometric image centers in RAS (FreeSurfer cras0-style).",
    )

    # ── output options ──────────────────────────────────────────────────────
    p.add_argument(
        "--mapmov",
        metavar="FILE",
        help="Save the mapped moving image resliced into reference geometry.",
    )
    p.add_argument(
        "--mapmovhdr",
        metavar="FILE",
        help="Save a header-only mapped moving image with no interpolation.",
    )
    p.add_argument(
        "--outliers",
        metavar="FILE",
        help=(
            "Save outlier map (1 - Tukey weights) to this file (MGZ format). "
            "High values indicate poorly registered regions (outliers), "
            "low values indicate well-registered regions. "
            "Use with heat colormap in freeview for visualization."
        ),
    )

    # ── misc ────────────────────────────────────────────────────────────────
    p.add_argument("--verbose", action="store_true", help="Enable INFO-level logging.")
    p.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging.")

    return p


def main(args=None) -> None:
    """Entry point for the ``robreg`` command."""
    import nibabel as nib

    from neuroreg.image import save_header_mapped_image, save_resliced_r2r_image
    from neuroreg.imreg.robreg import robreg
    from neuroreg.transforms import LTA

    parser = _build_parser()
    ns = parser.parse_args(args)
    ns.symmetric = getattr(ns, "symmetric", True)

    # ── logging ─────────────────────────────────────────────────────────────
    level = logging.DEBUG if ns.debug else (logging.INFO if ns.verbose else logging.WARNING)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("neuroreg.cli.robreg")

    # ── load images ─────────────────────────────────────────────────────────
    logger.info("Loading moving image:    %s", ns.mov)
    logger.info("Loading reference image: %s", ns.ref)
    try:
        mov_img = nib.load(ns.mov)
        ref_img = nib.load(ns.ref)
    except Exception as exc:
        print(f"ERROR loading image: {exc}", file=sys.stderr)
        sys.exit(1)

    mov_img = cast(Any, mov_img)
    ref_img = cast(Any, ref_img)

    # ── register ────────────────────────────────────────────────────────────
    logger.info("Starting IRLS registration (dof=%d, symmetric=%s) …", ns.dof, ns.symmetric)
    kwargs: dict[str, Any] = dict(
        return_v2v=False,
        dof=ns.dof,
        nmax=ns.nmax,
        sat=ns.sat,
        symmetric=ns.symmetric,
        isotropic=True,
        outliers_name=ns.outliers,
        verbose=ns.verbose or ns.debug,
    )
    if ns.init_type is not None:
        kwargs["init_type"] = ns.init_type
    Mr2r = robreg(
        mov_img,
        ref_img,
        **kwargs,
    )
    Mr2r_cpu = Mr2r.detach().cpu()

    # ── write LTA ───────────────────────────────────────────────────────────
    LTA.from_matrix(
        Mr2r_cpu.numpy(),
        ns.mov,
        mov_img,
        ns.ref,
        ref_img,
        lta_type=1,
    ).write(ns.out)
    logger.info("Wrote LTA: %s", ns.out)
    print(f"Transform: {ns.out}")

    # ── write mapped image if requested ─────────────────────────────────────
    if ns.mapmov:
        target_shape = cast(tuple[int, int, int], tuple(int(v) for v in ref_img.shape[:3]))
        save_resliced_r2r_image(
            mov_img,
            Mr2r_cpu.numpy(),
            ns.mapmov,
            target_affine=ref_img.affine,
            target_shape=target_shape,
            mode="linear",
        )
        logger.info("Wrote resliced mapped image: %s", ns.mapmov)
        print(f"MapMov:    {ns.mapmov}")

    if ns.mapmovhdr:
        save_header_mapped_image(mov_img, Mr2r_cpu.numpy(), ns.mapmovhdr)
        logger.info("Wrote header-mapped image: %s", ns.mapmovhdr)
        print(f"MapMovHdr: {ns.mapmovhdr}")

    # Outliers file is already saved by register_irls_pyramid if requested
    if ns.outliers:
        if Path(ns.outliers).exists():
            print(f"Outliers:  {ns.outliers}")
        else:
            logger.warning("Outlier map was requested but no file was written: %s", ns.outliers)

    logger.info("Registration complete")


if __name__ == "__main__":
    main()
