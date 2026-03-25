"""Command-line interface for IRLS robust registration (robreg_irls)."""

import argparse
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="robreg_irls",
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
        choices=[6, 12],
        metavar="{6,12}",
        help="Degrees of freedom: 6=rigid, 12=affine (IRLS currently supports 6 only).",
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
        "--symmetric",
        action="store_true",
        help="Use symmetric (halfway-space) registration (matches FreeSurfer default).",
    )

    # ── output options ──────────────────────────────────────────────────────
    p.add_argument(
        "--mapped",
        metavar="FILE",
        help="Save warped moving image to this file (MGZ format).",
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
    """Entry point for the ``robreg_irls`` command."""
    import nibabel as nib
    import torch

    from nireg.transforms import LTA
    from nireg.transforms.irls import register_irls_pyramid

    parser = _build_parser()
    ns = parser.parse_args(args)

    # ── logging ─────────────────────────────────────────────────────────────
    level = logging.DEBUG if ns.debug else (logging.INFO if ns.verbose else logging.WARNING)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("nireg.cli.robreg_irls")

    # ── load images ─────────────────────────────────────────────────────────
    logger.info("Loading moving image:    %s", ns.mov)
    logger.info("Loading reference image: %s", ns.ref)
    try:
        mov_img = nib.load(ns.mov)
        ref_img = nib.load(ns.ref)
    except Exception as exc:
        print(f"ERROR loading image: {exc}", file=sys.stderr)
        sys.exit(1)

    # Convert to tensors
    mov_data = torch.from_numpy(mov_img.get_fdata()).float()
    ref_data = torch.from_numpy(ref_img.get_fdata()).float()
    mov_affine = torch.from_numpy(mov_img.affine).float()
    ref_affine = torch.from_numpy(ref_img.affine).float()

    # ── register ────────────────────────────────────────────────────────────
    logger.info("Starting IRLS registration (dof=%d, symmetric=%s) …", 
                ns.dof, ns.symmetric)
    
    T_v2v, all_info = register_irls_pyramid(
        src=mov_data,
        trg=ref_data,
        src_affine=mov_affine,
        trg_affine=ref_affine,
        nmax=ns.nmax,
        sat=ns.sat,
        symmetric=ns.symmetric,
        isotropic=True,
        outliers_name=ns.outliers,  # Pass outliers filename
        verbose=ns.verbose or ns.debug,
    )

    # ── convert to r2r ──────────────────────────────────────────────────────
    # Convert v2v to r2r for LTA
    Mr2r = ref_affine.double() @ T_v2v.double() @ torch.inverse(mov_affine.double())

    # ── write LTA ───────────────────────────────────────────────────────────
    LTA.from_matrix(
        Mr2r.numpy(), 
        ns.mov, mov_img, 
        ns.ref, ref_img,
        lta_type=1  # RAS-to-RAS
    ).write(ns.out)
    logger.info("Wrote LTA: %s", ns.out)
    print(f"Transform: {ns.out}")

    # ── write mapped image if requested ─────────────────────────────────────
    if ns.mapped:
        from nireg.image.map import map_r2r
        
        mapped_data = map_r2r(
            mov_data,
            Mr2r.float(),
            source_affine=mov_affine,
            target_affine=ref_affine,
            target_shape=tuple(ref_img.shape[:3]),
            mode='bilinear'
        ).detach().numpy()
        
        mapped_img = nib.MGHImage(mapped_data, ref_img.affine, ref_img.header)
        mapped_img.to_filename(ns.mapped)
        logger.info("Wrote mapped image: %s", ns.mapped)
        print(f"Mapped:    {ns.mapped}")

    # Outliers file is already saved by register_irls_pyramid if requested
    if ns.outliers:
        print(f"Outliers:  {ns.outliers}")

    logger.info("Registration complete")


if __name__ == "__main__":
    main()

