"""Command-line interface for robust image-to-image registration (robreg)."""

import argparse
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="robreg",
        description=(
            "Robust 3-D image-to-image registration using PyTorch optimisation.\n"
            "Analogous to FreeSurfer's mri_robust_register."
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
    p.add_argument("--dof", type=int, default=6, choices=[3, 6, 9, 12],
                   metavar="{3,6,9,12}",
                   help="Degrees of freedom: 3=translation, 6=rigid, "
                        "9=rigid+scale, 12=affine.")

    # ── optimisation ────────────────────────────────────────────────────────
    p.add_argument("--n_iters", type=int, default=None, metavar="N",
                   help="Maximum number of optimisation iterations per pyramid level "
                        "(default: auto from register_pyramid).")
    p.add_argument("--device", default="cpu", metavar="DEVICE",
                   help="PyTorch device, e.g. 'cpu' or 'cuda'.")

    # ── misc ────────────────────────────────────────────────────────────────
    p.add_argument("--verbose", action="store_true",
                   help="Enable INFO-level logging.")
    p.add_argument("--debug", action="store_true",
                   help="Enable DEBUG-level logging.")

    return p


def main(args=None) -> None:
    """Entry point for the ``robreg`` command."""
    import nibabel as nib

    from nireg import register_pyramid
    from nireg.transforms import write_lta

    parser = _build_parser()
    ns = parser.parse_args(args)

    # ── logging ─────────────────────────────────────────────────────────────
    level = logging.DEBUG if ns.debug else (logging.INFO if ns.verbose else logging.WARNING)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("nireg.cli.robreg")

    # ── load images (needed for write_lta geometry metadata) ────────────────
    logger.info("Loading moving image:    %s", ns.mov)
    logger.info("Loading reference image: %s", ns.ref)
    try:
        mov_img = nib.load(ns.mov)
        ref_img = nib.load(ns.ref)
    except Exception as exc:
        print(f"ERROR loading image: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── register ────────────────────────────────────────────────────────────
    # Pass paths so register_pyramid handles loading internally.
    # return_v2v=True → vox-to-vox matrix, consistent with lta_type=0 below.
    kwargs = dict(dof=ns.dof, device=ns.device, return_v2v=True)
    if ns.n_iters is not None:
        kwargs["n"] = ns.n_iters

    logger.info("Starting image-to-image registration (dof=%d) …", ns.dof)
    v2v = register_pyramid(ns.mov, ns.ref, **kwargs)

    # ── write LTA ───────────────────────────────────────────────────────────
    # lta_type=0 (LINEAR_VOX_TO_VOX) matches the vox-to-vox matrix returned
    # by register_pyramid(..., return_v2v=True).
    write_lta(ns.out, v2v.numpy(), ns.mov, mov_img, ns.ref, ref_img, lta_type=0)
    logger.info("Wrote LTA: %s", ns.out)
    print(f"Output: {ns.out}")

