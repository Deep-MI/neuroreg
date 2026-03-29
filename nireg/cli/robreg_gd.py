#!/usr/bin/env python3
"""Command-line interface for legacy gradient-descent image registration (robreg_gd)."""

import argparse
import logging
import sys
from typing import Any, cast


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="robreg_gd",
        description=(
            "Legacy gradient-descent 3-D image-to-image registration using PyTorch optimisation.\n"
            "This is the pre-IRLS robreg path."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── required ────────────────────────────────────────────────────────────
    p.add_argument("--mov", required=True, metavar="FILE", help="Moving (source) image (NIfTI or MGZ).")
    p.add_argument("--ref", required=True, metavar="FILE", help="Reference (target/fixed) image (NIfTI or MGZ).")
    p.add_argument("--out", required=True, metavar="LTA", help="Output LTA file for the recovered transformation.")

    # ── transform ───────────────────────────────────────────────────────────
    p.add_argument(
        "--dof",
        type=int,
        default=6,
        choices=[3, 6, 9, 12],
        metavar="{3,6,9,12}",
        help="Degrees of freedom: 3=translation, 6=rigid, 9=rigid+scale, 12=affine.",
    )

    # ── optimisation ────────────────────────────────────────────────────────
    p.add_argument(
        "--n_iters",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of optimisation iterations per pyramid level (default: auto from register_pyramid).",
    )
    p.add_argument(
        "--noinit",
        action="store_true",
        help="Skip centroid-based initialization and start from identity (matches FreeSurfer --noinit).",
    )
    p.add_argument("--device", default="cpu", metavar="DEVICE", help="PyTorch device, e.g. 'cpu' or 'cuda'.")

    # ── misc ────────────────────────────────────────────────────────────────
    p.add_argument("--verbose", action="store_true", help="Enable INFO-level logging.")
    p.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging.")

    return p


def main(args=None) -> None:
    """Entry point for the ``robreg_gd`` command."""
    import nibabel as nib

    from nireg.imreg.robreg_gd import register_pyramid
    from nireg.transforms import LTA

    parser = _build_parser()
    ns = parser.parse_args(args)

    # ── logging ─────────────────────────────────────────────────────────────
    level = logging.DEBUG if ns.debug else (logging.INFO if ns.verbose else logging.WARNING)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("nireg.cli.robreg_gd")

    # ── load images (needed for LTA.write geometry metadata) ────────────────
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
    # Pass paths so register_pyramid handles loading internally.
    # return_v2v=True → vox-to-vox matrix, consistent with lta_type=0 below.
    kwargs = dict(dof=ns.dof, device=ns.device, return_v2v=True, centroid_init=not ns.noinit)
    if ns.n_iters is not None:
        kwargs["n"] = ns.n_iters

    logger.info("Starting image-to-image registration (dof=%d) …", ns.dof)
    v2v = register_pyramid(mov_img, ref_img, **kwargs)

    # ── write LTA ───────────────────────────────────────────────────────────
    # lta_type=0 (LINEAR_VOX_TO_VOX) matches the vox-to-vox matrix returned
    # by register_pyramid(..., return_v2v=True).
    # LTA.from_matrix() constructs the LTA object; .write() serialises it to disk.
    LTA.from_matrix(v2v.numpy(), ns.mov, cast(Any, mov_img), ns.ref, cast(Any, ref_img), lta_type=0).write(ns.out)
    logger.info("Wrote LTA: %s", ns.out)
    print(f"Output: {ns.out}")

<<<<<<< HEAD
if __name__ == "__main__":
    main()
=======

if __name__ == "__main__":
    main()

>>>>>>> 05f7c62 (auto call main)
