#!/usr/bin/env python3
"""Command-line interface for image-based cross-modal registration (coreg)."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, cast


def _parse_int_csv(value: str) -> list[int]:
    """Parse a coarse-to-fine pyramid iteration schedule from the CLI.

    The resulting list is forwarded to ``coreg(level_iters=...)``.
    Users can pass ``0`` for any intermediate level they want to skip.
    """
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers")
    try:
        return [int(part) for part in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers") from exc


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="coreg",
        description=(
            "Gradient-descent 3-D image-to-image registration using PyTorch optimisation.\n"
            "This is the main image-based cross-modal registration path."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--mov", required=True, metavar="FILE", help="Moving (source) image (NIfTI or MGZ).")
    p.add_argument("--ref", required=True, metavar="FILE", help="Reference (target/fixed) image (NIfTI or MGZ).")
    p.add_argument("--out", required=True, metavar="LTA", help="Output LTA file for the recovered transformation.")

    p.add_argument(
        "--dof",
        type=int,
        default=6,
        choices=[3, 6, 9, 12],
        metavar="{3,6,9,12}",
        help="Degrees of freedom: 3=translation, 6=rigid, 9=rigid+scale, 12=affine.",
    )

    p.add_argument(
        "--n_iters",
        type=int,
        default=None,
        metavar="N",
        help="Uniform number of optimisation iterations per pyramid level.",
    )
    p.add_argument(
        "--level-iters",
        type=_parse_int_csv,
        default=None,
        help="Comma-separated per-level iteration schedule in coarse->fine order. Use 0 to skip a level.",
    )
    p.add_argument("--lr", type=float, default=None, help="Optimizer step size used on every executed pyramid level.")
    p.add_argument("--min-voxels", type=int, default=16, help="Minimum pyramid level size.")
    p.add_argument(
        "--max-voxels",
        type=int,
        default=None,
        help="Largest allowed dimension of the finest pyramid level. Omit to run up to original resolution.",
    )
    p.add_argument(
        "--nosym",
        dest="symmetric",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Disable symmetric halfway-space registration and run directed registration.",
    )
    p.add_argument(
        "--noinit",
        action="store_true",
        help="Skip centroid-based initialization and start from identity (matches FreeSurfer --noinit).",
    )
    p.add_argument(
        "--isotropic",
        action="store_true",
        help="Enable shared isotropic preprocessing before building the pyramid.",
    )
    p.add_argument("--device", default="cpu", metavar="DEVICE", help="PyTorch device, e.g. 'cpu' or 'cuda'.")

    p.add_argument("--verbose", action="store_true", help="Enable INFO-level logging.")
    p.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging.")

    return p


def main(args=None) -> None:
    """Entry point for the ``coreg`` command-line interface.

    This wrapper exposes the image-based gradient-descent registration path used
    for cross-modal alignment when only images are available. It provides a
    small set of practical knobs: transform DOF, pyramid iteration schedule,
    optional centroid initialization, symmetric vs directed mode, and optional
    isotropic preprocessing. The written output LTA is a voxel-to-voxel
    transform in public ``moving -> reference`` direction.
    """
    import nibabel as nib

    from neuroreg.imreg.coreg import coreg
    from neuroreg.transforms import LTA

    parser = _build_parser()
    ns = parser.parse_args(args)
    ns.symmetric = getattr(ns, "symmetric", True)

    level = logging.DEBUG if ns.debug else (logging.INFO if ns.verbose else logging.WARNING)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("neuroreg.cli.coreg")

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

    kwargs = dict(
        dof=ns.dof,
        device=ns.device,
        return_v2v=True,
        centroid_init=not ns.noinit,
        symmetric=ns.symmetric,
        isotropic=ns.isotropic,
        level_iters=ns.level_iters,
        min_voxels=ns.min_voxels,
        max_voxels=ns.max_voxels,
        lr=ns.lr,
    )
    if ns.n_iters is not None:
        kwargs["n"] = ns.n_iters

    logger.info(
        (
            "Starting image-to-image registration "
            "(dof=%d, symmetric=%s, isotropic=%s, n=%s, "
            "level_iters=%s, lr=%s, min_voxels=%d, max_voxels=%s) ..."
        ),
        ns.dof,
        ns.symmetric,
        ns.isotropic,
        ns.n_iters,
        ns.level_iters,
        ns.lr,
        ns.min_voxels,
        ns.max_voxels,
    )
    v2v = coreg(mov_img, ref_img, **kwargs)
    v2v_cpu = v2v.detach().cpu()

    LTA.from_matrix(v2v_cpu.numpy(), ns.mov, cast(Any, mov_img), ns.ref, cast(Any, ref_img), lta_type=0).write(ns.out)
    logger.info("Wrote LTA: %s", ns.out)
    print(f"Output: {ns.out}")

if __name__ == "__main__":
    main()
