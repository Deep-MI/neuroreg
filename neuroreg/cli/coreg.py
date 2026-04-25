#!/usr/bin/env python3
"""Command-line interface for image-based cross-modal registration (coreg)."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, cast

from neuroreg.transforms import LINEAR_RAS_TO_RAS, LINEAR_VOX_TO_VOX, LTA, convert_transform_type


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
            "3-D image-to-image registration.\n"
            "Defaults to the MRI_coreg-style Powell path; use --method gd for the legacy PyTorch gradient-descent path."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--mov", required=True, metavar="FILE", help="Moving (source) image (NIfTI or MGZ).")
    p.add_argument("--ref", required=True, metavar="FILE", help="Reference (target/fixed) image (NIfTI or MGZ).")
    p.add_argument("--out", required=True, metavar="LTA", help="Output LTA file for the recovered transformation.")
    p.add_argument("--mapmov", metavar="FILE", help="Save the mapped moving image resliced into reference geometry.")
    p.add_argument(
        "--mapmovhdr",
        metavar="FILE",
        help="Save a header-only mapped moving image with no interpolation.",
    )

    p.add_argument(
        "--dof",
        type=int,
        default=6,
        choices=[3, 6, 9, 12],
        metavar="{3,6,9,12}",
        help="Degrees of freedom: 3=translation, 6=rigid, 9=rigid+scale, 12=affine.",
    )
    p.add_argument(
        "--method",
        choices=["powell", "gd"],
        default="powell",
        help="Registration backend: Powell or legacy gradient descent ('gd').",
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
        "--init-lta",
        dest="init_lta",
        metavar="FILE",
        help="Initialize from an existing LTA transform. When given, other init flags are ignored.",
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
    p.add_argument(
        "--isotropic",
        action="store_true",
        help="Enable shared isotropic preprocessing before building the pyramid.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        metavar="DEVICE",
        help=(
            "Torch device string, e.g. 'cpu', 'cuda', 'mps', or 'gpu'. The Powell backend currently falls back to CPU."
        ),
    )
    p.add_argument(
        "--powell-brute-limit",
        type=float,
        default=30.0,
        help="Initial search half-width for the Powell-style brute-force stage.",
    )
    p.add_argument(
        "--powell-brute-iters",
        type=int,
        default=1,
        help="Number of coarse-to-fine passes in the Powell-style brute-force stage.",
    )
    p.add_argument(
        "--powell-brute-samples",
        type=int,
        default=30,
        help="Number of samples per dimension in the Powell-style brute-force stage.",
    )
    p.add_argument(
        "--powell-maxiter",
        type=int,
        default=4,
        help="Maximum Powell iterations in the Powell-style refinement stage.",
    )
    p.add_argument(
        "--powell-sep",
        type=int,
        default=4,
        help="Sampling spacing for the Powell-style MRI_coreg evaluator.",
    )

    p.add_argument("--verbose", action="store_true", help="Enable INFO-level logging.")
    p.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging.")

    return p


def main(args=None) -> None:
    """Entry point for the ``coreg`` command-line interface.

    This wrapper exposes public image-based registration for cross-modal
    alignment when only images are available. It defaults to the FreeSurfer-
    style brute-force plus Powell path and keeps the legacy gradient-descent
    backend available via ``--method gd``. The written output LTA is a
    voxel-to-voxel transform in public ``moving -> reference`` direction.
    """
    from neuroreg.image import load_image, save_header_mapped_image
    from neuroreg.imreg.coreg import coreg

    parser = _build_parser()
    ns = parser.parse_args(args)
    ns.symmetric = getattr(ns, "symmetric", True)
    if ns.init_lta is not None and ns.init_type is not None:
        logging.getLogger("neuroreg.cli.coreg").warning(
            "Ignoring %s because --init-lta was provided.",
            ns.init_type,
        )

    level = logging.DEBUG if ns.debug else (logging.INFO if ns.verbose else logging.WARNING)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("neuroreg.cli.coreg")

    logger.info("Loading moving image:    %s", ns.mov)
    logger.info("Loading reference image: %s", ns.ref)
    try:
        mov_img = load_image(ns.mov)
        ref_img = load_image(ns.ref)
    except Exception as exc:
        print(f"ERROR loading image: {exc}", file=sys.stderr)
        sys.exit(1)

    mov_img = cast(Any, mov_img)
    ref_img = cast(Any, ref_img)

    kwargs = dict(
        dof=ns.dof,
        method=ns.method,
        device=ns.device,
        return_v2v=False,
        mapped_name=ns.mapmov,
        init_lta=ns.init_lta,
        symmetric=ns.symmetric,
        isotropic=ns.isotropic,
        level_iters=ns.level_iters,
        min_voxels=ns.min_voxels,
        max_voxels=ns.max_voxels,
        lr=ns.lr,
        powell_brute_force_limit=ns.powell_brute_limit,
        powell_brute_force_iters=ns.powell_brute_iters,
        powell_brute_force_samples=ns.powell_brute_samples,
        powell_maxiter=ns.powell_maxiter,
        powell_sep=ns.powell_sep,
    )
    if ns.init_lta is not None:
        logger.info("Using explicit LTA initialization: %s", ns.init_lta)
    elif ns.init_type is not None:
        kwargs["init_type"] = ns.init_type
    if ns.n_iters is not None:
        kwargs["n"] = ns.n_iters

    logger.info(
        (
            "Starting image-to-image registration "
            "(method=%s, dof=%d, symmetric=%s, isotropic=%s, n=%s, "
            "level_iters=%s, lr=%s, min_voxels=%d, max_voxels=%s, powell_sep=%d) ..."
        ),
        ns.method,
        ns.dof,
        ns.symmetric,
        ns.isotropic,
        ns.n_iters,
        ns.level_iters,
        ns.lr,
        ns.min_voxels,
        ns.max_voxels,
        ns.powell_sep,
    )
    r2r = coreg(mov_img, ref_img, **kwargs)
    r2r_cpu = r2r.detach().cpu()
    v2v = convert_transform_type(
        r2r_cpu.numpy(),
        src_affine=mov_img.affine,
        dst_affine=ref_img.affine,
        from_type=LINEAR_RAS_TO_RAS,
        to_type=LINEAR_VOX_TO_VOX,
    )

    LTA.from_matrix(v2v, ns.mov, mov_img, ns.ref, ref_img, lta_type=0).write(ns.out)
    logger.info("Wrote LTA: %s", ns.out)
    print(f"Output: {ns.out}")
    if ns.mapmov:
        logger.info("Wrote resliced mapped image: %s", ns.mapmov)
        print(f"MapMov:    {ns.mapmov}")
    if ns.mapmovhdr:
        save_header_mapped_image(mov_img, r2r_cpu.numpy(), ns.mapmovhdr)
        logger.info("Wrote header-mapped image: %s", ns.mapmovhdr)
        print(f"MapMovHdr: {ns.mapmovhdr}")


if __name__ == "__main__":
    main()
