#!/usr/bin/env python3
"""Unified image-utility CLI.

Small ``mri_*``-style volume utilities grouped under a single command, in the
same spirit as the ``lta`` transform CLI. Available subcommands are ``mask``
(analogous to FreeSurfer's ``mri_mask``) and ``info`` (analogous to
``mri_info``); ``diff`` and ``binarize`` are planned. Run ``mri --help`` or
``mri <subcommand> --help`` for the full command syntax.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from ..image import (
    describe_image,
    image_value_stats,
    load_image,
    mask_geometry_differs,
    reslice_and_apply_mask,
    save_image,
)

# ── parser ──────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``mri`` command.

    Returns
    -------
    argparse.ArgumentParser
        Configured CLI parser with one sub-parser per utility.
    """
    p = argparse.ArgumentParser(
        prog="mri",
        description="Image volume utilities (mask, info, ...).",
    )
    sub = p.add_subparsers(dest="command", metavar="COMMAND", required=True)

    # ── mask ────────────────────────────────────────────────────────────────
    mask_p = sub.add_parser(
        "mask",
        help="Apply a binary mask to a volume in its own geometry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Apply a binary mask to an image, keeping voxels where the mask\n"
            "value is strictly greater than --threshold and setting the rest to\n"
            "--oval. The mask is resampled with nearest-neighbor interpolation\n"
            "into the input geometry when its grid differs, so a mask given in a\n"
            "different geometry is handled like FreeSurfer's mri_mask. The input\n"
            "dtype is preserved; the output format follows the out extension.\n"
            "\n"
            "This is a single-space operation: it does not map between\n"
            "geometries. To mask before/after a transform, compose with vol2vol\n"
            "(mri mask ... then vol2vol ...  =  mask-then-map;  vol2vol ... then\n"
            "mri mask ...  =  map-then-mask)."
        ),
    )
    mask_p.add_argument("in_file", metavar="in", help="Input image to mask.")
    mask_p.add_argument("mask", metavar="mask", help="Binary mask image.")
    mask_p.add_argument("out", metavar="out", help="Output image filename (format from extension).")
    mask_p.add_argument(
        "-T",
        "--threshold",
        type=float,
        default=0.0,
        metavar="T",
        help="Voxels with mask value strictly greater than this are kept (default: 0).",
    )
    mask_p.add_argument(
        "--oval",
        type=float,
        default=0.0,
        metavar="V",
        help="Value assigned to voxels outside the mask (default: 0).",
    )

    # ── info ────────────────────────────────────────────────────────────────
    info_p = sub.add_parser(
        "info",
        help="Print header and geometry information for a volume.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Print header and geometry information for an image, analogous to\n"
            "FreeSurfer's mri_info. With no selector flags a full human-readable\n"
            "dump is printed. Selector flags print only the requested value(s),\n"
            "one per line, for scripting."
        ),
    )
    info_p.add_argument("file", metavar="FILE", help="Input image.")
    info_p.add_argument("--dim", action="store_true", help="Print dimensions: 'w h d'.")
    info_p.add_argument("--res", action="store_true", help="Print voxel sizes: 'x y z'.")
    info_p.add_argument("--voxvol", action="store_true", help="Print the voxel volume.")
    info_p.add_argument("--type", action="store_true", help="Print the data dtype.")
    info_p.add_argument("--nframes", action="store_true", help="Print the number of frames.")
    info_p.add_argument(
        "--orientation", "--ori", action="store_true", dest="orientation", help="Print the orientation string."
    )
    info_p.add_argument("--cras", action="store_true", help="Print the volume center RAS: 'c_r c_a c_s'.")
    info_p.add_argument("--vox2ras", action="store_true", help="Print the voxel-to-RAS (scanner) matrix.")
    info_p.add_argument("--ras2vox", action="store_true", help="Print the RAS-to-voxel matrix.")
    info_p.add_argument("--vox2ras-tkr", action="store_true", help="Print the voxel-to-tkRAS matrix.")
    info_p.add_argument("--stats", action="store_true", help="Print voxel value stats: 'min max mean'.")

    return p


_INFO_SELECTORS = (
    "dim",
    "res",
    "voxvol",
    "type",
    "nframes",
    "orientation",
    "cras",
    "vox2ras",
    "ras2vox",
    "vox2ras_tkr",
    "stats",
)


# ── subcommand handlers ───────────────────────────────────────────────────────


def _main_mask(ns: argparse.Namespace) -> None:
    try:
        image = load_image(ns.in_file)
        mask = load_image(ns.mask)
        target_affine = np.asarray(image.affine, dtype=np.float64)
        target_shape = tuple(int(v) for v in image.shape[:3])
        if mask_geometry_differs(mask, target_affine, target_shape):
            print("Mask:      geometry differs from input; resampling mask to input grid (nearest).")
        masked = reslice_and_apply_mask(image, mask, threshold=ns.threshold, fill=ns.oval)
        save_image(masked, ns.out)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Output: {ns.out}")


def _format_matrix(m: np.ndarray) -> str:
    """Format a 4x4 matrix as aligned, space-separated rows."""
    return "\n".join(" ".join(f"{v: .8f}" for v in row) for row in np.asarray(m, dtype=np.float64))


def _print_info_dump(d: dict) -> None:
    """Print a full FreeSurfer-style information dump."""
    shape = d["shape"]
    vs = d["voxel_sizes"]
    cras = d["cras"]
    print(f"Volume information for {d['fname']}")
    print(f"          type: {d['file_type']}")
    if d["nframes"] > 1:
        print(f"    dimensions: {shape[0]} x {shape[1]} x {shape[2]} x {d['nframes']}")
    else:
        print(f"    dimensions: {shape[0]} x {shape[1]} x {shape[2]}")
    print(f"   voxel sizes: {vs[0]:.6f}, {vs[1]:.6f}, {vs[2]:.6f}")
    print(f"     data type: {d['dtype']}")
    print(f"           fov: {d['fov']:.3f}")
    print(f"       nframes: {d['nframes']}")
    print(f"          cras: {cras[0]:.6f} {cras[1]:.6f} {cras[2]:.6f}")
    print(f"   Orientation: {d['orientation']}")
    print("\nvoxel to ras transform:")
    print(_format_matrix(d["vox2ras"]))
    print(f"\nvoxel-to-ras determinant: {d['determinant']:g}")
    print("\nras to voxel transform:")
    print(_format_matrix(d["ras2vox"]))


def _main_info(ns: argparse.Namespace) -> None:
    try:
        img = load_image(ns.file)
        d = describe_image(img, fname=ns.file)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if not any(getattr(ns, name) for name in _INFO_SELECTORS):
        _print_info_dump(d)
        return

    # Scriptable mode: print only the requested value(s), one per line.
    shape = d["shape"]
    vs = d["voxel_sizes"]
    cras = d["cras"]
    if ns.dim:
        print(f"{shape[0]} {shape[1]} {shape[2]}")
    if ns.res:
        print(f"{vs[0]:.6f} {vs[1]:.6f} {vs[2]:.6f}")
    if ns.voxvol:
        print(f"{d['voxvol']:g}")
    if ns.type:
        print(str(d["dtype"]))
    if ns.nframes:
        print(d["nframes"])
    if ns.orientation:
        print(d["orientation"])
    if ns.cras:
        print(f"{cras[0]:.6f} {cras[1]:.6f} {cras[2]:.6f}")
    if ns.vox2ras:
        print(_format_matrix(d["vox2ras"]))
    if ns.ras2vox:
        print(_format_matrix(d["ras2vox"]))
    if ns.vox2ras_tkr:
        print(_format_matrix(d["vox2ras_tkr"]))
    if ns.stats:
        stats = image_value_stats(img)
        print(f"{stats['min']:g} {stats['max']:g} {stats['mean']:g}")


# ── entry point ───────────────────────────────────────────────────────────────


def main(args=None) -> None:
    """Entry point for the ``mri`` command.

    Parameters
    ----------
    args : list of str or None, optional
        Command-line arguments. When ``None``, arguments are read from
        :data:`sys.argv`.

    Returns
    -------
    None
        This function is invoked for its side effects only.
    """
    parser = _build_parser()
    ns = parser.parse_args(args)

    if ns.command == "mask":
        _main_mask(ns)
    elif ns.command == "info":
        _main_info(ns)


if __name__ == "__main__":
    main()
