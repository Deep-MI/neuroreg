#!/usr/bin/env python3
"""Unified image-utility CLI.

Small ``mri_*``-style volume utilities grouped under a single command, in the
same spirit as the ``lta`` transform CLI. Available subcommands are ``mask``
(analogous to FreeSurfer's ``mri_mask``), ``info`` (``mri_info``), ``diff``
(``mri_diff``), ``binarize`` (``mri_binarize``), ``geom``, and ``tag``
(``mri_add_xform_to_header`` and embedding a colour table). Run ``mri --help``
or ``mri <subcommand> --help`` for the full command syntax.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from ..image import (
    MGHTags,
    binarize_image,
    build_grid_affine,
    compare_images,
    describe_image,
    direction_cosines_from_orientation,
    image_value_stats,
    inherit_tags,
    load_geometry_source,
    load_image,
    mask_geometry_differs,
    read_color_lut,
    read_image_tags,
    recognized_image_suffix,
    relative_xform_path,
    reslice_and_apply_mask,
    save_image,
    shape_from_fov,
    write_image_tags,
)
from ._args import NumberListParser, number_list
from ._outputs import output_history, validate_image_outputs

# ── parser ──────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``mri`` command.

    Returns
    -------
    argparse.ArgumentParser
        Configured CLI parser with one sub-parser per utility.
    """
    # NumberListParser so that a negative geometry component such as
    # "--cras -4,0,0" is read as a value rather than an unknown option; the
    # sub-parsers inherit the class via argparse's parser_class default.
    p = NumberListParser(
        prog="mri",
        description="Image volume utilities (mask, info, diff, binarize, geom, tag).",
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

    # ── diff ────────────────────────────────────────────────────────────────
    diff_p = sub.add_parser(
        "diff",
        help="Compare two volumes and exit nonzero when they differ.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Compare two volumes, analogous to FreeSurfer's mri_diff. Checks are\n"
            "run in order and (unless --no-exit-on-diff) the command exits at the\n"
            "first difference with a FreeSurfer-compatible status code:\n"
            "0   volumes are the same\n"
            "1   error (e.g. a file could not be read)\n"
            "101 dimensions differ (always exits)\n"
            "102 voxel resolution differs (> --res-thresh)\n"
            "104 geometry / vox2ras differs (> --geo-thresh)\n"
            "105 data type (precision) differs\n"
            "106 voxel values differ (max abs diff > --thresh and count > --count-thresh)\n"
            "\n"
            "Acquisition-parameter (TR/TE/TI/flip) checks are not performed."
        ),
    )
    diff_p.add_argument("vol1", metavar="vol1", help="First image.")
    diff_p.add_argument("vol2", metavar="vol2", help="Second image.")
    diff_p.add_argument(
        "--thresh", type=float, default=0.0, metavar="T", help="Voxel value difference threshold (default: 0)."
    )
    diff_p.add_argument(
        "--res-thresh", type=float, default=0.0, metavar="T", help="Voxel-size difference threshold (default: 0)."
    )
    diff_p.add_argument(
        "--geo-thresh", type=float, default=0.0, metavar="T", help="vox2ras element difference threshold (default: 0)."
    )
    diff_p.add_argument(
        "--count-thresh",
        type=int,
        default=0,
        metavar="N",
        help="Voxel values count as differing only when more than N voxels differ (default: 0).",
    )
    diff_p.add_argument("--count", action="store_true", help="Print the number of differing voxels.")
    diff_p.add_argument(
        "--no-exit-on-diff",
        action="store_false",
        dest="exit_on_diff",
        help="Report all differences instead of exiting at the first one.",
    )
    diff_p.add_argument(
        "--skip-res",
        "--notallow-res",
        action="store_true",
        dest="skip_res",
        help="Skip the voxel-resolution check.",
    )
    diff_p.add_argument(
        "--skip-geo",
        "--notallow-geo",
        action="store_true",
        dest="skip_geo",
        help="Skip the geometry / vox2ras check.",
    )
    diff_p.add_argument(
        "--skip-prec",
        "--notallow-prec",
        action="store_true",
        dest="skip_prec",
        help="Skip the data-type / precision check.",
    )
    diff_p.add_argument(
        "--skip-pix",
        "--notallow-pix",
        action="store_true",
        dest="skip_pix",
        help="Skip the pixel-value check.",
    )
    diff_p.add_argument(
        "--notallow-acq",
        action="store_true",
        dest="skip_acq",
        help="Accepted for FreeSurfer compatibility; acquisition-parameter checks are not performed.",
    )

    # ── binarize ──────────────────────────────────────────────────────────────
    bin_p = sub.add_parser(
        "binarize",
        help="Binarize a volume by intensity range or matched label values.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Binarize an image, analogous to FreeSurfer's mri_binarize. A voxel is\n"
            "selected when it matches one of --match (exact), or lies in the\n"
            "inclusive range [--min, --max] (either bound may be omitted).\n"
            "Selected voxels are set to --binval, the rest to --binvalnot; --inv\n"
            "swaps that assignment. At least one of --min, --max, or --match is\n"
            "required. Output is int32 by default (--uchar selects uint8)."
        ),
    )
    bin_p.add_argument("--i", "--in", dest="input_file", required=True, metavar="FILE", help="Input image.")
    bin_p.add_argument("--o", "--out", dest="out", required=True, metavar="FILE", help="Output image.")
    bin_p.add_argument("--min", dest="vmin", type=float, metavar="MIN", help="Inclusive lower intensity bound.")
    bin_p.add_argument("--max", dest="vmax", type=float, metavar="MAX", help="Inclusive upper intensity bound.")
    bin_p.add_argument(
        "--match", type=float, nargs="+", metavar="V", help="Match these values exactly (e.g. label ids)."
    )
    bin_p.add_argument("--binval", type=int, default=1, metavar="V", help="Value for selected voxels (default: 1).")
    bin_p.add_argument(
        "--binvalnot", type=int, default=0, metavar="V", help="Value for unselected voxels (default: 0)."
    )
    bin_p.add_argument("--inv", action="store_true", help="Swap the selected/unselected output values.")
    bin_p.add_argument("--abs", action="store_true", dest="use_abs", help="Take abs value before thresholding.")
    bin_p.add_argument("--frame", type=int, default=None, metavar="N", help="For 4D input, binarize this frame only.")
    bin_p.add_argument("--uchar", action="store_true", help="Write uint8 output instead of int32.")

    # ── geom ──────────────────────────────────────────────────────────────────
    geom_p = sub.add_parser(
        "geom",
        help="Write an empty volume that carries only a target geometry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Write a zero-filled volume whose header describes a requested target\n"
            "geometry, for use as a reference image by tools that accept only a\n"
            "reference file. This makes grids expressible that no stock file\n"
            "provides, for example the mni305.cor.mgz frame (256mm FOV, LIA,\n"
            "cras 0 0 0) at a native voxel size other than 1mm.\n"
            "\n"
            "A geometry has four components: image dimensions, voxel sizes,\n"
            "direction cosines, and placement (cras). Each is taken from its own\n"
            "flag if given, else from --like, else it is an error. The exception\n"
            "is cras, which defaults to 0,0,0. --shape and --fov are two ways to\n"
            "give the dimensions; --fov derives them from the voxel size, so a\n"
            "fixed 256mm field of view needs no division in the caller.\n"
            "\n"
            "--like supplies a field of view rather than image dimensions, so\n"
            "'--like sub.mgz --vox-size 0.5' keeps the extent sub.mgz covers and\n"
            "grows the dimensions to match, instead of cropping to sub.mgz's\n"
            "dimensions at a finer resolution. Pass --shape to fix them instead.\n"
            "\n"
            "--orientation sets a strict axis-aligned orientation and so discards\n"
            "any oblique rotation from --like. The output contains no image data."
        ),
    )
    geom_p.add_argument("--o", "--out", dest="out", required=True, metavar="FILE", help="Output image.")
    geom_p.add_argument(
        "--like",
        dest="like",
        metavar="FILE",
        help=(
            "Image or geometry JSON supplying any geometry component not given explicitly "
            "(extent, not dimensions)."
        ),
    )
    size_group = geom_p.add_mutually_exclusive_group()
    size_group.add_argument(
        "--shape",
        type=number_list("--shape", allow_scalar=True, cast=int, positive=True),
        metavar="I[,J,K]",
        help="Image dimensions in voxels, as one value (cube) or three.",
    )
    size_group.add_argument(
        "--fov",
        type=number_list("--fov", allow_scalar=True, positive=True),
        metavar="MM[,MM,MM]",
        help="Field of view in mm; the image dimensions are derived from the voxel size.",
    )
    geom_p.add_argument(
        "--vox-size",
        dest="vox_size",
        type=number_list("--vox-size", allow_scalar=True, positive=True),
        metavar="MM[,MM,MM]",
        help="Voxel size in mm, as one value (isotropic) or three. With --like the dimensions rescale.",
    )
    geom_p.add_argument(
        "--orientation",
        "--ori",
        dest="orientation",
        metavar="CODE",
        help="Strict axis-aligned orientation, e.g. LIA or RAS.",
    )
    geom_p.add_argument(
        "--cras",
        dest="cras",
        type=number_list("--cras"),
        metavar="X,Y,Z",
        help="World coordinate of the grid centre (default: 0,0,0).",
    )

    # ── tag ─────────────────────────────────────────────────────────────────
    tag_p = sub.add_parser(
        "tag",
        help="Record a transform path and/or embed a colour table in an MGH volume.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Copy an MGH or NIfTI volume, recording the path of its talairach\n"
            "transform and/or embedding a colour lookup table, as FreeSurfer's\n"
            "mri_add_xform_to_header and mri_convert --ctab do. Nothing else\n"
            "changes: an MGH volume is copied byte for byte apart from the\n"
            "requested tags, and a NIfTI volume keeps its header and voxels.\n"
            "OUT may be IN to tag a volume in place, and must share its format.\n"
            "\n"
            "An MGH volume keeps its tags after the voxel data, a NIfTI volume\n"
            "in FreeSurfer's header extension (code 14), which FreeSurfer 8 and\n"
            "surfa read. The other commands here carry tags forward: an output\n"
            "keeps the tags of the volume its values came from, except those\n"
            "tied to a coordinate frame it no longer lives in.\n"
            "\n"
            "--xform is recorded relative to the directory of OUT by default,\n"
            "so a subject directory stays valid when it is moved. FreeSurfer\n"
            "first looks for transforms/talairach.xfm next to the volume and\n"
            "uses that whenever it exists, which covers the standard layout;\n"
            "any other relative path it resolves against the current directory.\n"
            "--abs-xform-tag records the absolute path instead, which is how\n"
            "recon-all records it.\n"
            "\n"
            "--xform-from and --ctab-from copy the tag from another volume, for\n"
            "outputs of tools that write none. A copied transform path is\n"
            "resolved where it was recorded and then recorded like --xform.\n"
            "\n"
            "A tag IN already holds is only replaced with --force, so a tag set\n"
            "on purpose is not overwritten by accident. Asking for a value it\n"
            "already holds is not an error, so a pipeline can repeat the step;\n"
            "colour tables count as the same when their entries are."
        ),
    )
    tag_p.add_argument("in_file", metavar="in", help="Input MGH or NIfTI volume.")
    tag_p.add_argument("out", metavar="out", help="Output volume in the same format; may be the input.")
    xform_group = tag_p.add_mutually_exclusive_group()
    xform_group.add_argument(
        "--xform",
        metavar="FILE",
        help="Transform whose path to record, typically transforms/talairach.xfm.",
    )
    xform_group.add_argument(
        "--xform-from",
        metavar="VOL",
        help="MGH or NIfTI volume whose recorded transform path to copy.",
    )
    tag_p.add_argument(
        "--abs-xform-tag",
        action="store_true",
        help="Record the transform's absolute path rather than one relative to the directory of OUT.",
    )
    ctab_group = tag_p.add_mutually_exclusive_group()
    ctab_group.add_argument(
        "--ctab",
        metavar="LUT",
        help="ASCII colour lookup table to embed, such as FreeSurferColorLUT.txt.",
    )
    ctab_group.add_argument(
        "--ctab-from",
        metavar="VOL",
        help="MGH or NIfTI volume whose embedded colour table to copy.",
    )
    tag_p.add_argument(
        "--force",
        action="store_true",
        help="Replace a transform path or colour table IN already holds with a different one.",
    )

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


def _main_mask(ns: argparse.Namespace, history: str) -> None:
    try:
        image = load_image(ns.in_file)
        mask = load_image(ns.mask)
        target_affine = np.asarray(image.affine, dtype=np.float64)
        target_shape = tuple(int(v) for v in image.shape[:3])
        if mask_geometry_differs(mask, target_affine, target_shape):
            print("Mask:      geometry differs from input; resampling mask to input grid (nearest).")
        masked = reslice_and_apply_mask(image, mask, threshold=ns.threshold, fill=ns.oval)
        # Masking keeps the input's grid, frame and labels, so all its tags apply.
        save_image(masked, ns.out, tags=inherit_tags(ns.in_file, ns.out).with_history(history))
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


def _print_tags(path: str) -> None:
    """Print the transform path and colour table a volume carries, if any."""
    if recognized_image_suffix(path) not in _TAGGED_SUFFIXES:
        return
    try:
        tags = read_image_tags(path)
    except ValueError as exc:
        print(f"\n{'tags':>14}: unreadable ({exc})")
        return
    table = tags.color_table
    if tags.xform is None and table is None:
        return
    print()
    if tags.xform is not None:
        print(f"{'talairach xfm':>14}: {tags.xform}")
    if table is not None:
        print(f"{'color table':>14}: {len(table.entries)} entries ({table.fname})")


def _main_info(ns: argparse.Namespace) -> None:
    try:
        img = load_image(ns.file)
        d = describe_image(img, fname=ns.file)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if not any(getattr(ns, name) for name in _INFO_SELECTORS):
        _print_info_dump(d)
        _print_tags(ns.file)
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


def _main_diff(ns: argparse.Namespace) -> None:
    try:
        v1 = load_image(ns.vol1)
        v2 = load_image(ns.vol2)
        # Header-only first: voxel data are not materialized yet (nibabel is lazy).
        d = compare_images(v1, v2, compare_pixels=False)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Dimension mismatch makes the remaining checks meaningless; always exit.
    if not d.shape_match:
        print("Volumes differ in dimension")
        print(f"v1dim {' '.join(str(v) for v in d.shape1)}")
        print(f"v2dim {' '.join(str(v) for v in d.shape2)}")
        sys.exit(101)

    status = 0

    if not ns.skip_res and d.res_max_diff > ns.res_thresh:
        print("Volumes differ in resolution")
        print(f"v1res {d.voxsize1[0]:f} {d.voxsize1[1]:f} {d.voxsize1[2]:f}")
        print(f"v2res {d.voxsize2[0]:f} {d.voxsize2[1]:f} {d.voxsize2[2]:f}")
        status = 102
        if ns.exit_on_diff:
            sys.exit(status)

    if not ns.skip_geo and d.geo_max_diff > ns.geo_thresh:
        print(f"Volumes differ in geometry (max vox2ras element diff = {d.geo_max_diff:g})")
        status = 104
        if ns.exit_on_diff:
            sys.exit(status)

    if not ns.skip_prec and not d.dtype_match:
        print(f"Volumes differ in precision {d.dtype1} {d.dtype2}")
        status = 105
        if ns.exit_on_diff:
            sys.exit(status)

    # Only materialize pixel data once header checks have passed (or --no-exit-on-diff).
    if not ns.skip_pix:
        d = compare_images(v1, v2, pix_thresh=ns.thresh, compare_pixels=True)
        if ns.count:
            print(f"diffcount {d.n_voxels_differ}")
        if d.max_abs_diff > ns.thresh and d.n_voxels_differ > ns.count_thresh:
            print("Volumes differ in pixel data")
            loc = "" if d.max_diff_loc is None else " at " + " ".join(str(v) for v in d.max_diff_loc)
            print(f"maxdiff {d.max_abs_diff:g}{loc}")
            status = 106
            if ns.exit_on_diff:
                sys.exit(status)

    if status == 0:
        print("Volumes are the same")
    sys.exit(status)


def _main_binarize(ns: argparse.Namespace, history: str) -> None:
    try:
        img = load_image(ns.input_file)
        out = binarize_image(
            img,
            vmin=ns.vmin,
            vmax=ns.vmax,
            match=ns.match,
            binval=ns.binval,
            binvalnot=ns.binvalnot,
            invert=ns.inv,
            use_abs=ns.use_abs,
            frame=ns.frame,
            out_dtype=np.uint8 if ns.uchar else np.int32,
        )
        # The output holds new labels, so a colour table would name the wrong
        # structures; everything else still describes the same grid and frame.
        tags = inherit_tags(ns.input_file, ns.out, keep_color_table=False).with_history(history)
        save_image(out, ns.out, tags=tags)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Output: {ns.out}")


def _resolve_geom_components(
        ns: argparse.Namespace,
        parser: argparse.ArgumentParser,
) -> tuple[np.ndarray, tuple[int, int, int], np.ndarray, np.ndarray]:
    """Resolve the four geometry components for ``mri geom``.

    Each component comes from its own flag if given, else from ``--like``, else
    it is a usage error. Two components qualify that rule:

    * ``--cras`` defaults to the world origin rather than erroring, since a grid
      built from scratch has no other sensible placement.
    * ``--like`` supplies a *field of view* rather than image dimensions, so
      overriding the voxel size rescales the dimensions to keep the same
      extent. Pass ``--shape`` to fix the dimensions instead.

    Parameters
    ----------
    ns : argparse.Namespace
        Parsed ``geom`` arguments.
    parser : argparse.ArgumentParser
        Parser used to report user-facing validation errors.

    Returns
    -------
    tuple
        Direction cosines, image dimensions, voxel sizes, and grid centre.

    Raises
    ------
    SystemExit
        Via :meth:`argparse.ArgumentParser.error` when a component cannot be
        resolved or an orientation code is invalid.
    """
    like = load_geometry_source(ns.like, label="--like") if ns.like is not None else None
    like_zooms = None if like is None else np.asarray(like.header.get_zooms()[:3], dtype=np.float64)
    if like_zooms is not None and np.any(like_zooms <= 0):
        parser.error(f"--like has a non-positive voxel size: {like_zooms.tolist()}")

    vox_size = ns.vox_size
    if vox_size is None:
        if like_zooms is None:
            parser.error("--vox-size is required unless --like is given.")
        vox_size = like_zooms

    if ns.shape is not None:
        shape = tuple(int(v) for v in ns.shape)
    elif ns.fov is not None:
        shape = shape_from_fov(ns.fov, vox_size)
    elif like is not None:
        # --like preserves the extent, not the dimensions: with the voxel size
        # overridden the dimensions follow from the source field of view, so the
        # new grid still covers the same anatomy. Keeping the dimensions would
        # silently crop it. Because shape_from_fov snaps counts that are integer
        # within a relative tolerance, feeding back the source's own voxel size
        # returns the source dimensions unchanged.
        source_fov = np.asarray(like.shape[:3], dtype=np.float64) * like_zooms
        shape = shape_from_fov(source_fov, vox_size)
    else:
        parser.error("--shape or --fov is required unless --like is given.")

    if ns.orientation is not None:
        try:
            cosines = direction_cosines_from_orientation(ns.orientation)
        except ValueError as exc:
            parser.error(str(exc))
    elif like is not None:
        # Normalise the columns so the oblique rotation survives while the
        # voxel size is taken from whichever source resolved it above.
        cosines = np.asarray(like.affine, dtype=np.float64)[:3, :3] / like_zooms
    else:
        parser.error("--orientation is required unless --like is given.")

    if ns.cras is not None:
        cras = ns.cras
    elif like is not None:
        cras = describe_image(like, ns.like)["cras"]
    else:
        cras = np.zeros(3, dtype=np.float64)

    return cosines, shape, np.asarray(vox_size, dtype=np.float64), np.asarray(cras, dtype=np.float64)


def _main_geom(ns: argparse.Namespace, parser: argparse.ArgumentParser, history: str) -> None:
    try:
        # Inside the boundary so an unreadable --like reports like the other
        # subcommands. parser.error raises SystemExit, not Exception, so usage
        # errors still surface as argparse messages rather than "ERROR: ...".
        cosines, shape, vox_size, cras = _resolve_geom_components(ns, parser)
        affine = build_grid_affine(cosines=cosines, vox_size=vox_size, shape=shape, cras=cras)
        # uint8 zeros: the payload is irrelevant, only the header is, and this
        # keeps a 320^3 reference file a few KB once compressed.
        img = nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), affine, dtype=np.uint8)
        # A new volume with no values of its own: only how it was made applies.
        save_image(img, ns.out, tags=MGHTags().with_history(history))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Report the geometry as stored, so what is printed is what a consumer reads
    # back rather than what was requested.
    written = describe_image(load_image(ns.out), ns.out)
    out_shape, out_vox, out_cras = written["shape"], written["voxel_sizes"], written["cras"]
    print(f"Output: {ns.out}")
    print(f"    dimensions: {out_shape[0]} x {out_shape[1]} x {out_shape[2]}")
    print(f"   voxel sizes: {out_vox[0]:.6f}, {out_vox[1]:.6f}, {out_vox[2]:.6f}")
    print(f"          cras: {out_cras[0]:.6f} {out_cras[1]:.6f} {out_cras[2]:.6f}")
    print(f"   Orientation: {written['orientation']}")


#: The transform name FreeSurfer finds next to a volume regardless of the tag.
_STANDARD_XFORM = Path("transforms", "talairach.xfm")
_MGH_SUFFIXES = (".mgz", ".mgh")
_NIFTI_SUFFIXES = (".nii.gz", ".nii")
#: Formats that hold FreeSurfer tags: MGH after the voxels, NIfTI in an extension.
_TAGGED_SUFFIXES = (*_MGH_SUFFIXES, *_NIFTI_SUFFIXES)


def _validate_tag_args(ns: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Reject ``mri tag`` invocations that cannot do anything useful."""
    if ns.xform is None and ns.xform_from is None and ns.ctab is None and ns.ctab_from is None:
        parser.error(
            "mri tag needs --xform/--xform-from and/or --ctab/--ctab-from: there is nothing to record otherwise."
        )
    if ns.abs_xform_tag and ns.xform is None and ns.xform_from is None:
        parser.error("--abs-xform-tag only says how to record a transform path, and none was given.")
    for flag, source in (("--xform-from", ns.xform_from), ("--ctab-from", ns.ctab_from)):
        if source is not None and recognized_image_suffix(source) not in _TAGGED_SUFFIXES:
            parser.error(f"{flag} copies a tag from an MGH or NIfTI volume, the formats that hold one.")
    in_suffix, out_suffix = recognized_image_suffix(ns.in_file), recognized_image_suffix(ns.out)
    if in_suffix not in _TAGGED_SUFFIXES:
        parser.error(
            "mri tag reads MGH or NIfTI input only, the formats that hold tags; "
            "convert the volume first, for example with vol2vol."
        )
    if out_suffix not in _TAGGED_SUFFIXES:
        parser.error("tags can only be stored in MGH or NIfTI volumes, so OUT needs one of their extensions.")
    if (in_suffix in _MGH_SUFFIXES) != (out_suffix in _MGH_SUFFIXES):
        parser.error(
            "mri tag copies a volume and so keeps its format; vol2vol converts between MGH and NIfTI "
            "and carries the tags across."
        )


def _xform_tag_value(xform: str, out: str, absolute: bool) -> str:
    """Return the transform path to record for a volume written to ``out``.

    The relative form is relative to the directory ``out`` is written to, not
    the current one, since that is the only location the path can be resolved
    from later. See :func:`~neuroreg.image.mgh_tags.relative_xform_path` for
    when the absolute path is recorded instead.
    """
    if absolute:
        # Forward slashes, like the relative form, so the tag reads the same
        # wherever FreeSurfer runs.
        return Path(os.path.abspath(xform)).as_posix()
    return relative_xform_path(xform, os.path.dirname(os.path.abspath(out)))


def _requested_xform_target(ns: argparse.Namespace) -> str | None:
    """Return the absolute path of the transform to record, if one was requested.

    A path copied with --xform-from is resolved from the directory of the
    volume it was recorded in, the only place a relative one means anything.
    """
    if ns.xform is not None:
        return os.path.abspath(ns.xform)
    if ns.xform_from is None:
        return None
    recorded = read_image_tags(ns.xform_from).xform
    if recorded is None:
        raise ValueError(f"{ns.xform_from} records no talairach xform to copy")
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(ns.xform_from)), recorded))


def _requested_color_table(ns: argparse.Namespace):
    """Return the colour table to embed, if one was requested."""
    if ns.ctab is not None:
        return read_color_lut(ns.ctab)
    if ns.ctab_from is None:
        return None
    table = read_image_tags(ns.ctab_from).color_table
    if table is None:
        raise ValueError(f"{ns.ctab_from} embeds no colour table to copy")
    return table


def _main_tag(ns: argparse.Namespace) -> None:
    try:
        xform = None
        target = _requested_xform_target(ns)
        if target is not None:
            if not Path(target).is_file():
                # mri_add_xform_to_header -c records the name without checking
                # it either: the transform may legitimately be written later.
                print(f"WARNING: {target} does not exist; recording its path anyway.", file=sys.stderr)
            xform = _xform_tag_value(target, ns.out, ns.abs_xform_tag)
            if not os.path.isabs(xform) and Path(xform) != _STANDARD_XFORM:
                print(
                    f"WARNING: FreeSurfer resolves the relative path {xform} against its current "
                    f"directory, and only finds {_STANDARD_XFORM} next to the volume on its own. "
                    "Pass --abs-xform-tag if FreeSurfer has to find this transform from elsewhere.",
                    file=sys.stderr,
                )
        table = _requested_color_table(ns)

        # Compare against what OUT would inherit, which for another directory
        # is the rebased path, not the string stored in IN.
        existing = read_image_tags(ns.in_file).rebase_xform(
            os.path.dirname(os.path.abspath(ns.in_file)), os.path.dirname(os.path.abspath(ns.out))
        )
        conflicts = existing.conflicts(xform=xform, color_table=table)
        if conflicts and not ns.force:
            reasons = "; and ".join(conflicts)
            raise ValueError(f"not replacing tags of {ns.in_file}: {reasons}. Pass --force to replace.")
        # A value IN already holds is left as it is, so a repeated step changes
        # nothing, including the name a matching colour table was recorded with.
        if xform is not None and existing.xform == xform:
            xform = None
        existing_table = existing.color_table if table is not None else None
        if existing_table is not None and existing_table.entries == table.entries:
            table = None

        in_place = os.path.exists(ns.out) and os.path.samefile(ns.in_file, ns.out)
        unchanged = in_place and xform is None and table is None
        if not unchanged:
            write_image_tags(ns.in_file, ns.out, xform=xform, color_table=table)
        # Report the tags as stored, so what is printed is what a reader gets.
        written = read_image_tags(ns.out)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Output: {ns.out}" + (" (unchanged: it already holds these tags)" if unchanged else ""))
    if written.xform is not None:
        print(f"     xform tag: {written.xform}")
    if written.color_table is not None:
        table = written.color_table
        print(f"   color table: {len(table.entries)} entries ({table.fname})")


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
    validate_image_outputs(parser, ns, "out")

    history = output_history("mri", args)
    if ns.command == "mask":
        _main_mask(ns, history)
    elif ns.command == "info":
        _main_info(ns)
    elif ns.command == "diff":
        _main_diff(ns)
    elif ns.command == "binarize":
        _main_binarize(ns, history)
    elif ns.command == "geom":
        _main_geom(ns, parser, history)
    elif ns.command == "tag":
        # tag promises that only the requested tags change, so it records no
        # history of its own.
        _validate_tag_args(ns, parser)
        _main_tag(ns)


if __name__ == "__main__":
    main()
