#!/usr/bin/env python3
"""Command-line interface for segmentation-based centroid registration.

The CLI wraps :func:`neuroreg.segreg.segreg` and optional mapped-image output
helpers so users can fit centroid-based transforms and immediately export LTAs,
resliced images, header-only mappings, or centroid JSON files.
"""

from __future__ import annotations

import argparse
import logging

from neuroreg.image import load_image, save_header_mapped_image, save_resliced_r2r_image
from neuroreg.segreg import segreg
from neuroreg.segreg.atlas import load_atlas_centroids as load_bundled_atlas_centroids
from neuroreg.segreg.io import read_centroids_json, write_centroids_json
from neuroreg.segreg.register import export_segmentation_centroids, resolve_output_geometry
from neuroreg.transforms import LTA


def _parse_int_csv(value: str) -> list[int]:
    """Parse a comma-separated CLI list of integer label IDs."""
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers")
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers") from exc


def _build_parser() -> argparse.ArgumentParser:
    """Build the ``segreg`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="segreg",
        description=(
            "Segmentation-based registration via label centroids.\n"
            "Supports translation-only, rigid, similarity, no-shear anisotropic-scale, or affine centroid "
            "alignment between segmentations, to bundled atlas centroids, or to a left-right flipped self "
            "target for uprighting."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mov", required=True, metavar="FILE", help="Moving segmentation image (NIfTI or MGZ).")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--ref", metavar="FILE", help="Reference segmentation image (NIfTI or MGZ).")
    target_group.add_argument("--ref-centroids", metavar="FILE", help="Reference centroid JSON file.")
    target_group.add_argument("--atlas", choices=["fsaverage"], help="Bundled atlas centroid target.")
    target_group.add_argument("--flipped", action="store_true", help="Register to a left-right flipped self target.")

    parser.add_argument(
        "--movimg",
        metavar="FILE",
        help=(
            "Optional moving intensity image to map with --mapmov/--mapmovhdr. "
            "If omitted, those outputs map the moving segmentation itself."
        ),
    )
    parser.add_argument(
        "--ref-geom",
        metavar="FILE",
        help="Optional reference geometry image when using --ref-centroids.",
    )

    parser.add_argument("--lta", metavar="FILE", help="Output LTA transform.")
    parser.add_argument(
        "--mapmov",
        metavar="FILE",
        help=(
            "Write a resliced mapped image. Uses --movimg when provided; otherwise "
            "the moving segmentation itself is resliced."
        ),
    )
    parser.add_argument(
        "--mapmovhdr",
        metavar="FILE",
        help=(
            "Write a header-mapped image with no interpolation. Uses --movimg when "
            "provided; otherwise the moving segmentation itself is remapped."
        ),
    )
    parser.add_argument("--write-mov-centroids", metavar="FILE", help="Export moving centroids as JSON.")
    parser.add_argument("--write-ref-centroids", metavar="FILE", help="Export target centroids as JSON.")

    parser.add_argument(
        "--dof",
        type=int,
        default=6,
        choices=[3, 6, 7, 9, 12],
        metavar="{3,6,7,9,12}",
        help=(
            "3=translation only, 6=rigid, 7=rigid+global scale, "
            "9=rigid+anisotropic scale (no shear), 12=affine."
        ),
    )
    parser.add_argument("--labels", type=_parse_int_csv, default=None, help="Comma-separated label subset override.")
    parser.add_argument(
        "--label-set",
        choices=["all_shared", "fsaverage_centroids", "cortex_lr_pairs"],
        default=None,
        help="Named label preset. Defaults depend on the chosen target mode.",
    )
    parser.add_argument(
        "--min-common-labels",
        type=int,
        default=None,
        help="Minimum number of matched labels required.",
    )
    parser.add_argument("--midslice", type=float, default=None, help="Mid-sagittal x position for --flipped mode.")
    parser.add_argument(
        "--keep-geom",
        choices=["mov", "ref", "atlas"],
        default=None,
        help="Geometry used by --mapmov. Defaults to ref for --ref, atlas for --atlas, otherwise mov.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable INFO-level logging.")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging.")
    return parser


def _default_keep_geom(ns: argparse.Namespace) -> str:
    """Return the implicit output-geometry policy for mapped images."""
    if ns.keep_geom is not None:
        return ns.keep_geom
    if ns.atlas is not None:
        return "atlas"
    if ns.ref is not None or ns.ref_geom is not None:
        return "ref"
    return "mov"


def _validate_args(ns: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate cross-argument constraints after parsing."""
    if not any((ns.lta, ns.mapmov, ns.mapmovhdr, ns.write_mov_centroids, ns.write_ref_centroids)):
        parser.error(
            "At least one output is required: --lta, --mapmov, --mapmovhdr, "
            "--write-mov-centroids, or --write-ref-centroids."
        )

    if ns.flipped and ns.ref_geom is not None:
        parser.error("--ref-geom is not valid with --flipped.")
    if ns.flipped and ns.dof != 6:
        parser.error("--flipped currently supports only --dof 6.")

    keep_geom = _default_keep_geom(ns)
    if ns.flipped and keep_geom != "mov":
        parser.error("--flipped only supports --keep-geom mov.")
    if ns.atlas is None and keep_geom == "atlas":
        parser.error("--keep-geom atlas requires --atlas.")


def _write_ref_centroids(ns: argparse.Namespace) -> None:
    """Export the selected target centroids to JSON when requested."""
    if ns.write_ref_centroids is None:
        return
    if ns.ref is not None:
        export_segmentation_centroids(ns.ref, ns.write_ref_centroids, labels=ns.labels)
        return
    if ns.ref_centroids is not None:
        write_centroids_json(ns.write_ref_centroids, read_centroids_json(ns.ref_centroids))
        return
    if ns.atlas is not None:
        write_centroids_json(ns.write_ref_centroids, load_bundled_atlas_centroids(ns.atlas))
        return
    raise ValueError("No reference centroids are available for export in the requested mode.")


def main(args=None) -> None:
    """Run the ``segreg`` command-line entry point."""
    parser = _build_parser()
    ns = parser.parse_args(args)
    _validate_args(ns, parser)

    level = logging.DEBUG if ns.debug else (logging.INFO if ns.verbose else logging.WARNING)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("neuroreg.cli.segreg")

    if ns.write_mov_centroids:
        export_segmentation_centroids(ns.mov, ns.write_mov_centroids, labels=ns.labels)
    if ns.write_ref_centroids:
        _write_ref_centroids(ns)

    if not any((ns.lta, ns.mapmov, ns.mapmovhdr)):
        if ns.write_mov_centroids:
            print(f"MovCent:   {ns.write_mov_centroids}")
        if ns.write_ref_centroids:
            print(f"RefCent:   {ns.write_ref_centroids}")
        return

    result = segreg(
        ns.mov,
        ns.ref,
        ref_centroids=ns.ref_centroids,
        ref_geom=ns.ref_geom,
        atlas=ns.atlas,
        dof=ns.dof,
        labels=ns.labels,
        label_set=ns.label_set,
        min_common_labels=ns.min_common_labels,
        flipped=ns.flipped,
        midslice=ns.midslice,
    )

    mov_seg_img = load_image(ns.mov)
    if ns.lta:
        LTA.from_matrix(
            result.r2r,
            ns.mov,
            mov_seg_img,
            result.target_name,
            result.target_geometry,
            lta_type=1,
        ).write(ns.lta)
        print(f"LTA:       {ns.lta}")

    mapped_source_path = ns.movimg or ns.mov
    mapped_source_img = load_image(mapped_source_path) if (ns.mapmov or ns.mapmovhdr) else None

    ref_geom_img = None
    if ns.ref is not None:
        ref_geom_img = load_image(ns.ref)
    elif ns.ref_geom is not None:
        ref_geom_img = load_image(ns.ref_geom)

    if ns.mapmov and mapped_source_img is not None:
        keep_geom = _default_keep_geom(ns)
        target_affine, target_shape = resolve_output_geometry(
            result,
            keep_geom=keep_geom,
            mov_img=mapped_source_img,
            ref_img=ref_geom_img,
        )
        save_resliced_r2r_image(
            mapped_source_img,
            result.r2r,
            ns.mapmov,
            target_affine=target_affine,
            target_shape=target_shape,
            mode="linear" if ns.movimg is not None else None,
        )
        logger.info("Wrote resliced mapped image: %s", ns.mapmov)
        print(f"MapMov:    {ns.mapmov}")

    if ns.mapmovhdr and mapped_source_img is not None:
        save_header_mapped_image(mapped_source_img, result.r2r, ns.mapmovhdr)
        logger.info("Wrote header-mapped image: %s", ns.mapmovhdr)
        print(f"MapMovHdr: {ns.mapmovhdr}")

    if ns.write_mov_centroids:
        print(f"MovCent:   {ns.write_mov_centroids}")
    if ns.write_ref_centroids:
        print(f"RefCent:   {ns.write_ref_centroids}")


if __name__ == "__main__":
    main()
