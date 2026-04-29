#!/usr/bin/env python3
"""Command-line interface for segmentation-based centroid registration.

The CLI fits an LTA from a moving segmentation to either a target segmentation,
a centroid target JSON file (or bundled centroid target name), or a left-right
flipped self target.
"""

from __future__ import annotations

import argparse
import logging

from ..image import load_image
from ..segreg import segreg
from ..segreg.atlas import available_atlases
from ..transforms import LTA


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
    atlas_names = ", ".join(available_atlases())
    parser = argparse.ArgumentParser(
        prog="segreg",
        description=(
            "Segmentation-based registration via label centroids.\n"
            "Fits translation-only, rigid, similarity, no-shear anisotropic-scale, or affine transforms "
            "from a moving segmentation to a target segmentation, a centroid target file, or a left-right "
            "flipped self target."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seg", required=True, metavar="FILE", help="Moving segmentation image (NIfTI or MGZ).")

    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--target-seg",
        metavar="FILE",
        help="Target segmentation image used to derive centroid correspondences and LTA target geometry.",
    )
    target_group.add_argument(
        "--centroids",
        metavar="TARGET",
        help=f"Target centroid JSON file or bundled target name ({atlas_names}).",
    )
    target_group.add_argument("--flipped", action="store_true", help="Register to a left-right flipped self target.")

    parser.add_argument("--lta", required=True, metavar="FILE", help="Output LTA transform.")
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
        choices=["all_shared", "target_centroids", "cortex_lr_pairs"],
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
    parser.add_argument("--verbose", action="store_true", help="Enable INFO-level logging.")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging.")
    return parser


def _validate_args(ns: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate cross-argument constraints after parsing."""
    if ns.flipped and ns.dof != 6:
        parser.error("--flipped currently supports only --dof 6.")


def main(args=None) -> None:
    """Run the ``segreg`` command-line entry point."""
    parser = _build_parser()
    ns = parser.parse_args(args)
    _validate_args(ns, parser)

    level = logging.DEBUG if ns.debug else (logging.INFO if ns.verbose else logging.WARNING)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    result = segreg(
        ns.seg,
        ns.target_seg,
        centroids=ns.centroids,
        dof=ns.dof,
        labels=ns.labels,
        label_set=ns.label_set,
        min_common_labels=ns.min_common_labels,
        flipped=ns.flipped,
        midslice=ns.midslice,
    )

    mov_seg_img = load_image(ns.seg)
    LTA.from_matrix(
        result.r2r,
        ns.seg,
        mov_seg_img,
        result.target_name,
        result.target_geometry,
        lta_type=1,
    ).write(ns.lta)
    print(f"LTA:       {ns.lta}")


if __name__ == "__main__":
    main()
