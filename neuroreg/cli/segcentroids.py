#!/usr/bin/env python3
"""Command-line interface for writing centroid target JSON files."""

from __future__ import annotations

import argparse
import logging

from ..segreg.io import geometry_from_image, read_target_json, write_target_json
from ..segreg.register import export_segmentation_target


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
    """Build the ``segcentroids`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="segcentroids",
        description=(
            "Either compute label centroids from a segmentation or read an existing centroid target JSON file, "
            "then write a centroid target JSON file. When building from --seg, geometry metadata comes from "
            "--geometry when provided, otherwise from --seg. When editing an existing target with --input, "
            "--geometry supplies the geometry to add or replace."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--seg", metavar="FILE", help="Segmentation image used to compute centroids.")
    source_group.add_argument(
        "--input",
        metavar="FILE",
        help=(
            "Existing centroid target JSON file whose centroids should be preserved "
            "while geometry is added or replaced."
        ),
    )
    parser.add_argument(
        "--geometry",
        metavar="FILE",
        default=None,
        help="Optional image whose geometry metadata should be embedded in the output JSON.",
    )
    parser.add_argument("--out", required=True, metavar="FILE", help="Output centroid target JSON file.")
    parser.add_argument("--labels", type=_parse_int_csv, default=None, help="Comma-separated label subset override.")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO-level logging.")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging.")
    return parser


def _validate_args(ns: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate cross-argument constraints after parsing."""
    if ns.input is not None and ns.geometry is None:
        parser.error("--input requires --geometry so segcentroids knows what geometry to add or replace.")
    if ns.input is not None and ns.labels is not None:
        parser.error("--labels is only valid together with --seg.")


def main(args=None) -> None:
    """Run the ``segcentroids`` command-line entry point."""
    parser = _build_parser()
    ns = parser.parse_args(args)
    _validate_args(ns, parser)

    level = logging.DEBUG if ns.debug else (logging.INFO if ns.verbose else logging.WARNING)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    if ns.input is not None:
        target = read_target_json(ns.input)
        write_target_json(ns.out, target.centroids, geometry=geometry_from_image(ns.geometry))
    else:
        export_segmentation_target(ns.seg, ns.out, geometry=ns.geometry, labels=ns.labels)
    print(f"Target:    {ns.out}")


if __name__ == "__main__":
    main()
