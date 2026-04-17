#!/usr/bin/env python3
"""Unified LTA transform manipulation CLI.

Available subcommands are ``diff`` to compare transforms, ``invert`` to invert
an LTA, and ``concat`` to chain two LTAs. Run ``lta --help`` or
``lta <subcommand> --help`` for the full command syntax.
"""

import argparse
import sys

import numpy as np

from neuroreg.transforms import LTA, decompose_transform

# ── helpers ───────────────────────────────────────────────────────────────────


def _positive_float(value: str) -> float:
    """Argparse type that accepts only strictly positive floats."""
    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid float") from None
    if f <= 0.0:
        raise argparse.ArgumentTypeError(f"--normdiv must be a positive, non-zero value (got {f})")
    return f


_VOL_FIELDS = ("xras", "yras", "zras", "cras", "voxelsize", "volume")


def _check_vol_info(
    parser: argparse.ArgumentParser,
    lta: LTA,
    label: str,
    blocks: tuple[str, ...] = ("src", "dst"),
) -> None:
    """Emit a parser.error if any required volume-info field is absent."""
    for block in blocks:
        info = getattr(lta, block)
        missing = [k for k in _VOL_FIELDS if k not in info]
        if missing:
            parser.error(f"{label} {block} volume info is missing required fields: {missing}")


def _needs_vol_info(lta: LTA, dist: int) -> bool:
    """Return True when the chosen metric path will access volume-info fields.

    False only for R2R-stored LTAs on a plain r2r() call (fast path, no
    affine needed).  V2V conversion and dist-3 corner placement both require
    both src and dst volume info.
    """
    return dist == 3 or lta.type == 0


def _run_dist(ns: argparse.Namespace, lta1: LTA, lta2: LTA | None) -> None:
    """Execute the distance computation for the diff subcommand."""
    if ns.dist == 1:
        print(f"{lta1.rigid_dist(lta2) / ns.normdiv}")

    elif ns.dist == 2:
        print(f"{lta1.affine_dist(lta2, radius=ns.radius) / ns.normdiv}")

    elif ns.dist == 3:
        print(f"{lta1.corner_dist(lta2) / ns.normdiv}")

    elif ns.dist == 4:
        print(f"{lta1.sphere_dist(lta2, radius=ns.radius) / ns.normdiv}")

    elif ns.dist == 5:
        # det is multiplicative: det(M1 @ M2) == det(M1) * det(M2)
        result = lta1.det if lta2 is None else lta1.det * lta2.det
        print(f"{result / ns.normdiv}")

    elif ns.dist == 7:
        # Single transform: use the LTA method; two transforms: compose then decompose.
        d = lta1.decompose() if lta2 is None else decompose_transform(lta1.r2r() @ lta2.r2r())
        with np.printoptions(precision=10, suppress=False):
            print("\nDecompose into Rot · Shear · diag(Scales) + Trans:\n")
            print("Rot =")
            print(d["rotation"])
            print(f"\nRotVec   = {d['rot_vec']}  (rad)")
            print(f"RotAngle = {np.radians(d['rot_angle_deg']):.6f} rad  = {d['rot_angle_deg']:.6f} deg")
            print("\nShear =")
            print(d["shear"])
            print(f"\nScales   = {d['scales']}")
            print(f"\nTrans    = {d['translation']}")
            print(f"AbsTrans = {d['abs_trans']:.6f} mm")
            print(f"\nDeterminant = {d['determinant']:.6f}")


# ── parser ────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lta",
        description="LTA transform manipulation utilities.",
    )
    sub = p.add_subparsers(dest="command", metavar="COMMAND", required=True)

    # ── diff ──────────────────────────────────────────────────────────────────
    diff_p = sub.add_parser(
        "diff",
        help="Compute a distance metric between two LTAs, or one vs identity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Compute distance metrics between two LTA transforms, or between\n"
            "one transform and identity.\n"
            "\n"
            "All metrics operate on the RAS-to-RAS representation of the stored\n"
            "transforms (vox-to-vox LTAs are converted automatically).\n"
            "\n"
            "Distance types:\n"
            "  1   Rigid transform distance  sqrt(||log R_d||² + ||T_d||²)\n"
            "      D = inv(M1) @ M2  (or M1 vs identity)\n"
            "  2   Affine RMS distance (Jenkinson 1999)  [default]\n"
            "      sqrt(r²/5 · Tr(AᵀA) + ||T_d||²),  D = M1 − M2\n"
            "  3   8-corner mean displacement (mm, image-specific)\n"
            "  4   Max displacement on sphere of radius r (mm)\n"
            "      D = inv(M1) @ M2\n"
            "  5   Determinant of M1 (· M2 when given)\n"
            "  7   Polar decomposition: rotation, shear, scale, translation\n"
        ),
    )
    diff_p.add_argument("lta1", metavar="LTA1", help="First (or only) LTA transform file.")
    diff_p.add_argument(
        "lta2", metavar="LTA2", nargs="?", default=None, help="Second LTA file.  Omit to compare LTA1 against identity."
    )
    diff_p.add_argument(
        "--dist",
        type=int,
        default=2,
        choices=[1, 2, 3, 4, 5, 7],
        metavar="{1,2,3,4,5,7}",
        help="Distance type (default: 2).",
    )
    diff_p.add_argument(
        "--radius",
        type=float,
        default=100.0,
        metavar="MM",
        help="Sphere / RMS radius in mm (dist 2 and 4, default: 100).",
    )
    diff_p.add_argument(
        "--normdiv",
        type=_positive_float,
        default=1.0,
        metavar="FLOAT",
        help="Divide the final distance by this value (must be > 0, default: 1).",
    )
    diff_p.add_argument("--invert1", action="store_true", help="Invert the first transform before comparison.")
    diff_p.add_argument("--invert2", action="store_true", help="Invert the second transform before comparison.")

    # ── invert ────────────────────────────────────────────────────────────────
    inv_p = sub.add_parser(
        "invert",
        help="Invert an LTA transform and write it to a new file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Invert a FreeSurfer LTA transform.\n"
            "\n"
            "The output is always stored as LINEAR_RAS_TO_RAS (type 1)\n"
            "with src and dst geometry blocks swapped."
        ),
    )
    inv_p.add_argument("input", metavar="INPUT", help="Input LTA file.")
    inv_p.add_argument("output", metavar="OUTPUT", help="Output (inverted) LTA file.")

    # ── concat ────────────────────────────────────────────────────────────────
    cat_p = sub.add_parser(
        "concat",
        help="Concatenate two LTAs (A→B then B→C) into one (A→C).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Concatenate two LTA transforms.\n"
            "\n"
            "LTA1 maps A → B and LTA2 maps B → C.\n"
            "The output maps A → C with matrix  M_LTA2 @ M_LTA1.\n"
            "src geometry is taken from LTA1; dst geometry from LTA2.\n"
            "\n"
            "Equivalent to FreeSurfer's mri_concatenate_lta."
        ),
    )
    cat_p.add_argument("lta1", metavar="LTA1", help="First transform  (A → B).")
    cat_p.add_argument("lta2", metavar="LTA2", help="Second transform (B → C).")
    cat_p.add_argument("output", metavar="OUTPUT", help="Output LTA file  (A → C).")

    return p


# ── subcommand handlers ───────────────────────────────────────────────────────


def _main_diff(parser: argparse.ArgumentParser, ns: argparse.Namespace) -> None:
    if ns.invert2 and ns.lta2 is None:
        parser.error("lta diff: --invert2 requires a second LTA file.")

    try:
        lta1 = LTA.read(ns.lta1)
    except Exception as e:
        print(f"ERROR: cannot read {ns.lta1}: {e}", file=sys.stderr)
        sys.exit(1)

    lta2 = None
    if ns.lta2 is not None:
        try:
            lta2 = LTA.read(ns.lta2)
        except Exception as e:
            print(f"ERROR: cannot read {ns.lta2}: {e}", file=sys.stderr)
            sys.exit(1)

    ltas = [(lta1, "LTA1")] + ([(lta2, "LTA2")] if lta2 is not None else [])
    for lta, label in ltas:
        if _needs_vol_info(lta, ns.dist):
            _check_vol_info(parser, lta, label)

    if ns.invert1:
        lta1 = lta1.invert()
    if ns.invert2 and lta2 is not None:
        lta2 = lta2.invert()

    try:
        _run_dist(ns, lta1, lta2)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def _main_invert(ns: argparse.Namespace) -> None:
    try:
        lta = LTA.read(ns.input)
    except Exception as e:
        print(f"ERROR: cannot read {ns.input}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        lta.invert().write(ns.output)
    except Exception as e:
        print(f"ERROR: cannot write {ns.output}: {e}", file=sys.stderr)
        sys.exit(1)


def _main_concat(ns: argparse.Namespace) -> None:
    try:
        lta1 = LTA.read(ns.lta1)
    except Exception as e:
        print(f"ERROR: cannot read {ns.lta1}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        lta2 = LTA.read(ns.lta2)
    except Exception as e:
        print(f"ERROR: cannot read {ns.lta2}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        lta1.concat(lta2).write(ns.output)
    except Exception as e:
        print(f"ERROR: cannot write {ns.output}: {e}", file=sys.stderr)
        sys.exit(1)


# ── entry point ───────────────────────────────────────────────────────────────


def main(args=None) -> None:
    """Entry point for the ``lta`` command."""
    parser = _build_parser()
    ns = parser.parse_args(args)

    if ns.command == "diff":
        _main_diff(parser, ns)
    elif ns.command == "invert":
        _main_invert(ns)
    elif ns.command == "concat":
        _main_concat(ns)

if __name__ == "__main__":
    main()
