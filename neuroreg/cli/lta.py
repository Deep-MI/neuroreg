#!/usr/bin/env python3
"""Unified LTA transform manipulation CLI.

Available subcommands are ``diff`` to compare transforms, ``invert`` to invert
an LTA, ``concat`` to chain two LTAs, and ``convert`` to translate between
LTA, XFM, FSL, ITK/ANTs text affine, experimental ANTs Matlab affine,
experimental AFNI affine text, NiftyReg affine text matrices, and
tkregister ``register.dat`` transforms. Run ``lta --help`` or
``lta <subcommand> --help`` for the full command syntax.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from neuroreg.transforms import (
    LTA,
    XFM,
    AFNIAffine,
    ANTsMatTransform,
    FSLMat,
    ITKTransform,
    NiftyRegTransform,
    RegisterDat,
    decompose_transform,
)

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


_FORMATS = ("lta", "xfm", "fsl", "regdat", "itk", "antsmat", "afni", "niftyreg")


def _infer_transform_format(path: str, explicit: str | None = None) -> str:
    if explicit is not None:
        return explicit

    lower = path.lower()
    suffix = Path(lower).suffix
    if suffix == ".lta":
        return "lta"
    if suffix == ".xfm":
        return "xfm"
    if lower.endswith("genericaffine.mat"):
        return "antsmat"
    if suffix in {".mat", ".fslmat"}:
        return "fsl"
    if suffix in {".dat", ".reg"}:
        return "regdat"
    if suffix == ".tfm" or lower.endswith(".itk.txt") or lower.endswith(".ants.txt"):
        return "itk"
    if lower.endswith(".aff12.1d"):
        return "afni"
    if lower.endswith(".niftyreg.txt"):
        return "niftyreg"
    raise ValueError(
        f"Unsupported transform format for {path!r}; expected .lta, .xfm, .mat, .fslmat, "
        ".dat, .reg, .tfm, .aff12.1D, \\*GenericAffine.mat, or .niftyreg.txt. "
        "Use --in-format/--out-format for ambiguous text formats such as .txt, .1D, or .mat"
    )


def _read_transform_as_lta(
        path: str,
        src_img: str | None = None,
        dst_img: str | None = None,
        fmt: str | None = None,
) -> LTA:
    fmt = _infer_transform_format(path, explicit=fmt)
    if fmt == "lta":
        return LTA.read(path)
    if fmt == "xfm":
        return XFM.read(path).to_lta(src_fname=src_img, src_img=src_img, dst_fname=dst_img, dst_img=dst_img)
    if fmt == "itk":
        return ITKTransform.read(path).to_lta(src_fname=src_img, src_img=src_img, dst_fname=dst_img, dst_img=dst_img)
    if fmt == "antsmat":
        return ANTsMatTransform.read(path).to_lta(
            src_fname=src_img, src_img=src_img, dst_fname=dst_img, dst_img=dst_img
        )
    if fmt == "afni":
        return AFNIAffine.read(path).to_lta(src_fname=src_img, src_img=src_img, dst_fname=dst_img, dst_img=dst_img)
    if fmt == "niftyreg":
        return NiftyRegTransform.read(path).to_lta(
            src_fname=src_img, src_img=src_img, dst_fname=dst_img, dst_img=dst_img
        )
    if src_img is None or dst_img is None:
        kind = "FSL" if fmt == "fsl" else "register.dat"
        raise ValueError(f"{kind} conversion requires both --src-img and --dst-img")
    if fmt == "fsl":
        return FSLMat.read(path).to_lta(src_fname=src_img, src_img=src_img, dst_fname=dst_img, dst_img=dst_img)
    return RegisterDat.read(path).to_lta(src_fname=src_img, src_img=src_img, dst_fname=dst_img, dst_img=dst_img)


def _write_lta_as_transform(
        lta: LTA,
        output: str,
        *,
        output_format: str | None = None,
        out_type: str | None = None,
        subject: str | None = None,
        fscale: float | None = None,
        float2int: str = "round",
) -> None:
    fmt = _infer_transform_format(output, explicit=output_format)
    if fmt == "lta":
        lta_type = None if out_type is None else {"vox2vox": 0, "ras2ras": 1}[out_type]
        lta.write(output, lta_type=lta_type)
    elif fmt == "xfm":
        XFM.from_lta(lta).write(output)
    elif fmt == "fsl":
        FSLMat.from_lta(lta).write(output)
    elif fmt == "itk":
        ITKTransform.from_lta(lta).write(output)
    elif fmt == "antsmat":
        ANTsMatTransform.from_lta(lta).write(output)
    elif fmt == "afni":
        AFNIAffine.from_lta(lta).write(output)
    elif fmt == "niftyreg":
        NiftyRegTransform.from_lta(lta).write(output)
    else:
        RegisterDat.from_lta(lta, subject=subject, intensity=fscale, float2int=float2int).write(output)


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
        # Concatenation determinant: det(M1 @ M2) = det(M1) * det(M2).
        result = lta1.det if lta2 is None else lta1.det * lta2.det
        print(f"{result / ns.normdiv}")

    elif ns.dist == 7:
        # Decompose the concatenation M1 @ M2 (not the difference inv(M1) @ M2).
        # Single transform: decompose M1 alone.
        # Two transforms: compose M1 @ M2 first, then decompose.
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
            "  1   Rigid transform distance  sqrt(||log R||² + ||T||²)\n"
            "      D = inv(M1) @ M2  (or M1 vs identity)\n"
            "      R = upper-left 3×3 rotation block of D\n"
            "      T = upper-right 3×1 translation column of D\n"
            "      Units: mixed (mm and rad added in quadrature)\n"
            "  2   Affine RMS distance (Jenkinson 1999)  [default]\n"
            "      sqrt(r²/5 · Tr(AᵀA) + ‖T‖²),  D = M1 − M2  (or M1 − I)\n"
            "      A = upper-left 3×3 of D; r = --radius (default 100 mm)\n"
            "      T = upper-right 3×1 translation column of D\n"
            "      Units: mm (RMS displacement over a sphere of radius r)\n"
            "  3   8-corner mean displacement (mm, image-specific)\n"
            "      One transform:  mean‖M1·c − c‖ for each src corner c in RAS.\n"
            "      Two transforms: mean‖M1·c − M2·c‖  (same src corners).\n"
            "  4   Max displacement on a sphere of radius r (mm, image-independent)\n"
            "      Md = inv(M1) @ M2  (or M1 vs identity)\n"
            "      displacement(p) = ‖Md·p − p‖  over ~1600 sphere samples\n"
            "  5   Determinant  det(M1)  (or det(M1 @ M2) = det(M1)*det(M2) when M2 given)\n"
            "      Uses matrix concatenation (not the difference); det is order-independent.\n"
            "  7   Polar decomposition of M1 (or M1 @ M2, i.e. concatenation):\n"
            "      prints Rot, RotVec, RotAngle, Shear, Scales, Trans, abs(Trans), det\n"
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

    # ── convert ───────────────────────────────────────────────────────────────
    conv_p = sub.add_parser(
        "convert",
        help="Convert between LTA, XFM, FSL, ITK/ANTs, ANTs .mat, AFNI, NiftyReg, and register.dat transforms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Convert between FreeSurfer-adjacent linear transform formats.\n"
            "\n"
            "Supported formats are usually inferred from file suffixes:\n"
            "  .lta  FreeSurfer Linear Transform Array\n"
            "  .xfm  MNI/MINC linear transform\n"
            "  .mat/.fslmat  FSL FLIRT affine matrix\n"
            "  .dat/.reg  tkregister volumetric register.dat format\n"
            "  .tfm  ITK/ANTs 3D affine text transform\n"
            "  \\*GenericAffine.mat  experimental ANTs / ITK Matlab affine\n"
            "  .aff12.1D  experimental AFNI affine text matrix\n"
            "  .niftyreg.txt  NiftyReg 3D affine text matrix\n"
            "\n"
            "Use --in-format/--out-format for ambiguous text outputs such as .txt, .1D, or .mat.\n"
            "FSL and register.dat conversion require both --src-img and --dst-img\n"
            "because the stored matrices depend on image geometry rather than being\n"
            "plain scanner-RAS affines. ITK/ANTs text affines, experimental ANTs .mat,\n"
            "experimental AFNI affine text, and NiftyReg affine text matrices are\n"
            "scanner-space transforms and can be read without images, though\n"
            "--src-img/--dst-img still enrich the resulting LTA geometry blocks.\n"
            "ANTs .mat support is currently based on SciPy + ITK Matlab IO semantics\n"
            "and should be considered experimental until validated on real files.\n"
            "AFNI support currently targets affine text matrices in DICOM/LPS\n"
            "coordinates and should likewise be considered experimental.\n"
            "NiftyReg affine text matrices store the inverse target-to-source RAS matrix."
        ),
    )
    conv_p.add_argument("input", metavar="INPUT", help="Input transform file.")
    conv_p.add_argument("output", metavar="OUTPUT", help="Output transform file.")
    conv_p.add_argument("--in-format", choices=_FORMATS, help="Override input format inference for ambiguous files.")
    conv_p.add_argument("--out-format", choices=_FORMATS, help="Override output format inference for ambiguous files.")
    conv_p.add_argument("--src-img", help="Moving/source image geometry for conversion when needed.")
    conv_p.add_argument("--dst-img", help="Reference/target image geometry for conversion when needed.")
    conv_p.add_argument(
        "--out-type",
        choices=["ras2ras", "vox2vox"],
        help="Output LTA storage type when OUTPUT ends in .lta (default: preserve the input LTA storage type).",
    )
    conv_p.add_argument("--subject", help="Subject metadata to store when writing register.dat.")
    conv_p.add_argument(
        "--fscale",
        type=float,
        help="Intensity/fscale metadata to store when writing register.dat.",
    )
    conv_p.add_argument(
        "--float2int",
        choices=["tkregister", "round", "floor"],
        default="round",
        help="Float-to-int footer when writing register.dat (default: round).",
    )

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


def _main_convert(ns: argparse.Namespace) -> None:
    try:
        lta = _read_transform_as_lta(ns.input, src_img=ns.src_img, dst_img=ns.dst_img, fmt=ns.in_format)
    except Exception as e:
        print(f"ERROR: cannot read {ns.input}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        _write_lta_as_transform(
            lta,
            ns.output,
            output_format=ns.out_format,
            out_type=ns.out_type,
            subject=ns.subject,
            fscale=ns.fscale,
            float2int=ns.float2int,
        )
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
    elif ns.command == "convert":
        _main_convert(ns)


if __name__ == "__main__":
    main()
