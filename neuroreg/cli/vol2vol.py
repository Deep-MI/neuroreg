#!/usr/bin/env python3
"""Command-line image mapping and reslicing utility.

The target geometry for resampled output is assembled per component rather than
copied from one source wholesale. The components are the matrix size, the voxel
sizes, the direction cosines, and the placement in world space (``c_ras``).

Each component is resolved from the first available source in this order:

1. an explicit override flag (currently ``--ref-cras``, for placement only),
2. the ``--ref`` image,
3. the ``dst`` geometry stored in ``--transform``,
4. the input image.

Overrides apply on top of whichever base geometry was selected, so
``--ref sub.mgz --ref-cras 0,0,0`` means "the reference grid, recentred on the
world origin" and needs no reference file at that voxel size to exist.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from typing import Any

import numpy as np

from ._outputs import validate_image_outputs
from ..image import (
    check_dtype_storable,
    clip_and_cast_dtype,
    create_image_like,
    header_map_image,
    load_image,
    reslice_r2r_image,
    save_image,
)
from ..transforms import TRANSFORM_FORMATS, affine_from_volume_info, read_transform_as_lta


class _NumberListParser(argparse.ArgumentParser):
    """Parser that reads a leading-minus number list as a value, not a flag.

    ``argparse`` treats any token starting with ``-`` as an option unless it
    matches its private negative-number pattern, which covers ``-4`` and
    ``-4.5`` but not lists such as ``-4.25,6.5,0``. Negative coordinates are
    the common case for ``--ref-cras`` (a scanner ``c_ras`` is usually
    negative in at least one axis), and without this the natural
    ``--ref-cras -4,0,0`` fails with "expected one argument" and forces users
    to discover the ``--ref-cras=-4,0,0`` spelling.

    Widening the pattern to "minus followed by a digit or a decimal point" is
    safe here because every option this CLI defines is a ``--word``, so no
    flag can be shadowed. The check also runs only after argparse has failed
    to resolve the token as a known option.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # argparse assigns this in _ActionsContainer.__init__, so it is an
        # instance attribute and has to be replaced after the base class runs;
        # a class-level override would simply be overwritten.
        self._negative_number_matcher = re.compile(r"^-[\d.]")


def _parse_pad(value: str) -> str | float:
    """Parse a padding argument as a named mode or numeric constant.

    Parameters
    ----------
    value : str
        CLI argument passed to ``--pad``.

    Returns
    -------
    str or float
        One of ``"zero"``, ``"border"``, ``"reflection"``, ``"brightest"``,
        or a numeric constant.

    Raises
    ------
    argparse.ArgumentTypeError
        If the argument is neither a supported named mode nor a valid float.
    """
    lowered = value.strip().lower()
    if lowered in {"zero", "border", "reflection", "brightest"}:
        return lowered
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--pad must be one of zero, border, reflection, brightest, or a numeric constant"
        ) from exc


def _parse_out_dtype(value: str) -> str:
    """Normalize an output-dtype argument to a NumPy dtype name.

    Parameters
    ----------
    value : str
        CLI argument passed to ``--out-dtype``.

    Returns
    -------
    str
        Normalized dtype name or the literal string ``"input"``.

    Raises
    ------
    argparse.ArgumentTypeError
        If the argument cannot be resolved to a supported dtype.
    """
    lowered = value.strip().lower()
    if lowered == "input":
        return "input"
    aliases = {
        "uchar": "uint8",
        "short": "int16",
        "int": "int32",
        "float": "float32",
        "double": "float64",
    }
    try:
        return np.dtype(aliases.get(lowered, lowered)).name
    except TypeError as exc:
        raise argparse.ArgumentTypeError(f"Unsupported output dtype {value!r}.") from exc


def _parse_cras(value: str) -> np.ndarray:
    """Parse a target-centre argument into a world-coordinate 3-vector.

    Parameters
    ----------
    value : str
        CLI argument passed to ``--ref-cras`` as ``X,Y,Z``. Surrounding
        whitespace around the whole value or any component is ignored.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Requested world coordinate of the target grid centre.

    Raises
    ------
    argparse.ArgumentTypeError
        If the argument is not exactly three comma-separated finite numbers.
    """
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--ref-cras must be three comma-separated numbers (X,Y,Z), got {len(parts)}: {value!r}"
        )
    try:
        cras = np.array([float(part) for part in parts], dtype=np.float64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--ref-cras must be three comma-separated numbers (X,Y,Z), got: {value!r}"
        ) from exc
    if not np.all(np.isfinite(cras)):
        # A non-finite centre would poison the whole output affine silently.
        raise argparse.ArgumentTypeError(f"--ref-cras must be finite, got: {value!r}")
    return cras


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``vol2vol`` command.

    Returns
    -------
    argparse.ArgumentParser
        Configured CLI parser.
    """
    parser = _NumberListParser(
        prog="vol2vol",
        description=(
            "Apply a linear transform to an image, reslice into a target geometry,\n"
            "or update the header only. This is the project-native analogue of\n"
            "FreeSurfer's mri_vol2vol for linear transforms.\n"
            "\n"
            "With no --transform and no --ref the image is read and written as-is\n"
            "(no mapping or reslicing); the output format is taken from the --out\n"
            "file extension, so this converts between any formats nibabel supports\n"
            "(e.g. .mgz, .nii, .nii.gz, .img/.hdr). The dtype/scaling flags also\n"
            "apply on the native grid without reslicing. To mask a volume, use\n"
            "'mri mask'.\n"
            "\n"
            "The target geometry is taken from --ref, else the transform's dst\n"
            "geometry, else the input. Override flags such as --ref-cras apply on\n"
            "top of that base geometry."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--in", "--i", "--mov", required=True, dest="input_file", metavar="FILE", help="Input (moving) image."
    )
    parser.add_argument(
        "--transform",
        "--lta",
        metavar="FILE",
        dest="transform",
        help="Optional linear transform to apply (any format: .lta, .xfm, FSL .mat, …).",
    )
    parser.add_argument(
        "--transform-format",
        choices=TRANSFORM_FORMATS,
        help="Override transform format inference for ambiguous files such as .txt or .mat.",
    )
    parser.add_argument(
        "--ref",
        "--targ",
        metavar="FILE",
        dest="ref",
        help="Optional target/reference image geometry. Overrides geometry stored in the transform.",
    )
    parser.add_argument(
        "--ref-cras",
        type=_parse_cras,
        dest="ref_cras",
        metavar="X,Y,Z",
        help=(
            "Override the c_ras (world coordinate of the grid centre) of the target "
            "geometry. Orientation, voxel sizes and matrix size are unchanged; only "
            "where that grid sits in world space moves. Useful to reslice into a "
            "standard-space pose at the input's resolution."
        ),
    )
    parser.add_argument(
        "--out",
        "--o",
        required=True,
        metavar="FILE",
        help="Output image filename. The extension selects the output format and is required.",
    )
    interp_group = parser.add_mutually_exclusive_group()
    interp_group.add_argument(
        "--interp",
        choices=["linear", "cubic", "nearest"],
        default="linear",
        help="Interpolation mode for resampled output.",
    )
    interp_group.add_argument(
        "--trilin",
        dest="interp",
        action="store_const",
        const="linear",
        help="Trilinear interpolation. Alias for --interp linear.",
    )
    interp_group.add_argument(
        "--nearest",
        dest="interp",
        action="store_const",
        const="nearest",
        help="Nearest-neighbour interpolation. Alias for --interp nearest.",
    )
    interp_group.add_argument(
        "--cubic",
        dest="interp",
        action="store_const",
        const="cubic",
        help="Cubic interpolation. Alias for --interp cubic.",
    )
    parser.add_argument(
        "--pad",
        type=_parse_pad,
        default="zero",
        metavar="MODE|VALUE",
        help="Out-of-bounds padding: zero, border, reflection, brightest, or a numeric constant.",
    )
    parser.add_argument(
        "--header-only",
        "--no-resample",
        action="store_true",
        dest="header_only",
        help="Apply the transform to the header only and skip interpolation.",
    )
    parser.add_argument(
        "--inverse",
        "--inv",
        action="store_true",
        dest="inverse",
        help="Apply the inverse of the supplied transform.",
    )
    dtype_group = parser.add_mutually_exclusive_group()
    dtype_group.add_argument(
        "--out-dtype",
        type=_parse_out_dtype,
        metavar="DTYPE",
        help="Explicit final output dtype, or 'input' to preserve the moving-image dtype.",
    )
    dtype_group.add_argument(
        "--keep-dtype",
        "--keep-precision",
        action="store_true",
        dest="keep_dtype",
        help="Write output in the moving-image dtype. Equivalent to --out-dtype input.",
    )
    parser.add_argument(
        "--scale-mode",
        choices=["clamp", "rescale", "robust"],
        help="Optional final intensity policy before dtype conversion.",
    )
    parser.add_argument(
        "--target-max",
        type=float,
        help="Upper target value for zero-anchored rescale or robust-rescale output.",
    )
    parser.add_argument(
        "--robust-low",
        type=float,
        default=0.0,
        metavar="FRAC",
        help="Lower robust quantile used to trim the source intensity distribution.",
    )
    parser.add_argument(
        "--robust-high",
        type=float,
        default=0.999,
        metavar="FRAC",
        help="Upper robust quantile used to estimate the source intensity ceiling.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable INFO-level logging.")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging.")
    return parser


def _validate_args(ns: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate cross-argument constraints for the mapping CLI.

    Parameters
    ----------
    ns : argparse.Namespace
        Parsed CLI arguments.
    parser : argparse.ArgumentParser
        Parser used to report user-facing validation errors.

    Returns
    -------
    None
        Validation is performed for its side effects only.

    Raises
    ------
    SystemExit
        Raised indirectly via :meth:`argparse.ArgumentParser.error` when the
        argument combination is invalid.
    """
    if ns.transform_format is not None and ns.transform is None:
        parser.error("--transform-format requires --transform.")
    if ns.header_only:
        if ns.interp != "linear":
            parser.error("--header-only cannot be combined with --interp.")
        if ns.pad != "zero":
            parser.error("--header-only cannot be combined with --pad.")
        if ns.out_dtype is not None or ns.keep_dtype:
            parser.error("--header-only cannot be combined with output-dtype flags.")
        if ns.scale_mode is not None or ns.target_max is not None:
            parser.error("--header-only cannot be combined with scaling flags.")
        if ns.robust_low != 0.0 or ns.robust_high != 0.999:
            parser.error("--header-only cannot be combined with robust scaling flags.")
        if ns.ref_cras is not None:
            # --header-only rewrites the input's own header and never resamples,
            # so there is no target grid for a placement override to act on.
            parser.error("--header-only cannot be combined with --ref-cras.")
    if ns.scale_mode is None:
        if ns.target_max is not None:
            parser.error("--target-max requires --scale-mode rescale or --scale-mode robust.")
        if ns.robust_low != 0.0 or ns.robust_high != 0.999:
            parser.error("--robust-low and --robust-high require --scale-mode robust.")
    elif ns.scale_mode != "robust" and (ns.robust_low != 0.0 or ns.robust_high != 0.999):
        parser.error("--robust-low and --robust-high require --scale-mode robust.")
    if not 0.0 <= ns.robust_low <= ns.robust_high <= 1.0:
        parser.error("--robust-low and --robust-high must satisfy 0 <= low <= high <= 1.")


def _resolve_target_dtype(ns: argparse.Namespace, source_dtype: np.dtype) -> np.dtype | None:
    """Resolve the final output dtype requested by the user.

    Parameters
    ----------
    ns : argparse.Namespace
        Parsed CLI arguments.
    source_dtype : np.dtype
        Dtype of the moving/source image.

    Returns
    -------
    np.dtype or None
        Requested output dtype, or ``None`` when no explicit conversion was
        requested.
    """
    if ns.keep_dtype:
        return source_dtype
    if ns.out_dtype is None:
        return None
    return source_dtype if ns.out_dtype == "input" else np.dtype(ns.out_dtype)


def _apply_ref_cras(
        affine: np.ndarray,
        shape: tuple[int, int, int],
        cras: np.ndarray,
) -> np.ndarray:
    """Move a target grid so its centre sits at ``cras`` in world space.

    ``c_ras`` is the world coordinate of the grid centre, taken at
    ``shape / 2`` exactly rather than ``(shape - 1) / 2``, matching MGH and
    :func:`neuroreg.image.describe_image`. Only the translation column changes,
    so direction cosines, voxel sizes and matrix size are preserved by
    construction, including for anisotropic and oblique grids.

    Parameters
    ----------
    affine : numpy.ndarray, shape (4, 4)
        Base target voxel-to-RAS affine.
    shape : tuple of int
        Target spatial shape, used to locate the grid centre.
    cras : numpy.ndarray, shape (3,)
        Requested world coordinate of the grid centre.

    Returns
    -------
    numpy.ndarray, shape (4, 4)
        Copy of ``affine`` whose grid centre lands on ``cras``.
    """
    out = np.array(affine, dtype=np.float64, copy=True)
    center_vox = np.asarray(shape, dtype=np.float64) / 2.0
    out[:3, 3] = np.asarray(cras, dtype=np.float64) - out[:3, :3] @ center_vox
    return out


def _resolve_target_geometry(
        mov_img: Any,
        ref_img: Any | None,
        effective_lta: Any | None,
        *,
        ref_cras: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Assemble the target affine and shape for resampled output.

    A complete target geometry has four independent components: matrix size,
    voxel sizes, direction cosines, and placement in world space. Each is
    resolved from the first available source in this order: an explicit
    override keyword, the ``--ref`` image, the transform's ``dst`` geometry,
    then the input image. A base geometry is selected first and the overrides
    are then applied on top of it, so callers can ask for "the reference grid,
    but placed elsewhere" without needing such a file to exist.

    Parameters
    ----------
    mov_img : Any
        Loaded moving/source image.
    ref_img : Any or None
        Loaded reference image, if supplied.
    effective_lta : Any or None
        Effective transform as an LTA after applying any inversion requested by
        the user. Overrides therefore act on the post-inversion target.
    ref_cras : numpy.ndarray or None, optional
        Placement override: world coordinate of the target grid centre. When
        ``None`` the placement of the base geometry is kept.

    Returns
    -------
    tuple[numpy.ndarray, tuple[int, int, int]]
        Output affine and spatial shape.

    Raises
    ------
    ValueError
        If no valid target geometry can be resolved.
    """
    if ref_img is not None:
        affine = np.asarray(ref_img.affine, dtype=np.float64)
        shape = tuple(int(v) for v in ref_img.shape[:3])
    elif effective_lta is None:
        affine = np.asarray(mov_img.affine, dtype=np.float64)
        shape = tuple(int(v) for v in mov_img.shape[:3])
    else:
        info = effective_lta.dst
        affine = affine_from_volume_info(info)
        shape = tuple(int(v) for v in info["volume"])

    if ref_cras is not None:
        affine = _apply_ref_cras(affine, shape, ref_cras)
    return affine, shape


def _resolve_padding(pad: str | float, mov_img: Any) -> tuple[str, float | None]:
    """Resolve CLI padding configuration to mapping-helper arguments.

    Parameters
    ----------
    pad : str or float
        Parsed ``--pad`` setting.
    mov_img : Any
        Loaded moving/source image.

    Returns
    -------
    tuple[str, float | None]
        Padding mode for the shared mapping helpers and optional constant fill
        value.
    """
    if isinstance(pad, float):
        return "zeros", float(pad)
    if pad == "zero":
        return "zeros", None
    if pad == "brightest":
        image_max = _finite_max(np.asarray(mov_img.dataobj, dtype=np.float32))
        if image_max is None:
            raise ValueError("--pad brightest requires at least one finite source intensity.")
        return "zeros", image_max
    return pad, None


def _infer_target_max(target_dtype: np.dtype | None) -> float | None:
    """Infer a sensible zero-anchored target maximum from an output dtype.

    Parameters
    ----------
    target_dtype : np.dtype or None
        Requested output dtype.

    Returns
    -------
    float or None
        Inferred upper bound, or ``None`` when no obvious default exists.
    """
    if target_dtype is None:
        return None
    if np.issubdtype(target_dtype, np.bool_):
        return 1.0
    if np.issubdtype(target_dtype, np.integer):
        return float(np.iinfo(target_dtype).max)
    return None


def _finite_max(data: np.ndarray, *, positive_only: bool = False) -> float | None:
    """Return the maximum finite value, optionally restricted to positives.

    Parameters
    ----------
    data : np.ndarray
        Source values to inspect.
    positive_only : bool, default=False
        If ``True``, ignore zero and negative values in addition to non-finite
        samples.

    Returns
    -------
    float or None
        Maximum finite value satisfying the requested filter, or ``None`` when
        no such value exists.
    """
    values = np.asarray(data, dtype=np.float32)
    mask = np.isfinite(values)
    if positive_only:
        mask &= values > 0
    if not np.any(mask):
        return None
    return float(values[mask].max())


def _robust_upper_bound(data: np.ndarray, low: float, high: float) -> float:
    """Estimate a robust positive upper bound for zero-anchored rescaling.

    Parameters
    ----------
    data : np.ndarray
        Source image intensity data.
    low : float
        Lower quantile used to trim very dark positive intensities.
    high : float
        Upper quantile used to estimate the bright-end intensity ceiling.

    Returns
    -------
    float
        Robust upper bound. Returns ``0.0`` when the image has no positive
        finite intensities.
    """
    positive = np.asarray(data, dtype=np.float32)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if positive.size == 0:
        return 0.0
    if low > 0.0:
        lower = float(np.quantile(positive, low))
        positive = positive[positive >= lower]
        if positive.size == 0:
            return 0.0
    return float(np.quantile(positive, high))


def _convert_output_image(
        mapped_img: Any,
        source_img: Any,
        target_dtype: np.dtype | None,
        scale_mode: str | None,
        target_max: float | None,
        robust_low: float,
        robust_high: float,
) -> Any:
    """Apply final dtype conversion and optional intensity scaling.

    Parameters
    ----------
    mapped_img : Any
        Resampled image to convert.
    source_img : Any
        Original moving/source image. Its intensity distribution is used to
        derive scale factors for zero-anchored rescaling.
    target_dtype : np.dtype or None
        Requested output dtype. When ``None``, the mapped image is returned
        unchanged unless scaling is requested.
    scale_mode : {'clamp', 'rescale', 'robust'} or None
        Optional output intensity policy.
    target_max : float or None
        Requested zero-anchored output maximum.
    robust_low : float
        Lower robust quantile.
    robust_high : float
        Upper robust quantile.

    Returns
    -------
    Any
        Converted image instance matching the mapped image class.

    Raises
    ------
    ValueError
        If the scaling request is inconsistent with the available dtype or
        target-range information.
    """
    effective_mode = scale_mode
    if (
            effective_mode is None
            and target_dtype is not None
            and (np.issubdtype(target_dtype, np.bool_) or np.issubdtype(target_dtype, np.integer))
    ):
        effective_mode = "clamp"
    if effective_mode == "clamp" and target_dtype is None:
        raise ValueError("scale_mode='clamp' requires --out-dtype or --keep-dtype.")

    resolved_target_max = target_max if target_max is not None else _infer_target_max(target_dtype)
    if effective_mode in {"rescale", "robust"} and resolved_target_max is None:
        raise ValueError("rescale and robust scaling require --target-max or an integer/bool output dtype.")

    if effective_mode is None and target_dtype is None:
        return mapped_img

    working_dtype = np.float32 if effective_mode in {"rescale", "robust"} else np.float64
    mapped_np = np.asarray(mapped_img.dataobj, dtype=working_dtype)
    if effective_mode == "rescale":
        source_upper = _finite_max(np.asarray(source_img.dataobj, dtype=np.float32), positive_only=True)
        scale = 0.0 if source_upper is None or source_upper <= 0.0 else float(resolved_target_max) / source_upper
        mapped_np = np.clip(mapped_np * scale, 0.0, float(resolved_target_max))
    elif effective_mode == "robust":
        source_upper = _robust_upper_bound(np.asarray(source_img.dataobj, dtype=np.float32), robust_low, robust_high)
        scale = 0.0 if source_upper <= 0.0 else float(resolved_target_max) / source_upper
        mapped_np = np.clip(mapped_np * scale, 0.0, float(resolved_target_max))

    if target_dtype is None:
        return create_image_like(mapped_img, mapped_np.astype(np.float32, copy=False), np.asarray(mapped_img.affine))
    converted = clip_and_cast_dtype(mapped_np, target_dtype)
    return create_image_like(mapped_img, converted, np.asarray(mapped_img.affine, dtype=np.float64))


def main(args=None) -> None:
    """Entry point for the ``vol2vol`` command-line interface.

    Parameters
    ----------
    args : list of str or None, optional
        Command-line arguments. When ``None``, arguments are read from
        :data:`sys.argv`.

    Returns
    -------
    None
        This function is invoked for its side effects: loading images and
        transforms, mapping the image, and writing the output volume.

    Raises
    ------
    SystemExit
        If argument parsing fails, required geometry is unavailable, or image /
        transform loading or writing raises an exception.
    """
    parser = _build_parser()
    ns = parser.parse_args(args)
    validate_image_outputs(parser, ns, "out")
    _validate_args(ns, parser)

    level = logging.DEBUG if ns.debug else (logging.INFO if ns.verbose else logging.WARNING)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("neuroreg.cli.vol2vol")

    try:
        mov_img = load_image(ns.input_file)
        ref_img = load_image(ns.ref) if ns.ref is not None else None
        lta = (
            None
            if ns.transform is None
            else read_transform_as_lta(
                ns.transform,
                src_img=ns.input_file,
                dst_img=ns.ref,
                fmt=ns.transform_format,
            )
        )
        effective_lta = None if lta is None else (lta.invert() if ns.inverse else lta)
        r2r = np.eye(4, dtype=np.float64) if effective_lta is None else effective_lta.r2r()
        target_dtype = _resolve_target_dtype(ns, np.dtype(mov_img.get_data_dtype()))
        if target_dtype is not None:
            # --out-dtype/--keep-dtype name a type explicitly; refuse rather than
            # write a different one than was asked for.
            check_dtype_storable(target_dtype, ns.out)
        if ns.header_only:
            mapped_img = header_map_image(mov_img, r2r)
        elif ns.transform is None and ns.ref is None and ns.ref_cras is None:
            # No geometry change requested: read and write on the native grid
            # without resampling. This covers pure format conversion (chosen by
            # the --out extension), as well as dtype/scale-only operations.
            # Interpolation would only average voxels, so it is skipped here to
            # keep voxel values and dtype intact. --ref-cras is excluded because
            # it does request a different target grid, so it must resample: the
            # anatomy keeps its world position while the grid moves under it.
            mapped_img = mov_img
            mapped_img = _convert_output_image(
                mapped_img,
                mov_img,
                target_dtype=target_dtype,
                scale_mode=ns.scale_mode,
                target_max=ns.target_max,
                robust_low=ns.robust_low,
                robust_high=ns.robust_high,
            )
        else:
            target_affine, target_shape = _resolve_target_geometry(
                mov_img, ref_img, effective_lta, ref_cras=ns.ref_cras
            )
            padding_mode, padding_value = _resolve_padding(ns.pad, mov_img)
            mapped_img = reslice_r2r_image(
                mov_img,
                r2r,
                target_affine=target_affine,
                target_shape=target_shape,
                mode=ns.interp,
                padding_mode=padding_mode,
                padding_value=padding_value,
                keep_dtype=False,
            )
            mapped_img = _convert_output_image(
                mapped_img,
                mov_img,
                target_dtype=target_dtype,
                scale_mode=ns.scale_mode,
                target_max=ns.target_max,
                robust_low=ns.robust_low,
                robust_high=ns.robust_high,
            )
        save_image(mapped_img, ns.out)
    except Exception as exc:
        logger.debug("vol2vol failed", exc_info=True)
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Output: {ns.out}")


if __name__ == "__main__":
    main()
