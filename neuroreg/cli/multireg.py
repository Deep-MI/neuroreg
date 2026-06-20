"""Command-line interface for FreeSurfer-style multi-timepoint robust registration."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np

from ..image import load_image
from ..multireg import multireg
from ..transforms import LTA


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for ``multireg``.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with the supported ``multireg`` command-line options.
    """
    p = argparse.ArgumentParser(
        prog="multireg",
        description=(
            "FreeSurfer-style multi-timepoint robust registration.\n"
            "Registers all time points to a deterministic initial target,\n"
            "constructs an unbiased mean space, and iteratively refines the template."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mov", nargs="+", required=True, metavar="FILE", help="Input time-point images (NIfTI or MGZ).")
    p.add_argument(
        "--mov-mask",
        nargs="*",
        metavar="FILE",
        help="Optional per-time-point masks. When given, provide one mask per --mov image.",
    )
    p.add_argument("--template", required=True, metavar="FILE", help="Output template image.")
    p.add_argument(
        "--lta",
        nargs="*",
        metavar="FILE",
        help="Optional output LTAs. When given, provide one output path per --mov image.",
    )
    p.add_argument(
        "--mapmov-dir",
        metavar="DIR",
        help="Optional directory for writing mapped per-time-point images in template space.",
    )
    p.add_argument(
        "--ixforms",
        nargs="*",
        metavar="FILE",
        help=(
            "Optional input LTAs, one per --mov image. When given, reuse these "
            "transforms as the template-space mapping."
        ),
    )
    p.add_argument(
        "--average",
        default="median",
        metavar="MODE",
        help="Template aggregation mode: mean, median, 0 (=mean), or 1 (=median).",
    )
    p.add_argument(
        "--inittp",
        type=int,
        metavar="N",
        help=(
            "1-based initial target time point. When omitted, choose a deterministic "
            "pseudo-random target from the inputs."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="INT",
        help="Seed used for the deterministic pseudo-random initial target selection. 0 recomputes it from the inputs.",
    )
    p.add_argument(
        "--fixtp",
        action="store_true",
        help="Keep the chosen initial target as the output space instead of constructing an unbiased mean space.",
    )
    p.add_argument(
        "--cras-center",
        action="store_true",
        help="Center the template geometry at the average CRAS instead of the average mapped centroid.",
    )
    iter_group = p.add_mutually_exclusive_group()
    iter_group.add_argument(
        "--noit",
        action="store_true",
        help="Stop after the initial mean-space pass instead of iteratively refining the template.",
    )
    iter_group.add_argument(
        "--iterate",
        type=int,
        metavar="N",
        help=(
            "Maximum number of template-refinement iterations. Defaults to 6 for 3+ "
            "time points and 0 for 2 time points."
        ),
    )
    p.add_argument(
        "--template-eps",
        type=float,
        default=0.03,
        metavar="FLOAT",
        help="Stop template refinement when the maximum transform change falls below this threshold.",
    )
    p.add_argument(
        "--nmax",
        type=int,
        default=5,
        metavar="N",
        help="Maximum number of outer IRLS iterations per pairwise registration pyramid level.",
    )
    p.add_argument(
        "--sat",
        type=float,
        default=6.0,
        metavar="FLOAT",
        help="Tukey biweight saturation threshold for the pairwise robust registrations.",
    )
    p.add_argument(
        "--nosym",
        dest="symmetric",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Disable symmetric halfway-space pairwise registration.",
    )
    init_group = p.add_mutually_exclusive_group()
    init_group.add_argument(
        "--init-header",
        dest="init_type",
        action="store_const",
        const="header",
        help="Use header alignment only for the pairwise registrations.",
    )
    init_group.add_argument(
        "--init-centroid",
        dest="init_type",
        action="store_const",
        const="centroid",
        help="Initialize the pairwise registrations by aligning intensity centroids in RAS.",
    )
    init_group.add_argument(
        "--init-center",
        dest="init_type",
        action="store_const",
        const="image_center",
        help="Initialize the pairwise registrations by aligning geometric image centers in RAS.",
    )
    p.add_argument(
        "--device",
        default="gpu",
        metavar="DEVICE",
        help="Torch device string, e.g. 'cpu', 'cuda', 'mps', or 'gpu'.",
    )
    p.add_argument(
        "--keep-dtype",
        action="store_true",
        help="Write --mapmov-dir outputs in the source-image dtype instead of float32.",
    )
    p.add_argument("--verbose", action="store_true", help="Enable INFO-level logging.")
    p.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging.")
    return p


def _mapped_output_path(directory: Path, input_path: str, index: int) -> Path:
    """Resolve the output path for a mapped time-point image.

    Parameters
    ----------
    directory : pathlib.Path
        Output directory for mapped images.
    input_path : str
        Original input filename used to preserve the basename when possible.
    index : int
        Zero-based time-point index used for fallback naming.

    Returns
    -------
    pathlib.Path
        Output filename for the mapped image.
    """
    name = Path(input_path).name
    if name:
        return directory / name
    return directory / f"tp{index + 1}.nii.gz"


def _save_image(image: Any, output_path: str | Path) -> None:
    """Save an image while honoring the requested output format.

    Parameters
    ----------
    image : Any
        Image-like object returned by ``multireg``.
    output_path : str or pathlib.Path
        Destination filename.

    Returns
    -------
    None
        This function returns ``None`` after writing the image.
    """
    path = Path(output_path)
    if path.suffix.lower() not in {".mgz", ".mgh"}:
        image.to_filename(path)
        return

    if isinstance(image, nib.MGHImage):
        image.to_filename(path)
        return

    data = np.asanyarray(image.dataobj)
    if data.dtype not in {np.dtype(np.uint8), np.dtype(np.int16), np.dtype(np.int32), np.dtype(np.float32)}:
        data = data.astype(np.float32, copy=False)
    nib.MGHImage(data, np.asarray(image.affine, dtype=np.float64)).to_filename(path)


def main(args=None) -> None:
    """Run the ``multireg`` command-line interface.

    Parameters
    ----------
    args : sequence of str or None, optional
        Explicit argument list. When ``None``, parse arguments from
        ``sys.argv``.

    Returns
    -------
    None
        This function returns ``None`` after writing requested outputs.

    Raises
    ------
    SystemExit
        If argument parsing fails or an input image cannot be loaded.
    """
    parser = _build_parser()
    ns = parser.parse_args(args)
    ns.symmetric = getattr(ns, "symmetric", True)
    if ns.mov_mask is not None and len(ns.mov_mask) != len(ns.mov):
        parser.error("--mov-mask requires exactly one mask per --mov input.")
    if ns.lta is not None and len(ns.lta) != len(ns.mov):
        parser.error("--lta requires exactly one output path per --mov input.")
    if ns.ixforms is not None and len(ns.ixforms) != len(ns.mov):
        parser.error("--ixforms requires exactly one input LTA per --mov input.")
    if ns.inittp is not None and not 1 <= ns.inittp <= len(ns.mov):
        parser.error(f"--inittp must be in [1, {len(ns.mov)}].")
    level = logging.DEBUG if ns.debug else (logging.INFO if ns.verbose else logging.WARNING)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("neuroreg.cli.multireg")
    try:
        mov_imgs = [cast(Any, load_image(path)) for path in ns.mov]
        mov_masks = None
        if ns.mov_mask is not None:
            mov_masks = [cast(Any, load_image(path)) for path in ns.mov_mask]
    except Exception as exc:
        print(f"ERROR loading image: {exc}", file=sys.stderr)
        sys.exit(1)
    logger.info("Starting multireg with %d time points.", len(mov_imgs))
    template_iterations = 0 if ns.noit else ns.iterate
    result = multireg(
        mov_imgs,
        masks=mov_masks,
        init_ltas=ns.ixforms,
        average=ns.average,
        init_target_index=None if ns.inittp is None else ns.inittp - 1,
        seed=ns.seed,
        fix_target=ns.fixtp,
        init_type=ns.init_type,
        nmax=ns.nmax,
        sat=ns.sat,
        symmetric=ns.symmetric,
        device=ns.device,
        use_cras_center=ns.cras_center,
        template_iterations=template_iterations,
        template_eps=ns.template_eps,
        return_mapped=ns.mapmov_dir is not None,
        mapped_keep_dtype=ns.keep_dtype,
        verbose=ns.verbose or ns.debug,
    )
    _save_image(result.template_image, ns.template)
    print(f"InitialTP:   {result.initial_target_index + 1}")
    print(f"Seed:        {result.seed}")
    print(f"Iterations:  {result.template_iterations_run}")
    if result.iteration_distances:
        print(f"LastChange:  {result.iteration_distances[-1]:.6f}")
    print(f"Template:    {ns.template}")
    if ns.lta is not None:
        for lta_path, mov_path, mov_img, matrix in zip(ns.lta, ns.mov, mov_imgs, result.transforms_r2r, strict=False):
            LTA.from_matrix(matrix, mov_path, mov_img, ns.template, result.template_image, lta_type=1).write(lta_path)
        print(f"LTAs:        {len(ns.lta)}")
    if ns.mapmov_dir is not None:
        mapmov_dir = Path(ns.mapmov_dir)
        mapmov_dir.mkdir(parents=True, exist_ok=True)
        mapped_images = result.mapped_images if result.mapped_images is not None else []
        for index, (mov_path, mapped_image) in enumerate(zip(ns.mov, mapped_images, strict=False)):
            _save_image(mapped_image, _mapped_output_path(mapmov_dir, mov_path, index))
        print(f"MapMovDir:   {mapmov_dir}")


if __name__ == "__main__":
    main()
