"""Command-line interface for FreeSurfer-style multi-timepoint robust registration."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, cast

from ..image import MGHTags, inherit_tags, load_geometry_source, load_image, save_image
from ..multireg import multireg
from ..transforms import LTA
from ._outputs import output_history, validate_image_outputs


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
            "A pre-alignment registers all time points to a deterministic initial target\n"
            "and constructs an unbiased mean space; a global loop then registers them to\n"
            "the template and refines it. With two time points the pre-alignment alone is\n"
            "the registration."
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
        "--mapmov",
        nargs="*",
        metavar="FILE",
        help="Optional output paths for mapped per-time-point images in template space, one per --mov image.",
    )
    p.add_argument(
        "--ixforms",
        nargs="*",
        metavar="FILE",
        help=(
            "Optional input LTAs, one per --mov image, for example poses to an atlas. The "
            "pre-alignment starts from them and keeps their pose, so the template lands in the "
            "space they define, on their shared destination grid (only its orientation if its "
            "voxel size differs from the inputs'). Add --noit to use them exactly as given, as in "
            "FreeSurfer, or --no-prealign to start the global loop from them directly."
        ),
    )
    p.add_argument(
        "--template-geom",
        metavar="FILE",
        help=(
            "Image or JSON with a geometry block supplying the output template geometry, replacing "
            "the geometry multireg would otherwise derive. Only the header is read. With "
            "--ixforms the destination geometry of the input transforms is then ignored and "
            "need not be present."
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
        help=(
            "Keep the pre-alignment target's space instead of constructing an unbiased mean space. "
            "With --ixforms the template sits where the target's transform puts it. Needs the "
            "pre-alignment, and cannot be combined with --template-geom, which supplies the "
            "template space itself."
        ),
    )
    p.add_argument(
        "--cras-center",
        action="store_true",
        help=(
            "Center the template geometry at the average CRAS instead of the average mapped "
            "centroid. Cannot be combined with --ixforms, --template-geom or --fixtp, none of "
            "which derives a geometry to center."
        ),
    )
    iter_group = p.add_mutually_exclusive_group()
    iter_group.add_argument(
        "--noit",
        action="store_true",
        help=(
            "Skip the global loop and build the template from the pre-alignment. With --ixforms "
            "there is no pre-alignment either, so the inputs are only mapped, as in FreeSurfer."
        ),
    )
    iter_group.add_argument(
        "--iterate",
        type=int,
        metavar="N",
        help=(
            "Maximum number of global loop iterations, 6 by default. Only for 3 or more time "
            "points: with 2 the pre-alignment is symmetric and is the registration."
        ),
    )
    iter_group.add_argument(
        "--map-only",
        action="store_true",
        help=(
            "Do not register: map each input with its --ixforms transform, or through its header "
            "without --ixforms, and build the template from them."
        ),
    )
    p.add_argument(
        "--no-prealign",
        dest="prealign",
        action="store_false",
        default=None,
        help=(
            "Skip the pre-alignment and register only in the global loop, starting from --ixforms "
            "or from the image headers. Avoids choosing a target time point. Needs 3 or more time "
            "points, since with 2 the pre-alignment is the registration; use --map-only there."
        ),
    )
    p.add_argument(
        "--allow-diff-vox-size",
        action="store_true",
        help=(
            "Accept inputs with different voxel sizes, with a warning. By default they are "
            "rejected: the template takes the coarsest size, so finer time points are downsampled."
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
        help=(
            "Write --mapmov outputs in each input's own dtype and the --template "
            "output in the initial target time point's dtype, instead of float32."
        ),
    )
    p.add_argument("--verbose", action="store_true", help="Enable INFO-level logging.")
    p.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging.")
    return p


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
        If argument parsing fails, an input image cannot be loaded, or the
        registration or any output write fails.
    """
    parser = _build_parser()
    ns = parser.parse_args(args)
    validate_image_outputs(parser, ns, "template", "mapmov")
    ns.symmetric = getattr(ns, "symmetric", True)
    if ns.mov_mask is not None and len(ns.mov_mask) != len(ns.mov):
        parser.error("--mov-mask requires exactly one mask per --mov input.")
    if ns.lta is not None and len(ns.lta) != len(ns.mov):
        parser.error("--lta requires exactly one output path per --mov input.")
    if ns.ixforms is not None and len(ns.ixforms) != len(ns.mov):
        parser.error("--ixforms requires exactly one input LTA per --mov input.")
    maps_only = ns.map_only or (ns.ixforms is not None and ns.noit and ns.prealign is None)
    if ns.map_only and ns.prealign is False:
        parser.error("--map-only already skips the pre-alignment; --no-prealign is redundant.")
    if ns.fixtp and (maps_only or ns.prealign is False):
        # --fixtp keeps the pre-alignment target's space, so there must be a
        # pre-alignment, and --ixforms --noit, like --map-only, has none.
        source = "--no-prealign" if ns.prealign is False else "--map-only" if ns.map_only else "--ixforms with --noit"
        parser.error(f"--fixtp keeps the space of the pre-alignment target, and {source} skips the pre-alignment.")
    if len(ns.mov) == 2 and ns.iterate:
        parser.error(
            "--iterate applies to 3 or more time points: with 2 the pre-alignment is symmetric and is the "
            "registration, so there is no global loop to iterate."
        )
    if len(ns.mov) == 2 and ns.prealign is False and not ns.noit:
        parser.error(
            "With 2 time points the pre-alignment is the registration, so --no-prealign leaves nothing to "
            "register. Use --map-only to only map the inputs."
        )
    if ns.template_geom is not None and ns.fixtp:
        # Same reasoning as above: --fixtp keeps the initial target time point's
        # grid, --template-geom supplies a grid outright.
        parser.error("--template-geom and --fixtp both determine the template space; pass only one.")
    if ns.cras_center and (ns.ixforms is not None or ns.template_geom is not None or ns.fixtp):
        # --cras-center only selects how a derived geometry is centered, and none
        # of these derives one: each supplies a grid, placement included.
        if ns.ixforms is not None:
            source = "--ixforms"
        elif ns.template_geom is not None:
            source = "--template-geom"
        else:
            source = "--fixtp"
        parser.error(f"--cras-center has no effect when {source} supplies the template space; pass only one.")
    if ns.mapmov is not None and len(ns.mapmov) != len(ns.mov):
        parser.error("--mapmov requires exactly one output path per --mov input.")
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
        template_geom_img = (
            None
            if ns.template_geom is None
            else cast(Any, load_geometry_source(ns.template_geom, label="--template-geom"))
        )
    except Exception as exc:
        print(f"ERROR loading image: {exc}", file=sys.stderr)
        sys.exit(1)
    logger.info("Starting multireg with %d time points.", len(mov_imgs))
    template_iterations = 0 if ns.noit or ns.map_only else ns.iterate
    try:
        result = multireg(
            mov_imgs,
            masks=mov_masks,
            init_ltas=ns.ixforms,
            template_geometry=template_geom_img,
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
            prealign=False if ns.map_only else ns.prealign,
            allow_diff_vox_size=ns.allow_diff_vox_size,
            return_mapped=ns.mapmov is not None,
            mapped_keep_dtype=ns.keep_dtype,
            verbose=ns.verbose or ns.debug,
        )
        history = output_history("multireg", args)
        # The template's values are new, built from all time points, so it
        # inherits nothing; it only records how it was made.
        save_image(result.template_image, ns.template, tags=MGHTags().with_history(history))
        print(f"InitialTP:   {result.initial_target_index + 1}")
        print(f"Seed:        {result.seed}")
        print(f"Iterations:  {result.template_iterations_run}")
        if result.iteration_distances:
            print(f"LastChange:  {result.iteration_distances[-1]:.6f}")
        print(f"Template:    {ns.template}")
        if ns.lta is not None:
            for lta_path, mov_path, mov_img, matrix in zip(
                ns.lta, ns.mov, mov_imgs, result.transforms_r2r, strict=False
            ):
                LTA.from_matrix(
                    matrix, mov_path, mov_img, ns.template, result.template_image, lta_type=1
                ).write(lta_path)
            print(f"LTAs:        {len(ns.lta)}")
        if ns.mapmov is not None:
            mapped_images = result.mapped_images if result.mapped_images is not None else []
            for mapmov_path, mov_path, mapped_image, matrix in zip(
                ns.mapmov, ns.mov, mapped_images, result.transforms_r2r, strict=False
            ):
                # A mapped time point keeps its own values, so it inherits from
                # its own input, moved into template space by its transform.
                tags = inherit_tags(mov_path, mapmov_path, r2r=matrix).with_history(history)
                save_image(mapped_image, mapmov_path, tags=tags)
            print(f"MapMov:      {len(ns.mapmov)}")
    except Exception as exc:
        # Registration and output writing report like the other commands rather
        # than as a traceback; --debug still shows the full stack.
        logger.debug("multireg failed", exc_info=True)
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
