"""Shared output-path validation for the command-line entry points."""

from __future__ import annotations

import argparse


def validate_image_outputs(parser: argparse.ArgumentParser, ns: argparse.Namespace, *dests: str) -> None:
    """Reject requested image outputs whose extension names no writable format.

    Output formats are selected by file extension, so a missing or unrecognized
    one (``mapped``, ``mapped.txt``) has no sensible interpretation. Checking
    before any work starts turns a typo into an immediate usage error instead
    of a failure after a registration has already run.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser used to report the error.
    ns : argparse.Namespace
        Parsed arguments.
    *dests : str
        Namespace attributes holding output image paths. Each may be ``None``,
        absent, a single path, or a list of paths. Absent and ``None`` entries
        are skipped, so optional flags and subcommand-specific ones are safe to
        pass unconditionally.

    Returns
    -------
    None
        Returns if every requested output path is usable.

    Raises
    ------
    SystemExit
        Via :meth:`argparse.ArgumentParser.error` if a path is unusable.
    """
    from ..image import IMAGE_SUFFIXES, recognized_image_suffix

    for dest in dests:
        value = getattr(ns, dest, None)
        if value is None:
            continue
        paths = value if isinstance(value, (list, tuple)) else [value]
        for path in paths:
            if recognized_image_suffix(path) is None:
                parser.error(
                    f"--{dest.replace('_', '-')} needs a recognized image extension "
                    f"({', '.join(IMAGE_SUFFIXES)}), got: {path}"
                )
