"""Shared argument parsing helpers for the command-line entry points.

Several commands accept geometry components as comma-separated number lists
(``--ref-cras 0,0,0``, ``--vox-size 0.8``, ``--shape 320,320,320``). Both the
parsing and the argparse workaround needed to accept negative values live here
so the commands cannot drift apart. The same applies to the options that take a
whole grid from a file, which accept either an image or a geometry JSON.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


class NumberListParser(argparse.ArgumentParser):
    """Parser that reads a leading-minus number list as a value, not a flag.

    ``argparse`` treats any token starting with ``-`` as an option unless it
    matches its private negative-number pattern, which covers ``-4`` and
    ``-4.5`` but not lists such as ``-4.25,6.5,0``. Negative coordinates are
    the common case for a ``c_ras`` (a scanner centre is usually negative in at
    least one axis), and without this the natural ``--cras -4,0,0`` fails with
    "expected one argument" and forces users to discover the ``--cras=-4,0,0``
    spelling.

    Widening the pattern to "minus followed by a digit or a decimal point" is
    safe because every option these CLIs define is a ``--word``, so no flag can
    be shadowed. The check also runs only after argparse has failed to resolve
    the token as a known option.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # argparse assigns this in _ActionsContainer.__init__, so it is an
        # instance attribute and has to be replaced after the base class runs;
        # a class-level override would simply be overwritten.
        self._negative_number_matcher = re.compile(r"^-[\d.]")


def number_list(
        flag: str,
        *,
        length: int = 3,
        allow_scalar: bool = False,
        cast: Callable[[str], Any] = float,
        positive: bool = False,
) -> Callable[[str], np.ndarray]:
    """Build an argparse ``type`` that parses a comma-separated number list.

    Whitespace around the value and around each component is ignored. Values
    are rejected rather than silently coerced, since a malformed geometry
    component would otherwise produce a plausible-looking but wrong output
    grid.

    Parameters
    ----------
    flag : str
        Option name to quote in error messages, e.g. ``"--cras"``.
    length : int, default=3
        Number of components required.
    allow_scalar : bool, default=False
        If ``True``, a single value is accepted and broadcast to ``length``,
        so ``--vox-size 0.8`` means ``0.8,0.8,0.8``.
    cast : callable, default=float
        Per-component conversion, ``float`` or ``int``.
    positive : bool, default=False
        If ``True``, require every component to be strictly greater than zero.

    Returns
    -------
    callable
        Function suitable as an argparse ``type``, returning an array of
        ``length`` values.
    """
    counts = f"{1} or {length}" if allow_scalar else str(length)
    kind = "integers" if cast is int else "numbers"
    expected = f"{flag} must be {counts} comma-separated {kind}"

    def parse(value: str) -> np.ndarray:
        """Parse one CLI value into an array of numbers."""
        parts = [part.strip() for part in value.split(",")]
        if len(parts) not in ({1, length} if allow_scalar else {length}):
            raise argparse.ArgumentTypeError(f"{expected}, got {len(parts)}: {value!r}")
        try:
            parsed = [cast(part) for part in parts]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{expected}, got: {value!r}") from exc
        if len(parsed) == 1:
            parsed = parsed * length
        try:
            out = np.asarray(parsed, dtype=np.int64 if cast is int else np.float64)
        except OverflowError as exc:
            # Python integers are unbounded but the array dtype is not, and
            # argparse does not translate OverflowError into a usage error.
            raise argparse.ArgumentTypeError(f"{flag} is out of range, got: {value!r}") from exc
        if not np.all(np.isfinite(out.astype(np.float64))):
            # A non-finite component would poison the whole output affine silently.
            raise argparse.ArgumentTypeError(f"{flag} must be finite, got: {value!r}")
        if positive and np.any(out <= 0):
            raise argparse.ArgumentTypeError(f"{flag} must be positive, got: {value!r}")
        return out

    parse.__name__ = f"number_list{flag}"
    return parse


#: Suffix that selects the geometry JSON reader instead of the image reader.
GEOMETRY_JSON_SUFFIX = ".json"


def is_geometry_json(source: str | Path) -> bool:
    """Return whether a geometry option points at a geometry JSON rather than an image."""
    return Path(source).suffix.lower() == GEOMETRY_JSON_SUFFIX


def load_geometry_source(source: str | Path, *, flag: str) -> Any:
    """Load a target grid from either an image or a centroid target JSON.

    ``--ref``, ``--template-geom``, ``--src-img``, ``--dst-img`` and ``--like``
    read only the shape and affine of what they are given, so requiring an image
    forces callers to keep a file around purely for its header. The bundled
    centroid targets already serialize exactly those fields under ``geometry``,
    so the same grid can be named without one.

    The JSON case is returned as an image whose voxels are a zero-strided view
    rather than an allocation, because no caller reads them: this keeps every
    consumer on the one code path it already has.

    Parameters
    ----------
    source : str or Path
        Path to an image, or to a target JSON carrying a ``geometry`` block.
        A bundled atlas name is not accepted here; pass a path.
    flag : str
        Option name to quote in error messages, e.g. ``"--ref"``.

    Returns
    -------
    Any
        Loaded nibabel image, or a geometry-only image built from the JSON.

    Raises
    ------
    ValueError
        If the JSON carries no ``geometry`` block, and so names no grid.
    """
    # Imported lazily so the lighter CLIs do not pull in segreg at import time.
    import nibabel as nib

    from ..image import load_image
    from ..segreg.atlas import affine_from_header
    from ..segreg.io import read_target_json

    if not is_geometry_json(source):
        return load_image(source)

    geometry = read_target_json(source).geometry
    if geometry is None:
        raise ValueError(
            f"{flag} was given the target file '{source}', which carries centroids but no 'geometry' "
            "block, so it does not describe an output grid. Pass an image, or a target file with a "
            "geometry block such as the bundled atlas targets."
        )
    dims = tuple(int(v) for v in geometry["dims"])
    voxels = np.broadcast_to(np.uint8(0), dims)
    # No filename is set: nibabel validates it against the image type and would
    # reject a .json path. Consumers that record one fall back to empty, which
    # is what an LTA already carries for a geometry with no image behind it.
    return nib.MGHImage(voxels, affine_from_header(geometry))
