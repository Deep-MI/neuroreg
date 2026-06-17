"""Template construction helpers for multi-timepoint registration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..image import create_image_like, reslice_r2r_image
from ..transforms import LTA


def resolve_average_mode(average: str | int) -> str:
    """Normalize user-facing average mode aliases to internal names.

    Parameters
    ----------
    average : {"mean", "median", 0, 1}
        User-facing aggregation mode.

    Returns
    -------
    str
        Canonical internal mode name, either ``"mean"`` or ``"median"``.

    Raises
    ------
    ValueError
        If ``average`` is not a supported alias.
    """
    if isinstance(average, str):
        normalized = average.strip().lower()
    else:
        normalized = str(int(average))
    if normalized in {"mean", "0"}:
        return "mean"
    if normalized in {"median", "1"}:
        return "median"
    raise ValueError("average must be 'mean', 'median', 0, or 1.")


def build_template(
    images: Sequence[Any],
    transforms_r2r: Sequence[np.ndarray],
    *,
    average: str,
    template_affine: np.ndarray,
    template_shape: tuple[int, int, int],
    template_like: Any,
    return_mapped: bool,
    mapped_keep_dtype: bool,
) -> tuple[Any, list[Any] | None]:
    """Map all images into template space and aggregate them into a template.

    Parameters
    ----------
    images : sequence of Any
        Input images to resample into template space.
    transforms_r2r : sequence of numpy.ndarray
        Timepoint-to-template RAS transforms.
    average : {"mean", "median"}
        Canonical aggregation mode.
    template_affine : numpy.ndarray
        Output template voxel-to-RAS affine.
    template_shape : tuple of int
        Output template grid shape.
    template_like : Any
        Image-like object used to construct the output template image.
    return_mapped : bool
        If ``True``, return per-timepoint mapped images alongside the template.
    mapped_keep_dtype : bool
        If ``True``, preserve source dtypes for the returned mapped images.

    Returns
    -------
    template_image : Any
        Aggregated template image.
    mapped_images : list of Any or None
        Mapped per-timepoint images when requested, otherwise ``None``.

    Raises
    ------
    RuntimeError
        If no mapped images were produced.
    ValueError
        If ``average`` is not one of the supported canonical modes.
    """
    mapped_data_stack: list[np.ndarray] = []
    mapped_images = [] if return_mapped else None
    for image, matrix in zip(images, transforms_r2r, strict=False):
        mapped_float = reslice_r2r_image(
            image,
            matrix,
            target_affine=template_affine,
            target_shape=template_shape,
            mode="cubic",
            keep_dtype=False,
        )
        mapped_data = np.asarray(mapped_float.dataobj, dtype=np.float32)
        mapped_data_stack.append(mapped_data)
        if return_mapped:
            if mapped_keep_dtype:
                mapped_images.append(
                    reslice_r2r_image(
                        image,
                        matrix,
                        target_affine=template_affine,
                        target_shape=template_shape,
                        mode="cubic",
                        keep_dtype=True,
                    )
                )
            else:
                mapped_images.append(mapped_float)
    if not mapped_data_stack:
        raise RuntimeError("multireg did not produce any mapped images.")
    stacked = np.stack(mapped_data_stack, axis=0)
    if average == "mean":
        template_data = np.mean(stacked, axis=0, dtype=np.float32)
    elif average == "median":
        template_data = np.median(stacked, axis=0).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported average mode: {average!r}")
    return create_image_like(template_like, template_data, template_affine), mapped_images


def build_ltas(
    transforms_r2r: Sequence[np.ndarray],
    source_names: Sequence[str],
    images: Sequence[Any],
    template_image: Any,
) -> list[LTA]:
    """Construct LTAs for the final timepoint-to-template mappings.

    Parameters
    ----------
    transforms_r2r : sequence of numpy.ndarray
        Final timepoint-to-template RAS transforms.
    source_names : sequence of str
        Source filenames to embed in the generated LTAs.
    images : sequence of Any
        Source images corresponding to ``transforms_r2r``.
    template_image : Any
        Final template image used as the LTA destination geometry.

    Returns
    -------
    list of LTA
        Generated LTA objects for each time point.
    """
    return [
        LTA.from_matrix(matrix, src_name, image, "", template_image, lta_type=1)
        for matrix, src_name, image in zip(transforms_r2r, source_names, images, strict=False)
    ]
