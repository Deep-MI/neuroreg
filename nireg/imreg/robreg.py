"""Public IRLS-backed robust image-registration API."""

from __future__ import annotations

from typing import Any, cast
from pathlib import Path

import nibabel as nib
import torch
from torch import Tensor

from ..imreg.irls import register_irls_pyramid


ImageLike = str | Path | Any | Tensor


def _as_tensor_and_affine(
    image: ImageLike,
    affine: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if isinstance(image, (str, Path)):
        img = cast(Any, nib.load(str(image)))
        return torch.from_numpy(img.get_fdata()).float(), torch.from_numpy(img.affine).float()

    if hasattr(image, "get_fdata") and hasattr(image, "affine"):
        return torch.from_numpy(image.get_fdata()).float(), torch.from_numpy(image.affine).float()

    if isinstance(image, torch.Tensor):
        return image.float(), (affine.float() if affine is not None else torch.eye(4, dtype=torch.float32))

    raise TypeError(f"Unsupported image type: {type(image)!r}")


def register_pyramid(
    src: ImageLike,
    trg: ImageLike,
    *,
    src_affine: Tensor | None = None,
    trg_affine: Tensor | None = None,
    return_v2v: bool = False,
    centroid_init: bool = True,
    dof: int = 6,
    nmax: int = 5,
    sat: float = 6.0,
    symmetric: bool = False,
    isotropic: bool = True,
    adaptive_sat: bool = False,
    target_outlier_pct: float = 5.0,
    outliers_name: str | None = None,
    verbose: bool = False,
    device: str = "cpu",
) -> Tensor:
    """Register two images with the IRLS robust-registration path.

    Parameters are intentionally close to :func:`nireg.imreg.irls.register_irls_pyramid`,
    but this wrapper also accepts filenames and nibabel images.
    """
    if dof != 6:
        raise ValueError("IRLS robreg currently supports rigid registration only (dof=6).")

    src_data, src_aff = _as_tensor_and_affine(src, src_affine)
    trg_data, trg_aff = _as_tensor_and_affine(trg, trg_affine)

    src_data = src_data.to(device)
    trg_data = trg_data.to(device)
    src_aff = src_aff.to(device)
    trg_aff = trg_aff.to(device)

    T_v2v, _ = register_irls_pyramid(
        src=src_data,
        trg=trg_data,
        src_affine=src_aff,
        trg_affine=trg_aff,
        centroid_init=centroid_init,
        nmax=nmax,
        sat=sat,
        symmetric=symmetric,
        isotropic=isotropic,
        adaptive_sat=adaptive_sat,
        target_outlier_pct=target_outlier_pct,
        outliers_name=outliers_name,
        verbose=verbose,
    )

    if return_v2v:
        return T_v2v

    return trg_aff.double() @ T_v2v.double() @ torch.inverse(src_aff.double())


def register(*args, **kwargs) -> Tensor:
    """Alias for :func:`register_pyramid` for the public robreg path."""
    return register_pyramid(*args, **kwargs)


def register_sym(*args, **kwargs) -> Tensor:
    """Convenience wrapper for symmetric IRLS-backed robust registration."""
    kwargs["symmetric"] = True
    return register_pyramid(*args, **kwargs)


__all__ = ["register", "register_pyramid", "register_sym"]


