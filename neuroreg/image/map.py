"""Utilities for mapping (resampling) 3-D images via affine transforms."""

from typing import Any

import interpol
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

from ..transforms import matrices as trans

# FreeSurfer's MRItoBSpline/MRIsampleBSpline (used by ``mri_convert -rt cubic`` and
# ``mri_vol2vol --interp cubic``) prefilter the image into cubic B-spline coefficients
# and evaluate them with mirror ("DCT-I") boundary handling beyond the array edges.
# These settings were validated against locally built FreeSurfer binaries (both an
# axis-aligned resample and a full rigid rotation+translation) to reproduce
# ``mri_convert``/``mri_vol2vol`` cubic output to float32 precision (~1e-5 mean
# absolute difference) away from regions that fall entirely outside the source FOV.
_CUBIC_INTERPOL_KWARGS: dict[str, Any] = {
    "interpolation": 3,
    "bound": "dct1",
    "extrapolate": True,
    "prefilter": True,
}


def _normalize_interpolation_mode(mode: str) -> str:
    """Map public interpolation names to the backend names used by PyTorch."""
    if mode == "linear":
        return "bilinear"
    if mode == "nearest":
        return mode
    if mode == "cubic":
        return mode
    raise ValueError(f"mode must be 'linear', 'cubic', or 'nearest', got '{mode}'.")


def _grid_sample_grid_to_source_voxels(
    grid: torch.Tensor,
    source_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Convert an ``affine_grid`` (align_corners=False) grid to absolute source-voxel coordinates.

    Parameters
    ----------
    grid : torch.Tensor, shape (1, D, H, W, 3)
        Normalized sampling grid as produced by :func:`torch.nn.functional.affine_grid`,
        with the last dimension ordered ``(x, y, z)`` matching PyTorch's
        ``(W, H, D)`` axis convention.
    source_shape : tuple[int, int, int]
        Spatial shape ``(D, H, W)`` of the image being sampled from.

    Returns
    -------
    torch.Tensor, shape (1, D, H, W, 3)
        Absolute (unnormalized) source-voxel coordinates with the last dimension
        ordered ``(d, h, w)`` to match :func:`interpol.grid_pull`'s convention of
        indexing the grid's trailing dimension in the same order as the sampled
        tensor's spatial dimensions (no axis flip, unlike ``grid_sample``).
    """
    src_d, src_h, src_w = source_shape
    gx, gy, gz = grid[..., 0], grid[..., 1], grid[..., 2]
    vox_w = ((gx + 1) * src_w - 1) / 2
    vox_h = ((gy + 1) * src_h - 1) / 2
    vox_d = ((gz + 1) * src_d - 1) / 2
    return torch.stack([vox_d, vox_h, vox_w], dim=-1)


def _cubic_sample(
    input_image: torch.Tensor,
    grid: torch.Tensor,
    source_shape: tuple[int, int, int],
    padding_mode: str,
    padding_value_t: torch.Tensor | None,
) -> torch.Tensor:
    """Sample with FreeSurfer-matching cubic B-spline interpolation.

    The spline's own boundary handling (mirrored coefficients beyond the array
    edge) always follows FreeSurfer's convention, regardless of *padding_mode*.
    *padding_mode* only controls what is reported for sample positions that fall
    entirely outside the source volume:

    - ``"zeros"``: replaced with ``0`` (or *padding_value_t* if given).
    - ``"border"`` / ``"reflection"``: left as the mirror-extrapolated value,
      since the spline boundary condition already extends/reflects the image
      near the edge.
    """
    voxel_coords = _grid_sample_grid_to_source_voxels(grid, source_shape)
    sampled = interpol.grid_pull(input_image.double(), voxel_coords.double(), **_CUBIC_INTERPOL_KWARGS)
    sampled = sampled.to(input_image.dtype)
    if padding_mode == "zeros":
        # A small tolerance absorbs float32 round-trip noise in the affine_grid
        # normalize/denormalize conversion so exact-boundary samples (e.g. an
        # identity transform's edge voxels) are not spuriously treated as
        # out-of-bounds.
        eps = 1e-3
        src_d, src_h, src_w = source_shape
        vd, vh, vw = voxel_coords[..., 0], voxel_coords[..., 1], voxel_coords[..., 2]
        in_bounds = (
            (vd >= -eps)
            & (vd <= src_d - 1 + eps)
            & (vh >= -eps)
            & (vh <= src_h - 1 + eps)
            & (vw >= -eps)
            & (vw <= src_w - 1 + eps)
        ).unsqueeze(1)
        fill = (
            torch.zeros((), dtype=sampled.dtype, device=sampled.device) if padding_value_t is None else padding_value_t
        )
        sampled = torch.where(in_bounds, sampled, fill)
    return sampled


def coerce_image_data_3d(data: Any, *, name: str = "image") -> np.ndarray:
    """Return a 3-D image array, squeezing only singleton extra dimensions.

    Parameters
    ----------
    data : Any
        Array-like image payload.
    name : str, default="image"
        Human-readable label used in error messages.

    Returns
    -------
    np.ndarray
        A 3-D NumPy array view/copy of the input data.

    Raises
    ------
    ValueError
        If the input is not 3-D and cannot be reduced to 3-D by removing only
        singleton dimensions.
    """
    array = np.asarray(data)
    if array.ndim == 3:
        return array
    squeezed = np.squeeze(array)
    if squeezed.ndim == 3:
        return squeezed
    raise ValueError(
        f"{name} must be a 3-D volume or only have singleton extra dimensions; got shape {tuple(array.shape)}."
    )


def map(
    image: torch.Tensor,
    transform: torch.Tensor,
    is_torch_mat: bool = True,
    target_shape: tuple[int, int, int] | None = None,
    mode: str = "linear",
    padding_mode: str = "zeros",
    padding_value: float | None = None,
) -> torch.Tensor:
    """Map an input image to another space using the inverse transformation matrix.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor, shape ``(D, H, W)``.
    transform : torch.Tensor
        4 × 4 transformation matrix.
    is_torch_mat : bool, optional
        If ``True`` (default), *transform* is already in PyTorch grid-sample
        format (3 × 4 or 4 × 4).  If ``False``, it is treated as a vox-to-vox
        matrix and converted via :func:`~neuroreg.transforms.matrices.convert_v2v_to_torch`.
    target_shape : tuple[int, int, int], optional
        Shape of the output grid ``(D, H, W)``.  Only used when
        ``is_torch_mat=False``.  Defaults to the shape of *image*.
    mode : {'linear', 'cubic', 'nearest'}, optional
        Interpolation mode. ``'linear'`` and ``'nearest'`` are translated
        internally to PyTorch's ``grid_sample`` backend names (``'bilinear'``
        and ``'nearest'``). ``'cubic'`` uses a prefiltered cubic B-spline
        evaluated via :func:`interpol.grid_pull`, matching FreeSurfer's
        ``mri_convert -rt cubic`` / ``mri_vol2vol --interp cubic``
        (``MRItoBSpline`` + ``MRIsampleBSpline``). Default is ``'linear'``.
    padding_mode : {'zeros', 'border', 'reflection'}, optional
        Padding strategy for out-of-bounds coordinates. For ``'linear'`` and
        ``'nearest'`` this is passed directly to
        :func:`torch.nn.functional.grid_sample`. For ``'cubic'``, the spline's
        own boundary handling always mirrors values past the array edge
        (matching FreeSurfer); *padding_mode* only controls what is reported
        for samples that fall entirely outside the source volume: ``'zeros'``
        reports 0 (or *padding_value*) there, while ``'border'`` and
        ``'reflection'`` leave the mirror-extrapolated value in place.
        Default is ``'zeros'``.
    padding_value : float, optional
        Constant value used for out-of-bounds samples when ``padding_mode`` is
        ``"zeros"``. When omitted, PyTorch's standard zero padding is used.

    Returns
    -------
    torch.Tensor
        Resampled image with shape *target_shape* (or the shape of *image* if
        *target_shape* is ``None``).

    Raises
    ------
    ValueError
        If *mode* is not ``'linear'``, ``'cubic'``, or ``'nearest'``, or if
        *padding_mode* is not one of ``'zeros'``, ``'border'``, or
        ``'reflection'``. Also raised when *padding_value* is supplied with a
        non-``"zeros"`` padding mode.
    """
    torch_mode = _normalize_interpolation_mode(mode)
    if padding_mode not in ("zeros", "border", "reflection"):
        raise ValueError(f"padding_mode must be 'zeros', 'border', or 'reflection', got '{padding_mode}'.")
    if padding_value is not None and padding_mode != "zeros":
        raise ValueError("padding_value may only be used with padding_mode='zeros'.")
    if not is_torch_mat:
        torch_transform = trans.convert_v2v_to_torch(transform, image.shape, target_shape)
    else:
        torch_transform = transform[:3, :]
    torch_transform = torch_transform.to(device=image.device)
    out_shape = target_shape if target_shape is not None else image.shape
    grid_size = (1, 1) + tuple(out_shape)
    grid = nn.functional.affine_grid(torch_transform.unsqueeze(0).float(), grid_size, align_corners=False)
    input_image = image.unsqueeze(0).unsqueeze(0)
    padding_value_t = (
        None
        if padding_value is None
        else torch.as_tensor(padding_value, dtype=input_image.dtype, device=input_image.device)
    )
    if torch_mode == "cubic":
        source_shape = (int(image.shape[0]), int(image.shape[1]), int(image.shape[2]))
        return _cubic_sample(input_image, grid, source_shape, padding_mode, padding_value_t).squeeze(0).squeeze(0)
    if padding_value_t is None:
        return (
            nn.functional.grid_sample(
                input_image,
                grid,
                mode=torch_mode,
                padding_mode=padding_mode,
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )
    shifted = input_image - padding_value_t
    return (
        (
            nn.functional.grid_sample(
                shifted,
                grid,
                mode=torch_mode,
                padding_mode="zeros",
                align_corners=False,
            )
            + padding_value_t
        )
        .squeeze(0)
        .squeeze(0)
    )


def map_r2r(
    image: torch.Tensor,
    r2r: torch.Tensor,
    source_affine: torch.Tensor,
    target_affine: torch.Tensor,
    target_shape: tuple[int, int, int] | None = None,
    mode: str = "linear",
    padding_mode: str = "zeros",
    padding_value: float | None = None,
) -> torch.Tensor:
    """Map an image using a RAS-to-RAS transform without a v2v intermediate.

    Wrapper around :func:`map` that calls
    :func:`~neuroreg.transforms.matrices.convert_r2r_to_torch` to build the
    PyTorch grid transform directly from the physical-space (RAS) chain::

        trg_norm → trg_vox → trg_RAS → src_RAS → src_vox → src_norm

    This avoids creating the intermediate vox-to-vox matrix
    ``inv(target_affine) @ r2r @ source_affine``, whose off-diagonal elements
    reflect genuine anisotropy but look like shear and can mislead callers.

    Parameters
    ----------
    image : torch.Tensor
        Source image tensor, shape ``(D, H, W)``.
    r2r : torch.Tensor
        4 × 4 RAS-to-RAS transform (source_RAS → target_RAS).
    source_affine : torch.Tensor
        4 × 4 voxel-to-RAS affine of the source image.
    target_affine : torch.Tensor
        4 × 4 voxel-to-RAS affine of the target image.
    target_shape : tuple[int, int, int], optional
        Output shape ``(D, H, W)``.  Defaults to the shape of *image*.
    mode : {'linear', 'cubic', 'nearest'}, optional
        Interpolation mode. ``'linear'`` is translated internally to PyTorch's
        ``'bilinear'`` name. Default is ``'linear'``.
    padding_mode : {'zeros', 'border', 'reflection'}, optional
        Out-of-bounds padding.  Default is ``'zeros'``.
    padding_value : float, optional
        Constant value used for out-of-bounds samples when ``padding_mode`` is
        ``"zeros"``.

    Returns
    -------
    torch.Tensor
        Resampled image with shape *target_shape* (or source shape).
    """
    if target_shape is None:
        target_shape = image.shape
    torch_mat = trans.convert_r2r_to_torch(r2r, image.shape, source_affine, target_shape, target_affine)
    return map(
        image,
        torch_mat,
        is_torch_mat=True,
        target_shape=target_shape,
        mode=mode,
        padding_mode=padding_mode,
        padding_value=padding_value,
    )


def resample_isotropic(
    img: nib.Nifti1Image,
    iso: float,
    out_shape: tuple[int, int, int] | None = None,
    mode: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resample a NIfTI image to an isotropic grid.

    Creates an isotropic resampled version of the input image where all voxels
    have the same physical size (*iso* mm) in all three dimensions. The
    isotropic affine preserves the original image origin while rescaling the
    column vectors to have uniform length.

    This function is commonly used as a preprocessing step before multi-scale
    registration to ensure both images share a consistent voxel grid.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        Input image to resample.
    iso : float
        Target isotropic voxel size in millimeters.
    out_shape : tuple[int, int, int], optional
        Output image shape ``(D, H, W)``.  If ``None``, the output shape is
        computed automatically to cover the entire field of view of the
        original image at the specified isotropic resolution.
    mode : {'linear', 'cubic', 'nearest'}, optional
        Interpolation mode. ``'linear'`` is translated internally to PyTorch's
        ``'bilinear'`` name. Default is ``'linear'``.

    Returns
    -------
    data : torch.Tensor
        Resampled image data, shape *out_shape*.
    iso_affine : torch.Tensor, dtype float32
        Isotropic voxel-to-RAS affine (4 × 4).
    Rvox : torch.Tensor, dtype float32
        Voxel-to-voxel transform from the isotropic grid back to the original
        grid (4 × 4), computed as ``inv(orig_affine) @ iso_affine``.

    Notes
    -----
    The isotropic affine is constructed by normalizing each column of the
    original 3 × 3 rotation/scale block to unit length, then scaling by *iso*.
    The origin (fourth column) is preserved.

    Examples
    --------
    >>> img = nib.load("example.mgz")
    >>> data_iso, aff_iso, Rvox = resample_isotropic(img, iso=1.0)
    >>> print(data_iso.shape, aff_iso.shape)
    """
    orig_affine = torch.from_numpy(img.affine).double()

    # Build isotropic affine: same origin, isotropic voxels
    iso_affine = orig_affine.clone()
    for i in range(3):
        col_norm = orig_affine[:3, i].norm()
        if col_norm > 0:
            iso_affine[:3, i] = orig_affine[:3, i] / col_norm * iso

    # Compute output shape if not provided
    if out_shape is None:
        zooms = np.linalg.norm(img.affine[:3, :3], axis=0)  # column norms = voxel sizes
        shape = np.array(img.shape[:3])
        # eps=1e-4 matches FreeSurfer's findRightSize: prevents ceil from adding a phantom
        # voxel when (s * z / iso) is an integer but rounds up due to floating-point noise.
        out_shape = tuple(max(1, int(np.ceil(s * z / iso - 1e-4))) for s, z in zip(shape, zooms, strict=False))

    # Resample using identity RAS-to-RAS transform
    identity_r2r = torch.eye(4, dtype=torch.float64)
    orig_data = torch.from_numpy(coerce_image_data_3d(img.get_fdata(), name="resample_isotropic image")).float()

    resampled = map_r2r(
        orig_data,
        identity_r2r.float(),
        source_affine=orig_affine.float(),
        target_affine=iso_affine.float(),
        target_shape=out_shape,
        mode=mode,
    )

    # Rvox: isotropic vox → original vox
    Rvox = torch.inverse(orig_affine) @ iso_affine

    return resampled, iso_affine.float(), Rvox.float()


def resample_isotropic_tensor(
    img: torch.Tensor,
    affine: np.ndarray,
    iso: float,
    out_shape: tuple[int, int, int] | None = None,
    mode: str = "linear",
    padding_mode: str = "zeros",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resample a torch tensor to an isotropic grid.

    Like resample_isotropic but works with torch tensors instead of nibabel images.

    Parameters
    ----------
    img : torch.Tensor
        Input image tensor, shape (D, H, W).
    affine : np.ndarray
        4×4 voxel-to-RAS affine matrix.
    iso : float
        Target isotropic voxel size in millimeters.
    out_shape : tuple[int, int, int], optional
        Output image shape (D, H, W). If None, computed automatically.
    mode : {'linear', 'cubic', 'nearest'}, optional
        Interpolation mode. Default is ``'linear'``.

    Returns
    -------
    data : torch.Tensor
        Resampled image data.
    iso_affine : torch.Tensor
        Isotropic voxel-to-RAS affine (4×4).
    Rvox : torch.Tensor
        Voxel-to-voxel transform from isotropic grid to original grid (4×4).
    """
    orig_affine = torch.from_numpy(affine).double()

    # Build rotation/scale part: same direction cosines, isotropic voxel size
    iso_rot_scale = orig_affine.clone()
    for i in range(3):
        col_norm = orig_affine[:3, i].norm()
        if col_norm > 0:
            iso_rot_scale[:3, i] = orig_affine[:3, i] / col_norm * iso

    # Compute output shape if not provided
    if out_shape is None:
        zooms = np.linalg.norm(affine[:3, :3], axis=0)
        shape = np.array(img.shape[:3])
        # eps=1e-4 matches FreeSurfer's findRightSize: prevents ceil from adding a phantom
        # voxel when (s * z / iso) is an integer but rounds up due to floating-point noise.
        out_shape = tuple(max(1, int(np.ceil(s * z / iso - 1e-4))) for s, z in zip(shape, zooms, strict=False))

    # FreeSurfer's makeIsotropic preserves the image center (c_r, c_a, c_s), not voxel-0.
    # Match that: set origin so the center voxel maps to the same RAS position as before.
    orig_shape = img.shape[:3]
    orig_center = torch.tensor([orig_shape[0] // 2, orig_shape[1] // 2, orig_shape[2] // 2, 1], dtype=torch.float64)
    c_ras = orig_affine @ orig_center  # RAS of original center voxel

    iso_center = torch.tensor([out_shape[0] // 2, out_shape[1] // 2, out_shape[2] // 2], dtype=torch.float64)
    iso_affine = iso_rot_scale.clone()
    iso_affine[:3, 3] = c_ras[:3] - iso_rot_scale[:3, :3] @ iso_center

    # Resample using identity RAS-to-RAS transform
    identity_r2r = torch.eye(4, dtype=torch.float64)

    resampled = map_r2r(
        img,
        identity_r2r.float(),
        source_affine=orig_affine.float(),
        target_affine=iso_affine.float(),
        target_shape=out_shape,
        mode=mode,
        padding_mode=padding_mode,
    )

    # Rvox: isotropic vox → original vox
    Rvox = torch.inverse(orig_affine) @ iso_affine

    return resampled, iso_affine.float(), Rvox.float()


def create_image_like(template_img: Any, data: Any, affine: np.ndarray) -> Any:
    """Create a new image instance matching a template image class.

    Parameters
    ----------
    template_img : Any
        Nibabel-like image instance whose class and header template should be
        reused for the output image.
    data : Any
        Image payload to store in the output image. This may be a NumPy array
        or a nibabel array proxy when voxel samples should stay lazily loaded.
    affine : np.ndarray
        Output voxel-to-RAS affine.

    Returns
    -------
    Any
        New image instance of the same class as ``template_img``.
    """
    header = template_img.header.copy()
    if hasattr(header, "set_data_dtype"):
        data_dtype = np.dtype(getattr(data, "dtype", template_img.get_data_dtype()))
        header.set_data_dtype(data_dtype)
    return template_img.__class__(data, affine, header)


def header_map_image(image: Any, r2r: torch.Tensor | np.ndarray) -> Any:
    """Return a header-only mapped image using a RAS-to-RAS transform.

    This helper preserves the voxel samples exactly and updates only the output
    affine so the image occupies the transformed pose in RAS space.

    Parameters
    ----------
    image : Any
        Nibabel-like image object exposing ``dataobj``, ``affine``, and
        ``header``.
    r2r : torch.Tensor or np.ndarray, shape (4, 4)
        Source-to-target RAS transform.

    Returns
    -------
    Any
        New image instance with unchanged voxel samples and updated affine.
    """
    r2r_np = r2r.detach().cpu().numpy() if hasattr(r2r, "detach") else np.asarray(r2r, dtype=np.float64)
    affine = r2r_np @ np.asarray(image.affine, dtype=np.float64)
    return create_image_like(image, image.dataobj, affine)


def reslice_r2r_image(
    image: Any,
    r2r: torch.Tensor | np.ndarray,
    *,
    target_affine: np.ndarray,
    target_shape: tuple[int, int, int],
    mode: str = "linear",
    padding_mode: str = "zeros",
    padding_value: float | None = None,
    keep_dtype: bool = False,
) -> Any:
    """Reslice an image with a RAS-to-RAS transform into a target geometry.

    Parameters
    ----------
    image : Any
        Nibabel-like image object exposing ``get_fdata()``, ``affine``, and
        ``header``.
    r2r : torch.Tensor or np.ndarray, shape (4, 4)
        Source-to-target RAS transform.
    target_affine : np.ndarray, shape (4, 4)
        Voxel-to-RAS affine of the output grid.
    target_shape : tuple of int
        Spatial shape of the output grid.
    mode : {'linear', 'cubic', 'nearest'}, default='linear'
        Interpolation mode forwarded to :func:`map_r2r`. ``'linear'`` is
        translated internally to PyTorch's ``'bilinear'`` backend name.
    padding_mode : {'zeros', 'border', 'reflection'}, default='zeros'
        Out-of-bounds padding mode.
    padding_value : float, optional
        Constant out-of-bounds fill value used when ``padding_mode`` is
        ``"zeros"``.
    keep_dtype : bool, default=False
        If ``True``, cast the final mapped output back to the source image data
        type. This mirrors FreeSurfer's ``mri_vol2vol --keep-precision`` and
        can lose interpolation precision. Nearest-neighbor resampling of
        integer- or boolean-valued inputs always preserves the source dtype.
        Otherwise, when ``False``, the written mapped output uses ``float32``.

    Returns
    -------
    Any
        Resampled image in the requested target geometry.

    Raises
    ------
    ValueError
        If *mode* or *padding_mode* is invalid for :func:`map_r2r`.
    """
    image_data = np.asarray(image.get_fdata(), dtype=np.float32)
    r2r_t = torch.from_numpy(np.asarray(r2r, dtype=np.float64)).float() if not hasattr(r2r, "detach") else r2r.float()
    mapped = map_r2r(
        torch.from_numpy(image_data),
        r2r_t,
        source_affine=torch.from_numpy(np.asarray(image.affine, dtype=np.float64)).float(),
        target_affine=torch.from_numpy(np.asarray(target_affine, dtype=np.float64)).float(),
        target_shape=target_shape,
        mode=mode,
        padding_mode=padding_mode,
        padding_value=padding_value,
    ).detach()
    mapped_np = mapped.cpu().numpy()
    source_dtype = np.dtype(image.get_data_dtype())
    preserve_discrete_dtype = mode == "nearest" and (
        np.issubdtype(source_dtype, np.bool_) or np.issubdtype(source_dtype, np.integer)
    )
    if preserve_discrete_dtype or keep_dtype:
        if np.issubdtype(source_dtype, np.bool_):
            mapped_np = np.clip(np.rint(mapped_np), 0, 1).astype(source_dtype)
        elif np.issubdtype(source_dtype, np.integer):
            source_info = np.iinfo(source_dtype)
            mapped_np = np.clip(np.rint(mapped_np), source_info.min, source_info.max).astype(source_dtype)
        else:
            mapped_np = mapped_np.astype(source_dtype, copy=False)
    else:
        mapped_np = mapped_np.astype(np.float32, copy=False)
    return create_image_like(image, mapped_np, np.asarray(target_affine, dtype=np.float64))


def infer_image_reslice_mode(image: Any) -> str:
    """Infer a safe default interpolation mode for a nibabel-like image.

    Parameters
    ----------
    image : Any
        Nibabel-like image object exposing ``get_data_dtype()``.

    Returns
    -------
    str
        ``"nearest"`` for integer-valued images such as segmentations and
        labels, otherwise ``"cubic"`` for floating-point intensity images.
        This matches FreeSurfer's own default for final registered output
        (``mri_robust_register``'s ``finalsampletype`` and
        ``mri_robust_template``'s ``finalinterp`` both default to
        ``SAMPLE_CUBIC_BSPLINE``).
    """
    source_dtype = np.dtype(image.get_data_dtype())
    return "nearest" if np.issubdtype(source_dtype, np.integer) else "cubic"


def save_resliced_r2r_image(
    image: Any,
    r2r: torch.Tensor | np.ndarray,
    output_path: str,
    *,
    target_affine: np.ndarray,
    target_shape: tuple[int, int, int],
    mode: str | None = None,
    padding_mode: str = "zeros",
    padding_value: float | None = None,
    keep_dtype: bool = False,
) -> Any:
    """Reslice and write an image using a shared RAS-to-RAS mapping path.

    Parameters
    ----------
    image : Any
        Nibabel-like source image.
    r2r : torch.Tensor or np.ndarray, shape (4, 4)
        Source-to-target RAS transform.
    output_path : str
        Destination filename for the resliced output.
    target_affine : np.ndarray, shape (4, 4)
        Voxel-to-RAS affine of the output grid.
    target_shape : tuple of int
        Spatial shape of the output grid.
    mode : {'linear', 'cubic', 'nearest'} or None, optional
        Interpolation mode. When omitted, the mode is chosen automatically from
        the source image data type via :func:`infer_image_reslice_mode`.
    padding_mode : {'zeros', 'border', 'reflection'}, default='zeros'
        Out-of-bounds padding mode.
    padding_value : float, optional
        Constant out-of-bounds fill value used when ``padding_mode`` is
        ``"zeros"``.
    keep_dtype : bool, default=False
        If ``True``, cast the final written mapped image back to the source data
        type after interpolation. This matches FreeSurfer's
        ``mri_vol2vol --keep-precision`` behavior. Nearest-neighbor resampling
        of integer- or boolean-valued inputs always preserves the source dtype.
        Otherwise, when ``False``, the written mapped output uses ``float32``.

    Returns
    -------
    Any
        Written mapped image object.

    Raises
    ------
    ValueError
        If the resolved interpolation or padding mode is invalid for
        :func:`reslice_r2r_image`.
    """
    resolved_mode = infer_image_reslice_mode(image) if mode is None else mode
    mapped_img = reslice_r2r_image(
        image,
        r2r,
        target_affine=target_affine,
        target_shape=target_shape,
        mode=resolved_mode,
        padding_mode=padding_mode,
        padding_value=padding_value,
        keep_dtype=keep_dtype,
    )
    mapped_img.to_filename(output_path)
    return mapped_img


def save_header_mapped_image(
    image: Any,
    r2r: torch.Tensor | np.ndarray,
    output_path: str,
) -> Any:
    """Write a header-only mapped image using a shared helper.

    Parameters
    ----------
    image : Any
        Nibabel-like source image.
    r2r : torch.Tensor or np.ndarray, shape (4, 4)
        Source-to-target RAS transform.
    output_path : str
        Destination filename for the output image.

    Returns
    -------
    Any
        Written header-mapped image object.
    """
    mapped_img = header_map_image(image, r2r)
    mapped_img.to_filename(output_path)
    return mapped_img


def reslice_and_apply_mask(
    image: Any,
    mask: Any,
    *,
    threshold: float = 0.0,
    fill: float = 0.0,
) -> Any:
    """Apply a binary mask to an image, reslicing the mask onto the image grid.

    Voxels where the mask value is less than or equal to ``threshold`` are set
    to ``fill``. The mask is resampled with nearest-neighbor interpolation into
    the image geometry (out-of-bounds treated as outside the mask), so a mask
    given in a different geometry is handled like FreeSurfer's ``mri_mask``. When
    the mask already shares the image grid the nearest resample is exact.

    Parameters
    ----------
    image : Any
        Nibabel-like image to be masked. Its geometry defines the output grid.
    mask : Any
        Nibabel-like mask image.
    threshold : float, default=0.0
        Voxels with mask value strictly greater than this are kept.
    fill : float, default=0.0
        Value assigned to voxels outside the mask.

    Returns
    -------
    Any
        Masked image in the input image geometry (float32 payload).
    """
    target_affine = np.asarray(image.affine, dtype=np.float64)
    target_shape = tuple(int(v) for v in image.shape[:3])
    mask_resliced = reslice_r2r_image(
        mask,
        np.eye(4, dtype=np.float64),
        target_affine=target_affine,
        target_shape=target_shape,
        mode="nearest",
        padding_mode="zeros",
        padding_value=0.0,
    )
    keep = np.asarray(mask_resliced.dataobj) > threshold
    data = np.asarray(image.get_fdata(), dtype=np.float32).copy()
    data[~keep] = float(fill)
    return create_image_like(image, data, target_affine)


def mask_geometry_differs(mask: Any, target_affine: np.ndarray, target_shape: tuple[int, int, int]) -> bool:
    """Return ``True`` when a mask does not already share the target grid.

    Parameters
    ----------
    mask : Any
        Nibabel-like mask image.
    target_affine : np.ndarray, shape (4, 4)
        Voxel-to-RAS affine of the target grid.
    target_shape : tuple of int
        Spatial shape of the target grid.

    Returns
    -------
    bool
        ``True`` when the mask shape or affine differs from the target grid and
        a resample is therefore required.
    """
    same_shape = tuple(int(v) for v in mask.shape[:3]) == tuple(int(v) for v in target_shape)
    same_affine = np.allclose(np.asarray(mask.affine, dtype=np.float64), target_affine, atol=1e-4)
    return not (same_shape and same_affine)
