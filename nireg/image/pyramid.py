import torch

_PYRAMID_FILTER = torch.tensor([0.0625, 0.25, 0.375, 0.25, 0.0625])
"""FreeSurfer's saved-multiresolution smoothing kernel (Registration::buildGPLimits)."""


def _conv1d_along(
    vol: torch.Tensor,
    kernel: torch.Tensor,
    dim: int,
    padding_mode: str = "replicate",
) -> torch.Tensor:
    """Convolve a 3-D volume with a 1-D kernel along one spatial dimension."""
    K = kernel.shape[0]
    pad = K // 2
    k = kernel.to(dtype=vol.dtype, device=vol.device)

    if vol.shape[dim] == 1:
        return vol.clone()

    x = vol.unsqueeze(0).unsqueeze(0)
    if dim == 0:
        w = k.view(1, 1, K, 1, 1)
        pads = (0, 0, 0, 0, pad, pad)
    elif dim == 1:
        w = k.view(1, 1, 1, K, 1)
        pads = (0, 0, pad, pad, 0, 0)
    else:
        w = k.view(1, 1, 1, 1, K)
        pads = (pad, pad, 0, 0, 0, 0)

    padded = torch.nn.functional.pad(x, pads, mode=padding_mode)
    return torch.nn.functional.conv3d(padded, w).squeeze(0).squeeze(0)


def _smooth3d(vol: torch.Tensor, kernel: torch.Tensor, padding_mode: str = "replicate") -> torch.Tensor:
    """Apply a separable 1-D kernel along all three spatial axes."""
    return _conv1d_along(
        _conv1d_along(_conv1d_along(vol, kernel, 2, padding_mode), kernel, 1, padding_mode),
        kernel,
        0,
        padding_mode,
    )


def _downsample2_trilinear(vol: torch.Tensor) -> torch.Tensor:
    """Downsample a volume by approximately 2x using trilinear interpolation.

    Notes
    -----
    FreeSurfer's multiresolution pyramid uses ``MRIdownsample2BSpline``;
    we instead downsample the already smoothed image via trilinear interpolation.
    """
    D, H, W = vol.shape
    out_shape = (
        max(1, D // 2) if D > 1 else 1,
        max(1, H // 2) if H > 1 else 1,
        max(1, W // 2) if W > 1 else 1,
    )
    return torch.nn.functional.interpolate(
        vol.unsqueeze(0).unsqueeze(0),
        size=out_shape,
        mode="trilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)


def _downsample_affine(
    affine: torch.Tensor,
    in_shape: torch.Size,
    out_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Update a voxel-to-RAS affine after trilinear resizing to ``out_shape``."""
    out_affine = affine.clone()
    downM = torch.eye(4, dtype=affine.dtype, device=affine.device)
    for axis, (n_in, n_out) in enumerate(zip(in_shape[:3], out_shape, strict=True)):
        if n_in <= 1:
            scale = 1.0
            offset = 0.0
        else:
            scale = float(n_in) / float(n_out)
            offset = 0.5 * scale - 0.5
        downM[axis, axis] = scale
        downM[axis, 3] = offset
    out_affine = out_affine @ downM
    return out_affine


def get_pyramid_limits(
    shape1: torch.Size, shape2: torch.Size | None = None, minsize: int = 32, maxsize: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the minimum and maximum levels for a pyramid representation of shapes.

    Parameters
    ----------
    shape1 : torch.Size
        The primary shape of an input, typically a tensor size.
    shape2 : torch.Size, optional
        A secondary shape to compare with `shape1` for determining the limits.
        If `shape2` is not provided, `shape1` will be used for calculations.
    minsize : int, optional
        The minimum allowed size for the smallest dimension in the pyramid. Default is 32.
    maxsize : int, optional
        The maximum allowed size for the largest dimension in the finest pyramid
        level to retain. If not provided, the original/full-resolution level is
        kept. Default is None.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
          - `minsteps` (torch.Tensor): The minimum number of steps the pyramid can take before
            exceeding the `maxsize` (if given).
          - `maxsteps` (torch.Tensor): The maximum number of steps the pyramid can take before
            hitting the `minsize`.

    Example
    -------
    >>> shape = torch.Size([256, 256, 256])
    >>> limits = get_pyramid_limits(shape, shape2=shape, minsize=16, maxsize=200)
    >>> print(limits)
    (tensor(1), tensor(4))
    """
    dims1 = tuple(int(v) for v in shape1[:3])
    dims2 = dims1 if shape2 is None else tuple(int(v) for v in shape2[:3])

    smallest = min(dims1[0], dims2[0], dims1[1], dims2[1])
    if dims1[2] != 1 or dims2[2] != 1:
        smallest = min(smallest, dims1[2], dims2[2])
    if smallest < minsize:
        raise ValueError(f"Input image is smaller than minsize={minsize}: smallest dimension is {smallest}")

    temp = smallest // 2
    maxsteps = 0
    while temp >= minsize:
        maxsteps += 1
        temp //= 2

    if maxsize is None:
        return torch.tensor(0), torch.tensor(maxsteps)

    common_dims = (min(dims1[0], dims2[0]), min(dims1[1], dims2[1]), min(dims1[2], dims2[2]))
    temp = max(common_dims)
    minsteps = 0
    while minsteps < maxsteps:
        if temp <= maxsize:
            break
        temp //= 2
        minsteps += 1

    return torch.tensor(minsteps), torch.tensor(maxsteps)


def build_gaussian_pyramid(
    image: torch.Tensor,
    affine: torch.Tensor,
    limits: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Build a Gaussian pyramid for a 3D image, including its downsampled versions.

    The smoothing step uses the same 5-tap kernel FreeSurfer uses for its
    multiresolution pyramid. FreeSurfer then downsamples with
    ``MRIdownsample2BSpline``; we instead apply trilinear downsampling to
    the smoothed image.

    Parameters
    ----------
    image : torch.Tensor
        A 3D or higher-dimensional tensor representing the input image to build the pyramid from.
    affine : torch.Tensor
        The affine (vox2ras) transformation matrix (4x4) corresponding to the input image.
        It is updated for each downsampled image in the pyramid.
    limits : tuple[torch.Tensor, torch.Tensor], optional
        A tuple containing the minimum and maximum level indices for the pyramid.
        If not provided, they are computed using the `get_pyramid_limits` function based on the image size.

    Returns
    -------
    tuple[list[torch.Tensor], list[torch.Tensor]]
        A tuple containing:
          - `imgs` (list[torch.Tensor]): A list of downsampled and smoothed image tensors, starting
            from the input image (if applicable).
          - `affines` (list[torch.Tensor]): A list of affine matrices corresponding to each
            downsampled image.

    Example
    -------
    >>> image = torch.rand(256, 256, 256)  # Example 3D image
    >>> affine = torch.eye(4)  # Initial affine matrix
    >>> gaussian_pyramid, affine_pyramid = build_gaussian_pyramid(image, affine)
    >>> print(len(gaussian_pyramid))
    4  # Number of pyramid levels (example)
    """
    if limits is None:
        limits = get_pyramid_limits(image.shape)

    min_steps = int(limits[0].item())
    max_steps = int(limits[1].item())
    current = image.float().clone()
    current_affine = affine.clone().detach() if isinstance(affine, torch.Tensor) else torch.as_tensor(affine)

    imgs: list[torch.Tensor] = [current]
    affines: list[torch.Tensor] = [current_affine]

    for _ in range(max_steps):
        blurred = _smooth3d(current, _PYRAMID_FILTER, padding_mode="replicate")
        next_level = _downsample2_trilinear(blurred)
        next_shape = (int(next_level.shape[0]), int(next_level.shape[1]), int(next_level.shape[2]))
        next_affine = _downsample_affine(current_affine, current.shape, next_shape)
        current = next_level
        current_affine = next_affine
        imgs.append(current)
        affines.append(current_affine)

    return imgs[min_steps:max_steps + 1], affines[min_steps:max_steps + 1]
