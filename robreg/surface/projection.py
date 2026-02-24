"""
Surface vertex projection utilities.

Implements vertex normal computation and projection along normals,
following the approach used in lapy but with PyTorch for GPU support.
"""

import torch
from typing import Optional


def compute_vertex_normals(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute vertex normals from triangle mesh.

    Uses area-weighted face normals, following the lapy approach.

    Parameters
    ----------
    vertices : torch.Tensor, shape (N, 3)
        Vertex coordinates
    faces : torch.Tensor, shape (F, 3)
        Triangle face indices (0-indexed)
    normalize : bool
        Whether to normalize the normals to unit length

    Returns
    -------
    normals : torch.Tensor, shape (N, 3)
        Vertex normal vectors

    Notes
    -----
    Each vertex normal is the area-weighted average of adjacent face normals.
    Face normals are computed using cross product of edge vectors.
    """
    n_vertices = vertices.shape[0]
    device = vertices.device

    # Initialize normals to zero
    normals = torch.zeros((n_vertices, 3), dtype=vertices.dtype, device=device)

    # Get triangle vertices
    v0 = vertices[faces[:, 0]]  # (F, 3)
    v1 = vertices[faces[:, 1]]  # (F, 3)
    v2 = vertices[faces[:, 2]]  # (F, 3)

    # Compute edge vectors
    e1 = v1 - v0  # (F, 3)
    e2 = v2 - v0  # (F, 3)

    # Compute face normals via cross product
    # Cross product gives vector perpendicular to both edges
    # Its magnitude equals twice the triangle area
    face_normals = torch.cross(e1, e2, dim=1)  # (F, 3)

    # Accumulate face normals to vertices (area-weighted)
    # Each face contributes to its three vertices
    normals.index_add_(0, faces[:, 0], face_normals)
    normals.index_add_(0, faces[:, 1], face_normals)
    normals.index_add_(0, faces[:, 2], face_normals)

    # Normalize to unit length
    if normalize:
        norm = torch.norm(normals, dim=1, keepdim=True)
        # Avoid division by zero for isolated vertices
        normals = normals / (norm + 1e-10)

    return normals


def project_vertices(
    vertices: torch.Tensor,
    normals: torch.Tensor,
    distance: float = 0.0,
    thickness: Optional[torch.Tensor] = None,
    fraction: Optional[float] = None
) -> torch.Tensor:
    """
    Project vertices along surface normals.

    Parameters
    ----------
    vertices : torch.Tensor, shape (N, 3)
        Original vertex positions
    normals : torch.Tensor, shape (N, 3)
        Surface normal vectors (should be unit length)
    distance : float
        Absolute distance to project (mm)
        Positive = outward (along normal)
        Negative = inward (opposite to normal)
    thickness : torch.Tensor, shape (N,), optional
        Per-vertex cortical thickness values
    fraction : float, optional
        Fractional distance based on thickness
        If provided, effective distance = fraction * thickness

    Returns
    -------
    projected : torch.Tensor, shape (N, 3)
        Projected vertex positions

    Notes
    -----
    If both distance and fraction are provided, they are combined:
        effective_distance = distance + fraction * thickness

    Examples
    --------
    # Project 2mm into white matter (inward)
    wm_vertices = project_vertices(vertices, normals, distance=-2.0)

    # Project 50% of cortical thickness into gray matter (outward)
    gm_vertices = project_vertices(vertices, normals,
                                   thickness=thickness, fraction=0.5)
    """
    effective_distance = distance

    # Add fractional thickness component if requested
    if fraction is not None and thickness is not None:
        # Broadcast thickness to (N, 1) for addition
        effective_distance = distance + fraction * thickness.unsqueeze(1)
    elif fraction is not None:
        raise ValueError("fraction specified but thickness not provided")

    # Project: v_new = v_old + effective_distance * normal
    projected = vertices + effective_distance * normals

    return projected


def create_wm_gm_surfaces(
    white_vertices: torch.Tensor,
    faces: torch.Tensor,
    normals: Optional[torch.Tensor] = None,
    thickness: Optional[torch.Tensor] = None,
    wm_proj_abs: float = 2.0,
    gm_proj_frac: float = 0.5,
    gm_proj_abs: Optional[float] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create WM and GM surfaces from white surface for BBR.

    This follows the bbregister approach:
    - WM surface: project inward into white matter
    - GM surface: project outward into gray matter

    Parameters
    ----------
    white_vertices : torch.Tensor, shape (N, 3)
        White matter surface vertices
    faces : torch.Tensor, shape (F, 3)
        Triangle faces
    normals : torch.Tensor, shape (N, 3), optional
        Precomputed normals. If None, will be computed.
    thickness : torch.Tensor, shape (N,), optional
        Cortical thickness (required if using gm_proj_frac)
    wm_proj_abs : float
        Distance to project into WM (mm, default: 2.0)
    gm_proj_frac : float
        Fraction of thickness to project into GM (default: 0.5)
    gm_proj_abs : float, optional
        Absolute distance to project into GM (if None, uses gm_proj_frac)

    Returns
    -------
    wm_vertices : torch.Tensor, shape (N, 3)
        White matter surface vertices (projected inward)
    gm_vertices : torch.Tensor, shape (N, 3)
        Gray matter surface vertices (projected outward)
    """
    # Compute normals if not provided
    if normals is None:
        normals = compute_vertex_normals(white_vertices, faces)

    # Create WM surface (project inward)
    wm_vertices = project_vertices(white_vertices, normals, distance=-wm_proj_abs)

    # Create GM surface (project outward)
    if gm_proj_abs is not None:
        # Use absolute distance
        gm_vertices = project_vertices(white_vertices, normals, distance=gm_proj_abs)
    else:
        # Use fractional thickness
        if thickness is None:
            raise ValueError("thickness required when using gm_proj_frac")
        gm_vertices = project_vertices(
            white_vertices, normals,
            thickness=thickness,
            fraction=gm_proj_frac
        )

    return wm_vertices, gm_vertices

