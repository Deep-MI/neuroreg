"""
Surface I/O utilities for loading FreeSurfer surface meshes.

Handles loading surface geometry from FreeSurfer .surf files and associated
data like thickness, curvature, etc. Correctly manages tkRAS coordinate system.
"""

from pathlib import Path

import nibabel as nib  # noqa: F401 – used in string type annotations
import numpy as np
import torch
from nibabel.freesurfer import read_geometry, read_morph_data


def load_surface(
    surf_path: str,
    thickness_path: str | None = None,
    device: str = 'cpu'
) -> dict[str, torch.Tensor]:
    """
    Load a FreeSurfer surface file and optional associated data.

    Parameters
    ----------
    surf_path : str
        Path to the surface file (e.g., lh.white, rh.pial)
    thickness_path : str, optional
        Path to thickness file (e.g., lh.thickness)
    device : str
        Device to load tensors on ('cpu' or 'cuda')

    Returns
    -------
    dict with keys:
        vertices : torch.Tensor, shape (N, 3)
            Vertex coordinates in tkRAS space
        faces : torch.Tensor, shape (F, 3)
            Triangle face indices (0-indexed)
        thickness : torch.Tensor, shape (N,), optional
            Per-vertex cortical thickness values

    Notes
    -----
    Vertices are in FreeSurfer's tkRAS coordinate system, not scanner RAS.
    To transform to voxel space, use the vox2ras_tkr matrix from nibabel.
    """
    # Load surface geometry
    vertices, faces = read_geometry(surf_path, read_metadata=False)

    # Handle byte order - FreeSurfer files may have non-native byte order
    # Convert to native byte order to avoid PyTorch issues
    # NumPy 2.0 compatible approach
    if vertices.dtype.byteorder not in ('=', '|', '<' if np.little_endian else '>'):
        vertices = vertices.byteswap().view(vertices.dtype.newbyteorder())
    if faces.dtype.byteorder not in ('=', '|', '<' if np.little_endian else '>'):
        faces = faces.byteswap().view(faces.dtype.newbyteorder())

    # Convert to torch tensors
    vertices_tensor = torch.from_numpy(vertices).float().to(device)
    faces_tensor = torch.from_numpy(faces).long().to(device)

    result = {
        'vertices': vertices_tensor,
        'faces': faces_tensor,
    }

    # Load thickness if provided
    if thickness_path is not None:
        thickness = read_morph_data(thickness_path)
        # Handle byte order (NumPy 2.0 compatible)
        if thickness.dtype.byteorder not in ('=', '|', '<' if np.little_endian else '>'):
            thickness = thickness.byteswap().view(thickness.dtype.newbyteorder())
        result['thickness'] = torch.from_numpy(thickness).float().to(device)

    return result


def load_surface_pair(
    lh_surf_path: str,
    rh_surf_path: str,
    lh_thickness_path: str | None = None,
    rh_thickness_path: str | None = None,
    device: str = 'cpu'
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Load left and right hemisphere surfaces.

    Parameters
    ----------
    lh_surf_path : str
        Path to left hemisphere surface
    rh_surf_path : str
        Path to right hemisphere surface
    lh_thickness_path : str, optional
        Path to left hemisphere thickness
    rh_thickness_path : str, optional
        Path to right hemisphere thickness
    device : str
        Device to load tensors on

    Returns
    -------
    lh_surf : dict
        Left hemisphere surface data
    rh_surf : dict
        Right hemisphere surface data
    """
    lh_surf = load_surface(lh_surf_path, lh_thickness_path, device)
    rh_surf = load_surface(rh_surf_path, rh_thickness_path, device)

    return lh_surf, rh_surf


def load_surface_from_subject(
    subject_dir: str,
    hemi: str = 'lh',
    surf_name: str = 'white',
    load_thickness: bool = True,
    device: str = 'cpu'
) -> dict[str, torch.Tensor]:
    """
    Load surface from FreeSurfer subject directory structure.

    Parameters
    ----------
    subject_dir : str
        Path to FreeSurfer subject directory (e.g., /data/subjects/sub-01)
    hemi : str
        Hemisphere: 'lh' or 'rh'
    surf_name : str
        Surface name: 'white', 'pial', 'inflated', etc.
    load_thickness : bool
        Whether to load thickness data
    device : str
        Device to load tensors on

    Returns
    -------
    dict
        Surface data with vertices, faces, and optional thickness
    """
    subject_path = Path(subject_dir)
    surf_path = subject_path / 'surf' / f'{hemi}.{surf_name}'

    thickness_path = None
    if load_thickness:
        thickness_file = subject_path / 'surf' / f'{hemi}.thickness'
        if thickness_file.exists():
            thickness_path = str(thickness_file)

    return load_surface(str(surf_path), thickness_path, device)


_AnyVolRef = (
    "nib.Nifti1Image | nib.MGHImage"
    " | nib.nifti1.Nifti1Header | nib.freesurfer.mghformat.MGHHeader"
)


def get_vox2ras_tkr(ref_volume: _AnyVolRef) -> np.ndarray:
    """Get voxel-to-tkRAS transformation matrix for a reference volume.

    tkRAS is the FreeSurfer coordinate system in which surface vertices are
    stored.  It is a voxel-centred RAS space that ignores the scanner
    position but preserves the voxel size and orientation.

    Parameters
    ----------
    ref_volume : nibabel image or nibabel header
        Reference volume or its header (e.g., orig.mgz or T1.mgz).
        Passing a header avoids loading voxel data entirely.

    Returns
    -------
    vox2ras_tkr : np.ndarray, shape (4, 4)
        Voxel-to-tkRAS transformation matrix.

    Notes
    -----
    **MGH/MGZ files** store ``vox2ras_tkr`` explicitly in the header, so
    nibabel's ``MGHHeader.get_vox2ras_tkr()`` is used directly.

    **NIfTI files** have no such field.  ``nibabel.freesurfer.mghformat
    .MGHHeader.from_header()`` is *not* used here because it ignores the
    actual voxel size and shape of the NIfTI and returns an incorrect
    near-identity matrix.  Instead we compute the standard FreeSurfer
    tkRAS convention explicitly: a diagonal matrix scaled by voxel size
    with the origin at the image centre (``shape / 2``), with the x-axis
    negated to match RAS orientation.
    """
    # Accept either an image or a bare header
    if hasattr(ref_volume, 'header'):
        header = ref_volume.header
    else:
        header = ref_volume

    # MGH/MGZ: use the stored field directly
    if hasattr(header, 'get_vox2ras_tkr'):
        return header.get_vox2ras_tkr()

    # NIfTI (and any other format): compute from voxel size and shape
    shape   = header.get_data_shape()[:3]
    voxsize = header.get_zooms()[:3]

    M = np.eye(4)
    M[0, 0] = -voxsize[0]   # x flipped to match RAS convention
    M[1, 1] =  voxsize[1]
    M[2, 2] =  voxsize[2]
    M[0, 3] =  voxsize[0] * shape[0] / 2.0   # centre origin
    M[1, 3] = -voxsize[1] * shape[1] / 2.0
    M[2, 3] = -voxsize[2] * shape[2] / 2.0

    return M

