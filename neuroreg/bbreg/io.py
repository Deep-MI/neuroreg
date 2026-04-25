from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.freesurfer import read_geometry, read_morph_data

logger = logging.getLogger(__name__)


def load_surface(
        surf_path: str, thickness_path: str | None = None, cortex_label_path: str | None = None, device: str = "cpu"
) -> dict[str, torch.Tensor]:
    """Load a FreeSurfer surface file and optional associated data.

    Parameters
    ----------
    surf_path : str
        Path to the surface file (e.g., lh.white, rh.pial)
    thickness_path : str, optional
        Path to thickness file (e.g., lh.thickness)
    cortex_label_path : str, optional
        Path to cortex label file (e.g., lh.cortex.label).  When provided a
        boolean ``cortex_mask`` tensor is added to the result; vertices *not*
        in the label (medial wall, brainstem, etc.) will be masked out inside
        :class:`~neuroreg.bbreg.optimize.BBRModel` before the cost is computed.
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
        cortex_mask : torch.Tensor of bool, shape (N,), optional
            True for vertices that are part of the cortex label.

    Notes
    -----
    Vertices are in FreeSurfer's tkRAS coordinate system, not scanner RAS.
    To transform to voxel space, use the vox2ras_tkr matrix from nibabel.
    """
    # Load surface geometry
    vertices, faces = read_geometry(surf_path, read_metadata=False)

    # Handle byte order - FreeSurfer files may have non-native byte order
    if vertices.dtype.byteorder not in ("=", "|", "<" if np.little_endian else ">"):
        vertices = vertices.byteswap().view(vertices.dtype.newbyteorder())
    if faces.dtype.byteorder not in ("=", "|", "<" if np.little_endian else ">"):
        faces = faces.byteswap().view(faces.dtype.newbyteorder())

    # Convert to torch tensors
    vertices_tensor = torch.from_numpy(vertices).float().to(device)
    faces_tensor = torch.from_numpy(faces).long().to(device)

    result = {
        "vertices": vertices_tensor,
        "faces": faces_tensor,
    }

    # Load thickness if provided
    if thickness_path is not None:
        thickness = read_morph_data(thickness_path)
        if thickness.dtype.byteorder not in ("=", "|", "<" if np.little_endian else ">"):
            thickness = thickness.byteswap().view(thickness.dtype.newbyteorder())
        result["thickness"] = torch.from_numpy(thickness).float().to(device)

    # Load cortex label if provided
    if cortex_label_path is not None:
        cortex_indices = nib.freesurfer.read_label(cortex_label_path)
        n_verts = vertices_tensor.shape[0]
        cortex_mask = torch.zeros(n_verts, dtype=torch.bool, device=device)
        cortex_mask[cortex_indices] = True
        result["cortex_mask"] = cortex_mask
        logger.debug("Cortex label %s: %d / %d vertices in cortex", cortex_label_path, int(cortex_mask.sum()), n_verts)

    return result


def load_surface_pair(
        lh_surf_path: str,
        rh_surf_path: str,
        lh_thickness_path: str | None = None,
        rh_thickness_path: str | None = None,
        lh_cortex_label_path: str | None = None,
        rh_cortex_label_path: str | None = None,
        device: str = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Load left and right hemisphere surfaces.

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
    lh_cortex_label_path : str, optional
        Path to left hemisphere cortex label file
    rh_cortex_label_path : str, optional
        Path to right hemisphere cortex label file
    device : str
        Device to load tensors on

    Returns
    -------
    lh_surf : dict
        Left hemisphere surface data
    rh_surf : dict
        Right hemisphere surface data
    """
    lh_surf = load_surface(lh_surf_path, lh_thickness_path, lh_cortex_label_path, device)
    rh_surf = load_surface(rh_surf_path, rh_thickness_path, rh_cortex_label_path, device)
    return lh_surf, rh_surf


def load_surface_from_subject(
        subject_dir: str, hemi: str = "lh", surf_name: str = "white", load_thickness: bool = True, device: str = "cpu"
) -> dict[str, torch.Tensor]:
    """Load surface from FreeSurfer subject directory structure.

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
        Surface data with vertices, faces, optional thickness, and optional
        cortex_mask (loaded from ``label/?h.cortex.label`` when present).
    """
    subject_path = Path(subject_dir)
    surf_path = subject_path / "surf" / f"{hemi}.{surf_name}"

    thickness_path = None
    if load_thickness:
        thickness_file = subject_path / "surf" / f"{hemi}.thickness"
        if thickness_file.exists():
            thickness_path = str(thickness_file)

    cortex_label_path = None
    cortex_label_file = subject_path / "label" / f"{hemi}.cortex.label"
    if cortex_label_file.exists():
        cortex_label_path = str(cortex_label_file)
        logger.info("Loading cortex label from %s", cortex_label_path)
    else:
        logger.warning(
            "Cortex label not found: %s — all vertices will be used for BBR "
            "(may include medial wall / brainstem boundary vertices).",
            cortex_label_file,
        )

    return load_surface(str(surf_path), thickness_path, cortex_label_path, device)
