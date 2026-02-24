"""Helper functions for working with nibabel image headers."""

import numpy as np
import nibabel as nib


def header_to_dict(img: nib.Nifti1Image | nib.MGHImage) -> dict:
    """
    Convert nibabel image header to dictionary format for write_lta.

    Parameters
    ----------
    img : nibabel image
        Input image (Nifti1Image or MGHImage)

    Returns
    -------
    dict
        Dictionary with keys: dims, delta, Mdc, Pxyz_c
    """
    header = img.header
    shape = [int(x) for x in img.shape[:3]]

    # For MGH format, use the stored header fields directly
    if isinstance(img, nib.MGHImage) and 'Mdc' in header:
        # MGH files store these fields directly in the header
        return {
            'dims': shape,
            'delta': [float(x) for x in header['delta']],
            'Mdc': header['Mdc'].copy(),
            'Pxyz_c': header['Pxyz_c'].copy()
        }

    # For NIfTI format, compute from affine
    # Get voxel sizes (convert to float to avoid numpy type strings)
    voxel_sizes = [float(x) for x in header.get_zooms()[:3]]

    # Get direction cosines and center from affine
    affine = img.affine

    # Direction cosine matrix (normalized)
    Mdc = affine[:3, :3] / voxel_sizes

    # RAS center (c_ras in FreeSurfer terminology)
    # This is the RAS coordinates of the center voxel
    center_vox = np.array(shape) / 2.0
    Pxyz_c = affine[:3, :3] @ center_vox + affine[:3, 3]

    return {
        'dims': shape,
        'delta': voxel_sizes,
        'Mdc': Mdc,
        'Pxyz_c': Pxyz_c
    }


def ras_to_vox_transform(
    ras_transform: np.ndarray,
    src_affine: np.ndarray,
    dst_affine: np.ndarray
) -> np.ndarray:
    """
    Convert RAS-to-RAS transform to vox-to-vox transform.

    Formula: Mv2v = inv(dst_affine) @ Mr2r @ src_affine

    Parameters
    ----------
    ras_transform : np.ndarray (4, 4)
        RAS-to-RAS transformation matrix
    src_affine : np.ndarray (4, 4)
        Source image voxel-to-RAS affine matrix
    dst_affine : np.ndarray (4, 4)
        Destination image voxel-to-RAS affine matrix

    Returns
    -------
    np.ndarray (4, 4)
        Vox-to-vox transformation matrix

    Notes
    -----
    This follows the FreeSurfer convention:
    - RAS-to-RAS (type=1): Mr2r transforms between RAS coordinate systems
    - Vox-to-vox (type=0): Mv2v transforms between voxel coordinate systems

    Relationship: Mr2r = dst_affine @ Mv2v @ inv(src_affine)
    Therefore: Mv2v = inv(dst_affine) @ Mr2r @ src_affine
    """
    return np.linalg.inv(dst_affine) @ ras_transform @ src_affine






