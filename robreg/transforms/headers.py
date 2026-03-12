"""Helper functions for working with nibabel image headers."""

import nibabel as nib
import numpy as np


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

