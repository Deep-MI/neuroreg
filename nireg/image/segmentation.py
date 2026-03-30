"""Surface extraction from brain segmentations for boundary-based registration.

This module converts a FreeSurfer-style parcellation (``aseg``, ``aparc+aseg``,
or equivalent) into white-matter surfaces suitable for use with
:func:`nireg.bbreg.register.register_surface` / the ``bbreg`` CLI.

The workflow is entirely CPU-based (numpy + scipy + scikit-image) and does
**not** require PyTorch or a GPU.

Typical pipeline
----------------
1. :func:`simplify_segmentation` — merge all parcellation labels into a
   4-class volume: LH-WM / LH-GM / RH-WM / RH-GM.
2. :func:`extract_wm_surface` — run marching cubes on one WM label, keep the
   largest connected component, apply light Gaussian + iterative smoothing, and
   return vertices in **tkRAS** (FreeSurfer surface-RAS) space together with
   faces as numpy arrays.
3. :func:`surfaces_from_segmentation` — convenience wrapper that runs steps 1
   and 2 for both hemispheres and returns data dicts compatible with
   :func:`nireg.bbreg.register.register_surface`.

When surfaces are derived from a segmentation no thickness file is available.
The callers should pass ``thickness=None``; :class:`~nireg.bbreg.optimize.BBRModel`
will then fall back to projecting a fixed distance outside the white surface.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import nibabel as nib
import numpy as np
from scipy.ndimage import uniform_filter

from nireg.bbreg.io import get_vox2ras_tkr

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FreeSurfer label constants
# ---------------------------------------------------------------------------

#: Subcortical labels that belong to the *left* hemisphere.
ASEG_LEFT_CLASSES: tuple[int, ...] = (2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 30, 31)

#: Subcortical labels that belong to the *right* hemisphere.
ASEG_RIGHT_CLASSES: tuple[int, ...] = (41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63)

# Simplified segmentation label constants
_LH_WM: int = 2
_LH_GM: int = 3
_RH_WM: int = 41
_RH_GM: int = 42


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mask_in_array(
    arr: npt.NDArray[np.integer],
    items: npt.ArrayLike,
) -> npt.NDArray[np.bool_]:
    """Return a boolean mask that is *True* where ``arr`` is in ``items``.

    Uses a lookup table for O(N) performance instead of O(N*K) comparisons.

    Parameters
    ----------
    arr : ndarray of int
        Label array.
    items : array-like of int
        Labels to test membership for.

    Returns
    -------
    mask : ndarray of bool
        Same shape as *arr*.
    """
    _items = np.asarray(items)
    if _items.size == 0:
        return np.zeros_like(arr, dtype=bool)
    if _items.size == 1:
        return arr == _items.flat[0]
    max_index = max(int(np.max(_items)), int(np.max(arr)))
    lookup = np.zeros(max_index + 1, dtype=bool)
    lookup[_items] = True
    return lookup[arr]


def _mask_not_in_array(
    arr: npt.NDArray[np.integer],
    items: npt.ArrayLike,
) -> npt.NDArray[np.bool_]:
    """Inverse of :func:`_mask_in_array`."""
    _items = np.asarray(items)
    if _items.size == 0:
        return np.ones_like(arr, dtype=bool)
    if _items.size == 1:
        return arr != _items.flat[0]
    max_index = max(int(np.max(_items)), int(np.max(arr)))
    lookup = np.ones(max_index + 1, dtype=bool)
    lookup[_items] = False
    return lookup[arr]


def _hemi_masks(
    arr: npt.NDArray[np.integer],
    window_size: int = 7,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Determine per-voxel hemisphere assignment by local label voting.

    Parameters
    ----------
    arr : ndarray of int
        Segmentation array (may be partially simplified).
    window_size : int
        Box filter size used for the local vote.

    Returns
    -------
    mask_left, mask_right : ndarray of bool
        Voxels that are more likely left / right hemisphere.
    """
    leftness = uniform_filter(
        _mask_in_array(arr, ASEG_LEFT_CLASSES).astype(np.float32),
        size=window_size,
    )
    rightness = uniform_filter(
        _mask_in_array(arr, ASEG_RIGHT_CLASSES).astype(np.float32),
        size=window_size,
    )
    return leftness > rightness, rightness > leftness


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simplify_segmentation(
    seg_input: str | os.PathLike | Any,
    output_path: str | os.PathLike | None = None,
) -> np.ndarray:
    """Reduce a detailed parcellation to a 4-class WM/GM segmentation.

    Merges all FreeSurfer aparc+aseg (or plain aseg) labels into four
    tissue classes:

    * **2** — left-hemisphere white matter (LH-WM)
    * **3** — left-hemisphere grey matter  (LH-GM)
    * **41** — right-hemisphere white matter (RH-WM)
    * **42** — right-hemisphere grey matter  (RH-GM)
    * **0** — background / excluded structures

    Mapping rules (applied in order):

    1. **Cortical parcels** (aparc labels 1000–1999 / 2000–2999) → LH-GM / RH-GM.
    2. **Subcortical fill → WM** — a set of subcortical structures that sit
       *inside* or immediately adjacent to the WM volume are absorbed into the
       nearest hemisphere's WM label so that the extracted WM surface has no
       interior holes.  This group includes the **lateral ventricles**
       (labels 4 / 43), thalamus, caudate, putamen, pallidum, accumbens,
       ventral DC, vessel, and choroid plexus.  Filling the lateral ventricles
       is intentional: treating them as background would leave a void in the
       centre of the WM and cause marching cubes to generate spurious interior
       surfaces.
    3. **Corpus callosum** (251–255) → split between LH-WM / RH-WM by local
       neighbourhood vote; tied midline voxels → background.
    4. **WM hypointensities** (77) → LH-WM / RH-WM by local neighbourhood vote.
    5. **Everything else** (cerebellum, brainstem, hippocampus, amygdala,
       inferior lateral ventricles, CSF, and any unlabelled voxels) → **0**
       (background).

    Parameters
    ----------
    seg_input : str, Path, or nibabel image
        Input parcellation.  Accepts a file path **or** an already-loaded
        nibabel image object.
    output_path : str or Path, optional
        If given, the simplified volume is saved to this path.

    Returns
    -------
    seg_data : ndarray, shape (X, Y, Z), dtype int32
        Simplified label volume.

    Notes
    -----
    The function is fully CPU-based (numpy + scipy) and does not require
    PyTorch.
    """
    if isinstance(seg_input, str | os.PathLike):
        logger.debug("Loading segmentation from %s", seg_input)
        seg_img = nib.load(seg_input)
    else:
        seg_img = seg_input

    seg_data: np.ndarray = seg_img.get_fdata().astype(np.int32)

    # ---- cortical parcels → GM ----
    seg_data[(seg_data >= 2000) & (seg_data <= 2999)] = _RH_GM
    seg_data[(seg_data >= 1000) & (seg_data <= 1999)] = _LH_GM

    # ---- subcortical fill → WM ----
    # Lateral-Ventricle (4), Thalamus (10), Caudate (11), Putamen (12),
    # Pallidum (13), Accumbens (26), VentralDC (28), Vessel (30), Choroid (31)
    # Corpus callosum (251-255) is handled separately below via hemisphere vote.
    left_fill = (4, 10, 11, 12, 13, 26, 28, 30, 31)
    right_fill = (43, 49, 50, 51, 52, 58, 60, 62, 63)
    seg_data[_mask_in_array(seg_data, left_fill)] = _LH_WM
    seg_data[_mask_in_array(seg_data, right_fill)] = _RH_WM

    # ---- hemisphere vote (used for CC and WM hypointensities) ----
    # Called after LH/RH WM and GM labels are set so the local vote is driven
    # by the simplified anatomy (LH-WM/GM vs RH-WM/GM neighbours).
    lm, rm = _hemi_masks(seg_data)

    # ---- corpus callosum (251-255) → split by hemisphere proximity ----
    # The CC sits at the midline; assigning everything to one hemisphere would
    # create a surface boundary facing CSF or the opposite hemisphere, which
    # would corrupt the BBR cost.  Instead split by local neighbourhood vote.
    cc_mask = _mask_in_array(seg_data, (251, 252, 253, 254, 255))
    if cc_mask.any():
        seg_data[cc_mask & lm] = _LH_WM
        seg_data[cc_mask & rm] = _RH_WM
        # Voxels with an exactly tied vote sit precisely at the midline;
        # assign them to background so they do not create a spurious surface.
        tied = cc_mask & ~lm & ~rm
        if tied.any():
            seg_data[tied] = 0
        logger.debug(
            "Corpus callosum split: %d → LH-WM, %d → RH-WM, %d → background",
            int((cc_mask & lm).sum()),
            int((cc_mask & rm).sum()),
            int(tied.sum()),
        )

    # ---- WM hypointensities (77) → hemisphere by local vote ----
    seg_data[(seg_data == 77) & lm] = _LH_WM
    seg_data[(seg_data == 77) & rm] = _RH_WM

    # ---- warn about any unexpected remaining labels ----
    dontcare = (
        0,
        5,
        6,
        7,
        8,
        14,
        15,
        16,
        17,
        18,
        24,
        44,
        45,
        46,
        47,
        53,
        54,
        85,  # optic chiasm → background
        _LH_WM,
        _RH_WM,
        _LH_GM,
        _RH_GM,
    )
    unexpected_mask = _mask_not_in_array(seg_data, dontcare)
    if unexpected_mask.any():
        unique, counts = np.unique(seg_data[unexpected_mask], return_counts=True)
        logger.warning(
            "Unexpected labels after simplification: %s (counts: %s)",
            unique.tolist(),
            counts.tolist(),
        )

    # ---- everything else → background ----
    seg_data[_mask_not_in_array(seg_data, (_LH_WM, _RH_WM, _LH_GM, _RH_GM))] = 0

    if output_path is not None:
        out_img = nib.MGHImage(seg_data.astype(np.float32), seg_img.affine, seg_img.header)
        nib.save(out_img, str(output_path))
        logger.info("Simplified segmentation saved to %s", output_path)

    return seg_data


def extract_wm_surface(
    seg_data: np.ndarray,
    wm_label: int,
    seg_header: Any,
    smooth_sigma: float = 0.5,
    marching_cubes_level: float = 0.45,
    smooth_iterations: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a white-matter surface from a simplified segmentation volume.

    Runs marching cubes on the binarised WM label, keeps the largest
    connected component, applies a two-stage smoothing (Gaussian pre-blur
    before marching cubes + iterative Taubin-style Laplacian smoothing after),
    and computes a cortex mask that flags vertices lying at the WM-GM boundary.

    The cortex mask is determined by checking the 8 voxel corners that
    enclose each vertex position (the trilinear interpolation stencil).  A
    vertex is considered cortical if *any* of those 8 corners carries the
    matching GM label in *seg_data*.  This excludes corpus callosum borders,
    brainstem/cerebellum WM edges, and any other locations where WM abuts a
    non-GM tissue — exactly what ``?h.cortex.label`` achieves in bbregister.

    Parameters
    ----------
    seg_data : ndarray, shape (X, Y, Z)
        Simplified segmentation array (output of :func:`simplify_segmentation`).
    wm_label : int
        WM label to extract (``2`` for LH, ``41`` for RH).
    seg_header : nibabel header
        Header of the segmentation image.  Must expose
        ``get_vox2ras_tkr()`` (MGH/MGZ) so that voxel coordinates can be
        converted to tkRAS (FreeSurfer surface-RAS) space.
    smooth_sigma : float
        Sigma (in voxels) of the Gaussian blur applied to the binary mask
        *before* marching cubes.  Set to ``0`` to skip.
    marching_cubes_level : float
        Iso-level for marching cubes (default 0.45, slightly below 0.5 to
        place the surface just inside the WM boundary).
    smooth_iterations : int
        Number of Taubin-smoothing iterations applied *after* marching cubes.
        Each iteration consists of one shrink + one inflate step
        (λ = 0.6307, μ = −0.6732).

    Returns
    -------
    vertices : ndarray, shape (V, 3), float32
        Surface vertices in **tkRAS** (FreeSurfer surface-RAS) coordinates.
    faces : ndarray, shape (F, 3), int32
        Triangle face indices.
    cortex_mask : ndarray, shape (V,), bool
        ``True`` for vertices at the WM-GM boundary (i.e. those that should
        contribute to the BBR cost).  Equivalent to ``?h.cortex.label`` when
        surfaces are derived from a segmentation.

    Raises
    ------
    ImportError
        If ``scikit-image`` is not installed.
    ValueError
        If *wm_label* is not present in *seg_data* or no surface could be
        extracted.
    """
    try:
        from skimage.filters import gaussian as sk_gaussian
        from skimage.measure import marching_cubes
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for surface extraction from segmentations. "
            "Install it with:  pip install scikit-image"
        ) from exc

    if not np.any(seg_data == wm_label):
        raise ValueError(f"WM label {wm_label} not found in segmentation. Run simplify_segmentation() first.")

    # ---- binary mask + optional Gaussian pre-blur ----
    binary = (seg_data == wm_label).astype(np.float32)
    if smooth_sigma > 0.0:
        binary = sk_gaussian(binary, sigma=smooth_sigma, preserve_range=True)

    # ---- marching cubes ----
    verts_vox, faces, *_ = marching_cubes(
        binary,
        level=marching_cubes_level,
        spacing=(1.0, 1.0, 1.0),
        gradient_direction="descent",
    )
    logger.debug(
        "Marching cubes (label %d): %d vertices, %d faces",
        wm_label,
        len(verts_vox),
        len(faces),
    )

    # ---- keep largest connected component ----
    # Label each triangle by its connected component using vertex connectivity
    faces = faces.astype(np.int32)
    n_verts = len(verts_vox)

    # Build a simple union-find on vertices using scipy's connected-component
    # labelling on a sparse adjacency (edges from faces)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    adj = coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_verts, n_verts),
    )
    n_components, comp_labels = connected_components(adj, directed=False)
    if n_components > 1:
        comp_sizes = np.bincount(comp_labels)
        largest = int(np.argmax(comp_sizes))
        keep_verts = comp_labels == largest
        logger.debug(
            "Connected components: %d — keeping largest (%d/%d vertices)",
            n_components,
            keep_verts.sum(),
            n_verts,
        )
        # remap vertex indices
        old_to_new = np.full(n_verts, -1, dtype=np.int32)
        old_to_new[keep_verts] = np.arange(keep_verts.sum(), dtype=np.int32)
        face_mask = keep_verts[faces[:, 0]] & keep_verts[faces[:, 1]] & keep_verts[faces[:, 2]]
        verts_vox = verts_vox[keep_verts]
        faces = old_to_new[faces[face_mask]]

    # ---- Taubin smoothing (numpy, no pytorch) ----
    if smooth_iterations > 0:
        verts_vox = _taubin_smooth_numpy(verts_vox, faces, n_iters=smooth_iterations)

    # ---- cortex mask: check 8 voxel corners for GM label ----
    # Done in voxel space before the tkRAS conversion.  For each vertex at
    # position (x, y, z) the 8 corners of the enclosing unit cube are sampled
    # in the simplified segmentation.  A vertex is "cortical" if any corner
    # carries the matching GM label — i.e. it sits at a genuine WM-GM boundary.
    gm_label = _LH_GM if wm_label == _LH_WM else _RH_GM
    cortex_mask = _compute_cortex_mask_8neighbors(verts_vox, seg_data, gm_label)
    logger.info(
        "Cortex mask (label %d, 8-neighbor): %d / %d vertices (%.1f%%)",
        wm_label,
        cortex_mask.sum(),
        len(verts_vox),
        100.0 * cortex_mask.sum() / max(len(verts_vox), 1),
    )

    # ---- convert voxel → tkRAS ----
    vox2tkras = np.array(get_vox2ras_tkr(seg_header), dtype=np.float32)
    ones = np.ones((len(verts_vox), 1), dtype=np.float32)
    verts_hom = np.hstack([verts_vox.astype(np.float32), ones])  # (V, 4)
    verts_tkras = (vox2tkras @ verts_hom.T).T[:, :3].astype(np.float32)

    logger.info(
        "Extracted WM surface (label %d): %d vertices, %d faces",
        wm_label,
        len(verts_tkras),
        len(faces),
    )
    return verts_tkras, faces.astype(np.int32), cortex_mask


def surfaces_from_segmentation(
    seg_path: str | os.PathLike | Any,
    *,
    hemispheres: tuple[str, ...] = ("lh", "rh"),
    smooth_sigma: float = 0.5,
    marching_cubes_level: float = 0.45,
    smooth_iterations: int = 50,
    device: str = "cpu",
) -> tuple[dict | None, dict | None]:
    """Extract left and/or right WM surfaces from a parcellation file.

    Convenience wrapper that calls :func:`simplify_segmentation` followed by
    :func:`extract_wm_surface` for each requested hemisphere and returns
    data dicts in the same format that
    :func:`nireg.bbreg.io.load_surface_from_subject` produces, so they can
    be passed directly to :func:`nireg.bbreg.register.register_surface`.

    Because no thickness file is available the ``'thickness'`` key is set to
    ``None``; ``BBRModel`` will then use a fixed outward projection distance
    instead of cortical thickness.

    Parameters
    ----------
    seg_path : str, Path, or nibabel image
        Parcellation file (``aparc+aseg.mgz``, ``aseg.mgz``, or any
        compatible format).
    hemispheres : tuple of str
        Which hemispheres to extract.  Any subset of ``("lh", "rh")``.
    smooth_sigma : float
        Gaussian pre-blur sigma passed to :func:`extract_wm_surface`.
    marching_cubes_level : float
        Marching-cubes iso-level passed to :func:`extract_wm_surface`.
    smooth_iterations : int
        Post-marching-cubes Taubin smoothing iterations.
    device : str
        Torch device string.  Vertices and faces tensors will be placed here.

    Returns
    -------
    lh_data : dict or None
        ``{'vertices': Tensor(V,3), 'faces': Tensor(F,3), 'thickness': None}``
        for the left hemisphere, or ``None`` if not requested.
    rh_data : dict or None
        Same for the right hemisphere.

    Notes
    -----
    The segmentation image header is also used as the registration target
    header (``trg_header``) so no separate T1 image needs to be provided when
    calling ``register_surface`` via this path.  The caller should pass::

        trg_header = seg_img.header
    """
    import torch

    if isinstance(seg_path, str | os.PathLike):
        seg_img = nib.load(str(seg_path))
    else:
        seg_img = seg_path

    logger.info("Simplifying segmentation …")
    seg_data = simplify_segmentation(seg_img)
    seg_header = seg_img.header

    hemi_cfg = {
        "lh": (_LH_WM, "lh"),
        "rh": (_RH_WM, "rh"),
    }

    results: dict[str, dict | None] = {"lh": None, "rh": None}
    for hemi in hemispheres:
        wm_label, _ = hemi_cfg[hemi]
        logger.info("Extracting %s white surface …", hemi)
        verts_np, faces_np, cortex_mask_np = extract_wm_surface(
            seg_data,
            wm_label=wm_label,
            seg_header=seg_header,
            smooth_sigma=smooth_sigma,
            marching_cubes_level=marching_cubes_level,
            smooth_iterations=smooth_iterations,
        )
        results[hemi] = {
            "vertices": torch.from_numpy(verts_np).to(device),
            "faces": torch.from_numpy(faces_np).to(device),
            "thickness": None,  # not available from segmentation
            "cortex_mask": torch.from_numpy(cortex_mask_np).to(device),
        }

    return results.get("lh"), results.get("rh")


# ---------------------------------------------------------------------------
# Internal helpers (not exported)
# ---------------------------------------------------------------------------


def _compute_cortex_mask_8neighbors(
    verts_vox: npt.NDArray[np.floating],
    seg_data: npt.NDArray[np.integer],
    gm_label: int,
) -> npt.NDArray[np.bool_]:
    """Return a cortex mask by checking the 8 enclosing voxel corners for GM.

    For each vertex at floating-point voxel position ``(x, y, z)`` the eight
    corners of the surrounding unit cube — ``(floor(x), floor(y), floor(z))``
    through ``(floor(x)+1, floor(y)+1, floor(z)+1)`` — are looked up in
    *seg_data*.  The vertex is marked cortical if *any* corner equals
    *gm_label*.

    This is equivalent to asking "does this vertex touch a GM voxel?", which
    exactly captures genuine WM-GM boundaries while excluding corpus callosum
    borders, brainstem WM edges, and other non-cortical locations.

    No normal vectors or sampling distances are required.

    Parameters
    ----------
    verts_vox : ndarray, shape (V, 3)
        Vertex positions in voxel coordinates (may be non-integer after
        Taubin smoothing).
    seg_data : ndarray, shape (X, Y, Z)
        Simplified segmentation (output of :func:`simplify_segmentation`).
    gm_label : int
        GM label to test against (``3`` for LH, ``42`` for RH).

    Returns
    -------
    cortex_mask : ndarray of bool, shape (V,)
        ``True`` where a GM voxel is among the 8 enclosing corners.
    """
    shape = np.array(seg_data.shape[:3], dtype=np.int32)

    lo = np.floor(verts_vox).astype(np.int32)  # (V, 3)
    hi = np.clip(lo + 1, 0, shape - 1)  # (V, 3)  ceil, clamped
    lo = np.clip(lo, 0, shape - 1)  # (V, 3)  floor, clamped

    mask = np.zeros(len(verts_vox), dtype=bool)
    for xi in (lo[:, 0], hi[:, 0]):
        for yi in (lo[:, 1], hi[:, 1]):
            for zi in (lo[:, 2], hi[:, 2]):
                mask |= seg_data[xi, yi, zi] == gm_label

    return mask


def compute_cortex_mask(
    vertices_tkras: npt.NDArray[np.floating],
    seg_input: str | os.PathLike | Any,
    hemi: str,
) -> npt.NDArray[np.bool_]:
    """Compute a cortex mask for an existing surface by sampling a segmentation.

    Converts surface vertices from tkRAS to voxel space, then checks whether
    any of the 8 enclosing voxel corners in the simplified segmentation carry
    the GM label for the requested hemisphere.  Vertices where no GM neighbour
    exists (corpus callosum border, brainstem/cerebellum WM edge, medial wall,
    etc.) are excluded.

    This can be used to generate a ``?h.cortex.label``-equivalent mask for a
    FreeSurfer surface when only a segmentation file is available instead of a
    proper cortex label file.

    Parameters
    ----------
    vertices_tkras : ndarray, shape (V, 3)
        Surface vertices in FreeSurfer tkRAS space (as returned by
        ``nibabel.freesurfer.read_geometry``).
    seg_input : str, Path, or nibabel image
        Segmentation file (``aparc+aseg.mgz``, ``aseg.mgz``, or equivalent).
    hemi : str
        Hemisphere: ``'lh'`` or ``'rh'``.

    Returns
    -------
    cortex_mask : ndarray of bool, shape (V,)
        ``True`` for cortical vertices (WM-GM boundary present in the
        segmentation).
    """
    if isinstance(seg_input, str | os.PathLike):
        seg_img = nib.load(str(seg_input))
    else:
        seg_img = seg_input

    seg_data = simplify_segmentation(seg_img)
    gm_label = _LH_GM if hemi == "lh" else _RH_GM

    # Convert tkRAS → voxel using the inverse of vox2tkras
    vox2tkras = np.array(get_vox2ras_tkr(seg_img.header), dtype=np.float64)
    tkras2vox = np.linalg.inv(vox2tkras)

    ones = np.ones((len(vertices_tkras), 1), dtype=np.float64)
    verts_hom = np.hstack([vertices_tkras.astype(np.float64), ones])  # (V, 4)
    verts_vox = (tkras2vox @ verts_hom.T).T[:, :3]  # (V, 3)

    mask = _compute_cortex_mask_8neighbors(verts_vox, seg_data, gm_label)
    logger.info(
        "compute_cortex_mask (%s): %d / %d vertices (%.1f%%) at WM-GM boundary",
        hemi,
        mask.sum(),
        len(mask),
        100.0 * mask.sum() / max(len(mask), 1),
    )
    return mask


def _taubin_smooth_numpy(
    verts: np.ndarray,
    faces: np.ndarray,
    n_iters: int = 50,
    lam: float = 0.6307,
    mu: float = -0.6732,
) -> np.ndarray:
    """Volume-preserving Taubin smoothing (numpy, CPU only).

    Alternates between a shrink step (λ > 0) and an inflate step (μ < 0)
    to suppress high-frequency noise while approximately preserving the
    enclosed volume.

    Parameters
    ----------
    verts : ndarray, shape (V, 3)
        Vertex positions in voxel coordinates.
    faces : ndarray, shape (F, 3), int
        Triangle face indices.
    n_iters : int
        Number of Taubin cycles (each cycle = one shrink + one inflate).
    lam : float
        Shrink step size (positive).
    mu : float
        Inflate step size (negative, typically ``−lam − ε``).

    Returns
    -------
    verts : ndarray, shape (V, 3)
        Smoothed vertex positions.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components  # noqa: F401 (keep import visible)

    n = len(verts)

    # Build symmetric adjacency as a sparse CSR matrix (degree + neighbour sum)
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    adj = coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, n),
    ).tocsr()
    # Symmetric: sum duplicates automatically
    degree = np.asarray(adj.sum(axis=1)).ravel()  # (V,)
    degree = np.maximum(degree, 1.0)

    verts = verts.astype(np.float32, copy=True)

    for _ in range(n_iters):
        # shrink
        lap = (adj @ verts) / degree[:, None] - verts
        verts = verts + lam * lap
        # inflate
        lap = (adj @ verts) / degree[:, None] - verts
        verts = verts + mu * lap

    return verts
