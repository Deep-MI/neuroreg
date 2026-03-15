"""Command-line interface for LTA transform comparison (lta-diff).

Analogous to FreeSurfer's ``lta_diff`` utility.
"""

import argparse
import sys

import numpy as np

from nireg.transforms import (
    affine_dist,
    corner_diff,
    decompose_transform,
    read_lta,
    rigid_dist,
    sphere_diff,
)
from nireg.transforms.matrices import (
    LINEAR_RAS_TO_RAS,
    LINEAR_VOX_TO_VOX,
    convert_transform_type,
)

# ── helpers ─────────────────────────────────────────────────────────────────

def _affine_from_info(info: dict) -> np.ndarray:
    """Reconstruct a voxel-to-RAS affine from an LTA volume-info dict."""
    vs   = info['voxelsize']
    dims = np.array(info['volume'], dtype=float)
    A    = np.eye(4)
    A[:3, 0] = np.array(info['xras']) * vs[0]
    A[:3, 1] = np.array(info['yras']) * vs[1]
    A[:3, 2] = np.array(info['zras']) * vs[2]
    A[:3, 3] = np.array(info['cras']) - A[:3, :3] @ (dims / 2.0)
    return A


def _load(path: str) -> dict:
    """Read an LTA file; return the raw dict (type unchanged)."""
    return read_lta(path)


def _to_ras(lta: dict) -> np.ndarray:
    """Return the 4×4 RAS-to-RAS matrix for an LTA dict."""
    if lta['type'] == LINEAR_RAS_TO_RAS:
        return lta['matrix'].copy()
    src_aff = _affine_from_info(lta['src'])
    dst_aff = _affine_from_info(lta['dst'])
    return convert_transform_type(
        lta['matrix'], src_aff, dst_aff,
        from_type=LINEAR_VOX_TO_VOX, to_type=LINEAR_RAS_TO_RAS,
    )


def _to_iso_vox(lta: dict) -> np.ndarray:
    """Return the vox-to-vox matrix scaled to isotropic mm units (getIsoVOX).

    Equivalent to ``diag(dst_vs) @ V2V @ diag(src_vs)``, which is the
    matrix used by FreeSurfer's ``lta_diff --vox``.
    """
    src_aff = _affine_from_info(lta['src'])
    dst_aff = _affine_from_info(lta['dst'])
    if lta['type'] == LINEAR_RAS_TO_RAS:
        v2v = convert_transform_type(
            lta['matrix'], src_aff, dst_aff,
            from_type=LINEAR_RAS_TO_RAS, to_type=LINEAR_VOX_TO_VOX,
        )
    else:
        v2v = lta['matrix'].copy()
    src_vs = np.diag([*lta['src']['voxelsize'], 1.0])
    dst_vs = np.diag([*lta['dst']['voxelsize'], 1.0])
    return dst_vs @ v2v @ src_vs


def _invert(lta: dict) -> dict:
    """Invert an LTA (matrix inverted, src/dst geometries swapped)."""
    M_ras     = _to_ras(lta)
    M_ras_inv = np.linalg.inv(M_ras)
    inv_lta   = dict(lta)
    inv_lta['src'] = lta['dst']
    inv_lta['dst'] = lta['src']
    inv_lta['type'] = LINEAR_RAS_TO_RAS
    inv_lta['matrix'] = M_ras_inv
    return inv_lta


def _get_matrix(lta: dict, vox: bool) -> np.ndarray:
    """Return the comparison matrix honouring the ``--vox`` flag."""
    return _to_iso_vox(lta) if vox else _to_ras(lta)


def _diff_matrix(lta1: dict, lta2: dict | None, vox: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (M1, M2) ready for the distance functions."""
    M1 = _get_matrix(lta1, vox)
    M2 = _get_matrix(lta2, vox) if lta2 is not None else None
    return M1, M2


# ── parser ───────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='lta-diff',
        description=(
            'Compute distance metrics between two LTA transforms, or between\n'
            'one transform and identity.\n'
            '\n'
            'Analogous to FreeSurfer\'s lta_diff utility.\n'
            '\n'
            'Distance types:\n'
            '  1   Rigid transform distance  sqrt(||log R_d||² + ||T_d||²)\n'
            '      D = inv(M1) @ M2  (or M1 vs identity)\n'
            '  2   Affine RMS distance (Jenkinson 1999)  [default]\n'
            '      sqrt(r²/5 · Tr(AᵀA) + ||T_d||²),  D = M1 − M2\n'
            '  3   8-corner mean displacement (mm or vox with --vox)\n'
            '  4   Max displacement on sphere of radius r\n'
            '      D = inv(M1) @ M2\n'
            '  5   Determinant of M1 (· M2 when given)\n'
            '  7   Polar decomposition: rotation, shear, scale, translation\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument('lta1', metavar='LTA1',
                   help='First (or only) LTA transform file.')
    p.add_argument('lta2', metavar='LTA2', nargs='?', default=None,
                   help='Second LTA file.  Omit to compare LTA1 against identity.')

    p.add_argument('--dist', type=int, default=2, choices=[1, 2, 3, 4, 5, 7],
                   metavar='{1,2,3,4,5,7}',
                   help='Distance type (default: 2).')
    p.add_argument('--radius', type=float, default=100.0, metavar='MM',
                   help='Sphere / RMS radius in mm (dist 2 and 4, default: 100).')
    p.add_argument('--normdiv', type=float, default=1.0, metavar='FLOAT',
                   help='Divide the final distance by this value (default: 1).')
    p.add_argument('--invert1', action='store_true',
                   help='Invert the first transform before comparison.')
    p.add_argument('--invert2', action='store_true',
                   help='Invert the second transform before comparison.')
    p.add_argument('--vox', action='store_true',
                   help=(
                       'Work in voxel coordinates scaled to mm '
                       '(iso-vox, like FreeSurfer --vox).  '
                       'For dist 3, distances are reported in voxels.'
                   ))

    return p


# ── dist implementations ─────────────────────────────────────────────────────

def _run_dist(ns: argparse.Namespace, lta1: dict, lta2: dict | None) -> None:
    M1, M2 = _diff_matrix(lta1, lta2, ns.vox)

    if ns.dist == 1:
        result = rigid_dist(M1, M2)
        print(f'{result / ns.normdiv}')

    elif ns.dist == 2:
        result = affine_dist(M1, M2, radius=ns.radius)
        print(f'{result / ns.normdiv}')

    elif ns.dist == 3:
        src_info  = lta1['src']
        src_shape = tuple(src_info['volume'])
        src_affine = None if ns.vox else _affine_from_info(src_info)
        if ns.vox:
            # pass the raw V2V (not iso-vox) since corner_diff works in voxels
            src_aff = _affine_from_info(src_info)
            dst_aff = _affine_from_info(lta1['dst'])
            if lta1['type'] == LINEAR_RAS_TO_RAS:
                v2v1 = convert_transform_type(
                    lta1['matrix'], src_aff, dst_aff,
                    from_type=LINEAR_RAS_TO_RAS, to_type=LINEAR_VOX_TO_VOX,
                )
            else:
                v2v1 = lta1['matrix'].copy()
            v2v2 = None
            if lta2 is not None:
                src_aff2 = _affine_from_info(lta2['src'])
                dst_aff2 = _affine_from_info(lta2['dst'])
                if lta2['type'] == LINEAR_RAS_TO_RAS:
                    v2v2 = convert_transform_type(
                        lta2['matrix'], src_aff2, dst_aff2,
                        from_type=LINEAR_RAS_TO_RAS, to_type=LINEAR_VOX_TO_VOX,
                    )
                else:
                    v2v2 = lta2['matrix'].copy()
            result = corner_diff(v2v1, src_shape, M2=v2v2, src_affine=None)
        else:
            result = corner_diff(M1, src_shape, M2=M2, src_affine=src_affine)
        print(f'{result / ns.normdiv}')

    elif ns.dist == 4:
        result = sphere_diff(M1, M2, radius=ns.radius)
        print(f'{result / ns.normdiv}')

    elif ns.dist == 5:
        if M2 is None:
            result = np.linalg.det(M1)
        else:
            result = np.linalg.det(M1 @ M2)
        print(f'{result / ns.normdiv}')

    elif ns.dist == 7:
        M = M1 if M2 is None else M1 @ M2
        d = decompose_transform(M)
        np.set_printoptions(precision=10, suppress=False)
        print('\nDecompose into Rot · Shear · diag(Scales) + Trans:\n')
        print('Rot =')
        print(d['rotation'])
        print(f'\nRotVec   = {d["rot_vec"]}  (rad)')
        print(f'RotAngle = {np.radians(d["rot_angle_deg"]):.6f} rad  '
              f'= {d["rot_angle_deg"]:.6f} deg')
        print('\nShear =')
        print(d['shear'])
        print(f'\nScales   = {d["scales"]}')
        print(f'\nTrans    = {d["translation"]}')
        print(f'AbsTrans = {d["abs_trans"]:.6f} mm')
        print(f'\nDeterminant = {d["determinant"]:.6f}')


# ── entry point ───────────────────────────────────────────────────────────────

def main(args=None) -> None:
    """Entry point for the ``lta-diff`` command."""
    parser = _build_parser()
    ns     = parser.parse_args(args)

    # validate invert2 requires a second LTA
    if ns.invert2 and ns.lta2 is None:
        parser.error('--invert2 requires a second LTA file.')

    # load
    try:
        lta1 = _load(ns.lta1)
    except Exception as e:
        print(f'ERROR: cannot read {ns.lta1}: {e}', file=sys.stderr)
        sys.exit(1)

    lta2 = None
    if ns.lta2 is not None:
        try:
            lta2 = _load(ns.lta2)
        except Exception as e:
            print(f'ERROR: cannot read {ns.lta2}: {e}', file=sys.stderr)
            sys.exit(1)

    # validate src geometry present for corner_diff
    if ns.dist == 3:
        required = ('xras', 'yras', 'zras', 'cras', 'voxelsize', 'volume')
        missing  = [k for k in required if k not in lta1.get('src', {})]
        if missing:
            parser.error(
                f'dist 3 requires src volume info in {ns.lta1} '
                f'(missing: {missing}).'
            )

    # invert
    if ns.invert1:
        lta1 = _invert(lta1)
    if ns.invert2 and lta2 is not None:
        lta2 = _invert(lta2)

    try:
        _run_dist(ns, lta1, lta2)
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)

