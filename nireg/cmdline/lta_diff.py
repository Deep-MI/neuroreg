"""Command-line interface for LTA transform comparison (lta-diff).

Analogous to FreeSurfer's ``lta_diff`` utility.
"""

import argparse
import sys

import numpy as np

from nireg.transforms import LTA, decompose_transform


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

def _run_dist(ns: argparse.Namespace, lta1: LTA, lta2: LTA | None) -> None:

    if ns.dist == 1:
        print(f'{lta1.rigid_dist(lta2) / ns.normdiv}')

    elif ns.dist == 2:
        print(f'{lta1.affine_dist(lta2, radius=ns.radius) / ns.normdiv}')

    elif ns.dist == 3:
        print(f'{lta1.corner_dist(lta2, vox=ns.vox) / ns.normdiv}')

    elif ns.dist == 4:
        print(f'{lta1.sphere_dist(lta2, radius=ns.radius) / ns.normdiv}')

    elif ns.dist == 5:
        M1 = lta1.iso_vox() if ns.vox else lta1.r2r()
        M2 = (lta2.iso_vox() if ns.vox else lta2.r2r()) if lta2 is not None else None
        result = float(np.linalg.det(M1 if M2 is None else M1 @ M2))
        print(f'{result / ns.normdiv}')

    elif ns.dist == 7:
        M1 = lta1.iso_vox() if ns.vox else lta1.r2r()
        M  = M1 if lta2 is None else M1 @ (lta2.iso_vox() if ns.vox else lta2.r2r())
        d  = decompose_transform(M)
        with np.printoptions(precision=10, suppress=False):
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

    if ns.invert2 and ns.lta2 is None:
        parser.error('--invert2 requires a second LTA file.')

    try:
        lta1 = LTA.read(ns.lta1)
    except Exception as e:
        print(f'ERROR: cannot read {ns.lta1}: {e}', file=sys.stderr)
        sys.exit(1)

    lta2 = None
    if ns.lta2 is not None:
        try:
            lta2 = LTA.read(ns.lta2)
        except Exception as e:
            print(f'ERROR: cannot read {ns.lta2}: {e}', file=sys.stderr)
            sys.exit(1)

    if ns.dist == 3:
        required = ('xras', 'yras', 'zras', 'cras', 'voxelsize', 'volume')
        missing  = [k for k in required if k not in lta1.src]
        if missing:
            parser.error(
                f'dist 3 requires src volume info in {ns.lta1} '
                f'(missing: {missing}).'
            )

    if ns.invert1:
        lta1 = lta1.invert()
    if ns.invert2 and lta2 is not None:
        lta2 = lta2.invert()

    try:
        _run_dist(ns, lta1, lta2)
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)

