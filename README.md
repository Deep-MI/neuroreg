[![PyPI version](https://badge.fury.io/py/nireg.svg)](https://pypi.org/project/nireg/)
# nireg

Nireg is a tool for the robust registration of 3D neuroimaging data (e.g. MRI).
It uses PyTorch's automatic differentiation for gradient-based optimisation and
can run efficiently on a GPU.

The main user-facing tools are:

- **`robreg`** – IRLS-based robust image-to-image registration
  (the current public robust-registration path; analogous to FreeSurfer's `mri_robust_register`)
- **`robreg_gd`** – legacy gradient-descent image-to-image registration
  (kept for comparison and experimentation during early development)
- **`bbreg`** – boundary-based registration using cortical surface meshes
  (analogous to FreeSurfer's `bbregister` / `mri_segreg`)
- **`lta`** – transform comparison / inversion / concatenation utilities

This project is a work-in-progress in an early development stage.

## Installation

```bash
pip install nireg
```

## Command-line interface

### `robreg` — image-to-image registration

Registers a moving image to a reference image using a multi-resolution
IRLS-based robust registration path with Tukey weighting.

This path is currently intended for **same-contrast or very similar-contrast**
image pairs. General cross-sequence / cross-modal registration (for example
plain T2→T1 image-to-image registration without surfaces) is **not** currently
implemented in `robreg`.

```
robreg --mov <moving.nii.gz> --ref <reference.nii.gz> --out <output.lta> [options]
```

Run `robreg -h` for a full argument summary with defaults.

**Required arguments**

| Argument | Description |
|----------|-------------|
| `--mov FILE` | Moving (source) image (NIfTI or MGZ). |
| `--ref FILE` | Reference (target/fixed) image (NIfTI or MGZ). |
| `--out LTA` | Output LTA file for the recovered transformation. |

**Options**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dof {6}` | `6` | Degrees of freedom. The public `robreg` path is currently rigid-only. |
| `--nmax N` | `5` | Maximum number of outer IRLS iterations per pyramid level. |
| `--sat FLOAT` | `6.0` | Tukey biweight saturation threshold. |
| `--symmetric` | off | Use symmetric halfway-space registration. |
| `--noinit` | off | Skip centroid-based initialization and start from identity (like FreeSurfer `--noinit`). |
| `--mapped FILE` | — | Save the warped moving image. |
| `--outliers FILE` | — | Save an outlier map (`1 - Tukey weights`). |
| `--verbose` | off | Enable INFO-level logging. |
| `--debug` | off | Enable DEBUG-level logging. |

**Example**

```bash
robreg --mov T1_repeat.nii.gz --ref T1_baseline.mgz --out T1_repeat_to_T1_baseline.lta --symmetric --verbose
```

---

### `robreg_gd` — legacy image-to-image registration

This is the older gradient-descent registration path. It is still available
for comparison while the package architecture stabilizes, but it is no longer
the default/public `robreg` implementation.

Like the public `robreg` path, this legacy image-to-image method is currently
meant for **same-contrast or similar-contrast** registration, not general
cross-sequence T2→T1 image-only alignment.

```
robreg_gd --mov <moving.nii.gz> --ref <reference.nii.gz> --out <output.lta> [options]
```

**Options**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dof {3,6,9,12}` | `6` | Degrees of freedom: 3=translation, 6=rigid, 9=rigid+scale, 12=affine. |
| `--n_iters N` | auto | Maximum optimisation iterations per pyramid level. |
| `--noinit` | off | Skip centroid-based initialization and start from identity. |
| `--device DEVICE` | `cpu` | PyTorch device, e.g. `cpu` or `cuda`. |
| `--verbose` | off | Enable INFO-level logging. |
| `--debug` | off | Enable DEBUG-level logging. |

**Example**

```bash
robreg_gd --mov T1_repeat.nii.gz --ref T1_baseline.mgz --out T1_repeat_to_T1_legacy.lta --dof 6 --verbose
```

---

### `bbreg` — boundary-based registration

Registers a moving image to a T1 anatomy using cortical surface meshes
(white matter / grey matter boundary). The tissue contrast direction
(T1-like: WM > GM, or T2-like: GM > WM) is auto-detected from the image
when not specified.

Surface input can be provided in three ways:

**Mode A — FreeSurfer / FastSurfer subject directory** (recommended)

```
bbreg --mov <moving.nii.gz> --subject_dir <subject_dir> --out <output.lta> [options]
```

The subject directory must contain `surf/lh.white`, `surf/rh.white`, and
`mri/orig.mgz` (standard FreeSurfer / FastSurfer output layout).

**Mode B — explicit surface files**

```
bbreg --mov <moving.nii.gz> --lh_surf <lh.white> --rh_surf <rh.white> \
      --ref <T1.mgz> --out <output.lta> [options]
```

**Mode C — segmentation file (no pre-built surfaces needed)**

```
bbreg --mov <moving.nii.gz> --seg <aparc+aseg.mgz> --out <output.lta> [options]
```

The WM/GM boundary surface is extracted on-the-fly from a parcellation or
aseg file (e.g. `aparc+aseg.mgz`, `aseg.mgz`, or NIfTI) via marching cubes.
The segmentation header provides the reference geometry; `--ref` must *not* be
specified in this mode.

**Required arguments**

| Argument | Description |
|----------|-------------|
| `--mov FILE` | Moving image to register (NIfTI or MGZ). |
| `--out LTA` | Output LTA file for the recovered transformation. |
| `--subject_dir DIR` | *(Mode A)* Subject directory with surfaces and `mri/orig.mgz`. |
| `--lh_surf / --rh_surf FILE` | *(Mode B)* Explicit left / right white surface files. |
| `--ref FILE` | *(Mode B)* Reference T1 image (required with explicit surfaces). |
| `--seg FILE` | *(Mode C)* Parcellation / aseg file; surfaces extracted automatically. |

**Options**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dof {6,9,12}` | `6` | Degrees of freedom: 6=rigid, 9=rigid+scale, 12=affine. |
| `--contrast {t1,t2}` | auto | Tissue contrast: `t1` (WM>GM) or `t2` (GM>WM). Auto-detected if omitted. |
| `--cost {contrast,gradient,both}` | `contrast` | BBR cost term. |
| `--wm_proj_abs MM` | `1.4` | WM projection depth in mm. |
| `--gm_proj_frac FRAC` | `0.5` | GM projection as fraction of cortical thickness. |
| `--slope SLOPE` | `0.5` | Slope of the BBR sigmoid cost function. |
| `--gradient_weight W` | `0.0` | Weight for gradient cost term when `--cost=both`. |
| `--n_iters N` | `500` | Number of RMSprop optimisation iterations. |
| `--lr LR` | `0.005` | Optimiser learning rate. |
| `--subsample N` | `2` | Use every N-th surface vertex (1 = all). |
| `--lh_thickness / --rh_thickness FILE` | — | *(Mode B)* Cortical thickness files for GM projection. |
| `--seg_smooth_sigma SIGMA` | `0.5` | *(Mode C)* Gaussian pre-blur sigma (voxels) before marching cubes. |
| `--seg_mc_level LEVEL` | `0.45` | *(Mode C)* Marching-cubes iso-level. |
| `--seg_smooth_iters N` | `50` | *(Mode C)* Taubin smoothing iterations after marching cubes. |
| `--init_lta FILE` | — | Initialise from an existing LTA transform (e.g. from a prior `robreg` run or a previous `bbreg` pass). |
| `--device DEVICE` | `cpu` | PyTorch device, e.g. `cpu` or `cuda`. |
| `--verbose` | off | Enable INFO-level logging. |
| `--debug` | off | Enable DEBUG-level logging. |

**Examples**

```bash
# Mode A: register fMRI to T1 using a FastSurfer subject directory
bbreg --mov fMRI.nii.gz --subject_dir /data/subjects/sub-01 --out fMRI_to_T1.lta

# Mode B: register T2 to T1 with explicit surfaces and thickness files
bbreg --mov T2.nii.gz \
      --lh_surf /data/subjects/sub-01/surf/lh.white \
      --rh_surf /data/subjects/sub-01/surf/rh.white \
      --lh_thickness /data/subjects/sub-01/surf/lh.thickness \
      --rh_thickness /data/subjects/sub-01/surf/rh.thickness \
      --ref /data/subjects/sub-01/mri/orig.mgz \
      --out T2_to_T1.lta --contrast t2

# Mode C: register fMRI using a segmentation (no surface files required)
bbreg --mov fMRI.nii.gz --seg /data/subjects/sub-01/mri/aparc+aseg.mgz \
      --out fMRI_to_T1.lta
```

Run `bbreg -h` for a full argument summary with defaults.

---

### `lta` — LTA transform utilities

Unified command for manipulating FreeSurfer LTA transform files with three
subcommands: `diff`, `invert`, and `concat`.

```
lta diff    LTA1 [LTA2] [options]    # Compare transforms
lta invert  INPUT OUTPUT              # Invert a transform
lta concat  LTA1 LTA2 OUTPUT          # Concatenate two transforms
```

---

#### `lta diff` — Compare transforms

Computes distance metrics between two LTA transforms, or between a single
transform and identity. Replicates the functionality of FreeSurfer's `lta_diff`.

**Usage**

```
lta diff LTA1 [LTA2] [--dist {1,2,3,4,5,7}] [--radius MM] [--normdiv FLOAT]
                     [--invert1] [--invert2]
```

**Distance types**

| `--dist` | Metric |
|----------|--------|
| `1` | Rigid transform distance: √(‖log R_d‖² + ‖T_d‖²) |
| `2` | Affine RMS distance (Jenkinson 1999) — **default** |
| `3` | Mean displacement at the 8 corners of the source volume (mm) |
| `4` | Max displacement on a sphere of radius `r` |
| `5` | Determinant of M1 (or M1·M2 when two transforms are given) |
| `7` | Polar decomposition: rotation, shear, scales, translation, determinant |

**Options**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dist {1,2,3,4,5,7}` | `2` | Distance type (see table above). |
| `--radius MM` | `100` | Sphere / RMS radius in mm (used by dist 2 and 4). |
| `--normdiv FLOAT` | `1` | Divide the final distance by this value (must be > 0). |
| `--invert1` | off | Invert LTA1 before comparison. |
| `--invert2` | off | Invert LTA2 before comparison (requires a second LTA). |

**Examples**

```bash
# Affine RMS distance between two registrations (default metric)
lta diff myreg.lta ref.lta

# Rigid distance
lta diff myreg.lta ref.lta --dist 1

# Polar decomposition of a single transform
lta diff myreg.lta --dist 7
```

---

#### `lta invert` — Invert a transform

Inverts an LTA transform and writes the result to a new file. The output is
always stored as LINEAR_RAS_TO_RAS with src and dst geometry swapped.

**Usage**

```
lta invert INPUT OUTPUT
```

**Examples**

```bash
# Invert a registration
lta invert T1_to_MNI.lta MNI_to_T1.lta

# Invert twice to verify round-trip
lta invert fwd.lta inv.lta
lta invert inv.lta fwd_again.lta
```

---

#### `lta concat` — Concatenate transforms

Concatenates two LTA transforms. If `LTA1` maps A→B and `LTA2` maps B→C, the
output maps A→C. Equivalent to FreeSurfer's `mri_concatenate_lta`.

**Usage**

```
lta concat LTA1 LTA2 OUTPUT
```

**Examples**

```bash
# Concatenate fMRI→T1 and T1→MNI to get fMRI→MNI
lta concat fMRI_to_T1.lta T1_to_MNI.lta fMRI_to_MNI.lta

# Chain multiple registrations
lta concat moving_to_intermediate.lta intermediate_to_fixed.lta moving_to_fixed.lta
```

**Notes**

- All metrics are numerically matched to FreeSurfer's `lta_diff` (except for a
  known bug in the C++ single-transform corner metric, which is fixed here).
- When only one LTA is supplied to `diff`, the transform is compared against identity.
- Run `lta --help` or `lta <subcommand> --help` for detailed usage.

## Python API

```python
import nibabel as nib

from nireg import register_pyramid, register_sym, register_surface
from nireg.imreg.robreg_gd import register_pyramid as register_pyramid_gd

# Public robust image-to-image registration (IRLS-backed).
# Accepts file paths or pre-loaded nibabel images.
# Intended for same-/similar-contrast image pairs.
# Returns vox-to-vox when return_v2v=True, or RAS-to-RAS by default.
transform = register_pyramid("T1_repeat.nii.gz", "T1_baseline.mgz", return_v2v=True, dof=6)

# The same public robreg path with pre-loaded nibabel images.
mov_img = nib.load("T1_repeat.nii.gz")
ref_img = nib.load("T1_baseline.mgz")
transform_r2r = register_pyramid(mov_img, ref_img, dof=6)

# Symmetric public robreg convenience wrapper.
transform_sym = register_sym(mov_img, ref_img, dof=6)

# Legacy gradient-descent path (optional / transitional).
transform_gd = register_pyramid_gd("T1_repeat.nii.gz", "T1_baseline.mgz", return_v2v=True, dof=6)

# Surface-based (BBR) registration — Mode A: subject directory
transform, model = register_surface(
    mov="fMRI.nii.gz",
    subject_dir="/data/subjects/sub-01",
    lta_name="fMRI_to_T1.lta",
    dof=6,
)

# Surface-based (BBR) registration — Mode B: explicit surface files
transform, model = register_surface(
    mov="T2.nii.gz",
    lh_surf="/data/subjects/sub-01/surf/lh.white",
    rh_surf="/data/subjects/sub-01/surf/rh.white",
    lh_thickness="/data/subjects/sub-01/surf/lh.thickness",
    rh_thickness="/data/subjects/sub-01/surf/rh.thickness",
    ref="/data/subjects/sub-01/mri/orig.mgz",
    lta_name="T2_to_T1.lta",
    contrast="t2",
)

# Surface-based (BBR) registration — Mode C: segmentation (no surface files needed)
transform, model = register_surface(
    mov="fMRI.nii.gz",
    seg="/data/subjects/sub-01/mri/aparc+aseg.mgz",
    lta_name="fMRI_to_T1.lta",
)
```

## Degrees of freedom

Current DOF support is:

| Command / API path | Supported DOF |
|--------------------|---------------|
| `robreg` / public `register_pyramid()` | `6` only |
| `robreg_gd` / `nireg.imreg.robreg_gd.register_pyramid()` | `3`, `6`, `9`, `12` |
| `bbreg` | `6`, `9`, `12` |

The public `robreg` path is intentionally rigid-only for now because it tracks
the current IRLS implementation.

Also note that the current image-based `robreg` / `robreg_gd` costs are aimed
at same-/similar-contrast registration. For cross-sequence alignment such as
T2→T1, the currently supported option is `bbreg` when a segmentation or
surfaces are available.

## API Documentation

The API Documentation can be found at https://deep-mi.org/nireg .

## References

If you use this software for a publication please cite:

- Reuter, Rosas, Fischl (2010).
  Highly accurate inverse consistent registration: a robust approach.
  NeuroImage 53(4):1181-1196.
  https://doi.org/10.1016/j.neuroimage.2010.07.040

- Reuter, Schmansky, Rosas, Fischl (2012).
  Within-subject template estimation for unbiased longitudinal image analysis.
  NeuroImage 61(4):1402-1418.
  https://doi.org/10.1016/j.neuroimage.2012.02.084

- Reuter, Fischl (2011).
  Avoiding asymmetry-induced bias in longitudinal image processing.
  NeuroImage 57(1):19-21.
  https://doi.org/10.1016/j.neuroimage.2011.02.076

If you use `bbreg` specifically, please also cite:

- Greve, Fischl (2009).
  Accurate and robust brain image alignment using boundary-based registration.
  NeuroImage 48(1):63-72.
  https://doi.org/10.1016/j.neuroimage.2009.06.060

We invite you to check out our lab webpage at https://deep-mi.org
