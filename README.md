[![PyPI version](https://badge.fury.io/py/neuroreg.svg)](https://pypi.org/project/neuroreg/)

# NeuroReg

NeuroReg is a package for the robust registration of 3D neuroimaging data (e.g. MRI).
It supports both image-to-image (within and cross modal) as well as image to
landmark (WM-surface) or segmentation-based registration, with a focus on high accuracy
and speed.

The main user-facing tools are:

- **`robreg`** – Highly accurate and robust same modality image-to-image registration
  (IRLS based, analogous to FreeSurfer's `mri_robust_register`)
- **`coreg`** – Image-to-image cross-modal registration
  (Powell-based by default, analogous to FreeSurfer/SPM `mri_coreg`)
- **`bbreg`** – boundary-based registration using cortical WM surfaces or segmentations
  (extending FreeSurfer's `bbregister`)
- **`segreg`** – segmentation-based registration via label centroids
  (rigid/affine, including atlas-centroid and upright/self-flip modes)
- **`lta`** – transform comparison, inversion, concatenation, and conversion utilities

This project is a work-in-progress in an early development stage. It is developed by
the creator of FreeSurfer's `mri_robust_register` as an efficient pure Python
replacement (with GPU support) and cross-modal extensions to support all your medical
imaging registration needs. If you find it
useful for a publication, please cite the relevant papers (see [References](#References)).

## Installation

```bash
pip install neuroreg
```

## Command-line interface

### `robreg` — image-to-image registration

Registers a moving image to a reference image using a multi-resolution
IRLS-based robust registration path with Tukey weighting.

This path is currently intended for **same-contrast or very similar-contrast**
image pairs.
On Apple MPS, the current public IRLS path falls back to CPU with a warning due
to lack of double precision support on MPS.

```
robreg --mov <moving.nii.gz> --ref <reference.nii.gz> --out <output.lta> [options]
```

Run `robreg -h` for a full argument summary with defaults.

**Required arguments**

| Argument     | Description                                       |
|--------------|---------------------------------------------------|
| `--mov FILE` | Moving (source) image (NIfTI or MGZ).             |
| `--ref FILE` | Reference (target/fixed) image (NIfTI or MGZ).    |
| `--out LTA`  | Output LTA file for the recovered transformation. |

**Options**

| Argument          | Default | Description                                                                                                        |
|-------------------|---------|--------------------------------------------------------------------------------------------------------------------|
| `--dof {6}`       | `6`     | Degrees of freedom. The public `robreg` path is currently rigid-only.                                              |
| `--nmax N`        | `5`     | Maximum number of outer IRLS iterations per pyramid level.                                                         |
| `--sat FLOAT`     | `6.0`   | Tukey biweight saturation threshold.                                                                               |
| `--nosym`         | off     | Disable symmetric halfway-space registration and run directed registration. Symmetric registration is the default. |
| `--init-header`   | off     | Use header alignment only.                                                                                         |
| `--init-centroid` | default | Initialize by aligning intensity centroids in RAS.                                                                 |
| `--init-center`   | off     | Initialize by aligning geometric image centers in RAS.                                                             |
| `--mapped FILE`   | —       | Save the warped moving image.                                                                                      |
| `--outliers FILE` | —       | Save an outlier map (`1 - Tukey weights`).                                                                         |
| `--verbose`       | off     | Enable INFO-level logging.                                                                                         |
| `--debug`         | off     | Enable DEBUG-level logging.                                                                                        |

**Example**

```bash
robreg --mov T1_repeat.nii.gz --ref T1_baseline.mgz --out T1_repeat_to_T1_baseline.lta --verbose
```

---

### `coreg` — image-based cross-modal registration

This is the package's image-to-image registration path for
**cross-sequence / cross-modal** alignment when no surfaces or segmentation
are available. By default it runs a Powell-based MRI_coreg-style pipeline
(coarse brute-force search plus Powell refinement), similar in spirit to
FreeSurfer/SPM `mri_coreg`; the older PyTorch gradient-descent path remains
available via `--method gd`.

```
coreg --mov <moving.nii.gz> --ref <reference.nii.gz> --out <output.lta> [options]
```

**Options**

| Argument                 | Default  | Description                                                                         |
|--------------------------|----------|-------------------------------------------------------------------------------------|
| `--dof {3,6,9,12}`       | `6`      | Degrees of freedom: 3=translation, 6=rigid, 9=rigid+scale, 12=affine.               |
| `--method {powell,gd}`   | `powell` | Registration backend: Powell by default, or legacy gradient descent.                |
| `--n_iters N`            | auto     | Uniform optimisation iterations per executed pyramid level.                         |
| `--level-iters A,B,...`  | —        | Comma-separated coarse-to-fine per-level iteration schedule.                        |
| `--lr LR`                | auto     | Optimizer step size for the GD path.                                                |
| `--min-voxels N`         | `16`     | Minimum pyramid level size.                                                         |
| `--max-voxels N`         | full     | Largest allowed dimension of the finest processed pyramid level.                    |
| `--nosym`                | off      | Disable symmetric halfway-space registration for the GD path.                       |
| `--init-header`          | off      | Use header alignment only.                                                          |
| `--init-centroid`        | off      | Initialize by aligning intensity centroids in RAS.                                  |
| `--init-center`          | default  | Initialize by aligning geometric image centers in RAS.                              |
| `--isotropic`            | off      | Enable shared isotropic preprocessing before the GD pyramid.                        |
| `--device DEVICE`        | `cpu`    | Torch device string, e.g. `cpu`, `cuda`, `mps`, or `gpu`. Powell falls back to CPU. |
| `--powell-brute-limit`   | `30.0`   | Initial search half-width for the Powell brute-force stage.                         |
| `--powell-brute-iters`   | `1`      | Number of coarse-to-fine passes in the Powell brute-force stage.                    |
| `--powell-brute-samples` | `30`     | Samples per dimension in the Powell brute-force stage.                              |
| `--powell-maxiter`       | `4`      | Maximum Powell iterations in the refinement stage.                                  |
| `--powell-sep`           | `4`      | Sampling spacing for the Powell MRI_coreg-style evaluator.                          |
| `--verbose`              | off      | Enable INFO-level logging.                                                          |
| `--debug`                | off      | Enable DEBUG-level logging.                                                         |

**Example**

```bash
coreg --mov T2.nii.gz --ref T1.mgz --out T2_to_T1.lta --dof 6 --verbose
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
The segmentation header provides the reference geometry for the BBR stage.
An optional `--ref` can still be supplied in this mode to drive the coarse
Powell-based image NMI prealignment step.

**Required arguments**

| Argument                     | Description                                                            |
|------------------------------|------------------------------------------------------------------------|
| `--mov FILE`                 | Moving image to register (NIfTI or MGZ).                               |
| `--out LTA`                  | Output LTA file for the recovered transformation.                      |
| `--subject_dir DIR`          | *(Mode A)* Subject directory with surfaces and `mri/orig.mgz`.         |
| `--lh_surf / --rh_surf FILE` | *(Mode B)* Explicit left / right white surface files.                  |
| `--ref FILE`                 | *(Mode B)* Reference T1 image (required with explicit surfaces).       |
| `--seg FILE`                 | *(Mode C)* Parcellation / aseg file; surfaces extracted automatically. |

**Options**

| Argument                               | Default    | Description                                                              |
|----------------------------------------|------------|--------------------------------------------------------------------------|
| `--dof {6,9,12}`                       | `6`        | Degrees of freedom: 6=rigid, 9=rigid+scale, 12=affine.                   |
| `--contrast {t1,t2}`                   | auto       | Tissue contrast: `t1` (WM>GM) or `t2` (GM>WM). Auto-detected if omitted. |
| `--cost {contrast,gradient,both}`      | `contrast` | BBR cost term.                                                           |
| `--wm_proj_abs MM`                     | `1.4`      | WM projection depth in mm.                                               |
| `--gm_proj_frac FRAC`                  | `0.5`      | GM projection as fraction of cortical thickness.                         |
| `--slope SLOPE`                        | `0.5`      | Slope of the BBR sigmoid cost function.                                  |
| `--gradient_weight W`                  | `0.0`      | Weight for gradient cost term when `--cost=both`.                        |
| `--n_iters N`                          | `200`      | Number of RMSprop optimisation iterations.                               |
| `--lr LR`                              | `0.005`    | Optimiser learning rate.                                                 |
| `--subsample N`                        | `2`        | Use every N-th surface vertex (1 = all).                                 |
| `--lh_thickness / --rh_thickness FILE` | —          | *(Mode B)* Cortical thickness files for GM projection.                   |
| `--seg_smooth_sigma SIGMA`             | `0.5`      | *(Mode C)* Gaussian pre-blur sigma (voxels) before marching cubes.       |
| `--seg_mc_level LEVEL`                 | `0.45`     | *(Mode C)* Marching-cubes iso-level.                                     |
| `--seg_smooth_iters N`                 | `50`       | *(Mode C)* Taubin smoothing iterations after marching cubes.             |
| `--init-lta FILE`                      | —          | Initialize from an existing LTA transform.                               |
| `--init-header`                        | off        | Skip coarse NMI prealignment and rely on header geometry only.           |
| `--no-coreg-ref-mask`                  | off        | Disable the default aparc+aseg/aseg mask during coarse NMI prealignment. |
| `--device DEVICE`                      | `cpu`      | Torch device string, e.g. `cpu`, `cuda`, `mps`, or `gpu`.                |
| `--verbose`                            | off        | Enable INFO-level logging.                                               |
| `--debug`                              | off        | Enable DEBUG-level logging.                                              |

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

### `segreg` — segmentation-based registration

Registers a moving segmentation by aligning label centroids to another
segmentation, to bundled atlas centroids such as `fsaverage`, or to a
left-right flipped self target for upright/midspace workflows.

`segreg` can also write mapped outputs while the package does not yet have a
separate transform-application CLI. If `--movimg` is provided, `--mapmov` and
`--mapmovhdr` map that intensity image with the recovered transform; otherwise
those outputs map the moving segmentation itself.

```
segreg --mov <moving_seg.mgz> (--ref <ref_seg.mgz> | --ref-centroids <centroids.json> | --atlas fsaverage | --flipped) [options]
```

**Common outputs**

- `--lta FILE` writes the recovered transform as an LTA.
- `--movimg FILE` provides a separate intensity image for `--mapmov` or `--mapmovhdr`.
- `--mapmov FILE` writes a resliced mapped output. It maps `--movimg` when provided; otherwise it reslices the moving segmentation itself.
- `--mapmovhdr FILE` writes a header-only mapped output. It maps `--movimg` when provided; otherwise it remaps the moving segmentation itself.

**Examples**

```bash
# Register a subject segmentation to another segmentation
segreg --mov sub-01/aparc+aseg.mgz --ref sub-02/aparc+aseg.mgz \
       --lta sub01_to_sub02.lta

# Register a segmentation, then map a separate intensity image with the same transform
segreg --mov subj/mri/aparc+aseg.mgz --movimg subj/mri/orig.mgz --atlas fsaverage \
       --lta subj_to_fsaverage.lta --mapmov orig_in_fsaverage_space.mgz

# If --movimg is omitted, --mapmov maps the segmentation itself
segreg --mov subj/mri/aparc+aseg.mgz --atlas fsaverage \
       --mapmov aparc_aseg_in_fsaverage_space.mgz

# Export centroid JSON only
segreg --mov subj/mri/aparc+aseg.mgz --atlas fsaverage \
       --write-mov-centroids subj_centroids.json --write-ref-centroids fsaverage_centroids.json
```

Run `segreg -h` for a full argument summary with defaults.

---

### `lta` — LTA and linear-transform utilities

Unified command for manipulating FreeSurfer-adjacent linear transforms with four
subcommands: `diff`, `invert`, `concat`, and `convert`.

```
lta diff    LTA1 [LTA2] [options]                   # Compare transforms
lta invert  INPUT OUTPUT                            # Invert a transform
lta concat  LTA1 LTA2 OUTPUT                        # Concatenate two transforms
lta convert INPUT OUTPUT [--src-img SRC --dst-img DST]  # Convert formats
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

| `--dist` | Metric                                                                 |
|----------|------------------------------------------------------------------------|
| `1`      | Rigid transform distance: √(‖log R_d‖² + ‖T_d‖²)                       |
| `2`      | Affine RMS distance (Jenkinson 1999) — **default**                     |
| `3`      | Mean displacement at the 8 corners of the source volume (mm)           |
| `4`      | Max displacement on a sphere of radius `r`                             |
| `5`      | Determinant of M1 (or M1·M2 when two transforms are given)             |
| `7`      | Polar decomposition: rotation, shear, scales, translation, determinant |

**Options**

| Argument               | Default | Description                                            |
|------------------------|---------|--------------------------------------------------------|
| `--dist {1,2,3,4,5,7}` | `2`     | Distance type (see table above).                       |
| `--radius MM`          | `100`   | Sphere / RMS radius in mm (used by dist 2 and 4).      |
| `--normdiv FLOAT`      | `1`     | Divide the final distance by this value (must be > 0). |
| `--invert1`            | off     | Invert LTA1 before comparison.                         |
| `--invert2`            | off     | Invert LTA2 before comparison (requires a second LTA). |

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

---

#### `lta convert` — Convert between transform formats

Converts between `.lta`, `.xfm`, volumetric tkregister `.dat`/`.reg`, FSL `.mat`/`.fslmat`, ITK/ANTs 3D affine text transforms, and NiftyReg affine text matrices by normalizing through an internal RAS-to-RAS `LTA`.

**Usage**

```
lta convert INPUT OUTPUT [--in-format {lta,xfm,fsl,regdat,itk,niftyreg}]
                         [--out-format {lta,xfm,fsl,regdat,itk,niftyreg}]
                         [--src-img SRC] [--dst-img DST]
                         [--out-type {ras2ras,vox2vox}]
                         [--subject SUBJECT] [--fscale FSCALE]
                         [--float2int {tkregister,round,floor}]
```

**Supported formats**

- `.lta` — FreeSurfer Linear Transform Array
- `.xfm` — MNI/MINC linear transform
- `.dat` — old tkregister volumetric `register.dat` format
- `.mat` / `.fslmat` — FSL FLIRT affine matrix
- `.tfm` — ITK/ANTs 3D affine text transform
- `.niftyreg.txt` — NiftyReg 3D affine text matrix

**Notes**

- Reading `register.dat` requires both `--src-img` and `--dst-img`, because the
  stored matrix is defined in tkregister coordinates rather than scanner RAS.
- Reading FSL `.mat` / `.fslmat` also requires both `--src-img` and `--dst-img`,
  because the stored matrix is defined in FSL voxel conventions rather than
  scanner RAS.
- Reading `.xfm`, ITK/ANTs text affines, or NiftyReg text matrices without
  images still preserves the transform matrix, but the resulting LTA geometry
  blocks stay marked as `valid=0` unless you provide `--src-img` and `--dst-img`.
- Use `--in-format` / `--out-format` for ambiguous text filenames such as `.txt`.
- ITK/ANTs support currently targets 3D affine text transforms only, not binary or
  composite transform files.
- NiftyReg stores the inverse target-to-source RAS matrix in the file, matching
  FreeSurfer's `lta_convert` handling.
- `--subject`, `--fscale`, and `--float2int` apply when writing `register.dat`.
- `--out-type` applies when writing `.lta` output.

**Examples**

```bash
# XFM -> LTA with explicit image geometry
lta convert talairach.xfm talairach.lta --src-img mov.mgz --dst-img ref.mgz

# LTA -> XFM
lta convert sub01_to_mni.lta sub01_to_mni.xfm

# LTA -> tkregister register.dat
lta convert bold_to_orig.lta bold_to_orig.dat --subject sub-01 --fscale 0.1

# tkregister register.dat -> LTA
lta convert bold_to_orig.dat bold_to_orig.lta --src-img bold.nii.gz --dst-img orig.mgz

# LTA -> FSL FLIRT matrix
lta convert bold_to_orig.lta bold_to_orig.mat

# FSL FLIRT matrix -> LTA
lta convert bold_to_orig.mat bold_to_orig_from_fsl.lta --src-img bold.nii.gz --dst-img orig.mgz

# LTA -> ITK/ANTs text affine
lta convert bold_to_orig.lta bold_to_orig.tfm

# ITK/ANTs text affine (.txt requires explicit format override)
lta convert bold_to_orig.txt bold_to_orig_from_itk.lta --in-format itk --src-img bold.nii.gz --dst-img orig.mgz

# LTA -> NiftyReg affine text matrix (.txt requires explicit format override)
lta convert bold_to_orig.lta bold_to_orig.txt --out-format niftyreg

# NiftyReg affine text matrix -> LTA
lta convert bold_to_orig.txt bold_to_orig_from_niftyreg.lta --in-format niftyreg
```

**General notes**

- All metrics are numerically matched to FreeSurfer's `lta_diff` (except for a
  known bug in the C++ single-transform corner metric, which is fixed here).
- When only one LTA is supplied to `diff`, the transform is compared against identity.
- Run `lta --help` or `lta <subcommand> --help` for detailed usage.

## Python API

```python
import nibabel as nib

from neuroreg import bbreg, coreg, robreg

# Public robust image-to-image registration (IRLS-backed).
# Intended for same-/similar-contrast pairs; symmetric mode is the default.
# On Apple MPS, this path currently warns and falls back to CPU.
transform_robreg = robreg("T1_repeat.nii.gz", "T1_baseline.mgz")

# The same public robreg path with pre-loaded nibabel images.
mov_img = nib.load("T1_repeat.nii.gz")
ref_img = nib.load("T1_baseline.mgz")
transform_robreg_loaded = robreg(mov_img, ref_img)

# Image-based cross-modal registration path for cases where no
# white-matter surface or segmentation is available. By default this
# uses the Powell-based MRI_coreg-style path.
transform_coreg = coreg("T2.nii.gz", "T1.mgz")

# Surface-based (BBR) registration — Mode A: subject directory
transform_bbreg = bbreg(
    mov="fMRI.nii.gz",
    subject_dir="/data/subjects/sub-01",
    lta_name="fMRI_to_T1.lta",
    dof=6,
)

# Surface-based (BBR) registration — Mode B: explicit surface files
transform_bbreg = bbreg(
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
transform_bbreg = bbreg(
    mov="fMRI.nii.gz",
    seg="/data/subjects/sub-01/mri/aparc+aseg.mgz",
    lta_name="fMRI_to_T1.lta",
)

# For debugging or inspection, request the fitted model explicitly.
transform_bbreg, model_bbreg = bbreg(
    mov="fMRI.nii.gz",
    subject_dir="/data/subjects/sub-01",
    return_model=True,
)
```

## Degrees of freedom

Current DOF support is:

| Command / API path           | Supported DOF       |
|------------------------------|---------------------|
| `robreg` / public `robreg()` | `6` only            |
| `coreg` / public `coreg()`   | `3`, `6`, `9`, `12` |
| `bbreg` / public `bbreg()`   | `6`, `9`, `12`      |

The public `robreg` path is intentionally rigid-only for now because it tracks
the current IRLS implementation.

Also note that the current public `robreg` path is aimed at same-/similar-contrast
registration. For cross-sequence alignment such as T2→T1, the current options are
`coreg` (Powell-based by default, with legacy GD still available via `method="gd"`)
or `bbreg` when a segmentation or surfaces are available.

## API Documentation

The API Documentation can be found at [https://deep-mi.org/neuroreg](https://deep-mi.org/neuroreg).

## References

If you use this software for a publication please cite:

- Reuter, Rosas, Fischl (2010).
  Highly accurate inverse consistent registration: a robust approach.
  NeuroImage 53(4):1181-1196.
  [https://doi.org/10.1016/j.neuroimage.2010.07.040](https://doi.org/10.1016/j.neuroimage.2010.07.040)

- Reuter, Schmansky, Rosas, Fischl (2012).
  Within-subject template estimation for unbiased longitudinal image analysis.
  NeuroImage 61(4):1402-1418.
  [https://doi.org/10.1016/j.neuroimage.2012.02.084](https://doi.org/10.1016/j.neuroimage.2012.02.084)

- Reuter, Fischl (2011).
  Avoiding asymmetry-induced bias in longitudinal image processing.
  NeuroImage 57(1):19-21.
  [https://doi.org/10.1016/j.neuroimage.2011.02.076](https://doi.org/10.1016/j.neuroimage.2011.02.076)

If you use `bbreg` specifically, please also cite:

- Greve, Fischl (2009).
  Accurate and robust brain image alignment using boundary-based registration.
  NeuroImage 48(1):63-72.
  [https://doi.org/10.1016/j.neuroimage.2009.06.060](https://doi.org/10.1016/j.neuroimage.2009.06.060)

We invite you to check out our lab webpage at [https://deep-mi.org](https://deep-mi.org)
