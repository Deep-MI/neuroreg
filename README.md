[![PyPI version](https://badge.fury.io/py/nireg.svg)](https://pypi.org/project/nireg/)
# nireg

Nireg is a tool for the robust registration of 3D neuroimaging data (e.g. MRI).
It uses PyTorch's automatic differentiation for gradient-based optimisation and
can run efficiently on a GPU.

Two registration methods are available:

- **`robreg`** – robust image-to-image registration via a Gaussian pyramid
  (analogous to FreeSurfer's `mri_robust_register`)
- **`bbreg`** – boundary-based registration using cortical surface meshes
  (analogous to FreeSurfer's `bbregister` / `mri_segreg`)

This project is a work-in-progress in an early development stage.

## Installation

```bash
pip install nireg
```

## Command-line interface

### `robreg` — image-to-image registration

Registers a moving image to a reference image using a multi-resolution
Gaussian pyramid with robust cost functions.

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
| `--dof {3,6,9,12}` | `6` | Degrees of freedom: 3=translation, 6=rigid, 9=rigid+scale, 12=affine. |
| `--n_iters N` | auto | Maximum optimisation iterations per pyramid level. |
| `--device DEVICE` | `cpu` | PyTorch device, e.g. `cpu` or `cuda`. |
| `--verbose` | off | Enable INFO-level logging. |
| `--debug` | off | Enable DEBUG-level logging. |

**Example**

```bash
robreg --mov T2.nii.gz --ref T1.mgz --out T2_to_T1.lta --dof 6 --verbose
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

## Python API

```python
from nireg import register_pyramid, register_surface

# Image-to-image registration — pass file paths (or pre-loaded nibabel images).
# Returns a single vox-to-vox Tensor when return_v2v=True, or RAS-to-RAS by default.
transform = register_pyramid("T2.nii.gz", "T1.mgz", return_v2v=True, dof=6)

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

Both `robreg` and `bbreg` support the following `--dof` settings:

| `--dof` | Transform | Parameters |
|---------|-----------|------------|
| `3` | Translation only | 3 |
| `6` | Rigid (translation + rotation) | 6 |
| `9` | Rigid + isotropic scaling | 9 |
| `12` | Fully affine | 12 |

*Note: `--dof 3` is only available for `robreg`.*

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
