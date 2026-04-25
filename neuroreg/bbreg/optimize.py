"""Optimization model for surface-based registration."""

import logging
from typing import Literal

import torch
import torch.nn as nn

from .cost import bbr_contrast_cost, detect_contrast, gradient_magnitude_cost
from .projection import compute_vertex_normals, create_wm_gm_surfaces
from .sampling import (
    compute_volume_gradient,
    sample_gradient_at_vertices,
    sample_volume_at_vertices,
)

logger = logging.getLogger(__name__)


class BBRModel(nn.Module):
    """PyTorch model for surface-based boundary-based registration (BBR).

    Implements the BBR cost function as used in FreeSurfer's bbregister /
    mri_segreg.  Intensities are sampled at two sets of vertices projected
    along cortical surface normals:

    * **WM sample** – ``wm_proj_abs`` mm inward from the white surface.
    * **GM sample** – ``gm_proj_frac`` × cortical-thickness mm outward.

    The optimisable transform is parameterised as Euler-angle rotations plus
    translations in scanner-RAS space (trg_RAS → mov_RAS).

    Notes
    -----
    Surface vertices are stored in target-image tkRAS space. On every forward
    pass the full coordinate chain is assembled::

        trg_tkRAS  ->[trg_tkras2ras]->  trg_RAS
                   ->[ras2ras param]->  mov_RAS
                   ->[inv(mov_affine)]-> mov_vox -> grid_sample

    Parameters
    ----------
    moving_volume : torch.Tensor, shape (H, W, D)
        Moving (source) image data.
    lh_white_vertices : torch.Tensor, shape (N, 3), optional
        Left-hemisphere white-surface vertices in target tkRAS space.
    lh_faces : torch.Tensor, shape (F, 3), optional
        Left-hemisphere triangle face indices.
    rh_white_vertices : torch.Tensor, shape (N, 3), optional
        Right-hemisphere white-surface vertices in target tkRAS space.
    rh_faces : torch.Tensor, shape (F, 3), optional
        Right-hemisphere triangle face indices.
    lh_thickness : torch.Tensor, shape (N,), optional
        Left-hemisphere cortical thickness values (mm).
    rh_thickness : torch.Tensor, shape (N,), optional
        Right-hemisphere cortical thickness values (mm).
    lh_cortex_mask : torch.Tensor of bool, shape (N,), optional
        Boolean mask for left-hemisphere cortex vertices.  Vertices where the
        mask is ``False`` (medial wall, brainstem boundary, corpus callosum
        border, etc.) are excluded from the BBR cost, mirroring the behaviour
        of ``lh.cortex.label`` in bbregister.  When ``None`` all vertices are
        used.
    rh_cortex_mask : torch.Tensor of bool, shape (N,), optional
        Same as *lh_cortex_mask* for the right hemisphere.
    trg_tkras2ras : torch.Tensor, shape (4, 4)
        **Required.** Matrix mapping target tkRAS → scanner RAS.
        Computed as ``trg_affine @ inv(trg_vox2tkras)``.
    mov_affine : torch.Tensor, shape (4, 4)
        **Required.** Moving-image voxel-to-RAS affine (nibabel ``img.affine``).
        The inverse (RAS → vox) is used internally.
    dof : int
        Degrees of freedom: 6 (rigid), 9 (rigid + scale), 12 (affine).
    init_transform : torch.Tensor, shape (4, 4), optional
        Initial trg_RAS → mov_RAS transform.  Defaults to identity.
    contrast : {'t1', 't2'} or None
        Expected tissue contrast.  ``'t1'``: WM > GM; ``'t2'``: GM > WM.
        If ``None`` (default), the contrast direction is **auto-detected** by
        sampling WM and GM intensities at the surface vertices using the
        initial transform and choosing the direction of the majority signal.
    wm_proj_abs : float
        Absolute projection distance into white matter (mm). Default 1.4 mm.
    gm_proj_frac : float
        Fractional projection into grey matter relative to cortical thickness.
        Used when thickness is available and *gm_proj_abs* is ``None``.
    gm_proj_abs : float, optional
        Absolute projection depth into grey matter (mm).  When set, overrides
        *gm_proj_frac* regardless of whether thickness is available.  When
        ``None`` (default), *gm_proj_frac* × thickness is used if thickness
        is present; otherwise falls back to 1.4 mm.
    slope : float
        Slope of the BBR sigmoid cost function.
    cost_type : {'contrast', 'gradient', 'both'}
        Which cost term(s) to use.
    gradient_weight : float
        Weight for the gradient term when ``cost_type='both'``.
    subsample : int
        Use every *n*-th vertex (1 = all vertices).
    device : str
        PyTorch device string, e.g. ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
            self,
            moving_volume: torch.Tensor,
            lh_white_vertices: torch.Tensor | None = None,
            lh_faces: torch.Tensor | None = None,
            rh_white_vertices: torch.Tensor | None = None,
            rh_faces: torch.Tensor | None = None,
            lh_thickness: torch.Tensor | None = None,
            rh_thickness: torch.Tensor | None = None,
            lh_cortex_mask: torch.Tensor | None = None,
            rh_cortex_mask: torch.Tensor | None = None,
            trg_tkras2ras: torch.Tensor | None = None,
            mov_affine: torch.Tensor | None = None,
            dof: int = 6,
            init_transform: torch.Tensor | None = None,
            contrast: Literal["t1", "t2"] | None = None,
            wm_proj_abs: float = 1.4,
            gm_proj_frac: float = 0.5,
            gm_proj_abs: float | None = None,
            slope: float = 0.5,
            cost_type: Literal["contrast", "gradient", "both"] = "contrast",
            gradient_weight: float = 0.0,
            subsample: int = 1,
            device: str = "cpu",
    ):
        super().__init__()

        if trg_tkras2ras is None:
            raise ValueError("trg_tkras2ras is required. Compute it as: get_tkras2ras(trg_img)")
        if mov_affine is None:
            raise ValueError("mov_affine is required (moving image nibabel affine, i.e. img.affine).")

        self.device = device
        self.dof = dof
        self.contrast = contrast
        self.slope = slope
        self.cost_type = cost_type
        self.gradient_weight = gradient_weight
        self.subsample = subsample

        # Store volume
        self.moving_volume = moving_volume.to(device)

        # trg_tkras2ras : target tkRAS → scanner RAS
        self.trg_tkras2ras = trg_tkras2ras.to(device).float()

        # mov_ras2vox : scanner RAS → moving-image voxel  (inverse of affine)
        self.mov_ras2vox = torch.inverse(mov_affine.to(device).float())

        # Identity placeholder passed to sample_volume_at_vertices when the
        # full coordinate chain is already folded into the reg_matrix argument.
        self._identity = torch.eye(4, device=device, dtype=torch.float32)

        # ---------- optimisable parameters --------------------------------
        if dof == 6:
            self.transform_params = nn.Parameter(torch.zeros(6, dtype=torch.float32, device=device))
        elif dof == 9:
            self.transform_params = nn.Parameter(torch.zeros(9, dtype=torch.float32, device=device))
        elif dof == 12:
            self.transform_params = nn.Parameter(torch.zeros(12, dtype=torch.float32, device=device))
        else:
            raise ValueError(f"Unsupported DOF: {dof}. Must be 6, 9, or 12.")

        if init_transform is not None:
            self._set_params_from_matrix(init_transform)

        # ---------- surface preparation -----------------------------------
        self.use_lh = lh_white_vertices is not None and lh_faces is not None
        self.use_rh = rh_white_vertices is not None and rh_faces is not None

        if not self.use_lh and not self.use_rh:
            raise ValueError("At least one hemisphere (lh or rh) must be provided.")

        if self.use_lh:
            # Move all surface tensors to the target device before any computation.
            lh_white_vertices = lh_white_vertices.to(device=device, dtype=torch.float32)
            lh_faces = lh_faces.to(device=device)
            if lh_thickness is not None:
                lh_thickness = lh_thickness.to(device=device, dtype=torch.float32)
            lh_normals = compute_vertex_normals(lh_white_vertices, lh_faces)
            lh_wm, lh_gm = create_wm_gm_surfaces(
                lh_white_vertices,
                lh_faces,
                lh_normals,
                lh_thickness,
                wm_proj_abs=wm_proj_abs,
                gm_proj_frac=gm_proj_frac,
                gm_proj_abs=gm_proj_abs,
            )
            # Apply cortex label mask first (mimics lh.cortex.label in bbregister)
            if lh_cortex_mask is not None:
                mask = lh_cortex_mask.to(device=device, dtype=torch.bool)
                lh_wm = lh_wm[mask]
                lh_gm = lh_gm[mask]
                lh_normals = lh_normals[mask]
                logger.debug("LH cortex mask: %d / %d vertices retained", int(mask.sum()), lh_white_vertices.shape[0])
            if subsample > 1:
                idx = torch.arange(0, lh_wm.shape[0], subsample, device=device)
                self.lh_wm_vertices = lh_wm[idx]
                self.lh_gm_vertices = lh_gm[idx]
                self.lh_normals = lh_normals[idx]
            else:
                self.lh_wm_vertices = lh_wm
                self.lh_gm_vertices = lh_gm
                self.lh_normals = lh_normals
            logger.debug(
                "LH white surface: %d vertices (subsampled: %d)",
                lh_white_vertices.shape[0],
                self.lh_wm_vertices.shape[0],
            )

        if self.use_rh:
            # Move all surface tensors to the target device before any computation.
            rh_white_vertices = rh_white_vertices.to(device=device, dtype=torch.float32)
            rh_faces = rh_faces.to(device=device)
            if rh_thickness is not None:
                rh_thickness = rh_thickness.to(device=device, dtype=torch.float32)
            rh_normals = compute_vertex_normals(rh_white_vertices, rh_faces)
            rh_wm, rh_gm = create_wm_gm_surfaces(
                rh_white_vertices,
                rh_faces,
                rh_normals,
                rh_thickness,
                wm_proj_abs=wm_proj_abs,
                gm_proj_frac=gm_proj_frac,
                gm_proj_abs=gm_proj_abs,
            )
            # Apply cortex label mask first (mimics rh.cortex.label in bbregister)
            if rh_cortex_mask is not None:
                mask = rh_cortex_mask.to(device=device, dtype=torch.bool)
                rh_wm = rh_wm[mask]
                rh_gm = rh_gm[mask]
                rh_normals = rh_normals[mask]
                logger.debug("RH cortex mask: %d / %d vertices retained", int(mask.sum()), rh_white_vertices.shape[0])
            if subsample > 1:
                idx = torch.arange(0, rh_wm.shape[0], subsample, device=device)
                self.rh_wm_vertices = rh_wm[idx]
                self.rh_gm_vertices = rh_gm[idx]
                self.rh_normals = rh_normals[idx]
            else:
                self.rh_wm_vertices = rh_wm
                self.rh_gm_vertices = rh_gm
                self.rh_normals = rh_normals
            logger.debug(
                "RH white surface: %d vertices (subsampled: %d)",
                rh_white_vertices.shape[0],
                self.rh_wm_vertices.shape[0],
            )

        # ── contrast auto-detection ----------------------------------------
        # If contrast is not specified, sample both hemispheres with the
        # current (initial) transform and determine the majority direction.
        if contrast is None:
            combined = self.mov_ras2vox @ self._params_to_matrix() @ self.trg_tkras2ras
            wm_samples, gm_samples = [], []
            if self.use_lh:
                wm_samples.append(
                    sample_volume_at_vertices(
                        self.moving_volume, self.lh_wm_vertices, self._identity, combined, interpolation="trilinear"
                    )
                )
                gm_samples.append(
                    sample_volume_at_vertices(
                        self.moving_volume, self.lh_gm_vertices, self._identity, combined, interpolation="trilinear"
                    )
                )
            if self.use_rh:
                wm_samples.append(
                    sample_volume_at_vertices(
                        self.moving_volume, self.rh_wm_vertices, self._identity, combined, interpolation="trilinear"
                    )
                )
                gm_samples.append(
                    sample_volume_at_vertices(
                        self.moving_volume, self.rh_gm_vertices, self._identity, combined, interpolation="trilinear"
                    )
                )
            vwm_all = torch.cat(wm_samples)
            vgm_all = torch.cat(gm_samples)
            contrast = detect_contrast(vwm_all, vgm_all)

        self.contrast = contrast
        # +1 for t2 (GM > WM), -1 for t1 (WM > GM)
        self.contrast_sign = -1 if contrast == "t1" else 1

        # ── precompute gradient volume ─────────────────────────────────────
        # moving_volume is constant throughout optimisation — only
        # transform_params change.  Pre-computing the gradient here avoids
        # three full conv3d passes every iteration when cost_type involves
        # the gradient term.
        if self.cost_type in ("gradient", "both"):
            with torch.no_grad():
                self._moving_grad_volume: torch.Tensor = compute_volume_gradient(self.moving_volume).detach()
            logger.debug(
                "Precomputed gradient volume: shape %s",
                list(self._moving_grad_volume.shape),
            )
        else:
            self._moving_grad_volume = None

    # ------------------------------------------------------------------
    def forward(self) -> torch.Tensor:
        """Compute BBR cost for the current transform parameters.

        Returns
        -------
        torch.Tensor
            Scalar cost value.
        """
        ras2ras = self._params_to_matrix()

        total_cost = torch.tensor(0.0, device=self.device)
        n_hemispheres = 0

        if self.use_lh:
            total_cost = total_cost + self._compute_hemisphere_cost(
                self.lh_wm_vertices, self.lh_gm_vertices, self.lh_normals, ras2ras
            )
            n_hemispheres += 1

        if self.use_rh:
            total_cost = total_cost + self._compute_hemisphere_cost(
                self.rh_wm_vertices, self.rh_gm_vertices, self.rh_normals, ras2ras
            )
            n_hemispheres += 1

        return total_cost / n_hemispheres

    # ------------------------------------------------------------------
    def _compute_hemisphere_cost(
            self,
            wm_vertices: torch.Tensor,
            gm_vertices: torch.Tensor,
            normals: torch.Tensor,
            ras2ras: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BBR cost for one hemisphere.

        Assembles the full coordinate chain and samples intensities at WM and
        GM surface vertices, then evaluates the configured cost function.

        Parameters
        ----------
        wm_vertices : torch.Tensor, shape (N, 3)
            White-matter sample points in target tkRAS space.
        gm_vertices : torch.Tensor, shape (N, 3)
            Grey-matter sample points in target tkRAS space.
        normals : torch.Tensor, shape (N, 3)
            Surface normal vectors (used for gradient cost only).
        ras2ras : torch.Tensor, shape (4, 4)
            Current trg_RAS → mov_RAS transform (from optimisable parameters).

        Returns
        -------
        torch.Tensor
            Scalar cost value for this hemisphere.
        """
        # Full chain: trg_tkRAS → trg_RAS → mov_RAS → mov_vox
        combined = self.mov_ras2vox @ ras2ras @ self.trg_tkras2ras

        vwm = sample_volume_at_vertices(
            self.moving_volume, wm_vertices, self._identity, combined, interpolation="trilinear"
        )
        vgm = sample_volume_at_vertices(
            self.moving_volume, gm_vertices, self._identity, combined, interpolation="trilinear"
        )

        if self.cost_type == "contrast":
            return bbr_contrast_cost(vwm, vgm, slope=self.slope, contrast_sign=self.contrast_sign)

        if self.cost_type == "gradient":
            white_vertices = (wm_vertices + gm_vertices) / 2.0
            grad = sample_gradient_at_vertices(
                self.moving_volume,
                white_vertices,
                self._identity,
                combined,
                precomputed_grad=self._moving_grad_volume,
            )
            return gradient_magnitude_cost(grad, normals)

        if self.cost_type == "both":
            cost_c = bbr_contrast_cost(vwm, vgm, slope=self.slope, contrast_sign=self.contrast_sign)
            white_vertices = (wm_vertices + gm_vertices) / 2.0
            grad = sample_gradient_at_vertices(
                self.moving_volume,
                white_vertices,
                self._identity,
                combined,
                precomputed_grad=self._moving_grad_volume,
            )
            cost_g = gradient_magnitude_cost(grad, normals)
            return cost_c + self.gradient_weight * cost_g

        raise ValueError(f"Unknown cost_type: {self.cost_type}")

    # ------------------------------------------------------------------
    def get_transform_matrix(self) -> torch.Tensor:
        """Return the current trg_RAS → mov_RAS transform (4 × 4).

        Returns
        -------
        torch.Tensor, shape (4, 4)
        """
        return self._params_to_matrix()

    # ------------------------------------------------------------------
    def eval_cost_at_ras2ras(self, ras2ras: torch.Tensor) -> float:
        """Evaluate the BBR cost at an arbitrary trg_RAS → mov_RAS transform.

        Useful for comparing cost at our optimised solution against an
        external reference transform (e.g. a bbregister result) without
        altering the model's optimisable parameters.

        Parameters
        ----------
        ras2ras : torch.Tensor, shape (4, 4)
            trg_RAS → mov_RAS transform to evaluate.

        Returns
        -------
        float
            Scalar BBR cost value.
        """
        with torch.no_grad():
            ras2ras = ras2ras.to(device=self.device, dtype=torch.float32)
            total = torch.tensor(0.0, device=self.device)
            n = 0
            if self.use_lh:
                total = total + self._compute_hemisphere_cost(
                    self.lh_wm_vertices, self.lh_gm_vertices, self.lh_normals, ras2ras
                )
                n += 1
            if self.use_rh:
                total = total + self._compute_hemisphere_cost(
                    self.rh_wm_vertices, self.rh_gm_vertices, self.rh_normals, ras2ras
                )
                n += 1
            return float(total / n)

    # ------------------------------------------------------------------
    def _params_to_matrix(self) -> torch.Tensor:
        """Convert optimisable parameters to a 4 × 4 transform matrix.

        The parameterisation uses Euler angles (X → Y → Z) for rotation,
        which is fully differentiable with no conditional branches.

        Returns
        -------
        torch.Tensor, shape (4, 4)
            trg_RAS → mov_RAS transformation matrix.
        """
        params = self.transform_params

        if self.dof == 12:
            mat = torch.eye(4, device=self.device, dtype=params.dtype)
            mat[:3, :] = params.view(3, 4)
            return mat

        mat = torch.eye(4, device=self.device, dtype=params.dtype)
        mat[:3, 3] = params[:3]  # translation

        if self.dof >= 6:
            angles = params[3:6]
            cos = torch.cos(angles)
            sin = torch.sin(angles)
            zero = torch.zeros_like(angles[0])
            one = torch.ones_like(angles[0])

            R_x = torch.stack(
                [
                    one,
                    zero,
                    zero,
                    zero,
                    cos[0],
                    -sin[0],
                    zero,
                    sin[0],
                    cos[0],
                ]
            ).view(3, 3)

            R_y = torch.stack(
                [
                    cos[1],
                    zero,
                    sin[1],
                    zero,
                    one,
                    zero,
                    -sin[1],
                    zero,
                    cos[1],
                ]
            ).view(3, 3)

            R_z = torch.stack(
                [
                    cos[2],
                    -sin[2],
                    zero,
                    sin[2],
                    cos[2],
                    zero,
                    zero,
                    zero,
                    one,
                ]
            ).view(3, 3)

            mat[:3, :3] = R_x @ R_y @ R_z

        if self.dof >= 9:
            scale = torch.exp(params[6:9])
            mat[:3, :3] = mat[:3, :3] * scale.unsqueeze(0)

        return mat

    # ------------------------------------------------------------------
    def _set_params_from_matrix(self, matrix: torch.Tensor) -> None:
        """Initialise optimisable parameters from a 4 × 4 transform matrix.

        Extracts translation and Euler angles (and optionally log-scale) from
        *matrix*.  The rotation decomposition is exact for any rotation but may
        enter the gimbal-lock branch for rotations near ±90° around Y.

        Parameters
        ----------
        matrix : torch.Tensor, shape (4, 4)
            trg_RAS → mov_RAS transformation matrix to decompose.
        """
        with torch.no_grad():
            # Ensure matrix is on the same device/dtype as transform_params
            # before any .copy_() or arithmetic.  Callers (register_surface)
            # may pass a CPU tensor (e.g. torch.eye(4) or torch.from_numpy(...))
            # even when the model is on cuda/mps.
            matrix = matrix.to(device=self.device, dtype=torch.float32)

            if self.dof == 12:
                self.transform_params.copy_(matrix[:3, :].flatten())
                return

            self.transform_params[:3].copy_(matrix[:3, 3])

            if self.dof >= 6:
                R = matrix[:3, :3]
                sy = torch.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

                if sy > 1e-6:
                    rx = torch.atan2(R[2, 1], R[2, 2])
                    ry = torch.atan2(-R[2, 0], sy)
                    rz = torch.atan2(R[1, 0], R[0, 0])
                else:
                    logger.warning("Gimbal lock detected during Euler-angle decomposition.")
                    rx = torch.atan2(-R[1, 2], R[1, 1])
                    ry = torch.atan2(-R[2, 0], sy)
                    rz = torch.tensor(0.0, device=self.device)

                self.transform_params[3:6].copy_(torch.stack([rx, ry, rz]))

            if self.dof >= 9:
                scale = torch.sqrt((R ** 2).sum(dim=0))
                self.transform_params[6:9].copy_(torch.log(scale))
