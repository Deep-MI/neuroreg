"""Optimization model for surface-based registration."""

import torch
import torch.nn as nn
from typing import Optional, Literal

from ..nn.reg_model import RegModel
from .sampling import sample_volume_at_vertices, sample_gradient_at_vertices
from .cost import bbr_contrast_cost, gradient_magnitude_cost
from .projection import create_wm_gm_surfaces, compute_vertex_normals


class BBRModel(nn.Module):
    """PyTorch model for surface-based boundary registration (BBR)."""

    def __init__(
        self,
        moving_volume: torch.Tensor,
        lh_white_vertices: Optional[torch.Tensor],
        lh_faces: Optional[torch.Tensor],
        rh_white_vertices: Optional[torch.Tensor] = None,
        rh_faces: Optional[torch.Tensor] = None,
        lh_thickness: Optional[torch.Tensor] = None,
        rh_thickness: Optional[torch.Tensor] = None,
        vox2ras_tkr: Optional[torch.Tensor] = None,
        dof: int = 6,
        init_transform: Optional[torch.Tensor] = None,
        contrast: Literal['t1', 't2'] = 't2',
        wm_proj_abs: float = 2.0,
        gm_proj_frac: float = 0.5,
        slope: float = 0.5,
        cost_type: Literal['contrast', 'gradient', 'both'] = 'contrast',
        gradient_weight: float = 0.0,
        subsample: int = 1,
        device: str = 'cpu'
    ):
        super().__init__()

        self.device = device
        self.dof = dof
        self.contrast = contrast
        self.slope = slope
        self.cost_type = cost_type
        self.gradient_weight = gradient_weight
        self.subsample = subsample

        # Store volume and transformation
        self.moving_volume = moving_volume.to(device)
        if vox2ras_tkr is not None:
            self.vox2ras_tkr = vox2ras_tkr.to(device)
        else:
            # Default identity if not provided
            self.vox2ras_tkr = torch.eye(4, device=device, dtype=torch.float32)

        # Initialize transformation parameters directly (RAS-to-RAS space)
        # Parameters: [tx, ty, tz, rx, ry, rz, sx, sy, sz, ...]
        # Rotations are Euler angles (X, Y, Z) like in RegModel
        if dof == 6:
            # Rigid: 3 translations + 3 rotations (Euler angles)
            self.transform_params = torch.nn.Parameter(
                torch.zeros(6, dtype=torch.float32, device=device)
            )
        elif dof == 9:
            # Rigid + isotropic scale: 3 trans + 3 rot + 3 scale
            self.transform_params = torch.nn.Parameter(
                torch.zeros(9, dtype=torch.float32, device=device)
            )
        elif dof == 12:
            # Full affine: 12 parameters (3x4 matrix)
            self.transform_params = torch.nn.Parameter(
                torch.zeros(12, dtype=torch.float32, device=device)
            )
        else:
            raise ValueError(f"Unsupported DOF: {dof}. Must be 6, 9, or 12.")

        # Apply initial transform if provided
        if init_transform is not None:
            self._set_params_from_matrix(init_transform)

        # Prepare surfaces
        self.use_lh = lh_white_vertices is not None
        self.use_rh = rh_white_vertices is not None

        if self.use_lh and lh_white_vertices is not None and lh_faces is not None:
            lh_normals = compute_vertex_normals(lh_white_vertices, lh_faces)
            lh_wm, lh_gm = create_wm_gm_surfaces(
                lh_white_vertices, lh_faces, lh_normals, lh_thickness,
                wm_proj_abs=wm_proj_abs, gm_proj_frac=gm_proj_frac
            )

            if subsample > 1:
                idx = torch.arange(0, lh_wm.shape[0], subsample, device=device)
                self.lh_wm_vertices = lh_wm[idx]
                self.lh_gm_vertices = lh_gm[idx]
                self.lh_normals = lh_normals[idx]
            else:
                self.lh_wm_vertices = lh_wm
                self.lh_gm_vertices = lh_gm
                self.lh_normals = lh_normals

        if self.use_rh and rh_white_vertices is not None and rh_faces is not None:
            rh_normals = compute_vertex_normals(rh_white_vertices, rh_faces)
            rh_wm, rh_gm = create_wm_gm_surfaces(
                rh_white_vertices, rh_faces, rh_normals, rh_thickness,
                wm_proj_abs=wm_proj_abs, gm_proj_frac=gm_proj_frac
            )

            if subsample > 1:
                idx = torch.arange(0, rh_wm.shape[0], subsample, device=device)
                self.rh_wm_vertices = rh_wm[idx]
                self.rh_gm_vertices = rh_gm[idx]
                self.rh_normals = rh_normals[idx]
            else:
                self.rh_wm_vertices = rh_wm
                self.rh_gm_vertices = rh_gm
                self.rh_normals = rh_normals

        self.contrast_sign = -1 if contrast == 't1' else 1

    def forward(self) -> torch.Tensor:
        """Compute BBR cost for current registration parameters."""
        # Get RAS-to-RAS transformation from parameters
        transform_4x4 = self._params_to_matrix()

        total_cost = torch.tensor(0.0, device=self.device)
        n_hemispheres = 0

        if self.use_lh:
            cost_lh = self._compute_hemisphere_cost(
                self.lh_wm_vertices, self.lh_gm_vertices,
                self.lh_normals, transform_4x4
            )
            total_cost = total_cost + cost_lh
            n_hemispheres += 1

        if self.use_rh:
            cost_rh = self._compute_hemisphere_cost(
                self.rh_wm_vertices, self.rh_gm_vertices,
                self.rh_normals, transform_4x4
            )
            total_cost = total_cost + cost_rh
            n_hemispheres += 1

        if n_hemispheres > 0:
            total_cost = total_cost / n_hemispheres

        return total_cost

    def _compute_hemisphere_cost(
        self,
        wm_vertices: torch.Tensor,
        gm_vertices: torch.Tensor,
        normals: torch.Tensor,
        transform: torch.Tensor
    ) -> torch.Tensor:
        """Compute cost for one hemisphere."""
        vwm = sample_volume_at_vertices(
            self.moving_volume, wm_vertices, self.vox2ras_tkr,
            transform, interpolation='trilinear'
        )

        vgm = sample_volume_at_vertices(
            self.moving_volume, gm_vertices, self.vox2ras_tkr,
            transform, interpolation='trilinear'
        )

        if self.cost_type == 'contrast':
            cost = bbr_contrast_cost(
                vwm, vgm, slope=self.slope,
                contrast_sign=self.contrast_sign
            )
        elif self.cost_type == 'gradient':
            white_vertices = (wm_vertices + gm_vertices) / 2.0
            grad = sample_gradient_at_vertices(
                self.moving_volume, white_vertices,
                self.vox2ras_tkr, transform
            )
            cost = gradient_magnitude_cost(grad, normals)
        elif self.cost_type == 'both':
            cost_contrast = bbr_contrast_cost(
                vwm, vgm, slope=self.slope,
                contrast_sign=self.contrast_sign
            )
            white_vertices = (wm_vertices + gm_vertices) / 2.0
            grad = sample_gradient_at_vertices(
                self.moving_volume, white_vertices,
                self.vox2ras_tkr, transform
            )
            cost_grad = gradient_magnitude_cost(grad, normals)
            cost = cost_contrast + self.gradient_weight * cost_grad
        else:
            raise ValueError(f"Unknown cost_type: {self.cost_type}")

        return cost

    def get_transform_matrix(self) -> torch.Tensor:
        """Get current transformation matrix (4x4) in RAS-to-RAS space."""
        return self._params_to_matrix()

    def _params_to_matrix(self) -> torch.Tensor:
        """
        Convert transform parameters to 4x4 RAS-to-RAS transformation matrix.

        Uses Euler angles for rotation (same as RegModel) which has better
        gradient flow properties than axis-angle representation.

        Returns
        -------
        torch.Tensor (4, 4)
            RAS-to-RAS transformation matrix
        """
        params = self.transform_params

        if self.dof == 12:
            # Full affine: reshape 12 params to 3x4, add bottom row
            mat = torch.eye(4, device=self.device, dtype=params.dtype)
            mat[:3, :] = params.view(3, 4)
            return mat

        # Build transform from translation, rotation, and optional scale
        mat = torch.eye(4, device=self.device, dtype=params.dtype)

        # Translation (first 3 params)
        mat[:3, 3] = params[:3]

        if self.dof >= 6:
            # Rotation (params 3-6 as Euler angles X, Y, Z)
            # This is the same approach as RegModel.get_rotation_euler()
            angles = params[3:6]
            cos = torch.cos(angles)
            sin = torch.sin(angles)

            # Build rotation matrices for X, Y, Z axes
            # These are always differentiable (no conditionals!)
            zero = torch.zeros_like(angles[0])
            one = torch.ones_like(angles[0])

            # Rotation around X axis
            R_x = torch.stack([
                one, zero, zero,
                zero, cos[0], -sin[0],
                zero, sin[0], cos[0]
            ], dim=-1).view(3, 3)

            # Rotation around Y axis
            R_y = torch.stack([
                cos[1], zero, sin[1],
                zero, one, zero,
                -sin[1], zero, cos[1]
            ], dim=-1).view(3, 3)

            # Rotation around Z axis
            R_z = torch.stack([
                cos[2], -sin[2], zero,
                sin[2], cos[2], zero,
                zero, zero, one
            ], dim=-1).view(3, 3)

            # Combined rotation: R = Rx * Ry * Rz
            R = R_x @ R_y @ R_z
            mat[:3, :3] = R

        if self.dof >= 9:
            # Isotropic scaling (params 6-9)
            scale = torch.exp(params[6:9])  # Use exp to ensure positive
            mat[:3, :3] = mat[:3, :3] * scale.unsqueeze(0)

        return mat

    def _set_params_from_matrix(self, matrix: torch.Tensor) -> None:
        """
        Initialize parameters from a 4x4 transformation matrix.

        Extracts translation, Euler angles, and optionally scale from the matrix.

        Parameters
        ----------
        matrix : torch.Tensor (4, 4)
            RAS-to-RAS transformation matrix
        """
        with torch.no_grad():
            if self.dof == 12:
                # Full affine: flatten 3x4 to 12 params
                self.transform_params.copy_(matrix[:3, :].flatten())
            else:
                # Extract translation
                self.transform_params[:3].copy_(matrix[:3, 3])

                if self.dof >= 6:
                    # Extract rotation as Euler angles (X, Y, Z)
                    # This is an approximation for small rotations
                    # For better accuracy, could use scipy's rotation decomposition
                    R = matrix[:3, :3]

                    # Extract Euler angles using ZYX convention (reverse of XYZ)
                    # Based on: http://www.staff.city.ac.uk/~sbbh653/publications/euler.pdf

                    # Check for gimbal lock
                    sy = torch.sqrt(R[0, 0]**2 + R[1, 0]**2)

                    if sy > 1e-6:
                        # Not in gimbal lock
                        rx = torch.atan2(R[2, 1], R[2, 2])
                        ry = torch.atan2(-R[2, 0], sy)
                        rz = torch.atan2(R[1, 0], R[0, 0])
                    else:
                        # Gimbal lock case
                        rx = torch.atan2(-R[1, 2], R[1, 1])
                        ry = torch.atan2(-R[2, 0], sy)
                        rz = torch.tensor(0.0, device=self.device)

                    self.transform_params[3:6].copy_(torch.stack([rx, ry, rz]))

                if self.dof >= 9:
                    # Extract scale from rotation matrix (diagonal norm)
                    scale = torch.sqrt((R ** 2).sum(dim=0))
                    self.transform_params[6:9].copy_(torch.log(scale))

    def eval_cost_at_transform(self, transform: torch.Tensor) -> torch.Tensor:
        """
        Evaluate BBR cost at a specific transformation without modifying model weights.

        Parameters
        ----------
        transform : torch.Tensor (4, 4)
            Transformation matrix to evaluate (RAS-to-RAS space)

        Returns
        -------
        torch.Tensor
            BBR cost value (scalar)

        Notes
        -----
        This bypasses the RegModel and directly applies the given transform
        to compute the cost. Useful for debugging and comparing transforms.
        """
        # Extract 3x4 transform
        if transform.shape[0] == 4:
            transform_3x4 = transform[:3, :]
        else:
            transform_3x4 = transform

        transform_3x4 = transform_3x4.to(self.device)

        # Collect all WM and GM vertices
        all_wm_verts = []
        all_gm_verts = []

        if hasattr(self, 'lh_wm_vertices') and self.lh_wm_vertices is not None:
            all_wm_verts.append(self.lh_wm_vertices)
            all_gm_verts.append(self.lh_gm_vertices)

        if hasattr(self, 'rh_wm_vertices') and self.rh_wm_vertices is not None:
            all_wm_verts.append(self.rh_wm_vertices)
            all_gm_verts.append(self.rh_gm_vertices)

        wm_vertices = torch.cat(all_wm_verts, dim=0)
        gm_vertices = torch.cat(all_gm_verts, dim=0)

        # Apply transformation (using the provided transform)
        # Transform: RAS -> RAS, then need to convert to voxel space for sampling
        wm_verts_hom = torch.cat([wm_vertices, torch.ones(wm_vertices.shape[0], 1, device=self.device)], dim=1)
        gm_verts_hom = torch.cat([gm_vertices, torch.ones(gm_vertices.shape[0], 1, device=self.device)], dim=1)

        # Apply RAS-to-RAS transform
        wm_transformed_ras = (transform_3x4 @ wm_verts_hom.T).T[:, :3]
        gm_transformed_ras = (transform_3x4 @ gm_verts_hom.T).T[:, :3]

        # Convert from tkRAS to voxel coordinates for sampling
        # vox = inv(vox2ras_tkr) @ ras
        vox2ras_inv = torch.inverse(self.vox2ras_tkr)
        wm_verts_hom_ras = torch.cat([wm_transformed_ras, torch.ones(wm_transformed_ras.shape[0], 1, device=self.device)], dim=1)
        gm_verts_hom_ras = torch.cat([gm_transformed_ras, torch.ones(gm_transformed_ras.shape[0], 1, device=self.device)], dim=1)

        wm_voxels = (vox2ras_inv @ wm_verts_hom_ras.T).T[:, :3]
        gm_voxels = (vox2ras_inv @ gm_verts_hom_ras.T).T[:, :3]

        # Sample intensities
        from .sampling import sample_volume_at_vertices
        wm_intensities = sample_volume_at_vertices(self.moving_volume, wm_voxels, self.vox2ras_tkr)
        gm_intensities = sample_volume_at_vertices(self.moving_volume, gm_voxels, self.vox2ras_tkr)

        # Compute cost
        from .cost import bbr_contrast_cost, gradient_magnitude_cost

        if self.cost_type == 'contrast':
            cost = bbr_contrast_cost(
                wm_intensities, gm_intensities,
                slope=self.slope, contrast_sign=self.contrast_sign
            )
        elif self.cost_type == 'gradient':
            cost = gradient_magnitude_cost(wm_intensities, gm_intensities)
        else:  # 'both'
            cost_contrast = bbr_contrast_cost(
                wm_intensities, gm_intensities,
                slope=self.slope, contrast_sign=self.contrast_sign
            )
            cost_gradient = gradient_magnitude_cost(wm_intensities, gm_intensities)
            cost = cost_contrast + self.gradient_weight * cost_gradient

        return cost

