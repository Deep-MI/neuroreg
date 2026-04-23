from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

import neuroreg.transforms.matrices as trans
from neuroreg.image import map as image_map


class RegModel(nn.Module):
    """
    PyTorch model for 3D image registration.

    This model provides utility functions for working with translation, rotation,
    and scaling matrices in 3D, as well as conversion between different affine
    transformations for image registration tasks.
    """

    def __init__(
        self,
        dof: int = 6,
        v2v_init: torch.Tensor | None = None,
        source_shape: Any | None = None,
        target_shape: Any | None = None,
        device: str = "cpu",
        translation_weight_scale: float = 1.0,
        rotation_weight_scale: float = 1.0,
        scale_weight_scale: float = 1.0,
        shear_weight_scale: float = 1.0,
    ) -> None:
        """
        Initialize the registration model with degrees of freedom and optional parameters.

        Parameters
        ----------
        dof : int, optional
            Degrees of freedom for the transformation. Supported values are 3, 6, 9, or 12.
            Defaults to 6.
        v2v_init : Optional[torch.Tensor], optional
            An optional tensor for vertex-to-vertex (v2v) initialization. If provided, it is
            converted to the transformation matrix using the `source_shape` and `target_shape`.
            Defaults to None.
        source_shape : Optional, optional
            The shape of the source object. Only used if `v2v_init` is provided. Defaults to None.
        target_shape : Optional, optional
            The shape of the target object. Only used if `v2v_init` is provided. Defaults to None.
        device : str or torch.device, optional
            The device where the parameters will be stored. Accepts torch device strings such as
            'cpu', 'cuda', 'mps', or the generic alias 'gpu'. Defaults to 'cpu'.

        Attributes
        ----------
        weights : torch.nn.Parameter
            A learnable parameter tensor that holds the transformation weights, initialized based
            on the specified degrees of freedom (`dof`).
        ixform : Optional[torch.Tensor]
            The precomputed transformation matrix if `v2v_init` is provided. Otherwise, set to None.

        Functionality
        -------------
        1. Initializes the `self.weights` parameter tensor using the `init_weights` method
           based on the specified degrees of freedom (`dof`) and device.
        2. Uses the `v2v_init` tensor, if provided, to compute and store an initial transformation
           matrix (`ixform`) via the `convert_v2v_to_torch` method.

        Example
        -------
        >>> model = RegModel(dof=6, device='cuda')
        >>> print(model.weights.size())
        >>> print(model.ixform)

        Notes
        -----
        - The `weights` parameter is trainable and represents the transformation space.
        - If `v2v_init` is not provided, the transformation matrix (`ixform`) will be initialized as None.
        - Ensure that `source_shape` and `target_shape` are compatible with `v2v_init` if provided.
        """
        super().__init__()
        self.weights = nn.Parameter(self.init_weights(dof=dof, device=device))
        self.translation_weight_scale = float(translation_weight_scale)
        self.rotation_weight_scale = float(rotation_weight_scale)
        self.scale_weight_scale = float(scale_weight_scale)
        self.shear_weight_scale = float(shear_weight_scale)
        # Store shapes for rotation-space correction (see get_torch_transform_from_weights)
        self.source_shape: tuple[int, int, int] | None = source_shape
        self.target_shape: tuple[int, int, int] | None = target_shape if target_shape is not None else source_shape
        if v2v_init is not None:
            if source_shape is None or target_shape is None:
                raise ValueError("Source and target shapes must be provided if v2v_init is provided.")
            self.set_ixform(v2v_init, source_shape, target_shape, device=device)
        else:
            self.ixform = None
            self.v2v_init = None

    def init_weights(self, dof: int = 6, device: str | torch.device = "cpu") -> Tensor:
        """
        Initialize transformation weights based on degrees of freedom (DOF).

        Parameters
        ----------
        dof : int, optional
            Degrees of freedom for transformation. Values can only be:
            - 3: Translation
            - 6: Rigid (default)
            - 9: Rigid and scaling
            - 12: Full affine
        device : str or torch.device, optional
            The device on which to store the initialized weights.
            This can be a torch device string such as 'cpu', 'cuda', 'mps', or 'gpu',
            or a torch.device object. Defaults to 'cpu'.

        Returns
        -------
        Tensor
            A tensor containing the initialized weights.
        """
        w = torch.zeros(dof, device=device)
        # if 3 < dof < 12:
        #    w[3:6] = 0.0001
        if 6 < dof <= 12:
            w[6:9] = 1.0
        return w

    def reset(self, dof: int = None) -> None:
        """
        Reset the weights to their initial values based on DOF.

        Parameters
        ----------
        dof : int, optional
            Degrees of freedom for the transformation. If unset, it will use the current value from weights.

        Returns
        -------
        None
            This function modifies the `self.weights` tensor in-place, resetting it to the initialized values.

        Functionality
        -------------
        - The method initializes the transformation weights using the specified degrees of freedom (`dof`).
        - The weights are reset using the `self.init_weights` method, which generates initial values and
          assigns them to `self.weights.data`.

        Example
        -------
        >>> model.reset(dof=6)  # Reset the weights with 6 degrees of freedom
        >>> print(model.weights)

        Notes
        -----
        - The `dof` parameter determines the type of transformation (e.g., rigid body, affine, etc.).
        - This function directly modifies the `self.weights` tensor's underlying data.
        - Ensure that any existing computational graph involving `self.weights` is cleared before calling
          this function to avoid unexpected behavior.
        """
        if dof is None:
            dof = self.weights.size(0)
        self.weights.data = self.init_weights(dof=dof, device=self.weights.device)

    def set_ixform(
        self,
        v2v_init: torch.Tensor,
        source_shape: tuple[int, int, int],
        target_shape: tuple[int, int, int],
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Update the initial vox2vox transformation matrix.

        Converts a voxel-to-voxel transformation matrix into a format compatible with PyTorch
        and the target device. This method takes the given transformation matrix along with
        the dimensions of the source and target grids to compute the internal transformation
        matrix (`ixform`), ensuring compatibility with tensor operations on the target device.

        Parameters
        ----------
        v2v_init : torch.Tensor
            The initial voxel-to-voxel transformation matrix, provided as a PyTorch tensor.
        source_shape : tuple[int, int, int]
            The dimensions (height, width, depth) of the source grid (shape of nibabel image data)
        target_shape : tuple[int, int, int]
            The dimensions (height, width, depth) of the target grid (shape of nibabel image data).

        Returns
        -------
        None
        """
        self.v2v_init = v2v_init.type(torch.float32).to(device)
        self.ixform = (
            trans.convert_v2v_to_torch(self.v2v_init, source_shape, target_shape).type(torch.float32).to(device)
        )

    def _build_torch_affine_from_weights(self, weights: Tensor) -> torch.Tensor:
        translation = weights[0:3] * self.translation_weight_scale
        affine = torch.cat(
            (
                torch.eye(3, device=weights.device, dtype=weights.dtype),
                translation.unsqueeze(dim=1),
            ),
            dim=1,
        )
        if weights.size(0) > 3:
            rotation = weights[3:6] * self.rotation_weight_scale
            linear_v2v = trans.get_rotation_euler(rotation)[:3, :3]
            if weights.size(0) > 6:
                scale = 1.0 + (weights[6:9] - 1.0) * self.scale_weight_scale
                linear_v2v = linear_v2v @ torch.diag(scale)
            if weights.size(0) > 9:
                shear = torch.eye(3, device=weights.device, dtype=weights.dtype)
                shear[0, 1] = weights[9] * self.shear_weight_scale
                shear[0, 2] = weights[10] * self.shear_weight_scale
                shear[1, 2] = weights[11] * self.shear_weight_scale
                linear_v2v = linear_v2v @ shear
            if self.source_shape is not None and self.target_shape is not None:
                ss = weights.new_tensor([self.source_shape[2], self.source_shape[1], self.source_shape[0]]) / 2.0
                st = weights.new_tensor([self.target_shape[2], self.target_shape[1], self.target_shape[0]]) / 2.0
                affine[:3, :3] = torch.diag(1.0 / ss) @ linear_v2v.mT @ torch.diag(st)
            else:
                affine[:3, :3] = linear_v2v
        return affine

    def get_torch_transform_from_weights(self) -> torch.Tensor:
        """Construct a transformation matrix from the weights tensor."""
        affine = self._build_torch_affine_from_weights(self.weights)
        if self.v2v_init is None:
            return affine

        if self.source_shape is None or self.target_shape is None:
            raise ValueError("Source and target shapes must be provided if v2v_init is provided.")

        current_torch = torch.eye(4, device=self.weights.device, dtype=affine.dtype)
        current_torch[:3, :4] = affine
        base_weights = self.init_weights(
            dof=int(self.weights.size(0)),
            device=self.weights.device,
        ).to(dtype=self.weights.dtype)
        base_affine = self._build_torch_affine_from_weights(base_weights)
        base_torch = torch.eye(4, device=self.weights.device, dtype=base_affine.dtype)
        base_torch[:3, :4] = base_affine

        current_v2v = trans.convert_torch_to_v2v(current_torch, self.source_shape, self.target_shape)
        base_v2v = trans.convert_torch_to_v2v(base_torch, self.source_shape, self.target_shape)
        relative_v2v = current_v2v @ torch.inverse(base_v2v)
        composed_v2v = relative_v2v.to(dtype=self.v2v_init.dtype, device=self.v2v_init.device) @ self.v2v_init
        return trans.convert_v2v_to_torch(composed_v2v, self.source_shape, self.target_shape).to(affine.dtype)

    def get_v2v_from_weights(self, sshape: tuple[int, int, int], tshape: tuple[int, int, int] | None = None) -> Tensor:
        """
        Convert transformation weights into a voxel-to-voxel transformation matrix.

        This function generates a voxel-to-voxel (v2v) transformation matrix by
        combining the weights tensor into an affine transformation matrix (4x4)
        and adapting it to the source and target voxel shapes.

        Parameters
        ----------
        sshape : Tuple[int, int, int]
            Shape of the source volume (depth, height, width).
        tshape : Optional[Tuple[int, int, int]], optional
            Shape of the target volume (depth, height, width). Defaults to `None`.

        Returns
        -------
        Tensor
            The voxel-to-voxel transformation matrix as a 4x4 tensor. The resulting
            transformation matrix is detached and returned on the CPU.

        Functionality
        -------------
        1. Generates a 4x4 affine transformation matrix using `self.get_torch_transform_from_weights`.
        2. Converts the affine matrix into a voxel-to-voxel transformation matrix using the function
           `self.convert_torch_to_v2v`, taking into account the source (and optionally target) shapes.
        3. Outputs the resulting matrix as a detached CPU tensor.

        Example
        -------
        >>> self.weights = torch.tensor([1.0, 2.0, 3.0])  # Example of translation weights
        >>> sshape = (128, 128, 128)  # Source shape
        >>> tshape = (256, 256, 256)  # Target shape
        >>> v2v_matrix = self.get_v2v_from_weights(sshape, tshape)
        >>> print(v2v_matrix)
        tensor([[1.0000, 0.0000, 0.0000, 1.0000],
                [0.0000, 1.0000, 0.0000, 2.0000],
                [0.0000, 0.0000, 1.0000, 3.0000],
                [0.0000, 0.0000, 0.0000, 1.0000]])

        Notes
        -----
        - If `tshape` is not provided, it is assumed that the transformation applies within the
          same shape as the `sshape`.
        - The transformation matrix is returned as a CPU tensor detached from computational graphs.
        """
        Atorch = torch.eye(4, device=self.weights.device)
        Atorch[:3, :4] = self.get_torch_transform_from_weights()
        v2v = trans.convert_torch_to_v2v(Atorch, sshape, tshape)
        return v2v.detach().cpu()

    def get_r2r_from_weights(
        self, saffine: Tensor, taffine: Tensor, sshape: tuple[int, int, int], tshape: tuple[int, int, int] | None = None
    ) -> Tensor:
        """
        Compute the RAS-to-RAS (r2r) transformation matrix from transformation weights.

        This function calculates the r2r transformation, which transforms coordinates
        from the reference space of the source volume to the reference space of the
        target volume. It uses voxel-to-voxel transformations (`v2v`) along with the
        source and target affine matrices.

        Parameters
        ----------
        saffine : Tensor
            The 4x4 affine transformation matrix defining the orientation and position
            of the source volume in RAS space.
        taffine : Tensor
            The 4x4 affine transformation matrix defining the orientation and position
            of the target volume in RAS space.
        sshape : Tuple[int, int, int]
            Shape of the source volume (depth, height, width).
        tshape : Optional[Tuple[int, int, int]], optional
            Shape of the target volume (depth, height, width). Defaults to `None`.

        Returns
        -------
        Tensor
            A 4x4 RAS-to-RAS (r2r) transformation matrix as a PyTorch tensor.

        Functionality
        -------------
        1. Computes the voxel-to-voxel (`v2v`) transformation matrix between the source
           and target spaces by calling `self.get_v2v_from_weights`.
        2. Combines the affine matrices (`saffine` and `taffine`) with the voxel-to-voxel
           transformation to calculate the r2r transformation:
            r2r = taffine @ v2v @ saffine⁻¹

        Raises
        ------
        RuntimeError
            If `saffine` or `taffine` is not invertible.

        Example
        -------
        >>> saffine = torch.eye(4)
        >>> taffine = torch.eye(4)
        >>> sshape = (128, 128, 128)
        >>> tshape = (256, 256, 256)
        >>> r2r_matrix = self.get_r2r_from_weights(saffine, taffine, sshape, tshape)
        >>> print(r2r_matrix)
        tensor([[1.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 1.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 1.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 1.0000]])

        Notes
        -----
        - The affine matrices (`saffine` and `taffine`) must be invertible 4x4 tensors.
        - When `tshape` is not provided, it is assumed that the transformation applies to the
          same shape as `sshape`.
        - The resulting matrix transforms points directly in RAS space.

        """
        v2v = self.get_v2v_from_weights(sshape, tshape)
        dtype = v2v.dtype
        device = v2v.device
        taffine_ras = taffine.to(device=device)
        saffine_ras = saffine.to(device=device)
        if taffine_ras.dtype != dtype:
            taffine_ras = taffine_ras.to(dtype=dtype)
        if saffine_ras.dtype != dtype:
            saffine_ras = saffine_ras.to(dtype=dtype)
        return taffine_ras @ v2v @ torch.inverse(saffine_ras)

    def map_image(
        self,
        image: torch.Tensor,
        torch_transform: torch.Tensor | None = None,
        mode: str = "linear",
        target_shape: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        """
        Map an input image to another space using the inverse transformation matrix.

        Parameters
        ----------
        image : torch.Tensor
            The input image tensor to be transformed. Shape is typically (depth, height, width).
        torch_transform : Optional[torch.Tensor], optional
            The 4x4 torch transformation matrix used for mapping the image. If not provided,
            it defaults to the transformation matrix generated by `self.get_torch_transform_from_weights()`.
        mode : str, optional
            The interpolation mode to use for sampling. Options include 'linear' (default) and 'nearest'.
        target_shape : tuple[int, int, int], optional
            Output grid shape. When ``None`` (default), uses ``self.target_shape``
            if available; otherwise falls back to the input image shape.

        Returns
        -------
        torch.Tensor
            The transformed image tensor resampled on *target_shape*.

        Functionality
        -------------
        1. If no `torch_transform` is provided, the function computes the transformation matrix
           using `self.get_torch_transform_from_weights()`.
        2. Reshapes the transformation matrix into a 3x4 affine transformation matrix for grid generation.
        3. Uses `torch.nn.functional.affine_grid` to compute a sampling grid in the transformed space.
        4. Applies `torch.nn.functional.grid_sample` to map the input image to the new space based on
           the computed grid and transformation matrix.

        Example
        -------
        >>> image = torch.randn(64, 64, 64)  # Example 3D image
        >>> torch_transform = torch.eye(4)   # Identity transform
        >>> result = self.map_image(image, torch_transform=torch_transform)
        >>> print(result.shape)
        torch.Size([64, 64, 64])

        Notes
        -----
        - The `mode` parameter controls the interpolation strategy:
            - 'linear' for smooth sampling.
            - 'nearest' for nearest-neighbor sampling.
        - The input image tensor is assumed to be 3-dimensional (depth, height, width).
        - Padding outside the valid image region is set to zero.

        """
        if torch_transform is None:
            torch_transform = self.get_torch_transform_from_weights()
        out_shape = target_shape if target_shape is not None else self.target_shape
        return image_map(image, torch_transform, target_shape=out_shape, mode=mode)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass, mapping the input image according to the transformation weights.

        Parameters
        ----------
        X : torch.Tensor
            The input image tensor to be transformed. Typically a 3D tensor (depth, height, width).

        Returns
        -------
        torch.Tensor
            The transformed image tensor after applying the mapping.

        Functionality
        -------------
        - The function maps the input image to a new space by calling the `self.map_image`
          method, which applies the transformation based on the model's weights.

        Example
        -------
        >>> X = torch.randn(64, 64, 64)  # Example input image
        >>> transformed_X = self.forward(X)
        >>> print(transformed_X.shape)
        torch.Size([64, 64, 64])

        Notes
        -----
        - The functionality relies on the `self.map_image` method to perform the actual
          transformation on the input image `X`.

        """
        return self.map_image(X, target_shape=self.target_shape)
