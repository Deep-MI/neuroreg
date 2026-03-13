from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

import nireg.image.map
import nireg.transforms.matrices as trans


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
            device: str = 'cpu'
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
        device : str, optional
            The device where the parameters will be stored. Can be 'cpu' or 'cuda'. Defaults to
            'cpu'.

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
        if v2v_init is not None:
            if source_shape is None or target_shape is None:
                raise ValueError("Source and target shapes must be provided if v2v_init is provided.")
            self.set_ixform(v2v_init, source_shape, target_shape, device=device)
        else:
            self.ixform = None

    def init_weights(self, dof: int = 6, device: str | torch.device = 'cpu') -> Tensor:
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
        device : Union[str, torch.device], optional
            The device on which to store the initialized weights.
            This can be a string ('cpu', 'cuda') or a torch.device object.
            Defaults to 'cpu'.

        Returns
        -------
        Tensor
            A tensor containing the initialized weights.
        """
        w = torch.zeros(dof, device=device)
        #if 3 < dof < 12:
        #    w[3:6] = 0.0001
        if 6 < dof < 12:
            w[6:9] = 1.0
        if dof == 12:
            w[0], w[5], w[10] = 1.0, 1.0, 1.0
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
            device: str | torch.device = 'cpu'
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
        self.ixform = (
            trans.convert_v2v_to_torch(v2v_init, source_shape, target_shape)
            .type(torch.float32)
            .to(device)
        )

    def get_torch_transform_from_weights(self) -> torch.Tensor:
        """
        Construct a transformation matrix from the weights tensor.

        Depending on the number of weights in the `self.weights` tensor,
        the function generates a transformation matrix that includes translation,
        rotation, and optional scaling. If the weights contain 12 values, the function
        assumes a fully affine transformation (3x4 matrix). Additionally, if an initial
        transformation (`self.ixform`) is provided, it will be applied.

        Returns
        -------
        torch.Tensor
            A 3x4 or 4x4 transformation matrix constructed from the weights tensor.

        Functionality
        -------------
        - If `weights.size(0) == 12`:
            Assumes the weights are the entries of a fully affine transformation matrix
            (3x4), which is directly reshaped.
        - If `weights.size(0) > 3`:
            Treats weights as parameters for translation, rotation, and optional scaling:
            - The first 3 weights (0:3) correspond to translation in the fourth column.
            - The next 3 weights (3:6) define rotation as an axis-angle representation
              and are converted to a rotation matrix using `get_rotation_euler`.
            - The next 3 weights (6:9) are a diagonal matrix representing scaling factors,
              which are applied multiplicatively.
        - If `weights.size(0) <= 3`:
            Assumes the weights are only translation parameters.
        - If `self.ixform` is not `None`:
            Multiplies the generated affine matrix with the initial transformation matrix (`ixform`).

        Raises
        ------
        ValueError
            If the weights tensor is in an unsupported format or dimension that does not meet the conditions.

        Example
        -------
        >>> weights = torch.tensor([2.0, 3.0, 4.0])  # Translation only
        >>> self.weights = weights
        >>> transform = self.get_torch_transform_from_weights()
        >>> print(transform)
        tensor([[1.0, 0.0, 0.0, 2.0],
                [0.0, 1.0, 0.0, 3.0],
                [0.0, 0.0, 1.0, 4.0]])

        Notes
        -----
        - The resulting transformation matrix is a 3x4 matrix by default.
        - Scaling is only applied if the weight size is greater than 6.
        - Fully affine transformations are handled as a direct reshaping of the weights.

        """
        # trans = torch.cat((torch.eye(3,device=self.weights.device), self.weights[0:3].unsqueeze(dim=1)), dim=1)
        #trans = torch.eye(4, device=self.weights.device, dtype=self.weights.dtype)
        #trans[:3, 3] = self.weights
        if self.weights.size(0) == 12:
            # Fully affine: optimize 12 matrix entries as 3x4
            affine = self.weights.view((3,4))
        else:
            # For translation, use three weights as translation entries in 4th column
            affine = torch.cat((torch.eye(3,device=self.weights.device),
                                self.weights[0:3].unsqueeze(dim=1)), dim=1)
            if self.weights.size(0) > 3:
                # For rigid get rotation from Euler angles
                # and insert into 3x3 block
                affine[:3,:3] = trans.get_rotation_euler(self.weights[3:6])[:3,:3]
            if self.weights.size(0) > 6:
                # For rigid and scaling multiply with scaling diagonal matrix
                affine.mul_(self.weights[6:9].view(-1, 1))
        if self.ixform is not None:
            # affine and ixform are 3x4 so we have to multiply step wise:
            affine[:3, :3] = affine[:3, :3].type(self.ixform.dtype) @ self.ixform[:3, :3]
            affine[:3, 3] = (
                    affine[:3, :3].type(self.ixform.dtype) @ self.ixform[:3, 3]
                    + affine[:3, 3].type(self.ixform.dtype)
            )

        return affine


    def get_v2v_from_weights(self, sshape: tuple[int, int, int],
                             tshape: tuple[int, int, int] | None = None) -> Tensor:
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
                self,
                saffine: Tensor,
                taffine: Tensor,
                sshape: tuple[int, int, int],
                tshape: tuple[int, int, int] | None = None
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
        v2v = self.get_v2v_from_weights(sshape,tshape)
        return torch.tensor(taffine) @ v2v.double() @ torch.inverse(torch.tensor(saffine))

    def map_image(
            self,
            image: torch.Tensor,
            torch_transform: torch.Tensor | None = None,
            mode: str = 'bilinear'
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
            The interpolation mode to use for sampling. Options include 'bilinear' (default) and 'nearest'.

        Returns
        -------
        torch.Tensor
            The transformed image tensor with the same dimensionality as the input tensor.

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
            - 'bilinear' for smooth sampling.
            - 'nearest' for nearest-neighbor sampling.
        - The input image tensor is assumed to be 3-dimensional (depth, height, width).
        - Padding outside the valid image region is set to zero.

        """
        if torch_transform is None:
            torch_transform = self.get_torch_transform_from_weights()
        return nireg.image.map(image, torch_transform, mode=mode)


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
        return self.map_image(X)

