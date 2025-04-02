from typing import Any

import torch

from .reg_model import RegModel
from ..transforms.matrices import compute_sqrtm


class RegModelSym(RegModel):
    """
    A RegModel for symmetric registration.

    This class inherits from RegModel and extends it to add symmetric registration into a midspace.
    It will map and interpolate both src and trg images to the midspace in each step.
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
        """
        super().__init__(dof=dof, source_shape=source_shape, target_shape=target_shape, device=device)  # Call the parent constructor
        v2v_init_sqrt = compute_sqrtm(v2v_init)


    def additional_feature(self) -> None:
        """
        Implement additional functionality specific to this custom model.

        Returns
        -------
        None
        """
        print(f"The custom parameter is: {self.new_param}")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Override the forward method to include custom behavior.

        Parameters
        ----------
        X : torch.Tensor
            The input tensor to the model.

        Returns
        -------
        torch.Tensor
            The transformed tensor after applying the custom behavior.
        """
        print("Custom forward pass is being executed.")
        result = super().forward(X)  # Optionally, use the parent class implementation
        # Add custom logic here
        return result
