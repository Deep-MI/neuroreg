import torch
from reg_model import RegModel


class RegModelSym(RegModel):
    """
    A RegModel for symmetric registration.

    This class inherits from RegModel and extends it to add symmetric registration into a midspace. It will be
    slower as both src and trg images are mapped to the midspace in each step.
    """

    def __init__(self, new_param: int, *args, **kwargs):
        """
        Initialize the CustomRegModel with additional parameters.

        Parameters
        ----------
        new_param : int
            A custom parameter for the new model.
        *args : tuple
            Positional arguments to pass to the RegModel constructor.
        **kwargs : dict
            Keyword arguments to pass to the RegModel constructor.
        """
        super().__init__(*args, **kwargs)  # Call the parent constructor
        self.new_param = new_param

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
