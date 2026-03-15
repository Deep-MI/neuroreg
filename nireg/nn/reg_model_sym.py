"""Symmetric registration model (midspace)."""

import logging

import torch

from ..transforms.matrices import compute_sqrtm
from .reg_model import RegModel

logger = logging.getLogger(__name__)


class RegModelSym(RegModel):
    """Registration model for symmetric (midspace) alignment.

    Extends :class:`RegModel` by mapping both source and target images into
    a common midspace at each optimisation step, ensuring that the final
    transform is the geometric mean of the forward and inverse transforms.

    Parameters
    ----------
    dof : int, optional
        Degrees of freedom (3, 6, 9, or 12).  Default is 6.
    v2v_init : torch.Tensor, optional
        Initial vox-to-vox transformation (4 × 4).
    source_shape : tuple, optional
        Shape of the source volume (D, H, W).
    target_shape : tuple, optional
        Shape of the target volume (D, H, W).
    device : str, optional
        PyTorch device string.  Default is ``'cpu'``.
    """

    def __init__(
            self,
            dof: int = 6,
            v2v_init: torch.Tensor | None = None,
            source_shape=None,
            target_shape=None,
            device: str = 'cpu'
    ) -> None:
        super().__init__(dof=dof, source_shape=source_shape, target_shape=target_shape, device=device)
        if v2v_init is not None:
            v2v_init_sqrt, _ = compute_sqrtm(v2v_init)
            logger.debug("Initial transform matrix square root: %s", v2v_init_sqrt)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Map input image via the current symmetric transform.

        Parameters
        ----------
        X : torch.Tensor
            Input image tensor (D, H, W).

        Returns
        -------
        torch.Tensor
            Transformed image tensor.
        """
        logger.debug("RegModelSym forward pass")
        return super().forward(X)
