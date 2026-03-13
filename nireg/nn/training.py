"""Training loop for image-registration models."""

import logging
import time

import torch
import torch.nn as nn
from torch.functional import F

from .stopper import EarlyStopper

logger = logging.getLogger(__name__)


def training_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    src_image: torch.Tensor,
    trg_image: torch.Tensor,
    n: int = 10,
    loss_name: str = "mse",
    verbose: bool = False
) -> list[torch.Tensor]:
    """Optimise a registration model for a fixed number of iterations.

    Runs a training loop with optional early stopping triggered either by
    loss plateau or by the vox-to-vox transform becoming stable.

    Parameters
    ----------
    model : nn.Module
        Registration model with a ``get_v2v_from_weights`` method.
    optimizer : torch.optim.Optimizer
        Optimiser for the model parameters.
    src_image : torch.Tensor
        Source image tensor.
    trg_image : torch.Tensor
        Target image tensor.
    n : int, optional
        Maximum number of iterations.  Default is 10.
    loss_name : {'mse', 'huber', 'smooth_l1', 'l1'}, optional
        Loss function name.  Default is ``'mse'``.
    verbose : bool, optional
        If ``True``, log gradients and weight values at each iteration
        (``DEBUG`` level).  Default is ``False``.

    Returns
    -------
    list[torch.Tensor]
        Square-root loss value at each completed iteration.

    Raises
    ------
    ValueError
        If *loss_name* is not one of the supported options.
    """
    losses: list[torch.Tensor] = []
    early_stopper = EarlyStopper(patience=4, min_delta=0.001)
    last_v2v = model.get_v2v_from_weights(src_image.shape)

    for i in range(n):
        logger.debug("Iteration %d", i)
        start = time.perf_counter()
        optimizer.zero_grad()

        def closure():
            preds = model(src_image)
            if loss_name == "mse":
                loss = F.mse_loss(preds, trg_image.view(preds.size()))
            elif loss_name == "huber":
                loss = F.huber_loss(preds, trg_image.view(preds.size()))
            elif loss_name == "smooth_l1":
                loss = F.smooth_l1_loss(preds, trg_image.view(preds.size()))
            elif loss_name == "l1":
                loss = F.l1_loss(preds, trg_image.view(preds.size()))
            else:
                raise ValueError(
                    f"Unknown loss_name '{loss_name}'. "
                    "Choose from: 'mse', 'huber', 'smooth_l1', 'l1'."
                )
            loss.backward()
            losses.append(loss.sqrt())
            return loss

        optimizer.step(closure)

        if verbose:
            logger.debug("Loss (iter %d): %s", i, losses[-1].detach().numpy())
            for name, param in model.named_parameters():
                if param.grad is not None:
                    logger.debug("Gradient of %s: %s", name, param.grad)
                else:
                    logger.debug("No gradient for %s", name)

        if early_stopper.early_stop(losses[-1]):
            logger.info("Early stop at iter %d: loss plateau", i)
            break

        v2v = model.get_v2v_from_weights(src_image.shape)
        diff = torch.norm(last_v2v - v2v)
        last_v2v = v2v

        if verbose:
            logger.debug("weights: %s", model.weights)
            logger.debug("v2v: %s", v2v)
            logger.debug("sec per iter %d: %.3f", i, time.perf_counter() - start)

        logger.debug("iter %d  transform diff: %.6f", i, diff.item())

        if diff < 0.01:
            logger.info("Early stop at iter %d: vox2vox transform stable (diff=%.6f)", i, diff.item())
            break

    return losses

