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
    n: int = 30,
    loss_name: str = "mse",
    loss_beta: float | None = None,
    normalize: bool = True,
    optimizer_name: str = "adam",
    verbose: bool = False
) -> list[float]:
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
        Maximum number of iterations.  Default is 30.
    loss_name : {'mse', 'huber', 'smooth_l1', 'l1'}, optional
        Loss function name.  Default is ``'mse'``.
    loss_beta : float, optional
        Transition point for ``'smooth_l1'`` (beta) and ``'huber'`` (delta).
        Below this value the loss is L2; above it, L1.  Defaults to
        ``None``, which lets PyTorch use its built-in default (1.0).
        Rule of thumb for MRI: set to roughly 10–20 % of the image
        intensity std so that normal tissue residuals stay in the L2
        regime while large outlier errors are penalised linearly.
    normalize : bool, optional
        If ``True`` (default), both images are divided by the 99.5th
        percentile of *trg_image* before computing the loss.  This maps
        intensities to roughly [0, 1] so that gradient magnitudes are
        consistent across image pairs and pyramid levels, regardless of
        the scanner intensity scale.  The model still samples from the
        original *src_image* (the transform is unchanged); only the
        loss computation uses the scaled values.
    optimizer_name : str, optional
        Name of the optimizer being used ('adam' or 'lbfgs'). L-BFGS
        requires special handling as it may call the closure multiple times
        per iteration. Default is 'adam'.
    verbose : bool, optional
        If ``True``, log gradients and weight values at each iteration
        (``DEBUG`` level).  Default is ``False``.

    Returns
    -------
    list[float]
        Square-root loss value at each completed iteration.

    Raises
    ------
    ValueError
        If *loss_name* is not one of the supported options.
    """
    losses: list[float] = []
    last_v2v = model.get_v2v_from_weights(src_image.shape)

    # Pre-compute normalisation scale from the target once.
    # clamp(min=1) guards against background-only crops at coarse levels.
    if normalize:
        scale = torch.quantile(trg_image.abs().reshape(-1), 0.995).clamp(min=1.0)
        src_image = src_image / scale
        trg_image = trg_image / scale
        if loss_beta is not None:
            loss_beta = loss_beta / scale.item()

    is_lbfgs = optimizer_name.lower() == 'lbfgs'
    
    for i in range(n):
        logger.debug("Iteration %d", i)
        start = time.perf_counter()
        
        # For L-BFGS, we need a closure that can be called multiple times
        # For Adam, we only need to call it once
        def closure():
            optimizer.zero_grad()
            preds = model(src_image)
            t = trg_image.view(preds.size())
            if loss_name == "mse":
                loss = F.mse_loss(preds, t)
            elif loss_name == "huber":
                kw = {} if loss_beta is None else {"delta": loss_beta}
                loss = F.huber_loss(preds, t, **kw)
            elif loss_name == "smooth_l1":
                kw = {} if loss_beta is None else {"beta": loss_beta}
                loss = F.smooth_l1_loss(preds, t, **kw)
            elif loss_name == "l1":
                loss = F.l1_loss(preds, t)
            else:
                raise ValueError(
                    f"Unknown loss_name '{loss_name}'. "
                    "Choose from: 'mse', 'huber', 'smooth_l1', 'l1'."
                )
            loss.backward()
            return loss
        
        if is_lbfgs:
            # L-BFGS calls closure multiple times, returns final loss
            loss = optimizer.step(closure)
            losses.append(float(loss.sqrt()))
        else:
            # Adam-style: call closure once, step, record loss
            loss = closure()
            optimizer.step()
            losses.append(float(loss.sqrt()))


        if verbose:
            logger.debug("Loss (iter %d): %s", i, losses[-1])
            for name, param in model.named_parameters():
                if param.grad is not None:
                    logger.debug("Gradient of %s: %s", name, param.grad)
                else:
                    logger.debug("No gradient for %s", name)


        v2v = model.get_v2v_from_weights(src_image.shape)
        diff = torch.norm(last_v2v - v2v)
        last_v2v = v2v

        if verbose:
            logger.debug("weights: %s", model.weights)
            logger.debug("v2v: %s", v2v)
            logger.debug("sec per iter %d: %.3f", i, time.perf_counter() - start)

        logger.debug("iter %d  transform diff: %.6f", i, diff.item())


    return losses

