"""Optimization helpers for the legacy gradient-descent image-registration path."""

import logging
import time

import torch
import torch.nn as nn
from torch.functional import F

from .losses import mi_loss, ncc_loss, nmi_loss

logger = logging.getLogger(__name__)

# Intensity-domain losses whose input images can be percentile-normalized.
_INTENSITY_LOSS_NAMES = {"mse", "huber", "smooth_l1", "l1"}

# Only true squared-error losses should be reported as RMSE.
_RMSE_REPORTED_LOSS_NAMES = {"mse"}


class EarlyStopper:
    """Utility class for early stopping based on a monitored scalar value."""

    def __init__(self, patience: int = 1, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop_old(self, validation_loss: float) -> bool:
        """Deprecated early-stopping rule kept for compatibility."""
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def early_stop(self, validation_loss: float) -> bool:
        """Return True when the monitored value has plateaued long enough."""
        if abs(validation_loss - self.min_validation_loss) < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.counter = 0
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
        return False


def training_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    src_image: torch.Tensor,
    trg_image: torch.Tensor,
    n: int = 30,
    loss_name: str = "mse",
    loss_beta: float | None = None,
    loss_bins: int = 32,
    normalize: bool = True,
    optimizer_name: str = "adam",
    verbose: bool = False,
    trace_fn=None,
) -> list[float]:
    """Optimise a registration model for a fixed number of iterations.

    Parameters
    ----------
    model : nn.Module
        Registration model whose parameters are updated.
    optimizer : Optimizer
        PyTorch optimiser wrapping ``model.parameters()``.
    src_image, trg_image : Tensor
        Source (moving) and target (fixed) images, shape (D, H, W).
    n : int, default=30
        Number of optimiser iterations.
    loss_name : str, default="mse"
        Similarity metric.  Accepted values:

        * ``"mse"``       — mean squared error (same-modality)
        * ``"huber"``     — Huber loss; ``loss_beta`` sets the delta threshold
        * ``"smooth_l1"`` — smooth L1; ``loss_beta`` sets the beta threshold
        * ``"l1"``        — mean absolute error
        * ``"ncc"``       — local normalised cross-correlation; ``loss_beta``
          sets the window size in voxels (default 9)
        * ``"mi"``        — mutual information via Parzen windows (cross-modal);
          ``loss_beta`` sets the Parzen sigma in normalised [0,1] intensity
          units; ``None`` (default) auto-computes ``sigma = 0.5/(num_bins-1)``
          (half a bin spacing); ``loss_bins`` sets the number of histogram bins
        * ``"nmi"``       — normalised mutual information (same hyper-parameters
          as ``"mi"``)
    loss_beta : float, optional
        Primary hyper-parameter for the chosen loss (see ``loss_name``).
    loss_bins : int, default=32
        Number of intensity histogram bins used by ``"mi"`` and ``"nmi"``.
        Ignored for other loss functions.
    normalize : bool, default=True
        If ``True``, divide both images by the 99.5th-percentile of the target
        before computing intensity-based losses.  NCC / MI / NMI handle their
        own internal normalisation and are unaffected by this flag.
    optimizer_name : str, default="adam"
        Name of the optimiser; controls the LBFGS closure path.
    verbose : bool, default=False
        If ``True``, emit per-iteration debug information.
    trace_fn : callable, optional
        Optional callback invoked as ``trace_fn(event=..., **payload)``.
        When provided, the loop emits ``"iter_end"`` events with the current
        iteration index, scalar loss, v2v transform, and transform change.

    Returns
    -------
    list of float
        Loss values recorded after each iteration. ``mse`` is reported as RMSE,
        while ``huber``, ``smooth_l1``, ``l1``, ``ncc``, ``mi``, and ``nmi`` are
        reported as their raw scalar losses.
    """
    losses: list[float] = []
    last_v2v = model.get_v2v_from_weights(tuple(int(v) for v in src_image.shape))

    # Only scale loss_beta for intensity-based losses where beta has intensity units.
    # NCC win_size is in voxels; MI/NMI sigma is already in [0, 1] intensity units.
    _is_intensity_loss = loss_name in _INTENSITY_LOSS_NAMES
    _report_rmse = loss_name in _RMSE_REPORTED_LOSS_NAMES

    if normalize and _is_intensity_loss:
        scale = torch.quantile(trg_image.abs().reshape(-1), 0.995).clamp(min=1.0)
        src_image = src_image / scale
        trg_image = trg_image / scale
        if loss_beta is not None:
            loss_beta = loss_beta / scale.item()

    is_lbfgs = optimizer_name.lower() == "lbfgs"

    for i in range(n):
        logger.debug("Iteration %d", i)
        start = time.perf_counter()

        def closure():
            optimizer.zero_grad()
            preds = model(src_image)
            if tuple(int(v) for v in preds.shape) != tuple(int(v) for v in trg_image.shape):
                raise ValueError(
                    "Predicted image shape does not match target image shape: "
                    f"preds={tuple(preds.shape)} vs trg={tuple(trg_image.shape)}."
                )
            t = trg_image
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
            elif loss_name == "ncc":
                win = 9 if loss_beta is None else max(1, int(loss_beta))
                loss = ncc_loss(preds.squeeze(), t.squeeze(), win_size=win)
            elif loss_name == "mi":
                sigma = None if loss_beta is None else float(loss_beta)
                loss = mi_loss(preds.squeeze(), t.squeeze(), num_bins=loss_bins, sigma=sigma)
            elif loss_name == "nmi":
                sigma = None if loss_beta is None else float(loss_beta)
                loss = nmi_loss(preds.squeeze(), t.squeeze(), num_bins=loss_bins, sigma=sigma)
            else:
                raise ValueError(
                    f"Unknown loss_name '{loss_name}'. "
                    "Choose from: 'mse', 'huber', 'smooth_l1', 'l1', 'ncc', 'mi', 'nmi'."
                )
            loss.backward()
            return loss

        def lbfgs_closure() -> float:
            return float(closure())

        if is_lbfgs:
            loss = optimizer.step(lbfgs_closure)
            if isinstance(loss, torch.Tensor):
                loss_tensor = loss
            elif loss is None:
                loss_tensor = torch.tensor(0.0, device=src_image.device)
            else:
                loss_tensor = torch.tensor(float(loss), device=src_image.device)
            losses.append(float(loss_tensor.sqrt()) if _report_rmse else float(loss_tensor))
        else:
            loss = closure()
            optimizer.step()
            losses.append(float(loss.sqrt()) if _report_rmse else float(loss))

        if verbose:
            logger.debug("Loss (iter %d): %s", i, losses[-1])
            for name, param in model.named_parameters():
                if param.grad is not None:
                    logger.debug("Gradient of %s: %s", name, param.grad)
                else:
                    logger.debug("No gradient for %s", name)

        v2v = model.get_v2v_from_weights(tuple(int(v) for v in src_image.shape))
        diff = torch.norm(last_v2v - v2v)
        last_v2v = v2v

        if trace_fn is not None:
            trace_fn(
                event="iter_end",
                iteration=i,
                loss=losses[-1],
                v2v=v2v.detach().clone(),
                transform_diff=float(diff),
            )

        if verbose:
            logger.debug("weights: %s", model.weights)
            logger.debug("v2v: %s", v2v)
            logger.debug("sec per iter %d: %.3f", i, time.perf_counter() - start)

        logger.debug("iter %d  transform diff: %.6f", i, diff.item())

    return losses


__all__ = ["EarlyStopper", "training_loop"]
