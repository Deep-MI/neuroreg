"""Optimization helpers for the legacy gradient-descent image-registration path."""

import logging
import time

import torch
import torch.nn as nn
from torch.functional import F

logger = logging.getLogger(__name__)


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
    normalize: bool = True,
    optimizer_name: str = "adam",
    verbose: bool = False,
) -> list[float]:
    """Optimise a registration model for a fixed number of iterations."""
    losses: list[float] = []
    last_v2v = model.get_v2v_from_weights(tuple(int(v) for v in src_image.shape))

    if normalize:
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
                    f"Unknown loss_name '{loss_name}'. Choose from: 'mse', 'huber', 'smooth_l1', 'l1'."
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
            losses.append(float(loss_tensor.sqrt()))
        else:
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

        v2v = model.get_v2v_from_weights(tuple(int(v) for v in src_image.shape))
        diff = torch.norm(last_v2v - v2v)
        last_v2v = v2v

        if verbose:
            logger.debug("weights: %s", model.weights)
            logger.debug("v2v: %s", v2v)
            logger.debug("sec per iter %d: %.3f", i, time.perf_counter() - start)

        logger.debug("iter %d  transform diff: %.6f", i, diff.item())

    return losses


__all__ = ["EarlyStopper", "training_loop"]
