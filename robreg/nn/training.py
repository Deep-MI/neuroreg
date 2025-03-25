import time

import torch
import torch.nn as nn
from stopper import EarlyStopper
from torch.functional import F


def training_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    src_image: torch.Tensor,
    trg_image: torch.Tensor,
    n: int = 10,
    loss_name: str = "mse",
    verbose: bool = False
) -> list[torch.Tensor]:
    """
    Optimize a PyTorch model using an optimizer and a loss function for image registration.

    This function performs training for a specified number of iterations while monitoring
    a loss function. It tracks losses, applies early stopping, and stops the training
    if the computed transformation stabilizes over iterations.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to be trained. It must define a forward pass and relevant
        methods for transformation calculations.
    optimizer : torch.optim.Optimizer
        An optimizer object for optimizing the `model` parameters.
    src_image : torch.Tensor
        The tensor representing the source image.
    trg_image : torch.Tensor
        The tensor representing the target image.
    n : int, optional
        The number of training iterations to perform, by default 10.
    loss_name : str, optional
        The name of the loss function to be used for optimization, by default "mse".
        Supported options: "mse", "huber", "smooth_l1", "l1".
    verbose : bool, optional
        If True, prints detailed logging information during training, by default False.

    Returns
    -------
    list[torch.Tensor]
        A list containing the square root of the loss values at each iteration.

    Raises
    ------
    ValueError
        If an invalid `loss_name` is provided, i.e., not one of the supported options.

    Notes
    -----
    - The function uses an early stopping mechanism to terminate training early
      if the monitored loss changes very little over a specified patience period.
    - The training halts if the transformation matrix stabilizes (L2 norm difference
      between consecutive transformations is below a threshold).
    """
    losses: list[torch.Tensor] = []
    early_stopper = EarlyStopper(patience=4, min_delta=0.001)
    last_v2v = model.get_v2v_from_weights(src_image.shape)

    for i in range(n):
        print(f"Iteration: {i}")
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
                raise ValueError(f"Unknown loss name: {loss_name}")
            loss.backward()
            losses.append(loss.sqrt())
            return loss

        optimizer.step(closure)
        if verbose:
            print(f"Loss (it {i})",losses[-1].detach().numpy())
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Gradient of {name}: {param.grad}")
                else:
                    print(f"No gradient for {name}")

        if early_stopper.early_stop(losses[-1]):
            print("!!! Early Stop (loss stable) !!!")
            break

        v2v = model.get_v2v_from_weights(src_image.shape)
        diff = torch.norm(last_v2v - v2v)
        last_v2v = v2v

        if verbose:
            print("weights optimized:", model.weights)
            print("v2v:", v2v)
            print(f"sec per iteration ({i}): {time.perf_counter()-start}")
        print("  -- diff. to prev. transform: ", diff.detach().numpy())
        if diff < 0.01:
            print("!!! Early Stop (vox2vox transform stable) !!!")
            break

    return losses