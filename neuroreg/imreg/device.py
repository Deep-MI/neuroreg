"""Helpers for normalizing device selection across image-registration backends."""

from __future__ import annotations

import warnings

import torch


def resolve_torch_device(device: str | torch.device) -> torch.device:
    """Normalize a requested torch device specification.

    Parameters
    ----------
    device : str or torch.device
        Requested device for a torch-backed registration path. String inputs are
        normalized case-insensitively and may use the generic alias ``"gpu"``.

    Returns
    -------
    torch.device
        The resolved device. ``"gpu"`` prefers CUDA when available, then MPS,
        and otherwise falls back to CPU with a warning.

    Notes
    -----
    Existing ``torch.device`` instances are returned unchanged so callers can
    pass already-resolved devices without further normalization.
    """
    if not isinstance(device, str):
        return device

    normalized = str(device).strip().lower()
    if normalized == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        warnings.warn(
            "Requested device 'gpu' but neither CUDA nor MPS is available; falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")

    return torch.device(normalized)


def resolve_cpu_only_device(device: str | torch.device, *, backend_name: str) -> torch.device:
    """Resolve a device for a backend that currently runs on CPU only.

    Parameters
    ----------
    device : str or torch.device
        Requested device for the backend.
    backend_name : str
        Human-readable backend name used in the fallback warning.

    Returns
    -------
    torch.device
        ``cpu`` when the backend is CPU-only. Non-CPU requests are normalized
        first, then downgraded to CPU with an explicit warning.
    """
    resolved = resolve_torch_device(device)
    if resolved.type != "cpu":
        warnings.warn(
            f"{backend_name} currently runs on CPU; falling back from {resolved.type.upper()} to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    return resolved


__all__ = ["resolve_cpu_only_device", "resolve_torch_device"]
