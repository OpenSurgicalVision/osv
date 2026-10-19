"""Image-restoration metrics for M5 desmoking.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

PSNR and SSIM on [0, 1] RGB tensors, per image, so a split's score is the mean
over frames and a confidence interval can be computed from the same numbers.
SSIM is the standard Gaussian-window form (sigma 1.5, 11 px, K1 0.01, K2 0.03)
on luma, which is what desmoking papers report. Every report carries the
no-op baseline - the smoky input scored against the clean frame - because a
restoration number means nothing without the number for doing nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PSNR_CAP = 100.0  # identical images: a finite ceiling, never inf in a mean
LUMA = (0.299, 0.587, 0.114)
# Smoke density strata by the pair's mean effective alpha (osv.datasets.smoke_m5 index).
DENSITY_EDGES = (0.02, 0.06)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR in dB for N x C x H x W tensors in [0, 1]."""
    mse = (pred - target).pow(2).flatten(1).mean(dim=1)
    value = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))
    return value.clamp(max=PSNR_CAP)


def _luma(image: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor(LUMA, dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def _gaussian(size: int = 11, sigma: float = 1.5, *, device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return (kernel[:, None] * kernel[None, :]).view(1, 1, size, size)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image SSIM on luma, valid-region (no padding), in [-1, 1]."""
    x, y = _luma(pred), _luma(target)
    window = _gaussian(device=x.device, dtype=x.dtype)
    c1, c2 = 0.01**2, 0.03**2
    mu_x, mu_y = F.conv2d(x, window), F.conv2d(y, window)
    sigma_x = F.conv2d(x * x, window) - mu_x**2
    sigma_y = F.conv2d(y * y, window) - mu_y**2
    sigma_xy = F.conv2d(x * y, window) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator).flatten(1).mean(dim=1)


def density_stratum(alpha_mean: float) -> str:
    light, medium = DENSITY_EDGES
    return "light" if alpha_mean < light else "medium" if alpha_mean < medium else "dense"


def mean_ci95(values: Sequence[float]) -> dict[str, float]:
    """Mean and a normal-approximation 95 % interval over per-frame scores."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    half = 1.96 * array.std(ddof=1) / np.sqrt(array.size) if array.size > 1 else 0.0
    mean = float(array.mean())
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half, "n": int(array.size)}


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """rows: one dict per frame with psnr, ssim, psnr_input, ssim_input, stratum."""
    report: dict[str, Any] = {}
    groups = {"all": list(rows)}
    for row in rows:
        groups.setdefault(row["stratum"], []).append(row)
    for name, members in groups.items():
        report[name] = {
            key: mean_ci95([r[key] for r in members])
            for key in ("psnr", "ssim", "psnr_input", "ssim_input")
        }
    return report


__all__ = ["DENSITY_EDGES", "density_stratum", "mean_ci95", "psnr", "ssim", "summarize"]
