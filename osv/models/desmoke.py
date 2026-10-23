"""M5 baseline: residual U-Net that removes synthetic surgical smoke.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

A deliberately ordinary restoration network, the fixed reference later M5 work
is compared against: a 4-level U-Net (GroupNorm + SiLU) predicting a residual
that is added to the smoky input, plus a one-channel head for the smoke's
effective alpha. The alpha head is auxiliary supervision from the compositing
model (`osv/augment/smoke.py`); it tells the network where the smoke is, and
its output is reported, never used to gate the restored image.

Any H x W works: the input is reflect-padded to a multiple of 2**depth and the
outputs are cropped back, so 854 x 480 frames need no resizing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from osv.arch.registry import lookup

TASK = "M5 desmoking"


def _block(channels_in: int, channels_out: int) -> nn.Sequential:
    groups = 8 if channels_out % 8 == 0 else 1
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, 3, padding=1),
        nn.GroupNorm(groups, channels_out),
        nn.SiLU(inplace=True),
        nn.Conv2d(channels_out, channels_out, 3, padding=1),
        nn.GroupNorm(groups, channels_out),
        nn.SiLU(inplace=True),
    )


class ResidualUNet(nn.Module):
    def __init__(self, base: int = 32, depth: int = 4) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.depth = depth
        widths = [base * 2**level for level in range(depth + 1)]
        self.encoders = nn.ModuleList(
            [_block(3, widths[0])] + [_block(widths[i - 1], widths[i]) for i in range(1, depth + 1)]
        )
        self.ups = nn.ModuleList(
            [nn.ConvTranspose2d(widths[i], widths[i - 1], 2, stride=2) for i in range(depth, 0, -1)]
        )
        self.decoders = nn.ModuleList(
            [_block(widths[i - 1] * 2, widths[i - 1]) for i in range(depth, 0, -1)]
        )
        self.residual = nn.Conv2d(widths[0], 3, 1)
        self.alpha = nn.Conv2d(widths[0], 1, 1)
        # Start as the identity: an untrained network returns its input, not noise.
        nn.init.zeros_(self.residual.weight)
        if self.residual.bias is not None:
            nn.init.zeros_(self.residual.bias)

    def forward(self, smoky: torch.Tensor) -> dict[str, torch.Tensor]:
        height, width = smoky.shape[-2:]
        multiple = 2**self.depth
        pad_h, pad_w = (-height) % multiple, (-width) % multiple
        x = F.pad(smoky, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else smoky
        skips = []
        for level, encoder in enumerate(self.encoders):
            x = encoder(x if level == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, decoder in zip(self.ups, self.decoders, strict=True):
            x = decoder(torch.cat([up(x), skips.pop()], dim=1))
        x = x[..., :height, :width]
        clean = (smoky + self.residual(x)).clamp(0.0, 1.0)
        return {"clean": clean, "alpha": torch.sigmoid(self.alpha(x))}


def build_desmoke_model(*, architecture: str = "residual_unet", base: int = 32) -> nn.Module:
    """A supervised M5 model by registry name; `base` is the U-Net base width / NAFNet width."""
    entry = lookup(architecture, task=TASK)
    if entry.training != "supervised":
        raise ValueError(
            f"{architecture!r} is a GAN: osv.models.desmoke_gan.build_gan builds it and "
            "osv.train.desmoke_gan trains it"
        )
    if architecture == "nafnet":
        from osv.models.restoration_nets import NAFNet

        return NAFNet(width=base)
    if architecture == "dehazeformer_s":
        from osv.models.restoration_nets import DehazeFormer

        return DehazeFormer()  # fixed published dims; `base` does not apply
    return ResidualUNet(base=base)


__all__ = ["TASK", "ResidualUNet", "build_desmoke_model"]
