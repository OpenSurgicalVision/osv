"""M5 desmoking: NAFNet and DehazeFormer, the two published restoration baselines of session 2.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

Re-implemented from the papers and their official code, with three adaptations that
every M5 supervised model shares, so that only the architecture differs from the
residual U-Net (`osv/models/desmoke.py`):

* the output is a correction of the smoky input, and the layer producing it starts
  at zero, so an untrained network returns its input (the no-op baseline) instead
  of noise;
* a one-channel smoke-alpha head is added for the same auxiliary supervision the
  U-Net gets (reported, never used to gate the image);
* any frame size works: the input is reflect-padded and the outputs cropped back.

**NAFNet** - Chen, Chu, Zhang, Sun, *Simple Baselines for Image Restoration*, ECCV 2022
(github.com/megvii-research/NAFNet). The SIDD configuration at width 32: blocks
[2, 2, 4, 8] / 12 / [2, 2, 2, 2]; NAFBlocks with LayerNorm, depth-wise conv, the
SimpleGate in place of activations and simplified channel attention; the global
residual of the paper.

**DehazeFormer-S** - Song, He, Qian, Du, *Vision Transformers for Single Image
Dehazing*, IEEE TIP 32 (2023) 1927-1941 (github.com/IDKiro/DehazeFormer). Embedding
dims [24, 48, 96, 48, 24], depths [8, 8, 8, 4, 4], heads [2, 4, 6, 1, 1], attention in
the last [1/4, 1/2, 3/4, 0, 0] of each stage, window 8 shifted in every other block,
depth-wise conv branch, RescaleNorm, log-spaced relative positions, SK fusion of the
skips, and the soft reconstruction ``K * I - B + I`` from the atmospheric scattering
model - the same model the synthetic smoke is composited with. It works in [-1, 1]
as in the paper; the wrapper converts.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bias(conv: nn.Conv2d) -> torch.Tensor:
    if conv.bias is None:
        raise ValueError("expected a convolution with a bias")
    return conv.bias


def _zero(conv: nn.Conv2d) -> None:
    """Zero the output layer: the network starts as the identity on its input."""
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(_bias(conv))


def _pad_to(x: torch.Tensor, multiple: int) -> torch.Tensor:
    height, width = x.shape[-2:]
    pad_h, pad_w = (-height) % multiple, (-width) % multiple
    if not (pad_h or pad_w):
        return x
    mode = "reflect" if pad_h < height and pad_w < width else "replicate"
    return F.pad(x, (0, pad_w, 0, pad_h), mode=mode)


# --- NAFNet ------------------------------------------------------------------------------


class LayerNorm2d(nn.Module):
    """LayerNorm over channels of an N x C x H x W tensor, statistics in float32."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        y = (x - mean) / (var + self.eps).sqrt()
        y = self.weight.view(1, -1, 1, 1) * y + self.bias.view(1, -1, 1, 1)
        return y.to(dtype)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2) -> None:
        super().__init__()
        dw = channels * dw_expand
        self.conv1 = nn.Conv2d(channels, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.conv3 = nn.Conv2d(dw // 2, channels, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.sg = SimpleGate()
        ffn = channels * ffn_expand
        self.conv4 = nn.Conv2d(channels, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, channels, 1)
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.sg(self.conv2(self.conv1(self.norm1(inp))))
        x = self.conv3(x * self.sca(x))
        y = inp + x * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(
        self,
        width: int = 32,
        enc_blocks: tuple[int, ...] = (2, 2, 4, 8),
        middle_blocks: int = 12,
        dec_blocks: tuple[int, ...] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        channels = width
        for count in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(count)]))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2
        self.middle = nn.Sequential(*[NAFBlock(channels) for _ in range(middle_blocks)])
        self.ups, self.decoders = nn.ModuleList(), nn.ModuleList()
        for count in dec_blocks:
            self.ups.append(
                nn.Sequential(nn.Conv2d(channels, channels * 2, 1, bias=False), nn.PixelShuffle(2))
            )
            channels //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(count)]))
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.alpha = nn.Conv2d(width, 1, 3, padding=1)
        _zero(self.ending)
        self.multiple = 2 ** len(enc_blocks)

    def forward(self, smoky: torch.Tensor) -> dict[str, torch.Tensor]:
        height, width = smoky.shape[-2:]
        x = self.intro(_pad_to(smoky, self.multiple))
        skips = []
        for encoder, down in zip(self.encoders, self.downs, strict=True):
            x = encoder(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips), strict=True):
            x = decoder(up(x) + skip)
        x = x[..., :height, :width]
        clean = (smoky + self.ending(x)).clamp(0.0, 1.0)
        return {"clean": clean, "alpha": torch.sigmoid(self.alpha(x))}


# --- DehazeFormer ------------------------------------------------------------------------


class RLN(nn.Module):
    """Rescale LayerNorm: normalise over C, H, W and return the rescale / rebias terms."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)
        nn.init.trunc_normal_(self.meta1.weight, std=0.02)
        nn.init.constant_(_bias(self.meta1), 1)
        nn.init.trunc_normal_(self.meta2.weight, std=0.02)
        nn.init.constant_(_bias(self.meta2), 0)

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = ((x - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps).sqrt()
        out = (x - mean) / std * self.weight + self.bias
        rescale, rebias = self.meta1(std), self.meta2(mean)
        return out.to(dtype), rescale.to(dtype), rebias.to(dtype)


def _trunc_init(conv: nn.Conv2d, gain: float) -> None:
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(conv.weight)
    nn.init.trunc_normal_(conv.weight, std=gain * math.sqrt(2.0 / float(fan_in + fan_out)))
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class Mlp(nn.Module):
    def __init__(self, network_depth: int, dim: int, hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1), nn.ReLU(True), nn.Conv2d(hidden, dim, 1)
        )
        for module in self.mlp:
            if isinstance(module, nn.Conv2d):
                _trunc_init(module, (8 * network_depth) ** (-1 / 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def window_partition(x: torch.Tensor, window: int) -> torch.Tensor:
    batch, height, width, channels = x.shape
    x = x.view(batch, height // window, window, width // window, window, channels)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window * window, channels)


def window_reverse(windows: torch.Tensor, window: int, height: int, width: int) -> torch.Tensor:
    batch = windows.shape[0] // ((height // window) * (width // window))
    x = windows.view(batch, height // window, width // window, window, window, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)


def relative_positions(window: int) -> torch.Tensor:
    """Log-spaced (dy, dx) between every pair of positions in a window: N x N x 2."""
    coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij"))
    flat = torch.flatten(coords, 1)
    relative = (flat[:, :, None] - flat[:, None, :]).permute(1, 2, 0).contiguous().float()
    return torch.sign(relative) * torch.log1p(relative.abs())


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window: int, heads: int) -> None:
        super().__init__()
        self.dim, self.heads = dim, heads
        self.scale = (dim // heads) ** -0.5
        self.register_buffer("positions", relative_positions(window), persistent=False)
        self.meta = nn.Sequential(nn.Linear(2, 256), nn.ReLU(True), nn.Linear(256, heads))

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = qkv.shape
        qkv = qkv.reshape(batch, tokens, 3, self.heads, self.dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.meta(self.positions).permute(2, 0, 1).contiguous()
        attn = (attn + bias.unsqueeze(0).to(attn.dtype)).softmax(dim=-1)
        return (attn @ v).transpose(1, 2).reshape(batch, tokens, self.dim)


class Attention(nn.Module):
    def __init__(
        self, network_depth: int, dim: int, heads: int, window: int, shift: int, use_attn: bool
    ) -> None:
        super().__init__()
        self.dim, self.window, self.shift, self.use_attn = dim, window, shift, use_attn
        self.conv = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, padding_mode="reflect")
        self.V = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        if use_attn:
            self.QK = nn.Conv2d(dim, dim * 2, 1)
            self.attn = WindowAttention(dim, window, heads)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                is_qk = module.weight.shape[0] == dim * 2
                _trunc_init(module, 1.0 if is_qk else (8 * network_depth) ** (-1 / 4))

    def _pad(self, x: torch.Tensor, shift: bool) -> torch.Tensor:
        height, width = x.shape[-2:]
        pad_h, pad_w = (-height) % self.window, (-width) % self.window
        if shift:
            s = self.shift
            return F.pad(
                x,
                (
                    s,
                    (self.window - s + pad_w) % self.window,
                    s,
                    (self.window - s + pad_h) % self.window,
                ),
                mode="reflect",
            )
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        value = self.V(x)
        if not self.use_attn:
            return self.proj(self.conv(value))
        qkv = self._pad(torch.cat([self.QK(x), value], dim=1), self.shift > 0)
        padded_h, padded_w = qkv.shape[-2:]
        windows = window_partition(qkv.permute(0, 2, 3, 1), self.window)
        out = window_reverse(self.attn(windows), self.window, padded_h, padded_w)
        s = self.shift
        attn_out = out[:, s : s + height, s : s + width, :].permute(0, 3, 1, 2)
        return self.proj(self.conv(value) + attn_out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        network_depth: int,
        dim: int,
        heads: int,
        mlp_ratio: float,
        window: int,
        shift: int,
        use_attn: bool,
    ) -> None:
        super().__init__()
        self.use_attn = use_attn
        self.norm1 = RLN(dim) if use_attn else None
        self.attn = Attention(network_depth, dim, heads, window, shift, use_attn)
        self.mlp = Mlp(network_depth, dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.norm1 is not None:
            x, rescale, rebias = self.norm1(x)
            x = self.attn(x) * rescale + rebias
        else:
            x = self.attn(x)
        x = identity + x
        return x + self.mlp(x)


class BasicLayer(nn.Module):
    def __init__(
        self,
        network_depth: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
        window: int,
        attn_ratio: float,
    ) -> None:
        super().__init__()
        attn_depth = attn_ratio * depth
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    network_depth,
                    dim,
                    heads,
                    mlp_ratio,
                    window,
                    shift=0 if i % 2 == 0 else window // 2,
                    use_attn=i >= depth - attn_depth,  # attention in the last blocks
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


def _patch_embed(patch: int, in_chans: int, dim: int, kernel: int | None = None) -> nn.Conv2d:
    kernel = patch if kernel is None else kernel
    padding = (kernel - patch + 1) // 2
    return nn.Conv2d(in_chans, dim, kernel, stride=patch, padding=padding, padding_mode="reflect")


def _patch_unembed(patch: int, out_chans: int, dim: int, kernel: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(dim, out_chans * patch**2, kernel, padding=kernel // 2, padding_mode="reflect"),
        nn.PixelShuffle(patch),
    )


class SKFusion(nn.Module):
    def __init__(self, dim: int, height: int = 2, reduction: int = 8) -> None:
        super().__init__()
        self.height = height
        hidden = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, dim * height, 1, bias=False),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        batch, channels, height, width = features[0].shape
        stacked = torch.cat(features, dim=1).view(batch, self.height, channels, height, width)
        attn = self.mlp(self.avg_pool(stacked.sum(dim=1)))
        attn = attn.view(batch, self.height, channels, 1, 1).softmax(dim=1)
        return (stacked * attn).sum(dim=1)


class DehazeFormer(nn.Module):
    def __init__(
        self,
        embed_dims: tuple[int, ...] = (24, 48, 96, 48, 24),
        mlp_ratios: tuple[float, ...] = (2.0, 4.0, 4.0, 2.0, 2.0),
        depths: tuple[int, ...] = (8, 8, 8, 4, 4),
        heads: tuple[int, ...] = (2, 4, 6, 1, 1),
        attn_ratios: tuple[float, ...] = (1 / 4, 1 / 2, 3 / 4, 0.0, 0.0),
        window: int = 8,
    ) -> None:
        super().__init__()
        if embed_dims[1] != embed_dims[3] or embed_dims[0] != embed_dims[4]:
            raise ValueError("DehazeFormer's skips need embed_dims[1] == [3] and [0] == [4]")
        depth = sum(depths)

        def layer(i: int) -> BasicLayer:
            return BasicLayer(
                depth, embed_dims[i], depths[i], heads[i], mlp_ratios[i], window, attn_ratios[i]
            )

        self.patch_embed = _patch_embed(1, 3, embed_dims[0], kernel=3)
        self.layer1 = layer(0)
        self.patch_merge1 = _patch_embed(2, embed_dims[0], embed_dims[1])
        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)
        self.layer2 = layer(1)
        self.patch_merge2 = _patch_embed(2, embed_dims[1], embed_dims[2])
        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)
        self.layer3 = layer(2)
        self.patch_split1 = _patch_unembed(2, embed_dims[3], embed_dims[2])
        self.fusion1 = SKFusion(embed_dims[3])
        self.layer4 = layer(3)
        self.patch_split2 = _patch_unembed(2, embed_dims[4], embed_dims[3])
        self.fusion2 = SKFusion(embed_dims[4])
        self.layer5 = layer(4)
        # K (1), B (3) of the soft reconstruction, plus the smoke-alpha logit (1).
        self.patch_unembed = _patch_unembed(1, 5, embed_dims[4], kernel=3)
        head = self.patch_unembed[0]
        assert isinstance(head, nn.Conv2d)
        _zero(head)

    def forward(self, smoky: torch.Tensor) -> dict[str, torch.Tensor]:
        height, width = smoky.shape[-2:]
        x = _pad_to(smoky * 2 - 1, 4)
        feat = self.layer1(self.patch_embed(x))
        skip1 = feat
        feat = self.layer2(self.patch_merge1(feat))
        skip2 = feat
        feat = self.patch_split1(self.layer3(self.patch_merge2(feat)))
        feat = self.fusion1([feat, self.skip2(skip2)]) + feat
        feat = self.patch_split2(self.layer4(feat))
        feat = self.fusion2([feat, self.skip1(skip1)]) + feat
        feat = self.patch_unembed(self.layer5(feat))
        k, b, alpha = torch.split(feat, [1, 3, 1], dim=1)
        restored = (k * x - b + x)[..., :height, :width]
        clean = ((restored + 1) / 2).clamp(0.0, 1.0)
        return {"clean": clean, "alpha": torch.sigmoid(alpha[..., :height, :width])}


__all__ = ["DehazeFormer", "NAFNet"]
