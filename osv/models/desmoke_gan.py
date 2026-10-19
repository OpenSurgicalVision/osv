"""M5 GAN baselines: Pix2Pix, SRGAN (at scale 1) and CycleGAN for surgical desmoking.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

Faithful, ordinary reproductions of three published designs (rows in
`osv/arch/registry.py`), adapted only where desmoking differs from the paper's task:

* **Pix2Pix** (Isola et al. 2017): U-Net generator with skip connections and
  decoder dropout, conditional 70x70 PatchGAN on (input, output).
* **SRGAN** (Ledig et al. 2017) at scale 1: the SRResNet generator without its
  pixel-shuffle upsampling, the paper's VGG-style global discriminator, and a
  VGG19 relu5_4 perceptual loss. The generator predicts a residual on its input
  so an untrained network is the identity.
* **CycleGAN** (Zhu et al. 2017): ResNet-9 generators smoky->clean and
  clean->smoky, two 70x70 PatchGANs, trained on unpaired real smoky and clean
  CholecT50 frames.

Every generator is wrapped by `PixelRange`: callers pass and get [0, 1] images and
a dict with "clean", like the supervised baseline, so evaluation is shared. GAN
outputs can invent plausible texture where the input carries none; the shared
evaluation scores them against the clean frame for exactly that reason.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VGG19_RELU5_4 = 35  # index of relu5_4 in torchvision vgg19().features
VGG_FEATURE_SCALE = 1 / 12.75  # Ledig et al.: brings the VGG loss to the scale of MSE


class PixelRange(nn.Module):
    """[0, 1] in and out around a [-1, 1] generator; pads to `multiple` and crops back."""

    def __init__(self, generator: nn.Module, multiple: int) -> None:
        super().__init__()
        self.generator, self.multiple = generator, multiple

    def forward(self, smoky: torch.Tensor) -> dict[str, torch.Tensor]:
        height, width = smoky.shape[-2:]
        pad_h, pad_w = (-height) % self.multiple, (-width) % self.multiple
        x = smoky * 2 - 1
        if pad_h or pad_w:
            # Reflection needs the pad to be smaller than the side; tiny crops replicate.
            mode = "reflect" if pad_h < height and pad_w < width else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        y = self.generator(x)[..., :height, :width]
        return {"clean": ((y + 1) / 2).clamp(0.0, 1.0)}


# --- Pix2Pix -----------------------------------------------------------------------------


class _UnetSkip(nn.Module):
    def __init__(
        self,
        outer: int,
        inner: int,
        *,
        sub: nn.Module | None,
        outermost: bool = False,
        dropout: bool = False,
        in_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.outermost = outermost
        in_channels = in_channels or outer
        down = nn.Conv2d(in_channels, inner, 4, stride=2, padding=1, bias=False)
        layers: list[nn.Module | None]
        if outermost:
            up = nn.ConvTranspose2d(inner * 2, outer, 4, stride=2, padding=1)
            layers = [down, sub, nn.ReLU(True), up, nn.Tanh()]
        elif sub is None:  # innermost
            up = nn.ConvTranspose2d(inner, outer, 4, stride=2, padding=1, bias=False)
            layers = [nn.LeakyReLU(0.2, True), down, nn.ReLU(True), up, nn.BatchNorm2d(outer)]
        else:
            up = nn.ConvTranspose2d(inner * 2, outer, 4, stride=2, padding=1, bias=False)
            layers = [
                nn.LeakyReLU(0.2, True),
                down,
                nn.BatchNorm2d(inner),
                sub,
                nn.ReLU(True),
                up,
                nn.BatchNorm2d(outer),
            ]
            if dropout:
                layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*[layer for layer in layers if layer is not None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) if self.outermost else torch.cat([x, self.model(x)], dim=1)


def unet_generator(num_downs: int = 7, ngf: int = 64) -> nn.Module:
    block = _UnetSkip(ngf * 8, ngf * 8, sub=None)
    for _ in range(num_downs - 5):
        block = _UnetSkip(ngf * 8, ngf * 8, sub=block, dropout=True)
    for scale in (4, 2, 1):
        block = _UnetSkip(ngf * scale, ngf * scale * 2, sub=block)
    return _UnetSkip(3, ngf, sub=block, outermost=True, in_channels=3)


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN (n_layers=3): one real/fake logit per overlapping patch."""

    def __init__(self, in_channels: int, ndf: int = 64, *, norm: str = "batch") -> None:
        super().__init__()
        Norm = nn.BatchNorm2d if norm == "batch" else nn.InstanceNorm2d  # noqa: N806
        layers: list[nn.Module] = [nn.Conv2d(in_channels, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        width = ndf
        for stride, mult in ((2, 2), (2, 4), (1, 8)):
            layers += [
                nn.Conv2d(width, ndf * mult, 4, stride, 1, bias=False),
                Norm(ndf * mult),
                nn.LeakyReLU(0.2, True),
            ]
            width = ndf * mult
        layers.append(nn.Conv2d(width, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# --- SRGAN at scale 1 --------------------------------------------------------------------


class _SRResidual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class SRResNetGenerator(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 16) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(3, channels, 9, padding=4), nn.PReLU())
        self.body = nn.Sequential(
            *[_SRResidual(channels) for _ in range(blocks)],
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.tail = nn.Conv2d(channels, 3, 9, padding=4)
        nn.init.zeros_(self.tail.weight)
        if self.tail.bias is not None:
            nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.head(x)
        return (x + self.tail(features + self.body(features))).clamp(-1.0, 1.0)


class SRGANDiscriminator(nn.Module):
    """The paper's 8-conv discriminator; adaptive pooling lets it take any crop size."""

    def __init__(self, ndf: int = 64) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(3, ndf, 3, 1, 1), nn.LeakyReLU(0.2, True)]
        width = ndf
        for mult, stride in ((1, 2), (2, 1), (2, 2), (4, 1), (4, 2), (8, 1), (8, 2)):
            layers += [
                nn.Conv2d(width, ndf * mult, 3, stride, 1, bias=False),
                nn.BatchNorm2d(ndf * mult),
                nn.LeakyReLU(0.2, True),
            ]
            width = ndf * mult
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class VGGPerceptual(nn.Module):
    """MSE between rescaled VGG19 features (relu5_4 by default) of two [0, 1] images.

    Weights come from a local file, never a download (R-07 keeps download.pytorch.org
    out): torchvision's `vgg19-dcbb9e9d.pth` state dict, or `timm/vgg19.tv_in1k`
    model.safetensors when the safetensors package is installed.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self,
        weights: str | Path | None,
        *,
        layer: int = VGG19_RELU5_4,
        feature_scale: float = VGG_FEATURE_SCALE,
    ) -> None:
        super().__init__()
        self.feature_scale = feature_scale
        from torchvision.models.vgg import cfgs, make_layers

        # The convolutional trunk alone: vgg19()'s classifier is 120M unused parameters.
        self.features = make_layers(cfgs["E"])[: layer + 1].eval()
        self.weights_sha256: str | None = None
        if weights is not None:
            path = Path(weights)
            self.weights_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.suffix == ".safetensors":
                try:
                    from safetensors.torch import load_file  # type: ignore[import-not-found]
                except ImportError as error:
                    raise SystemExit(
                        f"{path} is a .safetensors file but safetensors is not installed: "
                        "pip install safetensors, or pass torchvision's vgg19 .pth"
                    ) from error
                state = load_file(str(path))
            else:
                state = torch.load(path, map_location="cpu", weights_only=True)
            own = {}
            for key, value in state.items():
                name = key.removeprefix("features.")
                if name.split(".")[0].isdigit() and int(name.split(".")[0]) <= layer:
                    own[name] = value
            missing = set(self.features.state_dict()) - set(own)
            if missing:
                raise ValueError(
                    f"VGG19 weights lack {sorted(missing)[:4]}...: not a vgg19 state dict"
                )
            self.features.load_state_dict(own)
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        a = self.features((pred - self.mean) / self.std) * self.feature_scale
        b = self.features((target - self.mean) / self.std) * self.feature_scale
        return F.mse_loss(a, b)


# --- CycleGAN ----------------------------------------------------------------------------


class _ResnetBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


def resnet_generator(ngf: int = 64, blocks: int = 9) -> nn.Module:
    layers: list[nn.Module] = [
        nn.ReflectionPad2d(3),
        nn.Conv2d(3, ngf, 7),
        nn.InstanceNorm2d(ngf),
        nn.ReLU(True),
    ]
    for mult in (1, 2):
        layers += [
            nn.Conv2d(ngf * mult, ngf * mult * 2, 3, 2, 1),
            nn.InstanceNorm2d(ngf * mult * 2),
            nn.ReLU(True),
        ]
    layers += [_ResnetBlock(ngf * 4) for _ in range(blocks)]
    for mult in (4, 2):
        layers += [
            nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(ngf * mult // 2),
            nn.ReLU(True),
        ]
    layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, 7), nn.Tanh()]
    return nn.Sequential(*layers)


class ImagePool:
    """CycleGAN's history of 50 generated images: discriminators see past fakes too."""

    def __init__(self, size: int = 50, *, generator: torch.Generator | None = None) -> None:
        self.size, self.random = size, generator or torch.Generator()
        self.images: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        out = []
        for image in images.detach():
            if len(self.images) < self.size:
                self.images.append(image)
                out.append(image)
            elif float(torch.rand(1, generator=self.random)) < 0.5:
                index = int(torch.randint(len(self.images), (1,), generator=self.random))
                out.append(self.images[index].clone())
                self.images[index] = image
            else:
                out.append(image)
        return torch.stack(out)


def init_normal(net: nn.Module, gain: float = 0.02) -> nn.Module:
    """The pix2pix / CycleGAN initialisation: N(0, 0.02) weights, N(1, 0.02) batch-norm scales."""
    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(module.weight, 0.0, gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, gain)
            nn.init.zeros_(module.bias)
    return net


def build_gan(architecture: str, *, width: int = 64) -> dict[str, nn.Module]:
    """Generator(s) wrapped in PixelRange plus discriminator(s), by registry name.

    CycleGAN's "generator" maps smoky to clean and is the one evaluated;
    "generator_back" maps clean to smoky. "discriminator" judges clean frames
    there, "discriminator_back" smoky ones.
    """
    if architecture == "pix2pix":
        return {
            "generator": PixelRange(init_normal(unet_generator(ngf=width)), 2**7),
            "discriminator": init_normal(PatchDiscriminator(6, width)),
        }
    if architecture == "srgan":  # default init: the zero tail keeps the generator the identity
        return {
            "generator": PixelRange(SRResNetGenerator(width), 1),
            "discriminator": SRGANDiscriminator(width),
        }
    if architecture == "cyclegan":
        return {
            "generator": PixelRange(init_normal(resnet_generator(width)), 4),
            "generator_back": PixelRange(init_normal(resnet_generator(width)), 4),
            "discriminator": init_normal(PatchDiscriminator(3, width, norm="instance")),
            "discriminator_back": init_normal(PatchDiscriminator(3, width, norm="instance")),
        }
    raise ValueError(f"{architecture!r} is not a GAN in osv/arch/registry.py")


__all__ = [
    "ImagePool",
    "PatchDiscriminator",
    "PixelRange",
    "SRGANDiscriminator",
    "SRResNetGenerator",
    "VGGPerceptual",
    "build_gan",
    "init_normal",
    "resnet_generator",
    "unet_generator",
]
