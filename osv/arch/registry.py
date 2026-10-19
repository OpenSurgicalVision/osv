"""Architectures an experiment may name (R-25).

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

The search space is this table and nothing else: a trainer looks an architecture
up here and refuses any name that is missing. Adding a row is a human-approved PR
against this file, never a flag on a training script. Each row records the task,
how the model is trained, and the paper it reproduces, so a reader can tell a
baseline from a novelty.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Architecture:
    name: str
    task: str
    training: str  # "supervised" | "paired_gan" | "unpaired_gan"
    reference: str


REGISTRY: dict[str, Architecture] = {
    a.name: a
    for a in (
        Architecture(
            "deeplabv3_resnet50",
            "M1 segmentation",
            "supervised",
            "Chen et al., Rethinking Atrous Convolution, arXiv:1706.05587",
        ),
        Architecture(
            "residual_unet",
            "M5 desmoking",
            "supervised",
            "Ronneberger et al., U-Net, MICCAI 2015 (residual output, L1 + SSIM)",
        ),
        Architecture(
            "pix2pix",
            "M5 desmoking",
            "paired_gan",
            "Isola et al., Image-to-Image Translation with Conditional Adversarial "
            "Networks, CVPR 2017 (U-Net generator, 70x70 PatchGAN, L1 weight 100)",
        ),
        Architecture(
            "srgan",
            "M5 desmoking",
            "paired_gan",
            "Ledig et al., Photo-Realistic Single Image Super-Resolution Using a "
            "Generative Adversarial Network, CVPR 2017 (SRResNet generator at scale 1, "
            "VGG19 perceptual loss, adversarial weight 1e-3)",
        ),
        Architecture(
            "cyclegan",
            "M5 desmoking",
            "unpaired_gan",
            "Zhu et al., Unpaired Image-to-Image Translation using Cycle-Consistent "
            "Adversarial Networks, ICCV 2017 (ResNet-9 generators, LSGAN, cycle 10, identity 5)",
        ),
    )
}


def lookup(name: str, *, task: str) -> Architecture:
    entry = REGISTRY.get(name)
    if entry is None or entry.task != task:
        raise ValueError(
            f"unsupported architecture {name!r} for {task}. Widening the search space is a "
            "human PR against osv/arch/registry.py (R-25), not a flag on this script."
        )
    return entry


__all__ = ["REGISTRY", "Architecture", "lookup"]
