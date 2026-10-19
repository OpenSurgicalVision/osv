"""L2 synthetic surgical smoke: composite a rendered smoke layer over a clean frame.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

The smoke layers are Blender volumetric renders saved as straight-alpha RGBA
(`osv/datasets/manifests/blender_smoke.yaml`): RGB is the smoke's own colour
where it exists, alpha its opacity. A clean frame J becomes

    I = J' * (1 - a) + C * a

the single-scattering haze model the desmoking literature trains on, where

* ``a`` is the render's alpha after an optical-depth gain: transmittance
  ``1 - alpha`` is raised to ``gain``, so doubling the gain doubles the smoke's
  optical thickness instead of clipping the alpha at 1;
* ``C`` is the render's colour scaled to the frame's own light level (the
  scope's lamp lights the smoke; a white render pasted at full brightness over
  a dim frame is the first thing a model learns to spot) and a small tint;
* ``J'`` is the clean frame softened behind dense smoke, the forward scatter
  that makes edges behind smoke blur rather than only fade;
* ``a`` is zero on the scope's black vignette (`field_mask`): smoke sits in
  the abdomen, not on the lens housing.

Everything is a pure function of its inputs and a ``SmokeParams`` drawn from a
seeded generator, so the fixed val/test pairs regenerate bit-for-bit and the
training pairs are reproducible per (seed, index). The clean frame is the
ground truth and is never altered beyond this model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

# Ranges are deliberately moderate: the renders already span light wisps to
# dense plumes (alpha up to 1.0), and the gain widens that, it does not invent it.
SCALE_RANGE = (0.8, 1.4)
SHIFT_RANGE = 0.25  # fraction of the frame, each axis
GAIN_RANGE = (0.6, 2.5)
BRIGHTNESS_RANGE = (0.75, 1.05)
TINT_RANGE = 0.04
BLUR_SIGMA_RANGE = (0.0, 2.5)  # pixels at 854 px width, scaled with the frame


@dataclass(frozen=True)
class SmokeParams:
    flip_h: bool
    flip_v: bool
    scale: float
    shift_x: float
    shift_y: float
    gain: float
    brightness: float
    tint: tuple[float, float, float]
    blur_sigma: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sample_params(rng: np.random.Generator) -> SmokeParams:
    """Draw one augmentation from ``rng``; the caller seeds it per pair."""
    tint = rng.uniform(-TINT_RANGE, TINT_RANGE, size=3)
    return SmokeParams(
        flip_h=bool(rng.integers(2)),
        flip_v=bool(rng.integers(2)),
        scale=float(rng.uniform(*SCALE_RANGE)),
        shift_x=float(rng.uniform(-SHIFT_RANGE, SHIFT_RANGE)),
        shift_y=float(rng.uniform(-SHIFT_RANGE, SHIFT_RANGE)),
        gain=float(rng.uniform(*GAIN_RANGE)),
        brightness=float(rng.uniform(*BRIGHTNESS_RANGE)),
        tint=(float(1 + tint[0]), float(1 + tint[1]), float(1 + tint[2])),
        blur_sigma=float(rng.uniform(*BLUR_SIGMA_RANGE)),
    )


def to_float(image: np.ndarray) -> np.ndarray:
    """uint8 HxWxC -> float32 in [0, 1]."""
    if image.dtype != np.uint8:
        raise TypeError(f"expected uint8 pixels, got {image.dtype}")
    return image.astype(np.float32) / 255.0


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)


def warp_smoke(smoke_rgba: np.ndarray, size_hw: tuple[int, int], params: SmokeParams) -> np.ndarray:
    """Fit the render to the frame (cover, keep aspect), then flip, scale and shift it.

    Pixels moved in from outside the render are fully transparent, never smeared
    edge pixels. Returns float32 HxWx4 straight alpha in [0, 1].
    """
    if smoke_rgba.ndim != 3 or smoke_rgba.shape[2] != 4:
        raise ValueError("smoke layer must be HxWx4 RGBA")
    height, width = size_hw
    layer = to_float(smoke_rgba)
    if params.flip_h:
        layer = layer[:, ::-1]
    if params.flip_v:
        layer = layer[::-1]
    src_h, src_w = layer.shape[:2]
    cover = max(width / src_w, height / src_h) * params.scale
    # Centre the scaled render on the frame, then shift by a fraction of the frame.
    tx = width / 2 - cover * src_w / 2 + params.shift_x * width
    ty = height / 2 - cover * src_h / 2 + params.shift_y * height
    matrix = np.array([[cover, 0, tx], [0, cover, ty]], dtype=np.float32)
    # Warp colour premultiplied so transparent pixels' RGB never bleeds into edges.
    premultiplied = np.concatenate([layer[..., :3] * layer[..., 3:], layer[..., 3:]], axis=2)
    warped = cv2.warpAffine(
        np.ascontiguousarray(premultiplied),
        matrix,
        (width, height),
        flags=cv2.INTER_AREA if cover < 1 else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    alpha = np.clip(warped[..., 3:], 0.0, 1.0)
    rgb = np.where(alpha > 1e-6, warped[..., :3] / np.maximum(alpha, 1e-6), 0.0)
    return np.concatenate([np.clip(rgb, 0.0, 1.0), alpha], axis=2).astype(np.float32)


def frame_light(clean: np.ndarray) -> float:
    """The frame's lamp level: 99th percentile of luma, floored so black frames stay sane."""
    luma = clean[..., 0] * 0.299 + clean[..., 1] * 0.587 + clean[..., 2] * 0.114
    return float(max(np.percentile(luma, 99), 0.2))


def field_mask(clean: np.ndarray) -> np.ndarray:
    """Soft mask of the scope's field of view: smoke cannot sit on the black vignette.

    The vignette is near-black and connected to the frame border; a lit dark
    patch inside the field is kept, since only border-connected dark regions go.
    """
    luma = clean[..., 0] * 0.299 + clean[..., 1] * 0.587 + clean[..., 2] * 0.114
    dark = (luma < 0.03).astype(np.uint8)
    count, labels = cv2.connectedComponents(dark, connectivity=4)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    vignette = np.isin(labels, border[border > 0]) & (dark > 0)
    mask = (~vignette).astype(np.float32)
    sigma = max(clean.shape[1] / 854.0 * 3.0, 0.5)
    return np.clip(cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma), 0.0, 1.0)


def composite(
    clean_rgb: np.ndarray, smoke_rgba: np.ndarray, params: SmokeParams
) -> tuple[np.ndarray, np.ndarray]:
    """Return (smoky uint8 HxWx3, effective alpha float32 HxW in [0, 1])."""
    if clean_rgb.ndim != 3 or clean_rgb.shape[2] != 3:
        raise ValueError("clean frame must be HxWx3 RGB")
    height, width = clean_rgb.shape[:2]
    clean = to_float(clean_rgb)
    layer = warp_smoke(smoke_rgba, (height, width), params)
    alpha = (1.0 - np.power(1.0 - layer[..., 3], params.gain)) * field_mask(clean)
    color = (
        layer[..., :3]
        * params.brightness
        * frame_light(clean)
        * np.array(params.tint, dtype=np.float32)
    )
    background = clean
    sigma = params.blur_sigma * width / 854.0
    if sigma > 0.05:
        blurred = cv2.GaussianBlur(clean, (0, 0), sigmaX=sigma, sigmaY=sigma)
        background = clean * (1 - alpha[..., None]) + blurred * alpha[..., None]
    smoky = background * (1 - alpha[..., None]) + np.clip(color, 0, 1) * alpha[..., None]
    return to_uint8(smoky), alpha.astype(np.float32)


def pair_rng(seed: int, index: int) -> np.random.Generator:
    """One independent stream per (seed, pair index): pairs don't shift when others change."""
    return np.random.default_rng([int(seed), int(index)])


__all__ = [
    "SmokeParams",
    "composite",
    "field_mask",
    "frame_light",
    "pair_rng",
    "sample_params",
    "to_float",
    "to_uint8",
    "warp_smoke",
]
