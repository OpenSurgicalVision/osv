"""M1 synthetic instruments: compositing a rendered instrument layer over a tool-free frame.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

The layer comes from `scripts/blender/instruments.py` (straight-alpha RGBA, a class map
with 0 background / 1 grasper / 2 L-hook, and the pose JSON); the background is a
reviewed tool-free CholecT50 frame (`osv.datasets.m1_backgrounds`). Everything that
moves or hides instrument pixels moves the label with them, so the mask stays exact:

* **geometry** - the layer is flipped horizontally and/or vertically (instruments enter
  from every border), with the tip coordinates flipped too;
* **light** - gain from the background's lamp level, per-channel colour gain, gamma;
* **optics** - motion blur along a random direction and defocus, applied to the
  premultiplied layer and to each class's coverage alike;
* **occlusion** - a soft ellipse of background over an instrument tip (tissue over the
  jaws), removing those pixels from the label;
* **field of view** - the scope's black vignette, taken from the background
  (`osv.augment.smoke.field_mask`), hides instrument pixels there, and they are labelled
  background, as CholecSeg8k does;
* **smoke** (optional) - an M5 smoke layer over the finished frame; the instrument is
  still there under it, so the label is unchanged;
* **sensor** - Gaussian noise and JPEG.

A pixel is labelled with the class whose blurred coverage is largest, if the total
coverage after occlusion and vignette is at least 0.5; otherwise background.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from osv.augment import smoke as S

NUM_CLASSES = 3  # background, grasper, hook


@dataclass(frozen=True)
class InstrumentParams:
    flip_h: bool
    flip_v: bool
    gain: float
    channel_gain: tuple[float, float, float]
    gamma: float
    motion_px: float
    motion_angle_deg: float
    defocus_sigma: float
    occlude: bool
    occlude_radius_px: float
    noise_sigma: float
    jpeg_quality: int
    smoke: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sample_params(rng: np.random.Generator, *, smoke_probability: float = 0.0) -> InstrumentParams:
    return InstrumentParams(
        flip_h=bool(rng.random() < 0.5),
        flip_v=bool(rng.random() < 0.5),
        gain=float(rng.uniform(0.8, 1.3)),
        channel_gain=tuple(float(g) for g in rng.uniform(0.92, 1.08, 3)),  # type: ignore[arg-type]
        gamma=float(rng.uniform(0.85, 1.15)),
        motion_px=float(rng.uniform(3, 15)) if rng.random() < 0.3 else 0.0,
        motion_angle_deg=float(rng.uniform(0, 180)),
        defocus_sigma=float(rng.uniform(0.5, 2.0)) if rng.random() < 0.3 else 0.0,
        occlude=bool(rng.random() < 0.3),
        occlude_radius_px=float(rng.uniform(8, 35)),
        noise_sigma=float(rng.uniform(0, 3.0)),
        jpeg_quality=int(rng.integers(60, 96)),
        smoke=bool(rng.random() < smoke_probability),
    )


def _motion_kernel(length: float, angle_deg: float) -> np.ndarray:
    size = max(int(round(length)) | 1, 3)
    kernel: np.ndarray = np.zeros((size, size), np.float32)
    kernel[size // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (size, size))
    return kernel / max(float(kernel.sum()), 1e-6)


def _optics(image: np.ndarray, params: InstrumentParams) -> np.ndarray:
    if params.motion_px > 0:
        image = cv2.filter2D(image, -1, _motion_kernel(params.motion_px, params.motion_angle_deg))
    if params.defocus_sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), params.defocus_sigma)
    return image


def tips(meta: dict[str, Any], *, width: int, height: int, params: InstrumentParams) -> list:
    """Instrument tip pixels after the flips."""
    points = []
    for instance in meta.get("instances", []):
        x, y = instance["tip_px"]
        if params.flip_h:
            x = width - 1 - x
        if params.flip_v:
            y = height - 1 - y
        points.append((float(x), float(y)))
    return points


def composite(
    background: np.ndarray,
    layer_rgba: np.ndarray,
    layer_class: np.ndarray,
    meta: dict[str, Any],
    params: InstrumentParams,
    rng: np.random.Generator,
    *,
    smoke_rgba: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(image uint8 HxWx3, label uint8 HxW in {0, 1, 2})."""
    height, width = background.shape[:2]
    if layer_rgba.shape[:2] != (height, width) or layer_class.shape != (height, width):
        raise ValueError("render and background must have the same size")
    rgb = layer_rgba[..., :3].astype(np.float32) / 255.0
    alpha = layer_rgba[..., 3].astype(np.float32) / 255.0
    cls = layer_class
    if params.flip_h:
        rgb, alpha, cls = rgb[:, ::-1], alpha[:, ::-1], cls[:, ::-1]
    if params.flip_v:
        rgb, alpha, cls = rgb[::-1], alpha[::-1], cls[::-1]

    back = background.astype(np.float32) / 255.0
    # The renders are lit for a bright frame; follow the frame's lamp level, within limits.
    light = float(np.clip(S.frame_light(back) / 0.85, 0.85, 1.15))
    rgb = np.clip(rgb * params.gain * light * np.array(params.channel_gain, np.float32), 0, 1)
    rgb = np.power(rgb, params.gamma)

    # Coverage per class, blurred like the image so label and pixels stay together.
    coverage = np.stack([alpha * (cls == k) for k in range(1, NUM_CLASSES)], axis=-1)
    premultiplied = _optics(np.ascontiguousarray(rgb * alpha[..., None]), params)
    coverage = _optics(np.ascontiguousarray(coverage), params)
    if coverage.ndim == 2:
        coverage = coverage[..., None]
    total = np.clip(coverage.sum(axis=-1), 0, 1)

    keep = S.field_mask(back)
    if params.occlude:
        hole: np.ndarray = np.ones((height, width), np.float32)
        points = tips(meta, width=width, height=height, params=params)
        if points:
            x, y = points[int(rng.integers(len(points)))]
            axes = (
                int(params.occlude_radius_px),
                int(params.occlude_radius_px * rng.uniform(0.5, 1)),
            )
            cv2.ellipse(hole, (int(x), int(y)), axes, float(rng.uniform(0, 180)), 0, 360, 0.0, -1)
            hole = cv2.GaussianBlur(hole, (0, 0), 2.0)
        keep = keep * hole
    scale = np.where(total > 0, keep, 0.0)
    premultiplied = premultiplied * scale[..., None]
    visible = total * scale

    image = premultiplied + back * (1 - visible[..., None])
    label = np.where(visible >= 0.5, coverage.argmax(axis=-1) + 1, 0).astype(np.uint8)

    out = np.clip(image * 255, 0, 255).astype(np.uint8)
    if params.smoke and smoke_rgba is not None:
        out, _ = S.composite(out, smoke_rgba, S.sample_params(rng))
    if params.noise_sigma > 0:
        noise = rng.normal(0, params.noise_sigma, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(
        ".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, params.jpeg_quality]
    )
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else None
    if decoded is not None:
        out = decoded[..., ::-1]
    return np.ascontiguousarray(out), label


__all__ = ["NUM_CLASSES", "InstrumentParams", "composite", "sample_params", "tips"]
