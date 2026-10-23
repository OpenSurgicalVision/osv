"""Per-epoch preview sheets for M5 desmoking: the same frames, restored after every epoch.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

The second M5 session tests whether PSNR and SSIM are the right yardstick at all: a
person looks at the restored frames after every epoch and judges them by eye. For
that the frames must be the same every epoch, chosen before training, and cover what
the metrics might miss. Three groups, each one PNG sheet per epoch:

* ``train`` - synthetic train pairs (one fixed composite each: set_epoch(0)), spread
  over the train videos: is the model learning what it is shown?
* ``val`` - fixed val pairs, chosen across the light / medium / dense strata: the
  held-out videos the metrics are computed on.
* ``real`` - the build's real hazy val frames (`real_smoke/val`), which have no clean
  target and no metric at all: the case that matters, judged only by eye.

The held-out **test** split never appears: looking at it after every epoch would be
choosing a model on test (R-25). Val is held out from training as well (other videos).

Rows are frames. Paired groups show input | output | target | error, where error is
``|output - target|`` amplified 4x; the caption has the frame's PSNR and SSIM for
input and output, so the eye and the metric can be compared tile by tile. The real
group shows input | output | change (``|output - input|`` x4).

Sheets go to ``<out>/previews/epoch_NNN_<group>.png`` (``epoch_init`` is the untrained
model, before the first step), next to an ``index.html`` that lays all epochs out for
a browser once the folder is downloaded from the pod. Nothing is uploaded anywhere:
the frames are CholecT50 data and stay on the pod until the owner copies them over SSH.
"""

from __future__ import annotations

import html
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from osv.datasets.desmoke import SyntheticSmokePairs
from osv.eval import restoration as R

GROUPS = ("train", "val", "real")
ERROR_GAIN = 4.0
PREVIEW_SPLIT = "val"


def _spread(items: list[Any], count: int, key: Callable[[Any], str], seed: int) -> list[Any]:
    """Up to `count` items, round-robin over `key` groups (videos, strata), seeded."""
    rng = random.Random(seed)
    buckets: dict[str, list[Any]] = {}
    for item in items:
        buckets.setdefault(key(item), []).append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    order = sorted(buckets)
    picked: list[Any] = []
    while len(picked) < count and any(buckets[name] for name in order):
        for name in order:
            if buckets[name] and len(picked) < count:
                picked.append(buckets[name].pop())
    return picked


class PreviewSet:
    """The fixed frames of one run's previews, chosen once from the build's index."""

    def __init__(self, processed_dir: Path, *, count: int, seed: int) -> None:
        self.root = Path(processed_dir)
        index = json.loads((self.root / "index.json").read_text())
        self.count = count
        train_video = {r["frame_id"]: r["video"] for r in index["clean"] if r["split"] == "train"}
        self.train = SyntheticSmokePairs("train", processed_dir=self.root, seed=seed)
        self.train.set_epoch(0)
        positions = list(range(len(self.train)))
        self.train_items = _spread(
            positions, count, key=lambda i: train_video.get(self.train.clean[i], "?"), seed=seed
        )
        self.val = SyntheticSmokePairs(PREVIEW_SPLIT, processed_dir=self.root)
        stratum = {p["pair_id"]: R.density_stratum(p["alpha_mean"]) for p in self.val.pairs}
        self.val_items = _spread(
            list(range(len(self.val))),
            count,
            key=lambda i: stratum[self.val.pairs[i]["pair_id"]],
            seed=seed,
        )
        real = sorted(
            r["frame_id"] for r in index.get("real_smoke", []) if r["split"] == PREVIEW_SPLIT
        )
        step = max(len(real) // max(count, 1), 1)
        self.real_items = real[::step][:count]

    def describe(self) -> dict[str, list[str]]:
        return {
            "train": [self.train.clean[i] for i in self.train_items],
            "val": [self.val.pairs[i]["pair_id"] for i in self.val_items],
            "real": list(self.real_items),
        }


def _to_uint8(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return np.rint(array * 255).astype(np.uint8)


def _difference(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    return _to_uint8(((a - b).abs() * ERROR_GAIN).clamp(0, 1))


def _sheet(
    rows: list[tuple[list[np.ndarray], str]], header: list[str], scale: float
) -> Image.Image:
    height, width = rows[0][0][0].shape[:2]
    tile_w, tile_h = int(width * scale), int(height * scale)
    caption_h, header_h = 26, 30
    sheet = Image.new(
        "RGB",
        (tile_w * len(header), header_h + len(rows) * (tile_h + caption_h)),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for column, title in enumerate(header):
        draw.text((column * tile_w + 8, 6), title, fill=(255, 255, 255), font=font)
    for row, (tiles, caption) in enumerate(rows):
        top = header_h + row * (tile_h + caption_h)
        draw.text((8, top + 4), caption, fill=(255, 220, 0), font=font)
        for column, tile in enumerate(tiles):
            image = Image.fromarray(tile).resize((tile_w, tile_h))
            sheet.paste(image, (column * tile_w, top + caption_h))
    return sheet


@torch.no_grad()
def render(
    model: torch.nn.Module, previews: PreviewSet, device: str, *, scale: float = 0.5
) -> dict[str, Image.Image]:
    """One sheet per group for the model as it is now; leaves it in eval mode."""
    model.eval()
    sheets: dict[str, Image.Image] = {}
    for group, dataset, items in (
        ("train", previews.train, previews.train_items),
        ("val", previews.val, previews.val_items),
    ):
        rows = []
        for item in items:
            sample = dataset[item]
            smoky = sample["smoky"][None].to(device)
            clean = sample["clean"][None].to(device)
            restored = model(smoky)["clean"].float().clamp(0, 1)
            name = sample["clean_id"] if group == "train" else sample["name"]
            caption = (
                f"{name}   input {float(R.psnr(smoky, clean)):.2f} dB / "
                f"SSIM {float(R.ssim(smoky, clean)):.3f}   output "
                f"{float(R.psnr(restored, clean)):.2f} dB / {float(R.ssim(restored, clean)):.3f}"
            )
            tiles = [_to_uint8(smoky[0]), _to_uint8(restored[0]), _to_uint8(clean[0])]
            rows.append(([*tiles, _difference(restored[0], clean[0])], caption))
        if rows:
            header = ["input (smoky)", "output", "target (clean)", "|output - target| x4"]
            sheets[group] = _sheet(rows, header, scale)
    rows = []
    for frame_id in previews.real_items:
        with Image.open(previews.root / "real_smoke" / PREVIEW_SPLIT / f"{frame_id}.png") as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        smoky = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(device)
        restored = model(smoky)["clean"].float().clamp(0, 1)
        tiles = [_to_uint8(smoky[0]), _to_uint8(restored[0])]
        rows.append(([*tiles, _difference(restored[0], smoky[0])], f"{frame_id}   no target"))
    if rows:
        sheets["real"] = _sheet(rows, ["input (real haze)", "output", "|output - input| x4"], scale)
    return sheets


def write_index(folder: Path, title: str) -> None:
    """A static page with every epoch's sheets, newest first, for a local browser."""
    labels = {p.name.rsplit("_", 1)[0] for p in folder.glob("epoch_*_*.png")}
    epochs = sorted(labels - {"epoch_init"}, reverse=True) + sorted(labels & {"epoch_init"})
    parts = [
        "<!doctype html><meta charset=utf-8>",
        f"<title>{html.escape(title)}</title>",
        "<style>body{background:#111;color:#eee;font:14px sans-serif}"
        "img{max-width:100%;margin:4px 0}h2{margin-top:32px}</style>",
        f"<h1>{html.escape(title)}</h1><p>RESEARCH USE ONLY - NOT FOR CLINICAL USE</p>",
    ]
    for epoch in epochs:
        parts.append(f"<h2>{html.escape(epoch)}</h2>")
        for group in GROUPS:
            name = f"{epoch}_{group}.png"
            if (folder / name).exists():
                parts.append(f"<h3>{group}</h3><img src='{html.escape(name)}' loading=lazy>")
    (folder / "index.html").write_text("\n".join(parts) + "\n")


def save(
    model: torch.nn.Module,
    previews: PreviewSet,
    folder: Path,
    *,
    epoch: int | None,
    device: str,
    title: str,
    scale: float = 0.5,
) -> list[Path]:
    """Render and write one epoch's sheets (epoch None: before training)."""
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "frames.json").exists():
        (folder / "frames.json").write_text(json.dumps(previews.describe(), indent=2) + "\n")
    label = "epoch_init" if epoch is None else f"epoch_{epoch:03d}"
    written = []
    for group, sheet in render(model, previews, device, scale=scale).items():
        path = folder / f"{label}_{group}.png"
        sheet.save(path)
        written.append(path)
    write_index(folder, title)
    return written


__all__ = ["GROUPS", "PreviewSet", "render", "save", "write_index"]
