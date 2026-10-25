"""DVC stage for M1 synthetic data: reviewed backgrounds, instrument renders, fixed val pairs.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

    python -m osv.datasets.m1_synth --backgrounds ~/osv-runs/m1_backgrounds.json \\
        --renders /outside/the/repo/m1_renders --out data/processed/m1_synth

Inputs come from outside the repository (R-01/R-02): the reviewed background list from
`osv.datasets.m1_backgrounds collect`, CholecT50 (CHOLECT50_RAW_ZIP or CHOLECT50_DIR),
the Blender renders of `scripts/blender/instruments.py`, and optionally the M5 build
(`--smoke-dir data/processed/smoke_m5`) for smoke layers.

What it writes under --out:

* ``backgrounds/<split>/<VIDnn_NNNNNN>.png`` - the accepted tool-free frames of the
  CholecT50 train and val videos, nothing else, resized to the renders' size where a
  video was recorded at another resolution (as `osv.datasets.smoke_m5` does for M5);
* ``renders/<split>/<id>_{rgba,class}.png`` and ``<id>.json`` - every 10th render
  (by index) goes to val, the rest to train, so no instrument layer is in both;
* ``pairs/val/<pair_id>.png`` and ``<pair_id>_label.png`` - fixed val composites (val
  backgrounds, val renders), identical on every run: arm A of RFC 0001 selects its
  checkpoint on these and on nothing real;
* ``index.json`` - inputs and their hashes, the lists, every pair's parameters;
* ``sheet_val.png`` - the first 12 val pairs, each as composite | composite with the
  label over it (grasper cyan, L-hook yellow): RFC 0001 milestone 2 has a person look
  at the composites over real backgrounds before any training.

Train composites are made on the fly (`osv.datasets.m1.SyntheticInstruments`), a new
background-render pairing and augmentation every epoch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from osv.augment import instruments as I
from osv.datasets.smoke_m5 import fit

SPLITS = ("train", "val")
VAL_EVERY = 10
DEFAULT_VAL_PAIRS = 500
DISCLAIMER = "RESEARCH USE ONLY - NOT FOR CLINICAL USE"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_ids(renders: Path) -> list[str]:
    """Render ids with all three files present, sorted by their index."""
    ids = []
    for meta in sorted(renders.glob("*.json")):
        sample = meta.stem
        if (renders / f"{sample}_rgba.png").is_file() and (
            renders / f"{sample}_class.png"
        ).is_file():
            ids.append(sample)
    if not ids:
        raise FileNotFoundError(
            f"no complete renders (<id>.json, _rgba.png, _class.png) in {renders}"
        )
    return ids


def render_split(ids: list[str]) -> dict[str, list[str]]:
    val = [r for n, r in enumerate(ids) if n % VAL_EVERY == 0]
    return {"train": [r for r in ids if r not in set(val)], "val": val}


def read_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"))


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def smoke_layers(smoke_dir: Path | None, split: str) -> list[Path]:
    if smoke_dir is None:
        return []
    return sorted((smoke_dir / "smoke" / split).glob("*.png"))


def make_pair(
    root: Path,
    split: str,
    background: str,
    render: str,
    rng: np.random.Generator,
    smoke: list[Path],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    params = I.sample_params(rng, smoke_probability=0.2 if smoke else 0.0)
    layer = read_rgba(root / "renders" / split / f"{render}_rgba.png")
    classes = read_gray(root / "renders" / split / f"{render}_class.png")
    meta = json.loads((root / "renders" / split / f"{render}.json").read_text())
    smoke_rgba = read_rgba(smoke[int(rng.integers(len(smoke)))]) if params.smoke else None
    image, label = I.composite(
        read_rgb(root / "backgrounds" / split / f"{background}.png"),
        layer,
        classes,
        meta,
        params,
        rng,
        smoke_rgba=smoke_rgba,
    )
    return image, label, params.as_dict()


OVERLAY = {1: (0, 220, 255), 2: (255, 220, 0)}


def contact_sheet(root: Path, count: int = 12, scale: float = 0.5) -> Image.Image:
    pairs = json.loads((root / "index.json").read_text())["pairs"][:count]
    tiles = []
    for pair in pairs:
        folder = root / "pairs" / "val"
        image = read_rgb(folder / f"{pair['pair_id']}.png")
        label = read_gray(folder / f"{pair['pair_id']}_label.png")
        overlay = image.astype(np.float32)
        for value, colour in OVERLAY.items():
            hit = label == value
            overlay[hit] = overlay[hit] * 0.45 + np.array(colour, np.float32) * 0.55
        tiles.append(np.concatenate([image, overlay.astype(np.uint8)], axis=1))
    height, width = tiles[0].shape[:2]
    size = (int(width * scale), int(height * scale))
    rows = [np.asarray(Image.fromarray(t).resize(size)) for t in tiles]
    return Image.fromarray(np.concatenate(rows, axis=0))


def copy_render(source: Path, target: Path) -> bool:
    """Copy unless an earlier run already did: on a network volume 60 000 copies take minutes."""
    if target.is_file() and target.stat().st_size == source.stat().st_size:
        return False
    shutil.copyfile(source, target)
    return True


def build(
    backgrounds_json: Path,
    renders: Path,
    out: Path,
    *,
    source,
    smoke_dir: Path | None,
    val_pairs: int,
    seed: int,
) -> dict[str, Any]:
    reviewed = json.loads(backgrounds_json.read_text())
    accepted = reviewed["accepted"]
    frames = {f.frame_id: f for f in source.frames()}
    missing = [f for split in SPLITS for f in accepted[split] if f not in frames]
    if missing:
        raise ValueError(f"{len(missing)} accepted backgrounds are not in CholecT50: {missing[:3]}")
    ids = render_ids(renders)
    height, width = read_gray(renders / f"{ids[0]}_class.png").shape
    out.mkdir(parents=True, exist_ok=True)
    resized: dict[str, list[int]] = {}
    for split in SPLITS:
        folder = out / "backgrounds" / split
        folder.mkdir(parents=True, exist_ok=True)
        for frame_id in accepted[split]:
            rgb = source.read(frames[frame_id])
            if rgb.shape[:2] != (height, width):
                resized[frame_id] = [rgb.shape[1], rgb.shape[0]]
            Image.fromarray(fit(rgb, (width, height))).save(folder / f"{frame_id}.png")

    split_renders = render_split(ids)
    for split, names in split_renders.items():
        folder = out / "renders" / split
        folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            for suffix in ("_rgba.png", "_class.png", ".json"):
                copy_render(renders / f"{name}{suffix}", folder / f"{name}{suffix}")

    smoke = smoke_layers(smoke_dir, "val")
    pairs_dir = out / "pairs" / "val"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for number in range(val_pairs):
        rng = np.random.default_rng([seed, number])
        background = accepted["val"][int(rng.integers(len(accepted["val"])))]
        render = split_renders["val"][int(rng.integers(len(split_renders["val"])))]
        image, label, params = make_pair(out, "val", background, render, rng, smoke)
        pair_id = f"m1val_{number:05d}"
        Image.fromarray(image).save(pairs_dir / f"{pair_id}.png")
        Image.fromarray(label).save(pairs_dir / f"{pair_id}_label.png")
        pairs.append(
            {
                "pair_id": pair_id,
                "background": background,
                "render": render,
                "params": params,
                "pixels": {str(k): int((label == k).sum()) for k in range(I.NUM_CLASSES)},
            }
        )

    index = {
        "disclaimer": DISCLAIMER,
        "inputs": {
            "backgrounds_json_sha256": _sha256(backgrounds_json),
            "backgrounds_reviewer": reviewed.get("reviewer"),
            "census_sha256": reviewed.get("census_sha256"),
            "renders": len(ids),
            "renders_list_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "smoke_dir": str(smoke_dir) if smoke_dir else None,
            "seed": seed,
        },
        "frame_size": [width, height],
        "backgrounds_resized_from": resized,
        "backgrounds": {split: list(accepted[split]) for split in SPLITS},
        "renders": split_renders,
        "pairs": pairs,
        "counts": {
            **{f"backgrounds_{s}": len(accepted[s]) for s in SPLITS},
            **{f"renders_{s}": len(split_renders[s]) for s in SPLITS},
            "val_pairs": len(pairs),
            "backgrounds_resized": len(resized),
        },
    }
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    contact_sheet(out).save(out / "sheet_val.png")
    return index


def main(argv: list[str] | None = None) -> int:
    from osv.datasets import cholect50 as C

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backgrounds", type=Path, required=True, help="m1_backgrounds collect")
    parser.add_argument("--renders", type=Path, required=True, help="Blender output folder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path, help="the M5 build, for smoke layers")
    parser.add_argument("--val-pairs", type=int, default=DEFAULT_VAL_PAIRS)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cholect50", help="default: CHOLECT50_RAW_ZIP or CHOLECT50_DIR")
    args = parser.parse_args(argv)
    source = C.FrameSource.from_env(args.cholect50)
    try:
        index = build(
            args.backgrounds,
            args.renders,
            args.out,
            source=source,
            smoke_dir=args.smoke_dir,
            val_pairs=args.val_pairs,
            seed=args.seed,
        )
    finally:
        source.close()
    print(f"[m1-synth] {index['counts']} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["build", "contact_sheet", "make_pair", "render_ids", "render_split"]
