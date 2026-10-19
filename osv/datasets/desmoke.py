"""Torch dataset of (smoky, clean, alpha) triples for M5 desmoking.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

Reads the `osv.datasets.smoke_m5` build. `val` and `test` return the fixed pairs
of `index.json`, identical on every run. `train` composites on the fly from train
clean frames and train smoke layers only (R-08 holds for both: whole videos and
whole smoke sequences), one independent random stream per (seed, epoch, index)
so a run is reproducible and every epoch sees new smoke.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from osv.augment import smoke as S

DEFAULT_PROCESSED_DIR = Path("data/processed/smoke_m5")


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _chw(array: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    return tensor.permute(2, 0, 1) if tensor.ndim == 3 else tensor.unsqueeze(0)


class SyntheticSmokePairs(Dataset):
    def __init__(
        self,
        split: str,
        *,
        processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
        seed: int = 1337,
        limit: int | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unknown split {split!r}")
        self.root = Path(processed_dir)
        self.index: dict[str, Any] = json.loads((self.root / "index.json").read_text())
        self.split, self.seed, self.epoch = split, int(seed), 0
        self.clean = [r["frame_id"] for r in self.index["clean"] if r["split"] == split]
        self.smoke = [r["frame_id"] for r in self.index["smoke"] if r["split"] == split]
        self.pairs = [p for p in self.index["pairs"] if p["split"] == split]
        if split == "train":
            if not self.clean or not self.smoke:
                raise ValueError("train split needs clean frames and smoke layers")
            self.length = len(self.clean)
        else:
            if not self.pairs:
                raise ValueError(f"no fixed pairs for split {split!r}")
            self.length = len(self.pairs)
        if limit is not None:
            self.length = min(self.length, int(limit))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item: int) -> dict[str, Any]:
        if self.split == "train":
            clean_id = self.clean[item]
            rng = np.random.default_rng([self.seed, self.epoch, item])
            smoke_id = self.smoke[int(rng.integers(len(self.smoke)))]
            clean = _rgb(self.root / "clean" / "train" / f"{clean_id}.png")
            with Image.open(self.root / "smoke" / "train" / f"{smoke_id}.png") as image:
                layer = np.asarray(image.convert("RGBA"))
            smoky, alpha = S.composite(clean, layer, S.sample_params(rng))
            name = f"train-{self.epoch}-{item}"
        else:
            pair = self.pairs[item]
            clean_id, name = pair["clean"], pair["pair_id"]
            folder = self.root / "pairs" / self.split
            smoky = _rgb(folder / f"{name}.png")
            with Image.open(folder / f"{name}_alpha.png") as image:
                alpha = np.asarray(image, dtype=np.float32) / 255.0
            clean = _rgb(self.root / "clean" / self.split / f"{clean_id}.png")
        return {
            "smoky": _chw(S.to_float(smoky)),
            "clean": _chw(S.to_float(clean)),
            "alpha": _chw(alpha.astype(np.float32)),
            "name": name,
            "clean_id": clean_id,
        }


__all__ = ["DEFAULT_PROCESSED_DIR", "SyntheticSmokePairs"]
