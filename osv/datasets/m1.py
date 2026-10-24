"""M1 datasets: synthetic composites and real CholecSeg8k frames, three classes.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

RFC 0001 segments **background (0), grasper (1), L-hook (2)**; the binary instrument
mask is ``label > 0``. Both datasets return what `osv.datasets.segmentation` returns -
an ImageNet-normalised image and a long label with 255 for ignored pixels - so one
trainer serves every arm.

* `SyntheticInstruments` reads the `osv.datasets.m1_synth` build: on-the-fly
  composites for train (a new background-render pairing every epoch) and the fixed
  val pairs.
* `RealInstruments` reads the onboarded CholecSeg8k frames through the approved class
  mapping (R-14) and keeps Grasper and L-hook Electrocautery; every other class
  becomes background, the watershed boundary stays ignored.
* `m1_split` is the RFC's split: the CholecSeg8k videos absent from CholecT50 are the
  only test videos (never a synthetic background); the overlapping ones are split
  into real train and val by frame count, 80 / 20, whole videos only (R-08).
* `fraction_of` keeps a share of real-train frames, uniform within every video and
  fixed per seed, for the fine-tuning arms C10 / C25 / C50.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from osv.datasets.cholecseg8k import build_patient_split
from osv.datasets.m1_census import admissible_test_videos
from osv.datasets.m1_synth import make_pair, read_gray, read_rgb, smoke_layers
from osv.datasets.segmentation import IMAGENET_MEAN, IMAGENET_STD, CholecSeg8kSegmentation
from osv.eval.metrics import IGNORE_INDEX

CLASSES = ("background", "grasper", "hook")
SEG_TO_M1 = {5: 1, 9: 2}  # CholecSeg8k approved ids: Grasper, L-hook Electrocautery
DEFAULT_SYNTH_DIR = Path("data/processed/m1_synth")
TRAIN_SAMPLES_PER_EPOCH = 10_000


def m1_split(frames_per_video: Mapping[str, int], cholect50_videos: Iterable[str]) -> dict:
    groups = admissible_test_videos(frames_per_video, list(cholect50_videos))
    pool = groups["overlap_with_cholect50"]["videos"]
    assignment = build_patient_split(
        {video: frames_per_video[video] for video in pool}, ratios={"train": 0.8, "val": 0.2}
    )
    return {
        "train": sorted(v for v, s in assignment.items() if s == "train"),
        "val": sorted(v for v, s in assignment.items() if s == "val"),
        "test": groups["test"]["videos"],
    }


def fraction_of(records: list[dict[str, Any]], fraction: float, seed: int) -> list[dict]:
    """ceil(fraction * n) frames of every video, the same ones for a given seed."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1:
        return list(records)
    by_video: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_video.setdefault(record["video_id"], []).append(record)
    rng = random.Random(seed)
    kept = []
    for video in sorted(by_video):
        frames = sorted(by_video[video], key=lambda r: r["file_name"])
        kept += rng.sample(frames, math.ceil(fraction * len(frames)))
    return sorted(kept, key=lambda r: r["file_name"])


def _normalised(rgb: np.ndarray) -> torch.Tensor:
    image = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


class RealInstruments(CholecSeg8kSegmentation):
    """CholecSeg8k frames of the given M1 videos, labels remapped to the three M1 classes."""

    def __init__(
        self,
        videos: Iterable[str],
        *,
        processed_dir: str | Path,
        source: str | Path | None = None,
        fraction: float = 1.0,
        seed: int = 1337,
        limit: int | None = None,
    ) -> None:
        # The parent loads the approved mapping and the converter's index; M1 then picks
        # its own videos instead of the converter's split.
        super().__init__("train", processed_dir=processed_dir, source=source)
        wanted = set(videos)
        doc = json.loads((self.processed_dir / "instances.json").read_text(encoding="utf-8"))
        records = sorted(
            (img for img in doc["images"] if img["video_id"] in wanted),
            key=lambda img: img["file_name"],
        )
        if not records:
            raise ValueError(f"no CholecSeg8k frames for videos {sorted(wanted)}")
        records = fraction_of(records, fraction, seed)
        self.images = records[:limit] if limit is not None else records
        self.num_classes = len(CLASSES)
        self.class_names = list(CLASSES)
        m1 = np.zeros(256, dtype=np.uint8)
        for seg_class, m1_class in SEG_TO_M1.items():
            m1[seg_class] = m1_class
        m1[IGNORE_INDEX] = IGNORE_INDEX
        self._to_m1 = m1

    def raw_pair(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        rgb, label = super().raw_pair(index)
        return rgb, self._to_m1[label]


class SyntheticInstruments(Dataset):
    """Composites of the m1_synth build: on the fly for train, the fixed pairs for val."""

    def __init__(
        self,
        split: str,
        *,
        root: str | Path = DEFAULT_SYNTH_DIR,
        seed: int = 1337,
        samples_per_epoch: int = TRAIN_SAMPLES_PER_EPOCH,
        limit: int | None = None,
        smoke_dir: str | Path | None = None,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("synthetic data has train and val only; the test is real")
        self.root, self.split, self.seed, self.epoch = Path(root), split, int(seed), 0
        self.index = json.loads((self.root / "index.json").read_text())
        self.backgrounds = self.index["backgrounds"][split]
        self.renders = self.index["renders"][split]
        self.pairs = self.index["pairs"] if split == "val" else []
        self.smoke = smoke_layers(Path(smoke_dir) if smoke_dir else None, split)
        self.length = len(self.pairs) if split == "val" else samples_per_epoch
        if limit is not None:
            self.length = min(self.length, limit)
        self.num_classes = len(CLASSES)
        self.class_names = list(CLASSES)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def raw_pair(self, item: int) -> tuple[np.ndarray, np.ndarray]:
        if self.split == "val":
            pair_id = self.pairs[item]["pair_id"]
            folder = self.root / "pairs" / "val"
            return read_rgb(folder / f"{pair_id}.png"), read_gray(folder / f"{pair_id}_label.png")
        rng = np.random.default_rng([self.seed, self.epoch, item])
        background = self.backgrounds[int(rng.integers(len(self.backgrounds)))]
        render = self.renders[int(rng.integers(len(self.renders)))]
        image, label, _ = make_pair(self.root, "train", background, render, rng, self.smoke)
        return image, label

    def __getitem__(self, item: int) -> dict[str, Any]:
        rgb, label = self.raw_pair(item)
        return {
            "image": _normalised(rgb),
            "label": torch.from_numpy(label.astype(np.int64)),
            "video_id": "synthetic",
            "file_name": f"{self.split}-{self.epoch}-{item}",
        }


__all__ = [
    "CLASSES",
    "SEG_TO_M1",
    "RealInstruments",
    "SyntheticInstruments",
    "fraction_of",
    "m1_split",
]
