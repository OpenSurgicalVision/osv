"""Torch dataset over the onboarded CholecSeg8k conversion.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

Reads the split and frame list from the DVC-stage output
(`data/processed/cholecseg8k/instances.json`) and the pixels from the
manifest-declared archive, which stays outside the repository tree (R-01/R-02).

The label mask comes from remapping the *watershed* mask through the mapping in
`osv/datasets/manifests/cholecseg8k.yaml`, not from rasterising the COCO
polygons: the polygons are a derived, lossy view (small regions below
`min_area` are dropped, contours are approximated), whereas the watershed mask
is the annotation as released. Any watershed value outside the approved mapping
- notably 255, the watershed boundary - becomes `IGNORE_INDEX` and is excluded
from both loss and metrics rather than silently absorbed into a class.

R-14 is inherited, not re-litigated here: this module never defines a class
mapping of its own, it calls `load_approved_mapping()` and fails if the
manifest is not signed off.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from osv.datasets.cholecseg8k import CATEGORIES, load_approved_mapping
from osv.eval.metrics import IGNORE_INDEX

DEFAULT_PROCESSED_DIR = Path("data/processed/cholecseg8k")

# ImageNet statistics - the torchvision segmentation backbones are pretrained
# with these, so they are a property of the checkpoint, not a tunable.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_source(source: str | Path | None = None) -> Path:
    """Locate the raw archive the same way the converter does."""
    candidate = source or os.environ.get("CHOLECSEG8K_RAW_ZIP")
    if not candidate:
        raise FileNotFoundError(
            "No CholecSeg8k source. Pass `source=` or set CHOLECSEG8K_RAW_ZIP to the "
            "archive verified against osv/datasets/manifests/cholecseg8k.yaml. The raw "
            "data is never stored in the repository tree (R-01/R-02)."
        )
    path = Path(candidate)
    if not path.exists():
        raise FileNotFoundError(f"CholecSeg8k source does not exist: {path}")
    return path


class CholecSeg8kSegmentation(Dataset):
    """One (image, label-mask) pair per annotated frame of a given split.

    `split` is the patient-level split written by the converter (R-08): the
    partition key is the source video, so no frame of a given operation can
    appear in two splits.
    """

    def __init__(
        self,
        split: str,
        *,
        processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
        source: str | Path | None = None,
        transform: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
        limit: int | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unknown split {split!r}")

        self.split = split
        self.processed_dir = Path(processed_dir)
        self.source = resolve_source(source)
        self.transform = transform

        instances_path = self.processed_dir / "instances.json"
        if not instances_path.is_file():
            raise FileNotFoundError(
                f"{instances_path} not found - run the `cholecseg8k_convert` DVC stage first."
            )
        doc = json.loads(instances_path.read_text(encoding="utf-8"))

        mapping, _ignore, reason = load_approved_mapping()
        if mapping is None:
            raise RuntimeError(
                f"Refusing to build a training set without an approved class mapping: {reason}. "
                "Anatomical classes are the Clinical Lead's call (R-14)."
            )
        self.mapping = mapping
        self.mapping_provenance = reason
        self.num_classes = len(CATEGORIES)
        self.class_names = [c["name"] for c in CATEGORIES]

        images = [img for img in doc["images"] if img.get("split") == split]
        images.sort(key=lambda img: img["file_name"])
        if limit is not None:
            images = images[:limit]
        if not images:
            raise ValueError(f"split {split!r} contains no images in {instances_path}")
        self.images = images

        # Re-opened per-process in _read() (see there): being merely lazy is
        # not enough, since anything that touches this dataset in the main
        # process before DataLoader forks its workers (e.g.
        # class_pixel_counts()) would still hand every worker a shared,
        # racy file offset via the inherited fd.
        self._zip: zipfile.ZipFile | None = None
        self._zip_pid: int | None = None

        self._lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
        for raw_value, class_id in self.mapping.items():
            self._lut[raw_value] = class_id

    def __len__(self) -> int:
        return len(self.images)

    @property
    def videos(self) -> list[str]:
        return sorted({img["video_id"] for img in self.images})

    def _read(self, name: str) -> np.ndarray:
        if self.source.is_dir():
            with Image.open(self.source / name) as im:
                return np.array(im)
        # A ZipFile opened in the parent process before DataLoader forks its
        # workers is inherited by every worker sharing the SAME underlying
        # file offset (fork() duplicates the fd, not the read position).
        # Concurrent workers then race seek()+read() on that one position,
        # so one worker silently reads another's byte range -- surfacing as
        # a "Bad CRC-32" on an effectively random member. Re-opening whenever
        # the current pid doesn't match the one that opened the handle keeps
        # every process on its own fd/offset, regardless of whether something
        # upstream (e.g. class_pixel_counts()) touched this dataset in the
        # main process before the workers existed.
        pid = os.getpid()
        if self._zip is None or self._zip_pid != pid:
            if self._zip is not None:
                self._zip.close()
            self._zip = zipfile.ZipFile(self.source)
            self._zip_pid = pid
        with self._zip.open(name) as fh:
            with Image.open(io.BytesIO(fh.read())) as im:
                return np.array(im)

    def raw_pair(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """(RGB uint8 HxWx3, label uint8 HxW) before any transform."""
        record = self.images[index]
        frame = record["file_name"]
        base = frame[: -len("_endo.png")]

        rgb = self._read(frame)
        if rgb.ndim == 3 and rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]

        watershed = self._read(f"{base}_endo_watershed_mask.png")
        if watershed.ndim == 3:
            watershed = watershed[:, :, 0]
        label = self._lut[watershed.astype(np.uint8)]
        return np.ascontiguousarray(rgb), np.ascontiguousarray(label)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rgb, label = self.raw_pair(index)
        if self.transform is not None:
            rgb, label = self.transform(rgb, label)

        image = rgb.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = np.transpose(image, (2, 0, 1))

        record = self.images[index]
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "label": torch.from_numpy(label.astype(np.int64)),
            "video_id": record["video_id"],
            "file_name": record["file_name"],
        }

    def class_pixel_counts(self, sample: int | None = None, *, spread: bool = False) -> np.ndarray:
        """Ground-truth pixel count per class over this split.

        `sample` bounds the scan to N frames; the full pass reads every mask in
        the split and is slow enough to be worth caching. By default these are
        the first N, which in a split ordered by video are one or two videos
        only; `spread` takes N frames evenly spaced over the whole split.
        """
        counts = np.zeros(self.num_classes, dtype=np.int64)
        n = len(self) if sample is None else min(sample, len(self))
        picks = (
            np.unique(np.linspace(0, len(self) - 1, n).round().astype(int))
            if spread and n < len(self)
            else range(n)
        )
        for i in picks:
            _, label = self.raw_pair(i)
            valid = label[label != IGNORE_INDEX]
            counts += np.bincount(valid, minlength=self.num_classes)[: self.num_classes]
        return counts


def inverse_frequency_weights(
    counts: np.ndarray,
    *,
    clip: float = 50.0,
    power: float = 1.0,
) -> torch.Tensor:
    """Class weights ~ 1/frequency, normalised to mean 1 and clipped.

    CholecSeg8k runs about 90:1 between Liver and the rare structures, so an
    unweighted loss simply learns to ignore them. Raw inverse frequency would
    swing as hard the other way and destabilise training, hence `clip`.

    A class with zero pixels gets weight 0: it cannot be learned from this
    split, and giving it a large weight would only amplify noise.

    `power` below 1 softens the balance: 0.5 weighs by 1/sqrt(frequency). With
    power 1 and clip 50 a class rare in the counts can outweigh a common one by
    hundreds of times (339x blood over liver on one synthetic split), enough for
    it to swallow every pixel that resembles it.
    """
    counts = np.asarray(counts, dtype=np.float64)
    weights = np.zeros_like(counts)
    present = counts > 0
    weights[present] = (counts[present].sum() / counts[present]) ** power
    if present.any():
        weights[present] /= weights[present].mean()
        weights[present] = np.clip(weights[present], 1.0 / clip, clip)
    return torch.tensor(weights, dtype=torch.float32)


__all__ = [
    "CholecSeg8kSegmentation",
    "DEFAULT_PROCESSED_DIR",
    "inverse_frequency_weights",
    "resolve_source",
]
