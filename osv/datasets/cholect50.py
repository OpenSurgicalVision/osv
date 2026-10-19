"""CholecT50 frames for OSV: layout, official video split and a haze score.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

Citation (CC BY-NC-SA 4.0 attribution clause; carry wherever the frames are used):

    C.I. Nwoye et al. Rendezvous: Attention Mechanisms for the Recognition of
    Surgical Action Triplets in Endoscopic Videos. Medical Image Analysis 78
    (2022) 102433. Splits: C.I. Nwoye, N. Padoy, arXiv:2204.05235 (2022).

The archive has no public URL (`osv/datasets/manifests/cholect50.yaml`): the
owner fetches it by hand and passes its path in CHOLECT50_RAW_ZIP, or the
extracted folder in CHOLECT50_DIR. Frames are read in place and never copied
into the repository tree (R-01/R-02). The split is the manifest's official
whole-video split (R-08); a video the table does not place is an error, not a
frame to drop or guess.

`haze_score` is the deterministic "is there already smoke in this frame"
measure the M5 pipeline uses to pick clean frames: the dark channel (per-pixel
minimum over RGB, eroded over a patch) is low on tissue, which is red, and
rises under grey-white smoke. It is a filter, not a smoke detector with a
claimed accuracy; its threshold is set by looking at frames (`smoke_m5
calibrate`) and recorded in the reviewed build config.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

MANIFEST_PATH = Path(__file__).resolve().parent / "manifests" / "cholect50.yaml"
FRAME = re.compile(r"(?:^|/)videos/(VID\d+)/(\d+)\.png$")
SPLITS = ("train", "val", "test")
CHUNK = 1 << 20


class ManifestNotApproved(RuntimeError):
    pass


def load_manifest(path: Path = MANIFEST_PATH, *, require_approved: bool = True) -> dict:
    manifest = yaml.safe_load(path.read_text())
    if require_approved and manifest.get("status") != "APPROVED":
        raise ManifestNotApproved(
            f"{path.name} is {manifest.get('status')!r}; a human owner approves it in review "
            "(R-23) before any frame is read."
        )
    return manifest


def official_split(manifest: Mapping) -> dict[str, str]:
    """video -> split, validated: every video in exactly one split."""
    table = manifest["split"]
    assignment: dict[str, str] = {}
    for split in SPLITS:
        for video in table[split]:
            if video in assignment:
                raise ValueError(f"{video} is in two splits")
            assignment[video] = split
    if not assignment or not table["train"]:
        raise ValueError("split table is empty")
    return assignment


@dataclass(frozen=True)
class Frame:
    video: str
    index: int
    member: str  # path inside the zip, or relative to the folder

    @property
    def frame_id(self) -> str:
        return f"{self.video}_{self.index:06d}"


class FrameSource:
    """Reads frames from the verified archive or an extracted folder."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._zip = zipfile.ZipFile(self.path) if self.path.is_file() else None

    @classmethod
    def from_env(cls, path: str | Path | None = None) -> FrameSource:
        candidate = path or os.environ.get("CHOLECT50_RAW_ZIP") or os.environ.get("CHOLECT50_DIR")
        if not candidate:
            raise FileNotFoundError(
                "No CholecT50 source: set CHOLECT50_RAW_ZIP to the archive verified against "
                "osv/datasets/manifests/cholect50.yaml, or CHOLECT50_DIR to its extracted folder."
            )
        if not Path(candidate).exists():
            raise FileNotFoundError(f"CholecT50 source does not exist: {candidate}")
        return cls(candidate)

    def frames(self) -> list[Frame]:
        if self._zip is not None:
            names = self._zip.namelist()
        else:
            names = [p.relative_to(self.path).as_posix() for p in self.path.rglob("*.png")]
        found = []
        for name in names:
            match = FRAME.search(name)
            if match:
                found.append(Frame(match[1], int(match[2]), name))
        return sorted(found, key=lambda f: (f.video, f.index))

    def read(self, frame: Frame) -> np.ndarray:
        if self._zip is not None:
            data = self._zip.read(frame.member)
            image = Image.open(io.BytesIO(data))
        else:
            image = Image.open(self.path / frame.member)
        return np.asarray(image.convert("RGB"))

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()


def verify_archive(path: Path, manifest: Mapping) -> None:
    """sha256 of the archive against the manifest; a folder is checked frame by frame instead."""
    if not path.is_file():
        return
    expected = {f["sha256"] for f in manifest["source"]["files"]}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    if digest.hexdigest() not in expected:
        raise ValueError(f"{path.name}: sha256 {digest.hexdigest()} is not the manifest's")


def assign_frames(frames: list[Frame], split_of: Mapping[str, str]) -> dict[str, list[Frame]]:
    unknown = sorted({f.video for f in frames} - set(split_of))
    if unknown:
        raise ValueError(f"videos missing from the official split: {unknown}")
    result: dict[str, list[Frame]] = {split: [] for split in SPLITS}
    for frame in frames:
        result[split_of[frame.video]].append(frame)
    return result


def haze_score(rgb: np.ndarray, *, patch: int = 15) -> float:
    """75th percentile of the eroded dark channel inside the field of view, in [0, 1].

    Black vignette pixels and small specular highlights are left out: the first
    would pull the score down, the second up, and neither is smoke.
    """
    image = rgb.astype(np.float32) / 255.0
    dark = cv2.erode(image.min(axis=2), np.ones((patch, patch), np.uint8))
    luma = image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    saturation = image.max(axis=2) - image.min(axis=2)
    specular = (luma > 0.92) & (saturation < 0.12)
    keep = (luma > 0.04) & ~cv2.dilate(specular.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(
        bool
    )
    if keep.sum() < 0.05 * keep.size:
        return 1.0  # nearly black or blown out: never a clean reference frame
    return float(np.percentile(dark[keep], 75))


def iter_split(
    source: FrameSource, split_of: Mapping[str, str], split: str, *, stride: int = 1
) -> Iterator[Frame]:
    """Every `stride`-th frame of each video in `split`, in a stable order."""
    for frame in assign_frames(source.frames(), split_of)[split]:
        if frame.index % stride == 0:
            yield frame


__all__ = [
    "Frame",
    "FrameSource",
    "ManifestNotApproved",
    "assign_frames",
    "haze_score",
    "iter_split",
    "load_manifest",
    "official_split",
    "verify_archive",
]
