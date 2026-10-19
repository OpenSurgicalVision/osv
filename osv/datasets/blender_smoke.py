"""Index and split the project's Blender smoke renders.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

Layout (`osv/datasets/manifests/blender_smoke.yaml`): one folder per simulated
sequence, `vNN/frames/vNN_fNNNN.png` (straight-alpha RGBA) plus `vNN/meta.json`,
and a root `manifest.jsonl` listing the frames the renderer kept. Only lines with
`quality == "ok"` are used, and only if the file is really there and RGBA.

Frames of one sequence are consecutive steps of one plume, so the split key is
the sequence, never the frame: a val/test plume never appears in training even
re-posed. Sequences are stratified by render generation (white shape-in-alpha
renders vs lit, tinted ones) so both kinds reach every split.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

MANIFEST_PATH = Path(__file__).resolve().parent / "manifests" / "blender_smoke.yaml"
FRAME = re.compile(r"(v\d{2,3})/frames/(v\d{2,3})_f(\d{4})\.png")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SmokeFrame:
    sequence: str
    index: int
    path: str  # relative to the render root

    @property
    def frame_id(self) -> str:
        return f"{self.sequence}_f{self.index:04d}"


def load_manifest(path: Path = MANIFEST_PATH, *, require_approved: bool = True) -> dict:
    manifest = yaml.safe_load(path.read_text())
    if require_approved and manifest.get("status") != "APPROVED":
        raise RuntimeError(
            f"{path.name} is {manifest.get('status')!r}; a human owner approves it in review "
            "(R-23) before the renders are used."
        )
    return manifest


def resolve_root(path: str | Path | None = None) -> Path:
    candidate = path or os.environ.get("OSV_BLENDER_SMOKE_DIR")
    if not candidate or not Path(candidate).is_dir():
        raise FileNotFoundError("Set OSV_BLENDER_SMOKE_DIR to the folder of Blender smoke renders.")
    return Path(candidate)


def index_renders(root: Path) -> dict[str, list[SmokeFrame]]:
    """sequence -> its kept frames in order, from manifest.jsonl, each checked on disk."""
    sequences: dict[str, list[SmokeFrame]] = {}
    for number, line in enumerate((root / "manifest.jsonl").read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("quality") != "ok":
            continue
        match = FRAME.fullmatch(record["file"])
        if not match or match[1] != match[2]:
            raise ValueError(f"manifest.jsonl:{number}: unexpected path {record['file']!r}")
        with Image.open(root / record["file"]) as image:
            if image.mode != "RGBA":
                raise ValueError(f"{record['file']}: mode {image.mode}, expected RGBA")
        sequences.setdefault(match[1], []).append(
            SmokeFrame(match[1], int(match[3]), record["file"])
        )
    if not sequences:
        raise ValueError("manifest.jsonl lists no usable frames")
    return {seq: sorted(frames, key=lambda f: f.index) for seq, frames in sorted(sequences.items())}


def generation(rgba: np.ndarray) -> str:
    """'white' when colour is constant white under the smoke (shape in alpha only), else 'lit'."""
    visible = rgba[..., 3] > 8
    if visible.sum() < 50:
        return "lit"
    rgb = rgba[..., :3][visible].astype(np.float32)
    return "white" if rgb.mean() > 250 and rgb.std() < 3 else "lit"


def split_sequences(
    strata: Mapping[str, str], *, salt: str, val_fraction: float, test_fraction: float
) -> dict[str, str]:
    """sequence -> split. Per stratum: order by salted hash, then test, val, the rest train."""
    by_stratum: dict[str, list[str]] = {}
    for sequence, stratum in strata.items():
        by_stratum.setdefault(stratum, []).append(sequence)
    assignment: dict[str, str] = {}
    for _stratum, sequences in sorted(by_stratum.items()):
        order = sorted(sequences, key=lambda s: hashlib.sha256(f"{salt}:{s}".encode()).hexdigest())
        count = len(order)
        n_test = max(1, round(count * test_fraction)) if count >= 3 else 0
        n_val = max(1, round(count * val_fraction)) if count >= 3 else 0
        for position, sequence in enumerate(order):
            assignment[sequence] = (
                "test" if position < n_test else "val" if position < n_test + n_val else "train"
            )
    return assignment


def sequence_strata(root: Path, sequences: Mapping[str, list[SmokeFrame]]) -> dict[str, str]:
    """Render generation per sequence, read from its densest kept frame."""
    strata = {}
    for sequence, frames in sequences.items():
        best = None
        for frame in frames:
            with Image.open(root / frame.path) as image:
                rgba = np.asarray(image)
            weight = float((rgba[..., 3] > 8).mean())
            if best is None or weight > best[0]:
                best = (weight, rgba)
        assert best is not None
        strata[sequence] = generation(best[1])
    return strata


__all__ = [
    "SmokeFrame",
    "generation",
    "index_renders",
    "load_manifest",
    "resolve_root",
    "sequence_strata",
    "split_sequences",
]
