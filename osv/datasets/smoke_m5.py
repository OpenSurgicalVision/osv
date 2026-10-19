"""DVC stage for M5 desmoking data: clean CholecT50 frames, smoke layers, fixed eval pairs.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

    python -m osv.datasets.smoke_m5 calibrate --preview-dir /outside/the/repo
    python -m osv.datasets.smoke_m5 build --out data/processed/smoke_m5

Inputs come from outside the repository (R-01/R-02): CHOLECT50_RAW_ZIP (or
CHOLECT50_DIR) and OSV_BLENDER_SMOKE_DIR. All three manifests must be APPROVED
(R-23): `cholect50.yaml`, `blender_smoke.yaml` and the build config
`smoke_m5.yaml`, whose `clean.haze_max` is set from `calibrate`.

What `build` writes under --out:

* ``clean/<split>/<VIDnn_NNNNNN>.png`` - every `stride`-th frame whose haze score
  is at most `haze_max`, in the official CholecT50 split (R-08: whole videos);
* ``smoke/<split>/<vNN_fNNNN>.png`` - smoke layers resized to the frame size,
  split by whole sequence (`osv.datasets.blender_smoke`);
* ``pairs/<val|test>/<pair_id>.png`` and ``<pair_id>_alpha.png`` - fixed smoky
  inputs and effective alpha for evaluation; the clean target is the clean frame
  named in the index. Train pairs are composited on the fly
  (`osv.datasets.desmoke`), from train frames and train smoke only;
* ``real_smoke/<split>/*.png`` - the hazy-est rejected real frames per split, with
  no ground truth: train for unpaired (CycleGAN) training, val/test for
  qualitative and no-reference evaluation;
* ``index.json`` - config and its hash, input hashes, both splits, every frame's
  haze score (kept and rejected), every pair's parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image

from osv.augment import smoke as S
from osv.datasets import blender_smoke as B
from osv.datasets import cholect50 as C

CONFIG_PATH = Path(__file__).resolve().parent / "manifests" / "smoke_m5.yaml"
REPO_ROOT = Path(__file__).resolve().parents[2]
DISCLAIMER = "RESEARCH USE ONLY - NOT FOR CLINICAL USE"


def load_config(path: Path = CONFIG_PATH, *, require_approved: bool = True) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if require_approved:
        if config.get("status") != "APPROVED":
            raise RuntimeError(
                f"{path.name} is {config.get('status')!r}; approve it in review (R-23)."
            )
        if config["clean"].get("haze_max") is None:
            raise RuntimeError("clean.haze_max is unset: run `calibrate` and record the threshold.")
    return config


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def fit(image: np.ndarray, size: Sequence[int]) -> np.ndarray:
    width, height = size
    if image.shape[1] == width and image.shape[0] == height:
        return image
    interpolation = cv2.INTER_AREA if image.shape[1] > width else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, optimize=False, compress_level=6)


def outside_repo(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ValueError("frame previews are never written inside the repository (R-01/R-05)")
    return resolved


def calibrate(
    source: C.FrameSource,
    split_of: dict[str, str],
    *,
    stride: int,
    sample: int,
    preview_dir: Path | None,
    per_bucket: int = 6,
) -> dict[str, Any]:
    """Haze-score quantiles over a deterministic sample, plus optional per-decile previews."""
    frames = [f for f in source.frames() if f.index % stride == 0 and f.video in split_of]
    rng = np.random.default_rng(0)
    chosen = sorted(rng.choice(len(frames), size=min(sample, len(frames)), replace=False))
    scored = [(C.haze_score(source.read(frames[i])), frames[i]) for i in chosen]
    scores = np.array([s for s, _ in scored])
    quantiles = {
        f"p{q}": round(float(np.percentile(scores, q)), 4) for q in (5, 10, 25, 50, 75, 90, 95)
    }
    if preview_dir is not None:
        out = outside_repo(preview_dir)
        out.mkdir(parents=True, exist_ok=True)
        ordered = sorted(scored, key=lambda pair: pair[0])
        for decile in range(10):
            bucket = ordered[decile * len(ordered) // 10 : (decile + 1) * len(ordered) // 10]
            tiles = []
            for score, frame in bucket[:: max(1, len(bucket) // per_bucket)][:per_bucket]:
                tile = cv2.resize(source.read(frame), (320, 180), interpolation=cv2.INTER_AREA)
                cv2.putText(
                    tile,
                    f"{score:.3f} {frame.frame_id}",
                    (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                tiles.append(tile)
            if tiles:
                save(out / f"haze_decile_{decile}.png", np.concatenate(tiles, axis=1))
    return {"sampled": len(scored), "quantiles": quantiles}


def _write_clean(
    out: Path,
    index: dict[str, Any],
    config: dict[str, Any],
    source: C.FrameSource,
    split_of: dict[str, str],
) -> None:
    """Clean frames by haze score; the hazy-est rejected frames become real-smoke frames.

    `real_smoke` maps each split to how many to keep (null: every rejected frame). Train
    real smoke is the unpaired smoky domain for CycleGAN; test real smoke is scored by
    haze drop. Black or blown-out frames (score 1.0) are never real smoke.
    """
    size, haze_max = config["size"], float(config["clean"]["haze_max"])
    real: dict[str, int | None] = config["real_smoke"]
    hazy: dict[str, list[tuple[float, C.Frame]]] = {split: [] for split in real}
    for split in C.SPLITS:
        for frame in C.iter_split(source, split_of, split, stride=int(config["clean"]["stride"])):
            rgb = source.read(frame)
            score = C.haze_score(rgb)
            record = {
                "frame_id": frame.frame_id,
                "video": frame.video,
                "split": split,
                "haze": round(score, 5),
            }
            if score <= haze_max:
                save(out / "clean" / split / f"{frame.frame_id}.png", fit(rgb, size))
                index["clean"].append(record)
                continue
            index["rejected"].append(record)
            if split in hazy and score < 1.0:
                hazy[split].append((score, frame))
    for split, count in real.items():
        ranked = sorted(hazy[split], key=lambda pair: (-pair[0], pair[1].frame_id))
        for score, frame in ranked if count is None else ranked[: int(count)]:
            save(
                out / "real_smoke" / split / f"{frame.frame_id}.png", fit(source.read(frame), size)
            )
            index["real_smoke"].append(
                {"frame_id": frame.frame_id, "split": split, "haze": round(score, 5)}
            )


def _write_smoke(
    out: Path,
    index: dict[str, Any],
    size: Sequence[int],
    smoke_root: Path,
    smoke_split: dict[str, str],
) -> None:
    for sequence, frames in B.index_renders(smoke_root).items():
        split = smoke_split[sequence]
        for frame in frames:
            with Image.open(smoke_root / frame.path) as image:
                rgba = np.asarray(image.convert("RGBA"))
            save(out / "smoke" / split / f"{frame.frame_id}.png", fit(rgba, size))
            index["smoke"].append(
                {"frame_id": frame.frame_id, "sequence": sequence, "split": split}
            )


def _write_pairs(out: Path, index: dict[str, Any], pairs_config: dict[str, Any]) -> None:
    """Fixed eval pairs: same-split clean frame and smoke layer, one stream per pair."""
    for split in pairs_config["splits"]:
        clean = [r["frame_id"] for r in index["clean"] if r["split"] == split]
        layers = [r["frame_id"] for r in index["smoke"] if r["split"] == split]
        if not clean or not layers:
            raise ValueError(f"split {split!r} has no clean frames or no smoke layers")
        number = 0
        for clean_id in clean:
            clean_rgb = np.asarray(
                Image.open(out / "clean" / split / f"{clean_id}.png").convert("RGB")
            )
            for _ in range(int(pairs_config["per_clean_frame"])):
                rng = S.pair_rng(int(pairs_config["seed"]), number)
                smoke_id = layers[int(rng.integers(len(layers)))]
                params = S.sample_params(rng)
                layer = np.asarray(
                    Image.open(out / "smoke" / split / f"{smoke_id}.png").convert("RGBA")
                )
                smoky, alpha = S.composite(clean_rgb, layer, params)
                pair_id = f"{split}-{number:06d}"
                save(out / "pairs" / split / f"{pair_id}.png", smoky)
                save(out / "pairs" / split / f"{pair_id}_alpha.png", S.to_uint8(alpha))
                index["pairs"].append(
                    {
                        "pair_id": pair_id,
                        "split": split,
                        "clean": clean_id,
                        "smoke": smoke_id,
                        "params": params.as_dict(),
                        "alpha_mean": round(float(alpha.mean()), 5),
                    }
                )
                number += 1


def build(
    out: Path,
    *,
    config: dict[str, Any],
    source: C.FrameSource,
    split_of: dict[str, str],
    smoke_root: Path,
    smoke_split: dict[str, str],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    index: dict[str, Any] = {
        "disclaimer": DISCLAIMER,
        "config": config,
        "config_sha256": config_hash(config),
        "inputs": inputs,
        "cholect50_split": split_of,
        "smoke_split": smoke_split,
        "clean": [],
        "rejected": [],
        "smoke": [],
        "pairs": [],
        "real_smoke": [],
    }
    _write_clean(out, index, config, source, split_of)
    _write_smoke(out, index, config["size"], smoke_root, smoke_split)
    _write_pairs(out, index, config["eval_pairs"])
    index["counts"] = {
        key: {split: sum(r["split"] == split for r in index[key]) for split in C.SPLITS}
        for key in ("clean", "rejected", "smoke", "pairs", "real_smoke")
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    return index


def _inputs(cholect: dict, smoke: dict, source_path: Path, smoke_root: Path) -> dict[str, Any]:
    return {
        "cholect50_source": source_path.name,
        "cholect50_sha256": [f["sha256"] for f in cholect["source"]["files"]],
        "blender_smoke_index_sha256": hashlib.sha256(
            (smoke_root / "manifest.jsonl").read_bytes()
        ).hexdigest(),
        "smoke_split_salt": smoke["split"]["salt"],
    }


def _smoke_split(smoke: dict, smoke_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    sequences = B.index_renders(smoke_root)
    strata = B.sequence_strata(smoke_root, sequences)
    split = B.split_sequences(
        strata,
        salt=smoke["split"]["salt"],
        val_fraction=float(smoke["split"]["val_fraction"]),
        test_fraction=float(smoke["split"]["test_fraction"]),
    )
    return split, strata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    cal = commands.add_parser("calibrate", help="haze-score quantiles and per-decile previews")
    cal.add_argument("--sample", type=int, default=500)
    cal.add_argument("--preview-dir", type=Path, help="absolute folder outside the repository")
    run = commands.add_parser("build", help="write clean frames, smoke layers, eval pairs, index")
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(DISCLAIMER, file=sys.stderr)
    cholect = C.load_manifest()
    split_of = C.official_split(cholect)
    source = C.FrameSource.from_env()
    try:
        if args.command == "calibrate":
            config = load_config(require_approved=False)
            report = calibrate(
                source,
                split_of,
                stride=int(config["clean"]["stride"]),
                sample=args.sample,
                preview_dir=args.preview_dir,
            )
            print(json.dumps(report, indent=1))
            return 0
        config = load_config()
        smoke = B.load_manifest()
        smoke_root = B.resolve_root()
        C.verify_archive(source.path, cholect)
        smoke_split, strata = _smoke_split(smoke, smoke_root)
        inputs = _inputs(cholect, smoke, source.path, smoke_root) | {"smoke_strata": strata}
        index = build(
            args.out,
            config=config,
            source=source,
            split_of=split_of,
            smoke_root=smoke_root,
            smoke_split=smoke_split,
            inputs=inputs,
        )
        print(json.dumps(index["counts"], indent=1))
        return 0
    finally:
        source.close()


if __name__ == "__main__":
    raise SystemExit(main())
