"""M5 synthetic-smoke data: compositing model, haze filter, splits and the build stage.

Every image here is generated in the test; no surgical frame is read or written.
"""

from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from osv.augment import smoke as S
from osv.datasets import blender_smoke as B
from osv.datasets import cholect50 as C
from osv.datasets import smoke_m5 as M
from osv.datasets.desmoke import SyntheticSmokePairs

SIZE = (96, 54)


def tissue(seed: int, w: int = 128, h: int = 72, *, vignette: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.zeros((h, w, 3), np.float32)
    base[..., 0] = rng.uniform(0.45, 0.8, (h, w))
    base[..., 1] = rng.uniform(0.08, 0.25, (h, w))
    base[..., 2] = rng.uniform(0.05, 0.2, (h, w))
    if vignette:
        yy, xx = np.mgrid[:h, :w]
        base[((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2 > 1.0] = 0
    return S.to_uint8(base)


def hazy(seed: int) -> np.ndarray:
    frame = S.to_float(tissue(seed))
    gray = np.full_like(frame, 0.75)
    return S.to_uint8(frame * 0.35 + gray * 0.65)


def plume(white: bool, seed: int, w: int = 160, h: int = 90) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
    alpha = np.exp(-(((xx - cx) / (0.25 * w)) ** 2 + ((yy - cy) / (0.3 * h)) ** 2))
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 3] = S.to_uint8(alpha * 0.9)
    rgba[..., :3] = 255 if white else S.to_uint8(np.stack([alpha * 0.4 + 0.3] * 3, axis=2))
    if not white:
        rgba[..., 2] = (rgba[..., 2] * 0.85).astype(np.uint8)
    rgba[rgba[..., 3] == 0, :3] = (24, 18, 10)  # straight alpha: colour under alpha 0 is junk
    return rgba


def params(**overrides):
    base = {
        "flip_h": False,
        "flip_v": False,
        "scale": 1.0,
        "shift_x": 0.0,
        "shift_y": 0.0,
        "gain": 1.0,
        "brightness": 1.0,
        "tint": (1.0, 1.0, 1.0),
        "blur_sigma": 0.0,
    }
    return S.SmokeParams(**(base | overrides))


# --- compositing -------------------------------------------------------------------------


def test_transparent_smoke_leaves_the_clean_frame_untouched():
    clean = tissue(1)
    layer = np.zeros((90, 160, 4), np.uint8)
    layer[..., :3] = 255
    smoky, alpha = S.composite(clean, layer, params(gain=2.0, blur_sigma=2.0))
    assert np.array_equal(smoky, clean) and alpha.max() == 0


def test_opaque_smoke_takes_the_lamp_scaled_colour_and_never_the_vignette():
    clean = tissue(2)
    layer = np.zeros((90, 160, 4), np.uint8)
    layer[...] = 255
    smoky, alpha = S.composite(clean, layer, params())
    light = S.frame_light(S.to_float(clean))
    centre = smoky[30:42, 55:75].astype(np.float32) / 255
    assert np.allclose(centre, light, atol=2 / 255)
    # The black corner outside the scope's field stays black: no smoke on the vignette.
    assert alpha[0, 0] < 0.05 and smoky[0, 0].max() <= 3


def test_gain_thickens_smoke_monotonically_without_clipping():
    clean, layer = tissue(3), plume(True, 3)
    alphas = [S.composite(clean, layer, params(gain=g))[1] for g in (0.6, 1.0, 2.5)]
    assert alphas[0].mean() < alphas[1].mean() < alphas[2].mean() < 1.0
    assert all(a.max() <= 1.0 and a.min() >= 0.0 for a in alphas)


def test_transparent_pixels_never_bleed_their_colour_into_the_plume_edge():
    layer = plume(True, 4)
    warped = S.warp_smoke(layer, (54, 96), params(scale=1.3, shift_x=0.1))
    edge = (warped[..., 3] > 0.01) & (warped[..., 3] < 0.2)
    assert edge.any() and warped[..., :3][edge].min() > 0.99  # still white, not the junk RGB


def test_pairs_are_reproducible_per_seed_and_index():
    first = S.sample_params(S.pair_rng(7, 3))
    assert first == S.sample_params(S.pair_rng(7, 3))
    assert first != S.sample_params(S.pair_rng(7, 4))
    assert S.GAIN_RANGE[0] <= first.gain <= S.GAIN_RANGE[1]


# --- haze filter and CholecT50 -----------------------------------------------------------


def test_haze_score_separates_tissue_from_smoke_and_drops_black_frames():
    assert C.haze_score(tissue(5)) < 0.25 < C.haze_score(hazy(5))
    assert C.haze_score(np.zeros((72, 128, 3), np.uint8)) == 1.0


def test_official_split_places_all_50_videos_once():
    split = C.official_split(C.load_manifest(require_approved=False))
    assert len(split) == 50
    assert sorted(set(split.values())) == ["test", "train", "val"]
    assert [v for v, s in split.items() if s == "val"] == [
        "VID08",
        "VID12",
        "VID29",
        "VID50",
        "VID78",
    ]


def test_only_approved_manifests_let_frames_be_read(tmp_path):
    assert C.load_manifest()["status"] == "APPROVED"
    assert B.load_manifest()["license"]["name"] == "CC BY 4.0"
    for path, loader in ((C.MANIFEST_PATH, C.load_manifest), (B.MANIFEST_PATH, B.load_manifest)):
        proposed = tmp_path / path.name
        proposed.write_text(path.read_text().replace("status: APPROVED", "status: PROPOSED"))
        with pytest.raises(RuntimeError, match="approve"):
            loader(proposed)
    # The build config needs its approval and a calibrated haze threshold.
    assert M.load_config()["clean"]["haze_max"] == 0.28
    proposed = tmp_path / "smoke_m5.yaml"
    proposed.write_text(M.CONFIG_PATH.read_text().replace("status: APPROVED", "status: PROPOSED"))
    with pytest.raises(RuntimeError, match="approve"):
        M.load_config(proposed)
    uncalibrated = tmp_path / "uncalibrated.yaml"
    uncalibrated.write_text(M.CONFIG_PATH.read_text().replace("haze_max: 0.28", "haze_max: null"))
    with pytest.raises(RuntimeError, match="haze_max is unset"):
        M.load_config(uncalibrated)


def write_cholect(root: Path, videos: dict[str, int]) -> None:
    for video, count in videos.items():
        for index in range(count):
            frame = hazy(index) if index == 2 else tissue(hash((video, index)) % 1000)
            path = root / "CholecT50" / "videos" / video / f"{index:06d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(path)


def test_frame_source_reads_folders_and_zips_alike(tmp_path):
    write_cholect(tmp_path / "dir", {"VID01": 2})
    archive = tmp_path / "CholecT50.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for path in sorted((tmp_path / "dir").rglob("*.png")):
            zipped.write(path, path.relative_to(tmp_path / "dir").as_posix())
    folder, packed = C.FrameSource(tmp_path / "dir"), C.FrameSource(archive)
    assert [f.frame_id for f in folder.frames()] == [f.frame_id for f in packed.frames()]
    assert np.array_equal(folder.read(folder.frames()[1]), packed.read(packed.frames()[1]))
    packed.close()
    with pytest.raises(ValueError, match="missing from the official split"):
        C.assign_frames(folder.frames(), {"VID02": "train"})


# --- smoke renders -----------------------------------------------------------------------


def write_renders(root: Path) -> None:
    lines = []
    for number in range(1, 7):
        sequence = f"v{number:02d}"
        for index in (10, 11):
            rel = f"{sequence}/frames/{sequence}_f{index:04d}.png"
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(plume(number <= 3, number * 100 + index)).save(root / rel)
            lines.append({"file": rel, "frame": index, "quality": "ok"})
        (root / sequence / "meta.json").write_text(json.dumps({"id": number, "test": False}))
    lines.append({"file": "v06/frames/v06_f0012.png", "frame": 12, "quality": "clipped"})
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(line) for line in lines))


def test_renders_are_indexed_from_ok_lines_and_split_by_whole_stratified_sequence(tmp_path):
    write_renders(tmp_path)
    sequences = B.index_renders(tmp_path)
    assert sorted(sequences) == [f"v{n:02d}" for n in range(1, 7)]
    assert all(len(frames) == 2 for frames in sequences.values())  # the clipped frame is dropped
    strata = B.sequence_strata(tmp_path, sequences)
    assert strata == {
        "v01": "white",
        "v02": "white",
        "v03": "white",
        "v04": "lit",
        "v05": "lit",
        "v06": "lit",
    }
    split = B.split_sequences(strata, salt="s", val_fraction=0.15, test_fraction=0.15)
    assert split == B.split_sequences(strata, salt="s", val_fraction=0.15, test_fraction=0.15)
    for stratum in ("white", "lit"):
        assert sorted(split[s] for s in strata if strata[s] == stratum) == ["test", "train", "val"]


def test_non_rgba_render_is_refused(tmp_path):
    write_renders(tmp_path)
    Image.fromarray(np.zeros((90, 160, 3), np.uint8)).save(tmp_path / "v01/frames/v01_f0010.png")
    with pytest.raises(ValueError, match="expected RGBA"):
        B.index_renders(tmp_path)


# --- build stage and torch dataset -------------------------------------------------------

CONFIG = {
    "name": "smoke_m5",
    "status": "APPROVED",
    "size": list(SIZE),
    "clean": {"stride": 1, "haze_max": 0.25},
    "real_smoke": {"split": "test", "count": 5},
    "eval_pairs": {"splits": ["val", "test"], "per_clean_frame": 1, "seed": 11},
}
SPLIT_OF = {"VID01": "train", "VID08": "val", "VID06": "test"}


def run_build(tmp_path: Path, name: str) -> dict:
    write_cholect(tmp_path / "cholect", {"VID01": 4, "VID08": 3, "VID06": 3})
    write_renders(tmp_path / "renders")
    sequences = B.index_renders(tmp_path / "renders")
    strata = B.sequence_strata(tmp_path / "renders", sequences)
    smoke_split = B.split_sequences(strata, salt="s", val_fraction=0.15, test_fraction=0.15)
    return M.build(
        tmp_path / name,
        config=copy.deepcopy(CONFIG),
        source=C.FrameSource(tmp_path / "cholect"),
        split_of=SPLIT_OF,
        smoke_root=tmp_path / "renders",
        smoke_split=smoke_split,
        inputs={"test": True},
    )


def test_build_keeps_clean_frames_and_never_mixes_splits(tmp_path):
    index = run_build(tmp_path, "out")
    # Frame 2 of every video is hazy: rejected, and the test split's becomes a real-smoke frame.
    assert index["counts"]["clean"] == {"train": 3, "val": 2, "test": 2}
    assert {r["frame_id"] for r in index["rejected"]} == {
        "VID01_000002",
        "VID08_000002",
        "VID06_000002",
    }
    assert [r["frame_id"] for r in index["real_smoke"]] == ["VID06_000002"]
    smoke_split = {r["frame_id"]: r["split"] for r in index["smoke"]}
    clean_split = {r["frame_id"]: r["split"] for r in index["clean"]}
    for pair in index["pairs"]:
        assert smoke_split[pair["smoke"]] == clean_split[pair["clean"]] == pair["split"]
    for record in index["clean"]:
        assert SPLIT_OF[record["video"]] == record["split"]
    out = tmp_path / "out"
    assert Image.open(out / "clean" / "val" / "VID08_000000.png").size == SIZE
    layer = next(r for r in index["smoke"] if r["split"] == "train")
    assert Image.open(out / "smoke" / "train" / f"{layer['frame_id']}.png").mode == "RGBA"
    assert json.loads((out / "index.json").read_text())["config_sha256"] == M.config_hash(CONFIG)


def test_build_is_bit_for_bit_reproducible(tmp_path):
    first, second = run_build(tmp_path, "a"), run_build(tmp_path, "b")
    assert first["pairs"] == second["pairs"]
    for pair in first["pairs"]:
        name = f"pairs/{pair['split']}/{pair['pair_id']}.png"
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_dataset_serves_fixed_eval_pairs_and_fresh_train_smoke_each_epoch(tmp_path):
    run_build(tmp_path, "out")
    train = SyntheticSmokePairs("train", processed_dir=tmp_path / "out", seed=5)
    item = train[0]
    assert item["smoky"].shape == (3, SIZE[1], SIZE[0]) and item["alpha"].shape == (
        1,
        SIZE[1],
        SIZE[0],
    )
    assert torch.equal(item["smoky"], train[0]["smoky"])
    train.set_epoch(1)
    assert not torch.equal(item["smoky"], train[0]["smoky"])
    assert torch.equal(item["clean"], train[0]["clean"])  # the target never changes
    val = SyntheticSmokePairs("val", processed_dir=tmp_path / "out")
    assert len(val) == 2 and val[0]["name"] == "val-000000"
    assert float(val[0]["smoky"].min()) >= 0.0 and float(val[0]["smoky"].max()) <= 1.0


def test_calibration_previews_stay_outside_the_repository(tmp_path):
    write_cholect(tmp_path / "cholect", {"VID01": 4})
    source = C.FrameSource(tmp_path / "cholect")
    report = M.calibrate(
        source, {"VID01": "train"}, stride=1, sample=4, preview_dir=tmp_path / "prev"
    )
    assert report["sampled"] == 4 and set(report["quantiles"]) >= {"p50", "p90"}
    assert list((tmp_path / "prev").glob("haze_decile_*.png"))
    with pytest.raises(ValueError, match="never written inside the repository"):
        M.calibrate(source, {"VID01": "train"}, stride=1, sample=4, preview_dir=M.REPO_ROOT / "tmp")
