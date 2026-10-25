"""M1 data: instrument compositing, the m1_synth build, real CholecSeg8k remapping, splits."""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import numpy as np
import pytest
from PIL import Image

from osv.augment import instruments as I
from osv.datasets import m1 as M
from osv.datasets import m1_synth as SY
from osv.datasets.cholect50 import FrameSource

H, W = 48, 80


def background(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.clip(np.array([150, 55, 45]) + rng.normal(0, 10, (H, W, 3)), 0, 255)
    frame[:, :6] = 0  # a strip of the scope's black vignette on the left
    return frame.astype(np.uint8)


def render(seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """A grasper bar across the middle and a hook bar lower down, reaching the vignette."""
    rgba = np.zeros((H, W, 4), np.uint8)
    classes = np.zeros((H, W), np.uint8)
    rgba[18:24, 0:60] = (200, 200, 205, 255)
    classes[18:24, 0:60] = 1
    rgba[34:38, 20:70] = (180, 180, 180, 255)
    classes[34:38, 20:70] = 2
    meta = {
        "instances": [
            {"class": "grasper", "tip_px": [59, 21]},
            {"class": "hook", "tip_px": [69, 36]},
        ]
    }
    return rgba, classes, meta


def plain(**changes) -> I.InstrumentParams:
    values: dict[str, Any] = {
        "flip_h": False,
        "flip_v": False,
        "gain": 1.0,
        "channel_gain": (1.0, 1.0, 1.0),
        "gamma": 1.0,
        "motion_px": 0.0,
        "motion_angle_deg": 0.0,
        "defocus_sigma": 0.0,
        "occlude": False,
        "occlude_radius_px": 10.0,
        "noise_sigma": 0.0,
        "jpeg_quality": 95,
        "smoke": False,
    }
    return I.InstrumentParams(**(values | changes))


def test_label_follows_the_layer_and_the_vignette_hides_instruments():
    rgba, classes, meta = render(0)
    image, label = I.composite(
        background(0), rgba, classes, meta, plain(), np.random.default_rng(0)
    )
    assert image.shape == (H, W, 3) and label.dtype == np.uint8
    assert set(np.unique(label)) == {0, 1, 2}
    assert (label[20, 10:58] == 1).all() and (label[36, 22:68] == 2).all()
    assert (label[:, :4] == 0).all()  # instrument pixels in the vignette are background


def test_flips_move_label_and_tips_together_and_occlusion_removes_pixels():
    rgba, classes, meta = render(0)
    flipped = plain(flip_h=True, flip_v=True)
    _, label = I.composite(background(0), rgba, classes, meta, flipped, np.random.default_rng(0))
    assert (label[H - 1 - 20, W - 1 - 30] == 1) and (label[H - 1 - 36, W - 1 - 40] == 2)
    assert I.tips(meta, width=W, height=H, params=flipped)[0] == (W - 1 - 59, H - 1 - 21)
    _, whole = I.composite(background(0), rgba, classes, meta, plain(), np.random.default_rng(1))
    _, hidden = I.composite(
        background(0), rgba, classes, meta, plain(occlude=True, occlude_radius_px=12),
        np.random.default_rng(1),
    )  # fmt: skip
    assert (hidden > 0).sum() < (whole > 0).sum()


def test_blur_keeps_every_label_on_a_visible_instrument_pixel():
    rgba, classes, meta = render(0)
    params = plain(
        motion_px=9, motion_angle_deg=30, defocus_sigma=1.5, noise_sigma=2, jpeg_quality=70
    )
    image, label = I.composite(background(0), rgba, classes, meta, params, np.random.default_rng(2))
    plain_image, _ = I.composite(background(0), np.zeros_like(rgba), np.zeros_like(classes), meta, plain(), np.random.default_rng(2))  # fmt: skip
    changed = np.abs(image.astype(int) - plain_image.astype(int)).sum(axis=-1) > 30
    assert changed[label > 0].mean() > 0.9  # labelled pixels really show an instrument


@pytest.fixture
def synth(tmp_path):
    """A CholecT50 folder, a reviewed background list and 20 renders, built into m1_synth."""
    source = tmp_path / "CholecT50"
    accepted = {"train": [], "val": []}
    for video, split in (("VID01", "train"), ("VID08", "val")):
        folder = source / "videos" / video
        folder.mkdir(parents=True)
        for index in range(4):
            frame = background(index)
            if (video, index) == ("VID08", 3):  # a video recorded at twice the resolution
                frame = np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)
            Image.fromarray(frame).save(folder / f"{index:06d}.png")
            accepted[split].append(f"{video}_{index:06d}")
    reviewed = tmp_path / "m1_backgrounds.json"
    reviewed.write_text(json.dumps({"accepted": accepted, "reviewer": "R", "census_sha256": "x"}))
    renders = tmp_path / "renders"
    renders.mkdir()
    for n in range(20):
        rgba, classes, meta = render(n)
        name = f"m1s_1337_{n:06d}"
        Image.fromarray(rgba, "RGBA").save(renders / f"{name}_rgba.png")
        Image.fromarray(classes).save(renders / f"{name}_class.png")
        (renders / f"{name}.json").write_text(json.dumps(meta))
    out = tmp_path / "m1_synth"
    index = SY.build(
        reviewed, renders, out, source=FrameSource(source), smoke_dir=None, val_pairs=6, seed=3
    )
    return out, index


def test_build_splits_renders_and_writes_fixed_val_pairs(synth):
    out, index = synth
    assert index["counts"] == {
        "backgrounds_train": 4, "backgrounds_val": 4, "renders_train": 18, "renders_val": 2,
        "val_pairs": 6, "backgrounds_resized": 1,
    }  # fmt: skip
    assert index["backgrounds_resized_from"] == {"VID08_000003": [2 * W, 2 * H]}
    with Image.open(out / "backgrounds" / "val" / "VID08_000003.png") as frame:
        assert frame.size == (W, H)
    assert not set(index["renders"]["train"]) & set(index["renders"]["val"])
    assert all(p["render"] in index["renders"]["val"] for p in index["pairs"])
    assert all(p["background"].startswith("VID08") for p in index["pairs"])
    with Image.open(out / "sheet_val.png") as sheet:
        assert sheet.size == (2 * int(W * 0.5), 6 * int(H * 0.5))
    label = np.asarray(Image.open(out / "pairs" / "val" / "m1val_00000_label.png"))
    assert set(np.unique(label)) <= {0, 1, 2}


def test_synthetic_dataset_is_reproducible_and_changes_with_the_epoch(synth):
    out, _ = synth
    train = M.SyntheticInstruments("train", root=out, seed=7, samples_per_epoch=5)
    first = train[2]
    assert first["image"].shape == (3, H, W) and first["label"].dtype.is_floating_point is False
    assert np.array_equal(train.raw_pair(2)[1], first["label"].numpy())
    train.set_epoch(1)
    assert not np.array_equal(
        train.raw_pair(2)[0], M.SyntheticInstruments("train", root=out, seed=7).raw_pair(2)[0]
    )
    val = M.SyntheticInstruments("val", root=out)
    assert len(val) == 6 and np.array_equal(val.raw_pair(0)[1], val.raw_pair(0)[1])
    with pytest.raises(ValueError, match="the test is real"):
        M.SyntheticInstruments("test", root=out)


def png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def test_real_frames_keep_grasper_and_hook_and_take_only_the_asked_videos(tmp_path):
    archive = tmp_path / "CholecSeg8k.zip"
    images = []
    with zipfile.ZipFile(archive, "w") as handle:
        for video in ("video01", "video09"):
            for n in range(4):
                base = f"CholecSeg8k/{video}/{video}_00000/frame_{n:05d}"
                watershed = np.full((8, 8), 50, np.uint8)  # 50 -> class 0 (black background)
                watershed[0:2] = 31  # Grasper (5) -> 1
                watershed[2:4] = 32  # L-hook (9) -> 2
                watershed[4:6] = 11  # Abdominal wall (1) -> 0
                watershed[6:8] = 255  # boundary -> ignored
                handle.writestr(f"{base}_endo.png", png(np.full((8, 8, 3), 90, np.uint8)))
                handle.writestr(f"{base}_endo_watershed_mask.png", png(watershed))
                images.append(
                    {"file_name": f"{base}_endo.png", "video_id": video, "split": "train"}
                )
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "instances.json").write_text(json.dumps({"images": images}))
    data = M.RealInstruments(["video09"], processed_dir=processed, source=archive)
    assert data.videos == ["video09"] and len(data) == 4
    label = data[0]["label"].numpy()
    assert (label[0:2] == 1).all() and (label[2:4] == 2).all()
    assert (label[4:6] == 0).all() and (label[6:8] == 255).all()
    half = M.RealInstruments(
        ["video01", "video09"], processed_dir=processed, source=archive, fraction=0.5
    )
    assert len(half) == 4 and {r["video_id"] for r in half.images} == {"video01", "video09"}


def test_m1_split_tests_only_on_videos_never_used_as_backgrounds():
    frames = {f"video{n:02d}": 100 + n for n in (1, 9, 12, 17, 18, 25, 26, 27, 35, 43)}
    split = M.m1_split(
        frames, ["VID01", "VID12", "VID18", "VID25", "VID26", "VID27", "VID35", "VID43"]
    )
    assert split["test"] == ["video09", "video17"]
    assert sorted(split["train"] + split["val"]) == ["video01", "video12", "video18", "video25", "video26", "video27", "video35", "video43"]  # fmt: skip
    assert split["val"] and not set(split["train"]) & set(split["val"])


def test_fraction_keeps_every_video_and_is_fixed_per_seed():
    records = [{"video_id": v, "file_name": f"{v}_{n:03d}"} for v in ("a", "b") for n in range(10)]
    quarter = M.fraction_of(records, 0.25, seed=1)
    assert len(quarter) == 6 and {r["video_id"] for r in quarter} == {"a", "b"}
    assert quarter == M.fraction_of(records, 0.25, seed=1) != M.fraction_of(records, 0.25, seed=2)
    with pytest.raises(ValueError):
        M.fraction_of(records, 0, seed=1)
