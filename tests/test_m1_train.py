"""M1 trainer and test evaluation: arms A / B / C on CPU fixtures, the touch ledger, the rule."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from test_m1_data import H, W, background, png, render

from osv.datasets import m1_synth as SY
from osv.datasets.cholect50 import FrameSource
from osv.eval import m1 as E
from osv.train import m1 as T

REAL_TRAIN_POOL = ("video01", "video12", "video18", "video25", "video26")  # all in CholecT50
REAL_TEST = ("video09",)  # not in CholecT50


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("m1")
    source = root / "CholecT50"
    accepted: dict[str, list[str]] = {"train": [], "val": []}
    for video, split in (("VID01", "train"), ("VID08", "val")):
        (source / "videos" / video).mkdir(parents=True)
        for index in range(3):
            Image.fromarray(background(index)).save(source / "videos" / video / f"{index:06d}.png")
            accepted[split].append(f"{video}_{index:06d}")
    reviewed = root / "m1_backgrounds.json"
    reviewed.write_text(json.dumps({"accepted": accepted, "reviewer": "R", "census_sha256": "x"}))
    renders = root / "renders"
    renders.mkdir()
    for n in range(10):
        rgba, classes, meta = render(n)
        Image.fromarray(rgba, "RGBA").save(renders / f"m1s_1337_{n:06d}_rgba.png")
        Image.fromarray(classes).save(renders / f"m1s_1337_{n:06d}_class.png")
        (renders / f"m1s_1337_{n:06d}.json").write_text(json.dumps(meta))
    synth = root / "m1_synth"
    SY.build(
        reviewed, renders, synth, source=FrameSource(source), smoke_dir=None, val_pairs=4, seed=3
    )

    archive, images = root / "CholecSeg8k.zip", []
    with zipfile.ZipFile(archive, "w") as handle:
        for video in REAL_TRAIN_POOL + REAL_TEST:
            for n in range(3):
                base = f"CholecSeg8k/{video}/{video}_00000/frame_{n:05d}"
                watershed = np.full((H, W), 50, np.uint8)
                watershed[18:24, 10:60] = 31  # grasper
                watershed[34:38, 20:70] = 32  # L-hook
                handle.writestr(f"{base}_endo.png", png(background(n)))
                handle.writestr(f"{base}_endo_watershed_mask.png", png(watershed))
                images.append(
                    {"file_name": f"{base}_endo.png", "video_id": video, "split": "train"}
                )
    processed = root / "processed"
    processed.mkdir()
    (processed / "instances.json").write_text(json.dumps({"images": images}))

    weights = root / "resnet50.pth"
    from torchvision.models import resnet50

    torch.save(resnet50(weights=None).state_dict(), weights)
    return {
        "synth": synth,
        "processed": processed,
        "archive": archive,
        "weights": weights,
        "root": root,
    }


def config(data, **changes) -> T.M1Config:
    values: dict[str, Any] = {
        "arm": "A",
        "seed": 1,
        "epochs": 1,
        "batch_size": 2,
        "train_limit": 4,
        "eval_limit": 4,
        "num_workers": 0,
        "smoke": False,
        "amp": False,
        "backbone_weights": str(data["weights"]),
        "synth_dir": str(data["synth"]),
        "processed_dir": str(data["processed"]),
        "source": str(data["archive"]),
    }
    return T.M1Config(**(values | changes))


def run(cfg, out):
    return T.train(cfg, device="cpu", out_dir=out, skip_quota=True, use_mlflow=False)


def test_the_real_split_tests_only_on_videos_absent_from_cholect50(data):
    split = T.real_split(data["processed"])
    assert split["test"] == ["video09"]
    assert sorted(split["train"] + split["val"]) == sorted(REAL_TRAIN_POOL)


def test_real_flips_move_the_label_with_the_image(data):
    from osv.datasets import m1 as D

    frames = D.RealInstruments(["video01"], processed_dir=data["processed"], source=data["archive"])
    flipped = T.Flipped(frames, seed=1)
    seen = set()
    for epoch in range(6):
        flipped.set_epoch(epoch)
        for item in range(len(frames)):
            plain, sample = frames[item], flipped[item]
            for dims in ([], [-1], [-2], [-1, -2]):
                image = plain["image"].flip(dims) if dims else plain["image"]
                if torch.equal(image, sample["image"]):
                    label = plain["label"].flip(dims) if dims else plain["label"]
                    assert torch.equal(label, sample["label"])
                    seen.add(tuple(dims))
                    break
            else:
                raise AssertionError("the image is not a flip of the frame")
    assert {(-1,), (-2,)} <= seen


def test_config_guards_the_arms():
    with pytest.raises(ValueError, match="arm C"):
        T.M1Config(arm="B", real_fraction=0.5)
    with pytest.raises(ValueError, match="--init"):
        T.M1Config(arm="C", real_fraction=0.5)
    with pytest.raises(ValueError, match="arm must be"):
        T.M1Config(arm="D")
    assert T.M1Config(arm="A").hash() != T.M1Config(arm="A", lr=2e-4).hash()
    assert T.M1Config(arm="A").hash() == T.M1Config(arm="A", num_workers=1, init="").hash()
    assert T.M1Config(arm="A").hash() != T.M1Config(arm="A", epochs=40).hash()  # cosine span


def test_backbone_comes_from_a_local_resnet50_file(data, tmp_path):
    with pytest.raises(SystemExit, match="R-07"):
        run(config(data, backbone_weights=""), tmp_path)
    other = tmp_path / "other.pth"
    torch.save({"weight": torch.zeros(1)}, other)
    with pytest.raises(SystemExit, match="not a ResNet-50"):
        T.build_model(architecture="deeplabv3_resnet50", backbone_weights=str(other))
    with pytest.raises(ValueError, match="R-25"):
        T.build_model(architecture="nafnet", backbone_weights=None)


@pytest.fixture(scope="module")
def trained(data):
    """Arms A and B, seed 1, trained once for the tests below."""
    out = data["root"] / "runs"
    reports = {arm: run(config(data, arm=arm), out) for arm in ("A", "B")}
    best = {arm: out / T.M1Config(**r["config"]).name / "best.pt" for arm, r in reports.items()}
    return {"out": out, "reports": reports, "best": best}


def test_arm_a_selects_on_synthetic_val_and_logs_real_val_for_information(data, trained, tmp_path):
    report = trained["reports"]["A"]
    (entry,) = report["history"]
    assert {"val_binary_iou", "info_real_val_binary_iou"} <= set(entry)
    best = torch.load(trained["best"]["A"], weights_only=False)
    assert best["selected_on"] == "synthetic val" and best["config"]["arm"] == "A"
    assert len(report["config"]["backbone_sha256"]) == 64
    assert len(report["config"]["data_sha256"]) == 64
    with pytest.raises(SystemExit, match="no smoke layers"):
        run(config(data, smoke=True), tmp_path / "smoke")


def test_arm_c_fine_tunes_arm_a_of_the_same_seed_only(data, trained, tmp_path):
    a_best, b_best = trained["best"]["A"], trained["best"]["B"]
    with pytest.raises(SystemExit, match="must fine-tune arm A seed 2"):
        run(config(data, arm="C", seed=2, real_fraction=0.5, init=str(a_best)), tmp_path)
    with pytest.raises(SystemExit, match="not arm A's selected checkpoint"):
        run(config(data, arm="C", real_fraction=0.5, init=str(b_best)), tmp_path)
    c = run(
        config(data, arm="C", real_fraction=0.5, init=str(a_best), backbone_weights=""), tmp_path
    )
    assert c["config"]["init_sha256"] == T.sha256_file(a_best)
    assert T.M1Config(**c["config"]).name.startswith("m1-C50-s1-")
    assert (
        torch.load(tmp_path / T.M1Config(**c["config"]).name / "best.pt", weights_only=False)[
            "selected_on"
        ]
        == "real val"
    )


def reviewed(tmp_path, **lists) -> Path:
    path = tmp_path / "m1_test_review.yaml"
    doc = {"status": "REVIEWED", "reviewer": "R", "excluded_clips": [], "excluded_frames": []}
    path.write_text(yaml.safe_dump(doc | lists))
    return path


def test_test_touches_once_and_is_counted(trained, tmp_path):
    best = trained["best"]["B"]
    ledger = tmp_path / "touches.jsonl"
    args = {"ledger": ledger, "out": tmp_path / "test", "device": "cpu", "batch_size": 2,
            "num_workers": 0, "processed_dir": None, "source": None}  # fmt: skip
    pending = reviewed(tmp_path, status="PENDING")
    with pytest.raises(SystemExit, match="PENDING"):
        E.test(best, **args, review_path=pending)
    assert not ledger.exists()
    args["review_path"] = reviewed(tmp_path, excluded_frames=["video09_00000/frame_2"])
    result = E.test(best, **args)
    assert result["touch_count"] == 1 and list(result["per_video"]) == ["video09"]
    assert result["per_video"]["video09"]["frames"] == 2
    assert result["touch"]["excluded_frames"] == 1 and len(result["touch"]["review_sha256"]) == 64
    assert 0 <= result["test"]["binary_iou"] <= 1
    with pytest.raises(SystemExit, match="once"):
        E.test(best, **args)
    assert len(ledger.read_text().splitlines()) == 1
    with pytest.raises(SystemExit, match="not a selected checkpoint"):
        E.test(best.parent / "last.pt", **args)


def test_review_sheets_show_every_test_clip_and_names_must_exist(data, tmp_path):
    (sheet,) = E.review(str(data["processed"]), str(data["archive"]), tmp_path, scale=1.0)
    assert sheet.name == "test_review_video09.png"
    with Image.open(sheet) as image:
        assert image.size == (6 * W, 28 + H)  # one clip: 3 frames x (image | label)
    listing = json.loads((tmp_path / "test_review.json").read_text())
    assert listing["clips"] == {"video09_00000": 3}
    images = [
        {"file_name": f"CholecSeg8k/video09/video09_00000/frame_{n:05d}_endo.png"} for n in range(3)
    ]
    assert len(E.apply_review(images, {"excluded_clips": ["video09_00000"]})) == 0
    with pytest.raises(SystemExit, match="not in the test"):
        E.apply_review(images, {"excluded_clips": ["video09_99999"]})


def test_boundary_f1_tolerates_three_pixels():
    target = np.zeros((40, 40), bool)
    target[10:30, 10:30] = True
    valid = np.ones_like(target)
    assert E.boundary_f1(E.boundary_counts(target, target, valid)) == 1.0
    shifted = np.roll(target, 2, axis=1)
    assert E.boundary_f1(E.boundary_counts(shifted, target, valid)) == 1.0
    far = np.roll(target, 8, axis=1)
    assert E.boundary_f1(E.boundary_counts(far, target, valid)) < 0.7


def fake_report(arm: str, seed: int, ious: dict[str, float], fraction: float = 1.0) -> dict:
    per_video = {}
    for video, value in ious.items():
        tp = int(1000 * value)
        per_video[video] = {"tp": tp, "fp": (1000 - tp) // 2, "fn": 1000 - tp - (1000 - tp) // 2}
    return {
        "config": {"arm": arm, "seed": seed, "real_fraction": fraction},
        "per_video": per_video,
        "test": {"boundary_f1": 0.5, "instrument_mean_iou": 0.5},
    }


def reports(a_shift: float) -> list[dict]:
    rng = np.random.default_rng(0)
    out = []
    for seed in (1, 2, 3):
        base = {f"video{n:02d}": 0.7 + 0.1 * rng.random() for n in (9, 17, 20, 24, 28, 37, 55)}
        out.append(fake_report("B", seed, base))
        out.append(fake_report("A", seed, {v: x + a_shift for v, x in base.items()}))
    return out


def test_decision_rule_follows_the_rfc():
    assert E.decide(reports(0.0), reps=500)["verdict"] == "SUPPORTED"
    assert E.decide(reports(-0.10), reps=500)["verdict"] == "REJECTED"
    noisy = reports(0.0)
    for r in noisy:
        if r["config"]["arm"] == "A":
            r["per_video"]["video09"]["tp"] = 100  # one bad video in arm A
    result = E.decide(noisy, reps=500)
    assert result["verdict"].startswith("INCONCLUSIVE")
    assert (
        result["delta_iou_points"]["ci95"][0]
        < -E.DELTA_POINTS
        < result["delta_iou_points"]["ci95"][1]
    )
    with pytest.raises(SystemExit, match="3 distinct seeds"):
        E.decide([r for r in reports(0.0) if r["config"]["seed"] != 3], reps=10)
