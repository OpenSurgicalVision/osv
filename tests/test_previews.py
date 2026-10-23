"""M5 per-epoch previews: fixed frames, train / val / real sheets, never the test split."""

from __future__ import annotations

import json

from PIL import Image
from test_desmoke_gan import gan_config
from test_desmoke_train import config_for
from test_smoke_m5 import run_build

from osv.train import desmoke as T
from osv.train import desmoke_gan as TG
from osv.train import previews as P


def run(config, out_dir, *, resume=False, gan=False):
    trainer = TG.train if gan else T.train
    return trainer(
        config, device="cpu", out_dir=out_dir, resume=resume, skip_quota=True, use_mlflow=False
    )


def test_previews_are_fixed_and_come_from_train_val_and_real_val_only(tmp_path):
    run_build(tmp_path, "out")
    index = json.loads((tmp_path / "out" / "index.json").read_text())
    first = P.PreviewSet(tmp_path / "out", count=3, seed=7).describe()
    assert first == P.PreviewSet(tmp_path / "out", count=3, seed=7).describe()
    assert 0 < len(first["train"]) <= 3 and 0 < len(first["val"]) <= 3
    val_pairs = {p["pair_id"] for p in index["pairs"] if p["split"] == "val"}
    assert set(first["val"]) <= val_pairs
    real_val = {r["frame_id"] for r in index["real_smoke"] if r["split"] == "val"}
    assert set(first["real"]) <= real_val
    test_ids = {p["pair_id"] for p in index["pairs"] if p["split"] == "test"}
    test_ids |= {r["frame_id"] for r in index["real_smoke"] if r["split"] == "test"}
    assert not test_ids & {frame for frames in first.values() for frame in frames}


def test_a_run_writes_sheets_for_the_untrained_model_and_every_epoch(tmp_path):
    run_build(tmp_path, "out")
    out = tmp_path / "run"
    result = run(config_for(tmp_path, epochs=2, preview_count=2), out)
    folder = out / "previews"
    names = sorted(p.name for p in folder.glob("*.png"))
    for label in ("epoch_init", "epoch_000", "epoch_001"):
        assert f"{label}_train.png" in names and f"{label}_val.png" in names
    frame = next((tmp_path / "out" / "pairs" / "val").glob("*_alpha.png"))
    with Image.open(frame) as image, Image.open(folder / "epoch_001_val.png") as sheet:
        assert sheet.width == 4 * int(image.width * 0.5)  # input | output | target | error
    page = (folder / "index.html").read_text()
    assert page.index("epoch_001") < page.index("epoch_000") < page.index("epoch_init")
    assert "NOT FOR CLINICAL USE" in page
    assert json.loads((folder / "frames.json").read_text())["val"]
    # Previews are looking, not training: they do not change the experiment.
    same = config_for(tmp_path, preview_count=0, preview_every=5, preview_scale=1.0)
    same.data_config_sha256 = result["config"]["data_config_sha256"]
    assert same.hash() == result["config_hash"]


def test_preview_every_and_off(tmp_path):
    run_build(tmp_path, "out")
    run(config_for(tmp_path, epochs=3, preview_count=1, preview_every=2), tmp_path / "a")
    labels = {p.name.rsplit("_", 1)[0] for p in (tmp_path / "a" / "previews").glob("*.png")}
    assert labels == {"epoch_init", "epoch_001", "epoch_002"}  # every 2nd, and the last
    run(config_for(tmp_path, preview_count=0), tmp_path / "b")
    assert not (tmp_path / "b" / "previews").exists()


def test_a_gan_run_writes_previews_of_its_generator(tmp_path):
    run_build(tmp_path, "out")
    run(gan_config(tmp_path, "pix2pix", preview_count=1), tmp_path / "gan", gan=True)
    names = {p.name for p in (tmp_path / "gan" / "previews").glob("*.png")}
    assert {"epoch_init_val.png", "epoch_000_val.png"} <= names
