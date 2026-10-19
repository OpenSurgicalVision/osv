"""M5 GAN baselines: registry, models, VGG loss and CPU training runs on generated fixtures."""

from __future__ import annotations

import json
import math

import pytest
import torch
from test_desmoke_train import label_real_smoke
from test_smoke_m5 import run_build
from torchvision.models.vgg import cfgs, make_layers

from osv.arch.registry import REGISTRY, lookup
from osv.datasets.desmoke import UnpairedSmoke
from osv.models import desmoke_gan as G
from osv.models.desmoke import build_desmoke_model
from osv.train import desmoke as T
from osv.train import desmoke_gan as TG


def test_the_registry_is_the_search_space():
    assert {n for n, a in REGISTRY.items() if a.task == "M5 desmoking"} == {
        "residual_unet",
        "pix2pix",
        "srgan",
        "cyclegan",
    }
    with pytest.raises(ValueError, match="R-25"):
        lookup("deeplabv3_resnet50", task="M5 desmoking")
    with pytest.raises(ValueError, match="R-25"):
        lookup("bigan", task="M5 desmoking")
    with pytest.raises(ValueError, match="is a GAN"):
        build_desmoke_model(architecture="pix2pix")


@pytest.mark.parametrize("architecture", ["pix2pix", "srgan", "cyclegan"])
def test_generators_keep_any_frame_size_and_the_pixel_range(architecture):
    nets = G.build_gan(architecture, width=8)
    smoky = torch.rand(2, 3, 54, 97)
    for net in nets.values():
        net.eval()
    with torch.no_grad():
        out = nets["generator"](smoky)["clean"]
        logits = nets["discriminator"](
            torch.cat([smoky, out], 1) if architecture == "pix2pix" else out
        )
    assert out.shape == smoky.shape
    assert float(out.min()) >= 0 and float(out.max()) <= 1
    assert torch.isfinite(logits).all()
    if architecture == "srgan":  # zero-initialised tail: the identity, like the U-Net baseline
        assert torch.allclose(out, smoky, atol=1e-6)
    if architecture == "cyclegan":
        assert set(nets) == {"generator", "generator_back", "discriminator", "discriminator_back"}


def test_training_output_is_not_clamped_so_saturated_pixels_still_learn():
    generator = G.build_gan("srgan", width=8)["generator"]
    tail = generator.generator.tail
    torch.nn.init.constant_(tail.bias, 3.0)  # every pixel far above white
    smoky = torch.rand(1, 3, 16, 16)
    generator.train()
    out = generator(smoky)["clean"]
    assert float(out.detach().min()) > 1
    out.mean().backward()
    assert tail.bias.grad is not None and bool((tail.bias.grad != 0).all())
    generator.eval()
    with torch.no_grad():
        assert float(generator(smoky)["clean"].max()) == 1.0


def test_only_srgan_has_an_mse_stage():
    assert TG.GanConfig(architecture="srgan").pretrain_epochs == TG.SRGAN_PRETRAIN_EPOCHS
    assert TG.GanConfig(architecture="srgan", pretrain_epochs=0).pretrain_epochs == 0
    assert TG.GanConfig(architecture="pix2pix").pretrain_epochs == 0
    with pytest.raises(ValueError, match="SRResNet stage"):
        TG.GanConfig(architecture="pix2pix", pretrain_epochs=2)


def test_image_pool_fills_then_mixes_history_with_new_fakes():
    pool = G.ImagePool(2, generator=torch.Generator().manual_seed(0))
    first = torch.zeros(2, 3, 4, 4)
    assert torch.equal(pool.query(first), first)
    later = pool.query(torch.ones(8, 3, 4, 4))
    assert later.shape == (8, 3, 4, 4)
    assert {float(image.mean()) for image in later} == {0.0, 1.0}


def vgg_weights(tmp_path):
    """A random VGG19 trunk saved like torchvision's, cut at relu1_2 to keep the test small."""
    trunk = make_layers(cfgs["E"])[:4]
    path = tmp_path / "vgg19.pt"
    torch.save({f"features.{k}": v for k, v in trunk.state_dict().items()}, path)
    return path


def test_vgg_loss_loads_a_local_file_records_its_hash_and_refuses_others(tmp_path):
    path = vgg_weights(tmp_path)
    loss = G.VGGPerceptual(path, layer=3)
    assert len(loss.weights_sha256 or "") == 64
    image = torch.rand(1, 3, 16, 16)
    assert float(loss(image, image)) == 0.0 and float(loss(image, 1 - image)) > 0
    assert not any(p.requires_grad for p in loss.parameters())
    other = tmp_path / "other.pt"
    torch.save({"0.weight": torch.zeros(1)}, other)
    with pytest.raises(ValueError, match="not a vgg19 state dict"):
        G.VGGPerceptual(other, layer=3)


def test_linear_decay_holds_then_reaches_zero():
    factor = TG.linear_decay(4)
    assert [factor(e) for e in range(4)] == pytest.approx([1.0, 1.0, 2 / 3, 1 / 3])


def test_unpaired_dataset_draws_labelled_real_smoke_and_clean_frames_of_one_split(tmp_path):
    run_build(tmp_path, "out")
    with pytest.raises(ValueError, match="none are labelled yet"):
        UnpairedSmoke("train", processed_dir=tmp_path / "out")
    label_real_smoke(tmp_path / "out")
    dataset = UnpairedSmoke("train", processed_dir=tmp_path / "out", seed=3)
    assert len(dataset) == 3  # 3 clean train frames walked, 1 real smoky frame drawn
    items = [dataset[i] for i in range(len(dataset))]
    assert {item["smoky_id"] for item in items} == {"VID01_000002"}
    assert len({item["clean_id"] for item in items}) == 3
    assert items[0]["smoky"].shape == items[0]["clean"].shape == (3, 54, 96)
    with pytest.raises(ValueError, match="real smoky frames"):
        UnpairedSmoke("val", processed_dir=tmp_path / "out")


def gan_config(tmp_path, architecture, **overrides):
    values = {
        "architecture": architecture,
        "epochs": 1,
        "batch_size": 2,
        "crop": 32,
        "base_channels": 8,
        "num_workers": 0,
        "processed_dir": str(tmp_path / "out"),
    }
    return TG.GanConfig(**(values | overrides))


def run(config, out_dir, *, resume=False):
    return TG.train(
        config, device="cpu", out_dir=out_dir, resume=resume, skip_quota=True, use_mlflow=False
    )


def test_pix2pix_trains_on_pairs_and_refuses_a_changed_resume(tmp_path):
    run_build(tmp_path, "out")
    result = run(gan_config(tmp_path, "pix2pix"), tmp_path / "run")
    epoch = result["history"][0]
    assert math.isfinite(epoch["loss"]) and math.isfinite(epoch["loss_d"])
    assert result["architecture"]["training"] == "paired_gan"
    saved = json.loads((tmp_path / "run" / f"report_{result['config_hash']}.json").read_text())
    assert saved["disclaimer"] == T.DISCLAIMER
    longer = run(gan_config(tmp_path, "pix2pix", epochs=2), tmp_path / "run", resume=True)
    assert [h["epoch"] for h in longer["history"]] == [1]
    changed = gan_config(tmp_path, "pix2pix", l1_weight=10.0)
    nets = G.build_gan("pix2pix", width=8)
    optimizers, schedulers = TG.build_optimizers(nets, changed)
    with pytest.raises(SystemExit, match="refusing to resume"):
        TG.load_checkpoint(
            tmp_path / "run" / f"ckpt_{result['config_hash']}.pt",
            nets=nets,
            optimizers=optimizers,
            schedulers=schedulers,
            config=changed,
        )


def test_srgan_needs_local_vgg_weights_and_hashes_them_into_the_experiment(tmp_path):
    run_build(tmp_path, "out")
    with pytest.raises(SystemExit, match="R-07"):
        run(gan_config(tmp_path, "srgan"), tmp_path / "run")
    weights = vgg_weights(tmp_path)
    stages = {"vgg_layer": 3, "epochs": 2, "pretrain_epochs": 1}
    config = gan_config(tmp_path, "srgan", vgg_weights=str(weights), **stages)
    result = run(config, tmp_path / "run")
    assert len(result["config"]["vgg_sha256"]) == 64
    mse, gan = result["history"]
    assert "loss_d" not in mse and math.isfinite(mse["content"])  # SRResNet stage
    assert math.isfinite(gan["loss_d"]) and math.isfinite(gan["content"])
    moved = tmp_path / "elsewhere.pt"
    moved.write_bytes(weights.read_bytes())
    same = gan_config(tmp_path, "srgan", vgg_weights=str(moved), **stages)
    same.vgg_sha256 = result["config"]["vgg_sha256"]
    same.data_config_sha256 = result["config"]["data_config_sha256"]
    assert same.hash() == result["config_hash"]  # the file's bytes, not its path, define the run


def test_cyclegan_trains_on_real_smoke_through_the_cli_and_scores_the_test_split(tmp_path):
    run_build(tmp_path, "out")
    label_real_smoke(tmp_path / "out")
    argv = [
        "--architecture", "cyclegan", "--epochs", "1", "--batch-size", "2", "--crop", "32",
        "--base-channels", "8", "--num-workers", "0", "--device", "cpu", "--eval-split", "test",
        "--data", str(tmp_path / "out"), "--out", str(tmp_path / "run"),
        "--skip-quota-check", "--no-mlflow",
    ]  # fmt: skip
    assert T.main(argv) == 0
    (report_path,) = (tmp_path / "run").glob("report_*.json")
    report = json.loads(report_path.read_text())
    assert report["config"]["experiment"] == "osv-m5-desmoke-gan"
    assert report["architecture"]["training"] == "unpaired_gan"
    assert {"cycle", "identity"} <= set(report["history"][0])
    assert report["real_smoke"]["haze_before"]["n"] == 1


def test_a_cuda_gan_run_never_skips_the_quota_gate(tmp_path):
    with pytest.raises(SystemExit, match="R-24"):
        TG.train(
            gan_config(tmp_path, "cyclegan"),
            device="cuda",
            out_dir=tmp_path,
            resume=False,
            skip_quota=True,
            use_mlflow=False,
        )
