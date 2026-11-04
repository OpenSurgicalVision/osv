"""Tests for the training entry point's guards (osv/train/segmentation.py).

Not the learning - the refusals. Each test below pins one way a training script
can quietly violate a project rule: spend GPU without passing the quota gate,
resume a checkpoint from a different configuration and present the result as
one curve, or widen the architecture search space through a CLI flag instead of
a reviewed PR.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from osv.train import segmentation as S  # noqa: E402

# --- config identity (R-24) -------------------------------------------------


def test_hash_is_stable_for_the_same_configuration():
    assert S.TrainConfig().hash() == S.TrainConfig().hash()


def test_running_longer_is_a_resumption_not_a_new_experiment():
    base = S.TrainConfig(epochs=10)
    assert replace(base, epochs=40).hash() == base.hash()


@pytest.mark.parametrize(
    "field,value",
    [("lr", 3e-4), ("seed", 7), ("batch_size", 16), ("architecture", "other"),
     ("pretrained", False), ("eval_split", "test")],
)
def test_changing_a_hyperparameter_changes_the_identity(field, value):
    base = S.TrainConfig()
    assert replace(base, **{field: value}).hash() != base.hash()


# --- the quota gate (R-24) --------------------------------------------------


def test_cuda_run_may_not_skip_the_quota_gate():
    with pytest.raises(SystemExit) as excinfo:
        S.preflight_quota(device="cuda", skip=True)
    assert "CPU run" in str(excinfo.value)


def test_cpu_run_does_not_invoke_the_gate(monkeypatch):
    # A CPU run spends no GPU-hours, so it must not depend on MLflow being up.
    called = False

    def explode(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("the quota script must not run for a CPU job")

    monkeypatch.setattr(S.subprocess, "run", explode)
    S.preflight_quota(device="cpu", skip=False)
    assert called is False


def test_cuda_run_stops_when_the_gate_reports_exhaustion(monkeypatch):
    class Result:
        returncode = 1
        stdout = "[osv.quota] STATUS: EXCEEDED\n"
        stderr = ""

    monkeypatch.setattr(S.subprocess, "run", lambda *a, **k: Result())
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    with pytest.raises(SystemExit) as excinfo:
        S.preflight_quota(device="cuda", skip=False)
    assert "not starting" in str(excinfo.value)


def test_cuda_run_stops_when_the_gate_cannot_reach_the_tracker(monkeypatch):
    # EXIT_UNDETERMINED must block too: a spend guard that fails open is decoration.
    class Result:
        returncode = 2
        stdout = ""
        stderr = "[osv.quota] could not reach MLflow\n"

    monkeypatch.setattr(S.subprocess, "run", lambda *a, **k: Result())
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    with pytest.raises(SystemExit):
        S.preflight_quota(device="cuda", skip=False)


def test_cuda_run_stops_when_the_gate_is_missing_entirely(monkeypatch):
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    with pytest.raises(SystemExit) as excinfo:
        S.preflight_quota(device="cuda", skip=False)
    assert "quota gate missing" in str(excinfo.value)


# --- resumption (R-24) ------------------------------------------------------


def test_resuming_a_differently_configured_checkpoint_is_refused(tmp_path):
    model = torch.nn.Conv2d(1, 1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    original = S.TrainConfig(lr=1e-4)
    ckpt = tmp_path / "ckpt.pt"
    S.save_checkpoint(ckpt, model=model, optimizer=optimizer, epoch=3, config=original)

    with pytest.raises(SystemExit) as excinfo:
        S.load_checkpoint(ckpt, model=model, optimizer=optimizer,
                          config=replace(original, lr=5e-4))
    assert "forbids resumption" in str(excinfo.value)


def test_resuming_the_same_configuration_continues_from_the_next_epoch(tmp_path):
    model = torch.nn.Conv2d(1, 1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = S.TrainConfig()
    ckpt = tmp_path / "ckpt.pt"
    S.save_checkpoint(ckpt, model=model, optimizer=optimizer, epoch=3, config=config)
    assert S.load_checkpoint(ckpt, model=model, optimizer=optimizer, config=config) == 4


def test_checkpoint_carries_the_research_use_disclaimer(tmp_path):
    # R-03: the notice travels with the artefact, not just the README.
    model = torch.nn.Conv2d(1, 1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ckpt = tmp_path / "ckpt.pt"
    S.save_checkpoint(ckpt, model=model, optimizer=optimizer, epoch=0, config=S.TrainConfig())
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert state["disclaimer"] == "RESEARCH USE ONLY - NOT FOR CLINICAL USE"


# --- the search space (R-25) ------------------------------------------------


def test_an_unlisted_architecture_cannot_be_selected_by_flag():
    with pytest.raises(ValueError) as excinfo:
        S.build_model(13, architecture="segformer_b5", pretrained=False)
    assert "human PR" in str(excinfo.value)


# --- egress (R-07) ----------------------------------------------------------


def _captured_kwargs(monkeypatch, *, pretrained: bool) -> dict:
    from torchvision.models import segmentation as tvseg

    seen: dict = {}
    real = tvseg.deeplabv3_resnet50

    def spy(**kwargs):
        seen.update(kwargs)
        return real(weights=None, weights_backbone=None, aux_loss=kwargs.get("aux_loss", True))

    monkeypatch.setattr(tvseg, "deeplabv3_resnet50", spy)
    S.build_model(13, architecture="deeplabv3_resnet50", pretrained=pretrained)
    return seen


def test_no_pretrained_also_silences_the_backbone_download(monkeypatch):
    # Regression: torchvision defaults weights_backbone to IMAGENET1K_V1 even when
    # weights=None, so an earlier version of build_model still fetched 97.8 MB from
    # download.pytorch.org - a host that is not on the R-07 allowlist - while
    # reporting itself as running without pretrained weights.
    seen = _captured_kwargs(monkeypatch, pretrained=False)
    assert seen["weights"] is None
    assert seen["weights_backbone"] is None


def test_pretrained_requests_both_sets_of_weights(monkeypatch):
    seen = _captured_kwargs(monkeypatch, pretrained=True)
    assert seen["weights"] == "DEFAULT"
    assert seen["weights_backbone"] == "DEFAULT"


# --- sim-to-real options ----------------------------------------------------


def test_sim2real_options_left_at_defaults_keep_the_identity():
    # Hashes of runs made before the options existed (pinned): their checkpoints resume.
    assert S.TrainConfig().hash() == "b57e7563b4986e0c"
    assert S.TrainConfig(batch_size=8, seed=1338, experiment="x").hash() == "f73268cd370f7a7d"


@pytest.mark.parametrize(
    "field,value",
    [
        ("class_weight_power", 0.5),
        ("class_counts_spread", True),
        ("augment", True),
        ("freeze_stem", True),
        ("frozen_bn", True),
        ("backbone_lr_scale", 0.1),
        ("keep_best", True),
    ],
)
def test_each_sim2real_option_is_a_different_experiment(field, value):
    base = S.TrainConfig()
    assert replace(base, **{field: value}).hash() != base.hash()


def test_weight_power_softens_the_balance():
    import numpy as np

    from osv.datasets.segmentation import inverse_frequency_weights

    counts = np.array([1000, 0, 1])  # a common class, an absent one, a rare one
    hard = inverse_frequency_weights(counts, clip=1e9).numpy()
    soft = inverse_frequency_weights(counts, clip=1e9, power=0.5).numpy()
    assert hard[1] == soft[1] == 0
    assert hard[2] / hard[0] == pytest.approx(1000)
    assert soft[2] / soft[0] == pytest.approx(1000**0.5)


def test_backbone_learning_rate_and_frozen_parts():
    model = S.build_model(4, architecture="deeplabv3_resnet50", pretrained=False)
    for name in ("conv1", "bn1", "layer1", "layer2"):
        for p in getattr(model.backbone, name).parameters():
            p.requires_grad_(False)
    backbone, head = S.param_groups(model, 1e-4, 0.1)
    assert backbone["lr"] == pytest.approx(1e-5) and head["lr"] == pytest.approx(1e-4)
    trainable = {id(p) for g in (backbone, head) for p in g["params"]}
    assert not any(id(p) in trainable for p in model.backbone.layer1.parameters())
    assert any(id(p) in trainable for p in model.backbone.layer4.parameters())
    model.train()
    S.freeze_backbone_bn(model)
    assert not model.backbone.bn1.training and model.classifier.training


def test_augmentation_moves_labels_with_the_image_and_ignores_uncovered_pixels():
    import random

    import numpy as np

    from osv.augment import IGNORE_LABEL, segmentation_train_transform

    random.seed(0)
    np.random.seed(0)
    rgb = np.zeros((96, 160, 3), np.uint8)
    rgb[:, :80] = (200, 40, 40)  # left half red ...
    label = np.zeros((96, 160), np.uint8)
    label[:, :80] = 2  # ... and labelled 2
    t = segmentation_train_transform()
    for _ in range(20):
        out_rgb, out_label = t(rgb, label)
        assert out_rgb.shape == rgb.shape and out_label.shape == label.shape
        assert set(np.unique(out_label)) <= {0, 2, IGNORE_LABEL}
        red = out_rgb[..., 0].astype(int) - out_rgb[..., 2].astype(int) > 60
        assert (out_label[red] == 2).mean() > 0.9  # red pixels keep their label
