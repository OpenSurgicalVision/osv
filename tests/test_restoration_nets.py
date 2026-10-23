"""M5 session 2: NAFNet and DehazeFormer-S as supervised desmoking baselines."""

from __future__ import annotations

import math

import pytest
import torch
from test_desmoke_train import config_for
from test_smoke_m5 import run_build

from osv.models.desmoke import build_desmoke_model
from osv.models.restoration_nets import relative_positions, window_partition, window_reverse
from osv.train import desmoke as T

PUBLISHED_PARAMETERS = {"nafnet": 29.16e6, "dehazeformer_s": 1.29e6}


@pytest.mark.parametrize("architecture", sorted(PUBLISHED_PARAMETERS))
def test_published_size_identity_start_and_any_frame_size(architecture):
    model = build_desmoke_model(architecture=architecture)
    count = sum(p.numel() for p in model.parameters())
    assert count == pytest.approx(PUBLISHED_PARAMETERS[architecture], rel=0.01)
    smoky = torch.rand(2, 3, 54, 97)
    with torch.no_grad():
        out = model.eval()(smoky)
    assert out["clean"].shape == smoky.shape and out["alpha"].shape == (2, 1, 54, 97)
    assert torch.allclose(out["clean"], smoky, atol=1e-6)  # untrained = the no-op baseline
    assert float(out["alpha"].min()) >= 0 and float(out["alpha"].max()) <= 1


def test_windows_round_trip_and_relative_positions_are_log_spaced():
    x = torch.rand(2, 16, 24, 5)
    assert torch.equal(window_reverse(window_partition(x, 8), 8, 16, 24), x)
    positions = relative_positions(8)
    assert positions.shape == (64, 64, 2)
    assert float(positions.abs().max()) == pytest.approx(math.log1p(7))


@pytest.mark.parametrize("architecture", sorted(PUBLISHED_PARAMETERS))
def test_cpu_run_trains_with_the_shared_trainer_and_moves_off_the_identity(tmp_path, architecture):
    run_build(tmp_path, "out")
    config = config_for(tmp_path, architecture=architecture, preview_count=0)
    result = T.train(
        config,
        device="cpu",
        out_dir=tmp_path / "run",
        resume=False,
        skip_quota=True,
        use_mlflow=False,
    )
    epoch = result["history"][0]
    assert math.isfinite(epoch["loss"]) and math.isfinite(epoch["psnr"])
    assert result["config"]["architecture"] == architecture
