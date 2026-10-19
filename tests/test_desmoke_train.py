"""M5 baseline: model, restoration metrics and a CPU training run on generated fixtures."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from test_smoke_m5 import run_build

from osv.eval import restoration as R
from osv.models.desmoke import ResidualUNet, build_desmoke_model
from osv.train import desmoke as T


def test_untrained_model_is_the_identity_at_any_frame_size():
    model = ResidualUNet(base=8, depth=3).eval()
    smoky = torch.rand(2, 3, 54, 97)  # not a multiple of 8 either way
    with torch.no_grad():
        out = model(smoky)
    assert torch.equal(out["clean"], smoky)
    assert out["alpha"].shape == (2, 1, 54, 97)
    assert float(out["alpha"].min()) >= 0 and float(out["alpha"].max()) <= 1
    with pytest.raises(ValueError, match="R-25"):
        build_desmoke_model(architecture="restormer")


def test_psnr_and_ssim_match_their_definitions():
    clean = torch.rand(1, 3, 32, 32)
    assert float(R.psnr(clean, clean)) == R.PSNR_CAP
    shifted = (clean + 0.1).clamp(0, 1)
    mse = float((shifted - clean).pow(2).mean())
    assert math.isclose(float(R.psnr(shifted, clean)), 10 * math.log10(1 / mse), rel_tol=1e-5)
    assert math.isclose(float(R.ssim(clean, clean)), 1.0, abs_tol=1e-6)
    noisy = (clean + 0.2 * torch.randn_like(clean)).clamp(0, 1)
    assert float(R.ssim(noisy, clean)) < 0.9


def test_reports_carry_intervals_strata_and_the_no_op_baseline():
    rows = [
        {
            "psnr": 30.0 + i,
            "ssim": 0.9,
            "psnr_input": 20.0,
            "ssim_input": 0.7,
            "stratum": R.density_stratum(a),
        }
        for i, a in enumerate([0.01, 0.03, 0.03, 0.2])
    ]
    report = R.summarize(rows)
    assert set(report) == {"all", "light", "medium", "dense"}
    assert report["all"]["psnr"]["n"] == 4 and report["all"]["psnr_input"]["mean"] == 20.0
    assert report["all"]["psnr"]["ci95_low"] < 31.5 < report["all"]["psnr"]["ci95_high"]
    assert report["dense"]["psnr"]["ci95_low"] == report["dense"]["psnr"]["mean"]  # n == 1


def test_one_crop_window_is_shared_by_input_target_and_alpha():
    batch = {"smoky": torch.arange(2 * 3 * 10 * 12, dtype=torch.float32).view(2, 3, 10, 12)}
    batch["clean"], batch["alpha"] = batch["smoky"].clone(), batch["smoky"][:, :1].clone()
    cropped = T.random_crop(batch, 4, torch.Generator().manual_seed(0))
    assert cropped["smoky"].shape[-2:] == (4, 4)
    assert torch.equal(cropped["smoky"], cropped["clean"])
    assert torch.equal(cropped["smoky"][:, :1], cropped["alpha"])
    assert T.random_crop(batch, None, torch.Generator()) is batch


def config_for(tmp_path, **overrides):
    values = {
        "epochs": 1,
        "batch_size": 2,
        "crop": 32,
        "base_channels": 8,
        "num_workers": 0,
        "amp": False,
        "processed_dir": str(tmp_path / "out"),
    }
    return T.DesmokeConfig(**(values | overrides))


def test_cpu_run_trains_evaluates_against_the_no_op_and_refuses_a_changed_resume(tmp_path):
    run_build(tmp_path, "out")
    result = T.train(
        config_for(tmp_path),
        device="cpu",
        out_dir=tmp_path / "run",
        resume=False,
        skip_quota=True,
        use_mlflow=False,
    )
    epoch = result["history"][0]
    assert math.isfinite(epoch["loss"]) and epoch["psnr_input"] < R.PSNR_CAP
    assert result["videos"] == {"train": ["VID01"], "val": ["VID08"]}
    assert result["config"]["data_config_sha256"]
    saved = json.loads((tmp_path / "run" / f"report_{result['config_hash']}.json").read_text())
    assert saved["disclaimer"] == T.DISCLAIMER and saved["real_smoke"] is None
    # Same experiment, more epochs: a resumption. Different learning rate: refused.
    longer = T.train(
        config_for(tmp_path, epochs=2),
        device="cpu",
        out_dir=tmp_path / "run",
        resume=True,
        skip_quota=True,
        use_mlflow=False,
    )
    assert [h["epoch"] for h in longer["history"]] == [1]
    changed = config_for(tmp_path, lr=1e-3)
    ckpt = tmp_path / "run" / f"ckpt_{result['config_hash']}.pt"
    model = build_desmoke_model(base=8)
    with pytest.raises(SystemExit, match="refusing to resume"):
        T.load_checkpoint(
            ckpt, model=model, optimizer=torch.optim.AdamW(model.parameters()), config=changed
        )


def test_test_split_run_also_scores_real_smoke_by_haze_drop(tmp_path):
    run_build(tmp_path, "out")
    result = T.train(
        config_for(tmp_path, eval_split="test"),
        device="cpu",
        out_dir=tmp_path / "run",
        resume=False,
        skip_quota=True,
        use_mlflow=False,
    )
    real = result["real_smoke"]
    assert real["haze_before"]["n"] == 1 and np.isfinite(real["haze_drop"]["mean"])


def test_a_cuda_run_never_skips_the_quota_gate_or_the_tracker(tmp_path):
    with pytest.raises(SystemExit, match="R-24"):
        T.train(
            config_for(tmp_path),
            device="cuda",
            out_dir=tmp_path,
            resume=False,
            skip_quota=True,
            use_mlflow=False,
        )
