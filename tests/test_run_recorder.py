"""MLflow run names: explicit, because a pinned seed gives every run the same generated one."""

from __future__ import annotations

import sys
import types

from osv.train import desmoke as T
from osv.train import desmoke_gan as TG
from osv.train import segmentation as S


def test_run_name_is_the_architecture_and_the_config_hash():
    config = T.DesmokeConfig()
    assert S.run_name(config) == f"residual_unet-{config.hash()}"
    gan = TG.GanConfig(architecture="srgan")
    assert S.run_name(gan) == f"srgan-{gan.hash()}"
    assert S.run_name(TG.GanConfig(architecture="pix2pix")) != S.run_name(gan)


def test_the_recorder_starts_the_run_under_that_name(monkeypatch):
    started = {}

    class Run:
        info = types.SimpleNamespace(run_id="r1")

    fake = types.ModuleType("mlflow")
    fake.set_experiment = lambda name: None
    fake.start_run = lambda run_name=None: started.setdefault("name", run_name) and Run()
    fake.end_run = lambda status: started.setdefault("status", status)
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    with S.RunRecorder(enabled=True, device="cpu", experiment="e", run_name="pix2pix-abc"):
        pass
    assert started == {"name": "pix2pix-abc", "status": "FINISHED"}
