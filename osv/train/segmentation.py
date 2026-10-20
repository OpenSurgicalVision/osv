"""Baseline semantic-segmentation training for CholecSeg8k.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

A deliberately ordinary baseline: torchvision's DeepLabV3-ResNet50 with the
head resized to the 13 published classes. Nothing here is a novel architecture,
and it is not meant to be - `osv/arch/registry.py` and the `arch-runner` agent
own architecture search under R-25, and this module is the fixed reference
those experiments are compared against.

What it does take seriously:

* **R-24 (quota).** A CUDA run refuses to start until
  `scripts/check_compute_quota.py` returns OK. Exhaustion is a clean exit with
  state preserved, not a crash. A CPU run may skip the gate explicitly, since
  it spends no GPU-hours - that opt-out is the only one and it is checked, not
  trusted.
* **R-24 (resumption).** Every epoch writes a checkpoint carrying the
  `config_hash`. Resuming against a different hash is refused outright: silently
  continuing a run under changed hyper-parameters produces a curve that
  describes no single experiment.
* **Reproducibility.** Seeds pinned across python/numpy/torch, cuDNN put in
  deterministic mode, and the resolved config plus its hash logged to MLflow so
  a number in a paper can be traced back to the run that produced it (R-15).
* **Honest metrics.** Per-class IoU/Dice via `osv.eval.metrics`, where a class
  absent from the split scores N/A rather than 0.0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from osv.datasets.segmentation import (
    CholecSeg8kSegmentation,
    inverse_frequency_weights,
)
from osv.eval import metrics as M

DISCLAIMER = "RESEARCH USE ONLY - NOT FOR CLINICAL USE"
QUOTA_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_compute_quota.py"


@dataclass
class TrainConfig:
    seed: int = 1337
    epochs: int = 20
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    architecture: str = "deeplabv3_resnet50"
    pretrained: bool = True
    class_weight_clip: float = 50.0
    train_limit: int | None = None
    eval_limit: int | None = None
    eval_split: str = "val"
    amp: bool = True
    num_workers: int = 0
    experiment: str = "osv-cholecseg8k-baseline"
    extra: dict[str, Any] = field(default_factory=dict)

    def hash(self) -> str:
        """Stable identity of this configuration.

        `epochs` is excluded on purpose: continuing the *same* experiment for
        longer is a resumption, whereas changing the learning rate or the
        architecture is a different experiment wearing the same name.
        """
        payload = {k: v for k, v in asdict(self).items() if k not in {"epochs", "extra"}}
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def pin_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preflight_quota(*, device: str, skip: bool, experiment: str | None = None) -> None:
    """R-24 gate. Raises SystemExit rather than proceeding on doubt.

    `experiment` is forwarded to the gate so the budget is read from the same
    place the run will be written to. Letting the two default independently is
    how a quota check ends up summing an experiment nobody trains into, and
    waving every run through on a confident zero.
    """
    if device != "cuda":
        if not skip:
            print("[osv.train] CPU run: no GPU-hours spent, quota gate not applicable.")
        return
    if skip:
        raise SystemExit(
            "[osv.train] --skip-quota-check is only permitted for a CPU run. A CUDA run "
            "spends the weekly budget and R-24 makes that gate mandatory."
        )
    if not QUOTA_SCRIPT.is_file():
        raise SystemExit(f"[osv.train] quota gate missing at {QUOTA_SCRIPT} - refusing to start.")

    argv = [sys.executable, str(QUOTA_SCRIPT)]
    if experiment:
        argv += ["--experiment", experiment]
    result = subprocess.run(argv, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise SystemExit(
            f"[osv.train] quota gate returned {result.returncode}; not starting. "
            "Under R-24 an exhausted or undetermined budget is a normal stop, not a crash."
        )


def run_name(config: Any) -> str:
    """`<architecture>-<config_hash>`: the name a resumed run shares with its first part."""
    return f"{config.architecture}-{config.hash()}"


class RunRecorder:
    """MLflow logging for one training run, and the invariant behind it.

    A GPU run that is not recorded does not merely lose its history - it makes
    the next quota pre-flight under-count, which is the exact failure R-24's
    gate exists to prevent. So on CUDA an unreachable tracker is fatal, not a
    warning. On CPU (no GPU-hours at stake) logging is best-effort.

    `gpu_count` is logged explicitly because the gate multiplies duration by it
    and assumes 1 when the parameter is missing - which would silently charge a
    CPU smoke run a GPU-hour per hour.

    `run_name` is set explicitly: with the seed pinned, MLflow's generated name
    is the same for every run, so a list of runs could not be told apart.
    """

    def __init__(
        self, *, enabled: bool, device: str, experiment: str, run_name: str | None = None
    ) -> None:
        self.device = device
        self.experiment = experiment
        self.run_name = run_name
        self.active = False
        self._mlflow = None

        if not enabled:
            if device == "cuda":
                raise SystemExit(
                    "[osv.train] --no-mlflow is only permitted for a CPU run: an unrecorded "
                    "GPU run makes the next R-24 quota check under-count the week."
                )
            print("[osv.train] MLflow logging disabled (CPU run).")
            return

        try:
            import mlflow
        except ImportError as exc:
            message = f"[osv.train] MLflow is not installed ({exc})."
            if device == "cuda":
                raise SystemExit(message + " A CUDA run must be recorded (R-24).") from exc
            print(message + " Continuing without tracking (CPU run).")
            return

        self._mlflow = mlflow
        try:
            mlflow.set_experiment(experiment)
            self.active = True
        except Exception as exc:  # noqa: BLE001 - any tracker failure is the same verdict
            message = f"[osv.train] MLflow tracker unreachable: {exc}."
            if device == "cuda":
                raise SystemExit(message + " Refusing to spend unrecorded GPU-hours (R-24).") from exc
            print(message + " Continuing without tracking (CPU run).")

    def __enter__(self) -> RunRecorder:
        if self.active:
            self._run = self._mlflow.start_run(run_name=self.run_name)
            print(f"[osv.train] MLflow run {self._run.info.run_id} in {self.experiment!r}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.active:
            status = "FINISHED" if exc_type is None else "FAILED"
            self._mlflow.end_run(status=status)

    def log_params(self, params: dict[str, Any]) -> None:
        if self.active:
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.active:
            self._mlflow.log_metrics(metrics, step=step)


def build_model(num_classes: int, *, architecture: str, pretrained: bool) -> nn.Module:
    from torchvision.models import segmentation as tvseg

    if architecture != "deeplabv3_resnet50":
        raise ValueError(
            f"unsupported architecture {architecture!r}. Widening the search space is a "
            "human PR against osv/arch/registry.py (R-25), not a flag on this script."
        )
    # `weights_backbone` must be silenced explicitly: torchvision defaults it to
    # IMAGENET1K_V1 regardless of `weights`, so passing weights=None alone still
    # reaches out to download.pytorch.org. That host is not on the R-07 egress
    # allowlist, and "no pretrained" has to mean no network, not almost none.
    weights = "DEFAULT" if pretrained else None
    weights_backbone = "DEFAULT" if pretrained else None
    model = tvseg.deeplabv3_resnet50(
        weights=weights, weights_backbone=weights_backbone, aux_loss=True
    )
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


@torch.no_grad()
def evaluate(model, loader, device, num_classes, class_names) -> dict[str, Any]:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].numpy()
        logits = model(images)["out"]
        pred = logits.argmax(dim=1).cpu().numpy()
        cm += M.confusion_matrix(pred, labels, num_classes)
    return M.summarize(cm, class_names)


def save_checkpoint(path: Path, *, model, optimizer, epoch: int, config: TrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config_hash": config.hash(),
            "config": asdict(config),
            "disclaimer": DISCLAIMER,
        },
        path,
    )


def load_checkpoint(path: Path, *, model, optimizer, config: TrainConfig) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    stored = state.get("config_hash")
    if stored != config.hash():
        raise SystemExit(
            f"[osv.train] refusing to resume: checkpoint config_hash {stored} != current "
            f"{config.hash()}. Under R-24 a changed configuration forbids resumption - "
            "start a new run rather than splicing two experiments into one curve."
        )
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return int(state["epoch"]) + 1


def train(
    config: TrainConfig,
    *,
    device: str,
    out_dir: Path,
    resume: bool,
    skip_quota: bool,
    use_mlflow: bool = True,
) -> dict[str, Any]:
    print(f"[osv.train] {DISCLAIMER}")
    preflight_quota(device=device, skip=skip_quota, experiment=config.experiment)
    pin_seed(config.seed)

    train_ds = CholecSeg8kSegmentation("train", limit=config.train_limit)
    eval_ds = CholecSeg8kSegmentation(config.eval_split, limit=config.eval_limit)
    print(
        f"[osv.train] train {len(train_ds)} frames / {len(train_ds.videos)} videos; "
        f"{config.eval_split} {len(eval_ds)} frames / {len(eval_ds.videos)} videos"
    )
    print(f"[osv.train] class mapping: {train_ds.mapping_provenance}")

    overlap = set(train_ds.videos) & set(eval_ds.videos)
    if overlap:
        raise SystemExit(
            f"[osv.train] patient leakage: video(s) {sorted(overlap)} appear in both splits. "
            "R-08 makes this a critical bug, not a warning."
        )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=False,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,
    )

    counts = train_ds.class_pixel_counts(sample=config.train_limit or 200)
    weights = inverse_frequency_weights(counts, clip=config.class_weight_clip).to(device)
    absent = [train_ds.class_names[i] for i, c in enumerate(counts) if c == 0]
    if absent:
        print(f"[osv.train] absent from the training split, weight 0: {absent}")

    model = build_model(train_ds.num_classes, architecture=config.architecture,
                        pretrained=config.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=M.IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler(device, enabled=config.amp and device == "cuda")

    ckpt_path = out_dir / f"ckpt_{config.hash()}.pt"
    start_epoch = 0
    if resume and ckpt_path.is_file():
        start_epoch = load_checkpoint(ckpt_path, model=model, optimizer=optimizer, config=config)
        print(f"[osv.train] resumed from {ckpt_path} at epoch {start_epoch}")

    recorder = RunRecorder(
        enabled=use_mlflow, device=device, experiment=config.experiment, run_name=run_name(config)
    )
    history: list[dict[str, Any]] = []
    with recorder:
        recorder.log_params(
            {
                **{k: v for k, v in asdict(config).items() if k != "extra"},
                "config_hash": config.hash(),
                "device": device,
                # The quota gate multiplies duration by this and assumes 1 when
                # absent; a CPU run must therefore say 0 out loud (R-24).
                "gpu_count": torch.cuda.device_count() if device == "cuda" else 0,
                "train_videos": ",".join(train_ds.videos),
                f"{config.eval_split}_videos": ",".join(eval_ds.videos),
                "train_frames": len(train_ds),
                "eval_frames": len(eval_ds),
                "disclaimer": DISCLAIMER,
            }
        )
        report = _run_epochs(
            config=config, device=device, model=model, criterion=criterion,
            optimizer=optimizer, scaler=scaler, train_loader=train_loader,
            eval_loader=eval_loader, train_ds=train_ds, start_epoch=start_epoch,
            ckpt_path=ckpt_path, history=history, recorder=recorder,
        )

    final = {
        "disclaimer": DISCLAIMER,
        "config": asdict(config),
        "config_hash": config.hash(),
        "device": device,
        "train_videos": train_ds.videos,
        f"{config.eval_split}_videos": eval_ds.videos,
        "history": history,
        "final_report": report,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"report_{config.hash()}.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    return final


def _run_epochs(
    *, config, device, model, criterion, optimizer, scaler,
    train_loader, eval_loader, train_ds, start_epoch, ckpt_path, history, recorder,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for epoch in range(start_epoch, config.epochs):
        model.train()
        running, seen, t0 = 0.0, 0, time.time()
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=config.amp and device == "cuda"):
                out = model(images)
                loss = criterion(out["out"], labels)
                if out.get("aux") is not None:
                    loss = loss + 0.4 * criterion(out["aux"], labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * images.size(0)
            seen += images.size(0)

        train_loss = running / max(seen, 1)
        report = evaluate(model, eval_loader, device, train_ds.num_classes, train_ds.class_names)
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "seconds": round(time.time() - t0, 1),
            **{k: v for k, v in report.items() if k != "per_class"},
        }
        history.append(entry)
        print(
            f"[osv.train] epoch {epoch}: loss {train_loss:.4f} | macro_iou "
            f"{report['macro_iou']:.4f} over {report['classes_scored']} classes "
            f"({report['classes_skipped_absent']} N/A) | {entry['seconds']}s"
        )
        save_checkpoint(ckpt_path, model=model, optimizer=optimizer, epoch=epoch, config=config)
        recorder.log_metrics(
            {
                "train_loss": train_loss,
                "epoch_seconds": entry["seconds"],
                **M.flatten_for_tracker(report),
            },
            step=epoch,
        )

    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="runs/cholecseg8k")
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--eval-limit", type=int, default=None)
    p.add_argument("--eval-split", default="val", choices=["val", "test"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--skip-quota-check",
        action="store_true",
        help="CPU runs only: a CUDA run may never skip the R-24 gate.",
    )
    p.add_argument(
        "--no-mlflow",
        action="store_true",
        help="CPU runs only: an unrecorded GPU run makes the next quota check under-count.",
    )
    p.add_argument("--experiment", default=TrainConfig.experiment)
    args = p.parse_args(argv)

    if args.eval_split == "test":
        print(
            "[osv.train] WARNING: evaluating on the held-out test split. Under R-25 the test "
            "set is a separate track an agent has no read access to; every touch belongs in "
            "the counter in docs/BENCHMARK.md."
        )

    config = TrainConfig(
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        pretrained=not args.no_pretrained, amp=not args.no_amp,
        train_limit=args.train_limit, eval_limit=args.eval_limit,
        eval_split=args.eval_split, num_workers=args.num_workers,
        experiment=args.experiment,
    )
    train(config, device=args.device, out_dir=Path(args.out),
          resume=args.resume, skip_quota=args.skip_quota_check,
          use_mlflow=not args.no_mlflow)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["TrainConfig", "build_model", "evaluate", "train", "preflight_quota", "pin_seed"]
