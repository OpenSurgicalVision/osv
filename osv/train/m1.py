"""M1 training for RFC 0001: synthetic instruments against real labels.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

One trainer for every arm of the RFC, so that only the training data differs:

* **A** - synthetic composites only (`osv.datasets.m1.SyntheticInstruments`); the
  checkpoint is selected on the fixed **synthetic** val pairs. Its real-val score is
  logged under ``info_real_val_*`` for information and never used: selecting on real
  frames would leak real labels into the synthetic arm.
* **B** - CholecSeg8k real-train, selected on real-val.
* **C** - arm A's selected checkpoint of the same seed, fine-tuned on ``--real-fraction``
  of the real-train frames (0.1 / 0.25 / 0.5 / 1.0), selected on real-val.

Fixed in every arm (RFC 0001, section 2):

* `deeplabv3_resnet50` from `osv/arch/registry.py` with a 3-class head (background,
  grasper, L-hook) and the ImageNet ResNet-50 backbone from a **local** file (R-07);
* full 854x480 frames, AdamW with cosine decay, unweighted cross-entropy plus 0.4 x the
  auxiliary head, the same epochs and the same number of samples per epoch: arm A draws
  as many fresh composites per epoch as real-train has frames, so A and B take the same
  number of optimiser steps;
* real-train frames get random flips, drawn per (seed, epoch, frame); the composites
  already carry flips among their augmentations.

The selection metric is the RFC's primary one, the IoU of the binary instrument mask
over all pixels of the val frames. The held-out test videos are never read here:
`osv.eval.m1 test` scores a selected checkpoint once and counts the touch (R-25).

    python -m osv.train.m1 --arm A --seed 1 --backbone-weights /path/resnet50-0676ba61.pth
    python -m osv.train.m1 --arm B --seed 1 --backbone-weights ...
    python -m osv.train.m1 --arm C --real-fraction 0.25 --seed 1 \\
        --init runs/m1/A_s1/best.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from osv.arch.registry import lookup
from osv.datasets import m1 as D
from osv.eval import metrics as M
from osv.train.segmentation import RunRecorder, pin_seed, preflight_quota

DISCLAIMER = "RESEARCH USE ONLY - NOT FOR CLINICAL USE"
TASK = "M1 segmentation"
ARMS = ("A", "B", "C")
AUX_WEIGHT = 0.4
RESNET50_FC = ("fc.weight", "fc.bias")


@dataclass
class M1Config:
    arm: str = "A"
    seed: int = 1
    epochs: int = 20
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    architecture: str = "deeplabv3_resnet50"
    real_fraction: float = 1.0
    smoke: bool = True
    amp: bool = True
    train_limit: int | None = None
    eval_limit: int | None = None
    num_workers: int = 8
    experiment: str = "osv-m1-synthetic"
    # Inputs: the paths are where they happen to be, the hashes are what they are.
    backbone_weights: str = ""
    backbone_sha256: str = ""
    init: str = ""
    init_sha256: str = ""
    synth_dir: str = str(D.DEFAULT_SYNTH_DIR)
    processed_dir: str = "data/processed/cholecseg8k"
    source: str = ""
    smoke_dir: str = "data/processed/smoke_m5"
    data_sha256: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # `epochs` is hashed: the cosine schedule spans it, so a longer run is another run.
    HASH_SKIP = frozenset(
        {
            "extra",
            "num_workers",
            "backbone_weights",
            "init",
            "synth_dir",
            "processed_dir",
            "source",
            "smoke_dir",
        }
    )

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {self.arm!r}")
        if self.arm != "C" and self.real_fraction != 1.0:
            raise ValueError("--real-fraction is for arm C; B always uses all real-train frames")
        if self.arm == "C" and not self.init:
            raise ValueError("arm C fine-tunes arm A's selected checkpoint: pass --init")
        if self.arm != "C" and self.init:
            raise ValueError("--init is for arm C only")

    def hash(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k not in self.HASH_SKIP}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    @property
    def name(self) -> str:
        fraction = f"{round(self.real_fraction * 100)}" if self.arm == "C" else ""
        return f"m1-{self.arm}{fraction}-s{self.seed}-{self.hash()}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(*, architecture: str, backbone_weights: str | None) -> nn.Module:
    """DeepLabV3-ResNet50, 3 classes; the backbone from a local ImageNet file or random."""
    from torchvision.models import segmentation as tvseg

    lookup(architecture, task=TASK)
    model = tvseg.deeplabv3_resnet50(
        weights=None, weights_backbone=None, num_classes=len(D.CLASSES), aux_loss=True
    )
    if backbone_weights:
        state = torch.load(backbone_weights, map_location="cpu", weights_only=True)
        missing, unexpected = model.backbone.load_state_dict(
            {k: v for k, v in state.items() if k not in RESNET50_FC}, strict=False
        )
        if missing or unexpected:
            raise SystemExit(
                f"[osv.train.m1] {backbone_weights} is not a ResNet-50 state dict "
                f"(missing {missing[:3]}, unexpected {unexpected[:3]})"
            )
    return model


class Flipped(Dataset):
    """Random horizontal and vertical flips of real frames, fixed per (seed, epoch, frame)."""

    def __init__(self, dataset: D.RealInstruments, seed: int) -> None:
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample = self.dataset[item]
        rng = np.random.default_rng([self.seed, self.epoch, item])
        dims = [d for d, flip in ((-1, rng.random() < 0.5), (-2, rng.random() < 0.5)) if flip]
        if dims:
            sample["image"] = sample["image"].flip(dims)
            sample["label"] = sample["label"].flip(dims)  # (H, W): the same last two axes
        return sample


def real_split(processed_dir: str | Path) -> dict[str, list[str]]:
    from osv.datasets import cholect50 as C

    doc = json.loads((Path(processed_dir) / "instances.json").read_text(encoding="utf-8"))
    frames: dict[str, int] = {}
    for image in doc["images"]:
        frames[image["video_id"]] = frames.get(image["video_id"], 0) + 1
    return D.m1_split(frames, C.official_split(C.load_manifest()))


def datasets(config: M1Config) -> dict[str, Any]:
    split = real_split(config.processed_dir)
    real: dict[str, Any] = {"processed_dir": config.processed_dir, "source": config.source or None}
    real_val = D.RealInstruments(split["val"], limit=config.eval_limit, **real)
    if config.arm == "A":
        real_frames = len(D.RealInstruments(split["train"], **real))
        train: Dataset = D.SyntheticInstruments(
            "train",
            root=config.synth_dir,
            seed=config.seed,
            samples_per_epoch=config.train_limit or real_frames,
            smoke_dir=config.smoke_dir if config.smoke else None,
        )
        if config.smoke and not train.smoke:  # type: ignore[attr-defined]
            raise SystemExit(f"[osv.train.m1] no smoke layers under {config.smoke_dir}/smoke")
        selection = D.SyntheticInstruments("val", root=config.synth_dir, limit=config.eval_limit)
        evals = {"val": selection, "info_real_val": real_val}
    else:
        fraction = config.real_fraction
        real_train = D.RealInstruments(
            split["train"], fraction=fraction, seed=config.seed, limit=config.train_limit, **real
        )
        train = Flipped(real_train, config.seed)
        evals = {"val": real_val}
    return {"split": split, "train": train, "evals": evals}


def data_sha256(config: M1Config) -> str:
    """The CholecSeg8k index, and for arm A the synthetic build (C has it through --init)."""
    parts = [sha256_file(Path(config.processed_dir) / "instances.json")]
    if config.arm == "A":
        parts.append(sha256_file(Path(config.synth_dir) / "index.json"))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def binary_iou(cm: np.ndarray) -> float:
    """IoU of instrument (any class > 0) against background, from a 3x3 confusion matrix."""
    tp = float(cm[1:, 1:].sum())
    fp = float(cm[0, 1:].sum())
    fn = float(cm[1:, 0].sum())
    return tp / (tp + fp + fn) if tp + fp + fn else float("nan")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, amp: bool) -> dict[str, Any]:
    model.eval()
    cm = np.zeros((len(D.CLASSES), len(D.CLASSES)), dtype=np.int64)
    for batch in loader:
        with torch.amp.autocast(device, enabled=amp and device == "cuda"):
            logits = model(batch["image"].to(device, non_blocking=True))["out"]
        cm += M.confusion_matrix(logits.argmax(1).cpu().numpy(), batch["label"].numpy(), len(cm))
    report = M.summarize(cm, D.CLASSES)
    report["binary_iou"] = binary_iou(cm)
    report["confusion"] = cm.tolist()
    return report


def tracker_metrics(report: dict[str, Any], prefix: str) -> dict[str, float]:
    metrics = M.flatten_for_tracker(report, prefix=f"{prefix}_")
    if not np.isnan(report["binary_iou"]):
        metrics[f"{prefix}_binary_iou"] = float(report["binary_iou"])
    return metrics


def initial_state(config: M1Config) -> dict[str, Any] | None:
    """Arm C: arm A's selected checkpoint of the same seed, checked, with its hash recorded."""
    if config.arm != "C":
        return None
    state = torch.load(config.init, map_location="cpu", weights_only=False)
    parent = state.get("config", {})
    if parent.get("arm") != "A" or not state.get("selected_on"):
        raise SystemExit(f"[osv.train.m1] {config.init} is not arm A's selected checkpoint")
    if parent.get("seed") != config.seed:
        raise SystemExit(
            f"[osv.train.m1] arm C seed {config.seed} must fine-tune arm A seed {config.seed}, "
            f"not seed {parent.get('seed')}"
        )
    config.init_sha256 = sha256_file(config.init)
    return state


def train_epoch(model, loader, criterion, optimizer, scaler, *, device: str, amp: bool) -> float:
    """One pass over `loader`; the mean loss per sample."""
    model.train()
    running, seen = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device, enabled=amp and device == "cuda"):
            out = model(images)
            loss = criterion(out["out"], labels) + AUX_WEIGHT * criterion(out["aux"], labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.detach()) * images.size(0)
        seen += images.size(0)
    return running / max(seen, 1)


def resume_from(path: Path, config: M1Config, **parts: Any) -> dict[str, Any]:
    """Load model, optimizer, scheduler and scaler from `last.pt`; refuse another config."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["config_hash"] != config.hash():
        raise SystemExit(
            f"[osv.train.m1] refusing to resume: {path} has config_hash "
            f"{state['config_hash']}, this run {config.hash()} (R-24)"
        )
    for name, part in parts.items():
        part.load_state_dict(state[name])
    print(f"[osv.train.m1] resumed {config.name} at epoch {state['epoch'] + 1}")
    return state


def save(path: Path, **state: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({**state, "disclaimer": DISCLAIMER}, tmp)
    tmp.replace(path)


def train(
    config: M1Config,
    *,
    device: str,
    out_dir: Path,
    resume: bool = False,
    skip_quota: bool = False,
    use_mlflow: bool = True,
) -> dict[str, Any]:
    print(f"[osv.train.m1] {DISCLAIMER}")
    preflight_quota(device=device, skip=skip_quota, experiment=config.experiment)
    if config.backbone_weights:
        config.backbone_sha256 = sha256_file(config.backbone_weights)
    elif config.arm != "C":
        raise SystemExit(
            "[osv.train.m1] pass --backbone-weights: the RFC starts arms A and B from the "
            "ImageNet ResNet-50, read from a local file (R-07)"
        )
    parent = initial_state(config)
    config.data_sha256 = data_sha256(config)
    pin_seed(config.seed)

    data = datasets(config)
    split, train_ds = data["split"], data["train"]
    print(
        f"[osv.train.m1] arm {config.arm}: train {len(train_ds)} samples/epoch; "
        + ", ".join(f"{k} {len(v)}" for k, v in data["evals"].items())
        + f"; real videos train {split['train']}, val {split['val']} (test never read here)"
    )

    model = build_model(
        architecture=config.architecture,
        backbone_weights=None if parent else config.backbone_weights,
    )
    if parent:
        model.load_state_dict(parent["model"])
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=M.IGNORE_INDEX)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler(device, enabled=config.amp and device == "cuda")

    loader_args: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_args)
    eval_loaders = {
        k: DataLoader(v, shuffle=False, **loader_args) for k, v in data["evals"].items()
    }

    run_dir = out_dir / config.name
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    history: list[dict[str, Any]] = []
    start, best = 0, {"epoch": None, "binary_iou": -1.0}
    if resume and last_path.is_file():
        state = resume_from(
            last_path, config, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler
        )
        start, history, best = state["epoch"] + 1, state["history"], state["best"]

    recorder = RunRecorder(
        enabled=use_mlflow, device=device, experiment=config.experiment, run_name=config.name
    )
    with recorder:
        recorder.log_params(
            {
                **{k: v for k, v in asdict(config).items() if k != "extra"},
                "config_hash": config.hash(),
                "device": device,
                "gpu_count": torch.cuda.device_count() if device == "cuda" else 0,
                "real_train_videos": ",".join(split["train"]),
                "real_val_videos": ",".join(split["val"]),
                "train_samples_per_epoch": len(train_ds),
                "disclaimer": DISCLAIMER,
            }
        )
        if config.arm != "A":
            frames = "\n".join(r["file_name"] for r in train_ds.dataset.images)  # type: ignore[attr-defined]
            recorder.log_text(frames, "real_train_frames.txt")

        for epoch in range(start, config.epochs):
            train_ds.set_epoch(epoch)  # type: ignore[attr-defined]
            t0 = time.time()
            loss = train_epoch(
                model, train_loader, criterion, optimizer, scaler, device=device, amp=config.amp
            )
            scheduler.step()
            reports = {k: evaluate(model, v, device, config.amp) for k, v in eval_loaders.items()}
            entry = {
                "epoch": epoch,
                "train_loss": loss,
                "seconds": round(time.time() - t0, 1),
                **{f"{k}_binary_iou": r["binary_iou"] for k, r in reports.items()},
            }
            history.append(entry)
            selected = reports["val"]["binary_iou"] > best["binary_iou"]
            if selected:
                best = {"epoch": epoch, "binary_iou": reports["val"]["binary_iou"]}
                save(
                    best_path,
                    model=model.state_dict(),
                    epoch=epoch,
                    config=asdict(config),
                    config_hash=config.hash(),
                    selected_on="synthetic val" if config.arm == "A" else "real val",
                    val=reports["val"],
                )
            save(
                last_path,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),
                epoch=epoch,
                history=history,
                best=best,
                config=asdict(config),
                config_hash=config.hash(),
            )
            metrics = {"train_loss": entry["train_loss"], "epoch_seconds": entry["seconds"]}
            for k, r in reports.items():
                metrics |= tracker_metrics(r, k)
            recorder.log_metrics(metrics, step=epoch)
            print(
                f"[osv.train.m1] epoch {epoch}: loss {entry['train_loss']:.4f} | "
                + " | ".join(f"{k} binary IoU {r['binary_iou']:.4f}" for k, r in reports.items())
                + f" | {entry['seconds']}s{' | selected' if selected else ''}"
            )
        recorder.log_metrics(
            {
                "selected_epoch": float(best["epoch"] or 0),
                "selected_val_binary_iou": float(best["binary_iou"] or 0),
            },
            step=config.epochs,
        )

    report = {
        "disclaimer": DISCLAIMER,
        "config": asdict(config),
        "config_hash": config.hash(),
        "device": device,
        "real_split": split,
        "history": history,
        "selected": best,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--real-fraction", type=float, default=1.0, help="arm C only")
    p.add_argument("--init", default="", help="arm C: arm A's best.pt of the same seed")
    p.add_argument("--backbone-weights", default="", help="local torchvision resnet50 file")
    p.add_argument("--epochs", type=int, default=M1Config.epochs)
    p.add_argument("--batch-size", type=int, default=M1Config.batch_size)
    p.add_argument("--lr", type=float, default=M1Config.lr)
    p.add_argument("--no-smoke", action="store_true", help="arm A without M5 smoke layers")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--eval-limit", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=M1Config.num_workers)
    p.add_argument("--synth-dir", default=M1Config.synth_dir)
    p.add_argument("--processed-dir", default=M1Config.processed_dir)
    p.add_argument("--source", default="", help="CholecSeg8k zip (default CHOLECSEG8K_RAW_ZIP)")
    p.add_argument("--smoke-dir", default=M1Config.smoke_dir)
    p.add_argument("--experiment", default=M1Config.experiment)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("runs/m1"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-quota-check", action="store_true", help="CPU runs only")
    p.add_argument("--no-mlflow", action="store_true", help="CPU runs only")
    args = p.parse_args(argv)
    config = M1Config(
        arm=args.arm,
        seed=args.seed,
        real_fraction=args.real_fraction,
        init=args.init,
        backbone_weights=args.backbone_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        smoke=not args.no_smoke,
        amp=not args.no_amp,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        num_workers=args.num_workers,
        synth_dir=args.synth_dir,
        processed_dir=args.processed_dir,
        source=args.source,
        smoke_dir=args.smoke_dir,
        experiment=args.experiment,
    )
    train(
        config,
        device=args.device,
        out_dir=args.out,
        resume=args.resume,
        skip_quota=args.skip_quota_check,
        use_mlflow=not args.no_mlflow,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["M1Config", "binary_iou", "build_model", "evaluate", "real_split", "train"]
