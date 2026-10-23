"""M5 baseline training: residual U-Net desmoking on synthetic smoke over CholecT50.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

    python -m osv.train.desmoke --epochs 40 --batch-size 8 --crop 256
    python -m osv.train.desmoke --architecture pix2pix ...   # GANs: osv.train.desmoke_gan

`--architecture` is looked up in `osv/arch/registry.py` (R-25); the supervised
residual U-Net trains here, the GAN rows in `osv.train.desmoke_gan`.

Data: the `smoke_m5_build` DVC stage (`osv/datasets/smoke_m5.py`). Train pairs
are composited on the fly, new smoke every epoch; val/test are the fixed pairs.

The run discipline is the segmentation baseline's, imported rather than copied
(`osv/train/segmentation.py`): the R-24 quota gate before any CUDA run, a
`config_hash` that forbids resuming a changed experiment, seeds pinned, and every
run recorded in MLflow (fatal on CUDA when the tracker is unreachable).

Loss: L1 on the restored frame + 0.2 * (1 - SSIM) + 0.1 * L1 on the predicted
smoke alpha. Every evaluation reports the model next to the no-op baseline (the
smoky input itself), overall and per smoke-density stratum, each with a 95 %
interval over frames. `--eval-split test` also scores the real test frames labelled
as smoke by the drop in their haze score - an indicator with no ground truth, and a
touch of the held-out split that belongs in the R-25 counter. None are labelled yet
(`osv.datasets.desmoke.confirmed_smoke`), so for now it is skipped.
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
from PIL import Image
from torch.utils.data import DataLoader

from osv.arch.registry import REGISTRY, lookup
from osv.datasets.cholect50 import haze_score
from osv.datasets.desmoke import DEFAULT_PROCESSED_DIR, SyntheticSmokePairs, confirmed_smoke
from osv.eval import restoration as R
from osv.models.desmoke import TASK, build_desmoke_model
from osv.train import previews as P
from osv.train.segmentation import RunRecorder, pin_seed, preflight_quota, run_name

DISCLAIMER = "RESEARCH USE ONLY - NOT FOR CLINICAL USE"


@dataclass
class DesmokeConfig:
    seed: int = 1337
    epochs: int = 40
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-4
    architecture: str = "residual_unet"
    base_channels: int = 32
    crop: int | None = 256
    ssim_weight: float = 0.2
    alpha_weight: float = 0.1
    train_limit: int | None = None
    eval_limit: int | None = None
    eval_split: str = "val"
    amp: bool = True
    num_workers: int = 4
    processed_dir: str = str(DEFAULT_PROCESSED_DIR)
    data_config_sha256: str = ""
    experiment: str = "osv-m5-desmoke-baseline"
    # Per-epoch preview sheets (osv.train.previews): looking, not training, so not hashed.
    preview_count: int = 8
    preview_every: int = 1
    preview_scale: float = 0.5
    extra: dict[str, Any] = field(default_factory=dict)

    HASH_SKIP = frozenset(
        {
            "epochs",
            "extra",
            "num_workers",
            "processed_dir",
            "preview_count",
            "preview_every",
            "preview_scale",
        }
    )

    def hash(self) -> str:
        """Like TrainConfig.hash: epochs, workers and paths do not change the experiment."""
        payload = {k: v for k, v in asdict(self).items() if k not in self.HASH_SKIP}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]


def random_crop(
    batch: dict[str, Any], size: int | None, generator: torch.Generator
) -> dict[str, Any]:
    """One window for the whole batch, the same for input, target and alpha (when present)."""
    if size is None:
        return batch
    height, width = batch["smoky"].shape[-2:]
    size_h, size_w = min(size, height), min(size, width)
    top = int(torch.randint(0, height - size_h + 1, (1,), generator=generator))
    left = int(torch.randint(0, width - size_w + 1, (1,), generator=generator))
    return {
        **batch,
        **{
            k: batch[k][..., top : top + size_h, left : left + size_w]
            for k in ("smoky", "clean", "alpha")
            if k in batch
        },
    }


def loss_terms(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], config: DesmokeConfig):
    l1 = (out["clean"] - batch["clean"]).abs().mean()
    structure = 1 - R.ssim(out["clean"], batch["clean"]).mean()
    alpha = (out["alpha"] - batch["alpha"]).abs().mean()
    total = l1 + config.ssim_weight * structure + config.alpha_weight * alpha
    terms = {"l1": l1, "ssim_loss": structure, "alpha_l1": alpha}
    return total, {name: float(value.detach()) for name, value in terms.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module, dataset: SyntheticSmokePairs, device: str, *, batch_size: int
) -> dict[str, Any]:
    model.eval()
    alpha_mean = {p["pair_id"]: p["alpha_mean"] for p in dataset.pairs}
    rows: list[dict[str, Any]] = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        smoky, clean = batch["smoky"].to(device), batch["clean"].to(device)
        restored = model(smoky)["clean"]
        scores = zip(
            R.psnr(restored, clean).tolist(),
            R.ssim(restored, clean).tolist(),
            R.psnr(smoky, clean).tolist(),
            R.ssim(smoky, clean).tolist(),
            batch["name"],
            strict=True,
        )
        for p, s, p_in, s_in, name in scores:
            rows.append(
                {
                    "name": name,
                    "psnr": p,
                    "ssim": s,
                    "psnr_input": p_in,
                    "ssim_input": s_in,
                    "stratum": R.density_stratum(alpha_mean[name]),
                }
            )
    return R.summarize(rows)


@torch.no_grad()
def evaluate_real_smoke(
    model: nn.Module, processed_dir: Path, split: str, device: str
) -> dict[str, Any] | None:
    """Haze score before and after on labelled real smoky frames: an indicator only.

    None when no frame of the split is labelled as smoke (`confirmed_smoke`).
    """
    model.eval()
    index = json.loads((processed_dir / "index.json").read_text())
    frame_ids = confirmed_smoke(index, split)
    if not frame_ids:
        print(f"[osv.train] no {split} frame labelled as real smoke: haze drop skipped.")
        return None
    before, after = [], []
    for frame_id in frame_ids:
        with Image.open(processed_dir / "real_smoke" / split / f"{frame_id}.png") as image:
            rgb = np.asarray(image.convert("RGB"))
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1)[None].to(device)
        restored = model(tensor)["clean"][0].permute(1, 2, 0).cpu().numpy()
        before.append(haze_score(rgb))
        after.append(haze_score(np.clip(np.rint(restored * 255), 0, 255).astype(np.uint8)))
    return {
        "haze_before": R.mean_ci95(before),
        "haze_after": R.mean_ci95(after),
        "haze_drop": R.mean_ci95([b - a for b, a in zip(before, after, strict=True)]),
    }


def split_videos(index: dict[str, Any], eval_split: str) -> dict[str, list[str]]:
    videos = {
        split: sorted({r["video"] for r in index["clean"] if r["split"] == split})
        for split in ("train", eval_split)
    }
    if set(videos["train"]) & set(videos[eval_split]):
        raise SystemExit("[osv.train] patient leakage between train and eval videos (R-08).")
    return videos


def save_checkpoint(
    path: Path, *, model, optimizer, scheduler, epoch: int, config: DesmokeConfig
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "config_hash": config.hash(),
            "config": asdict(config),
            "disclaimer": DISCLAIMER,
        },
        path,
    )


def load_checkpoint(path: Path, *, model, optimizer, config: DesmokeConfig, scheduler=None) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("config_hash") != config.hash():
        raise SystemExit(
            f"[osv.train] refusing to resume: checkpoint config_hash {state.get('config_hash')} != "
            f"current {config.hash()}. A changed configuration is a new run (R-24)."
        )
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    return int(state["epoch"]) + 1


def preview_hook(config: DesmokeConfig, processed: Path, out_dir: Path, device: str) -> Any:
    """`hook(model, epoch)` writing preview sheets after the epochs `preview_every` picks;
    `epoch=None` is the untrained model. A no-op when `preview_count` is 0."""
    if config.preview_count <= 0:
        return lambda model, epoch: None
    previews = P.PreviewSet(processed, count=config.preview_count, seed=config.seed)
    folder = out_dir / "previews"
    title = f"{run_name(config)} - M5 previews"

    def hook(model: nn.Module, epoch: int | None) -> None:
        last = epoch is not None and epoch == config.epochs - 1
        if epoch is None or last or (epoch + 1) % max(config.preview_every, 1) == 0:
            P.save(
                model,
                previews,
                folder,
                epoch=epoch,
                device=device,
                title=title,
                scale=config.preview_scale,
            )

    return hook


def _flatten(report: dict[str, Any], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}{group}_{metric}": values["mean"]
        for group, metrics in report.items()
        for metric, values in metrics.items()
        if isinstance(values, dict) and np.isfinite(values["mean"])
    }


def train(
    config: DesmokeConfig,
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
    processed = Path(config.processed_dir)
    index = json.loads((processed / "index.json").read_text())
    config.data_config_sha256 = index["config_sha256"]
    train_ds = SyntheticSmokePairs(
        "train", processed_dir=processed, seed=config.seed, limit=config.train_limit
    )
    eval_ds = SyntheticSmokePairs(
        config.eval_split, processed_dir=processed, limit=config.eval_limit
    )
    videos = split_videos(index, config.eval_split)
    print(
        f"[osv.train] train {len(train_ds)} frames / {len(videos['train'])} videos; "
        f"{config.eval_split} {len(eval_ds)} pairs / {len(videos[config.eval_split])} videos"
    )

    model = build_desmoke_model(architecture=config.architecture, base=config.base_channels).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    scaler = torch.amp.GradScaler(device, enabled=config.amp and device == "cuda")
    ckpt_path = out_dir / f"ckpt_{config.hash()}.pt"
    start_epoch = 0
    if resume and ckpt_path.is_file():
        start_epoch = load_checkpoint(
            ckpt_path, model=model, optimizer=optimizer, config=config, scheduler=scheduler
        )
        print(f"[osv.train] resumed from {ckpt_path} at epoch {start_epoch}")
    preview = preview_hook(config, processed, out_dir, device)
    if start_epoch == 0:
        preview(model, None)

    loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    crops = torch.Generator().manual_seed(config.seed + 1)
    history: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    recorder = RunRecorder(
        enabled=use_mlflow, device=device, experiment=config.experiment, run_name=run_name(config)
    )
    with recorder:
        recorder.log_params(
            {
                **{k: v for k, v in asdict(config).items() if k != "extra"},
                "config_hash": config.hash(),
                "device": device,
                "gpu_count": torch.cuda.device_count() if device == "cuda" else 0,
                "train_videos": ",".join(videos["train"]),
                f"{config.eval_split}_videos": ",".join(videos[config.eval_split]),
                "train_frames": len(train_ds),
                "eval_pairs": len(eval_ds),
                "disclaimer": DISCLAIMER,
            }
        )
        for epoch in range(start_epoch, config.epochs):
            model.train()
            train_ds.set_epoch(epoch)
            sums: dict[str, float] = {}
            seen, started = 0, time.time()
            for raw in loader:
                batch = random_crop(raw, config.crop, crops)
                batch = {
                    k: batch[k].to(device, non_blocking=True) for k in ("smoky", "clean", "alpha")
                }
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device, enabled=config.amp and device == "cuda"):
                    out = model(batch["smoky"])
                loss, terms = loss_terms({k: v.float() for k, v in out.items()}, batch, config)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                for key, value in {"loss": float(loss.detach()), **terms}.items():
                    sums[key] = sums.get(key, 0.0) + value
                seen += 1
            scheduler.step()
            report = evaluate(model, eval_ds, device, batch_size=config.batch_size)
            entry = {
                "epoch": epoch,
                **{k: v / max(seen, 1) for k, v in sums.items()},
                "seconds": round(time.time() - started, 1),
                "psnr": report["all"]["psnr"]["mean"],
                "psnr_input": report["all"]["psnr_input"]["mean"],
                "ssim": report["all"]["ssim"]["mean"],
            }
            history.append(entry)
            print(
                f"[osv.train] epoch {epoch}: loss {entry.get('loss', float('nan')):.4f} | "
                f"{config.eval_split} PSNR {entry['psnr']:.2f} dB (input {entry['psnr_input']:.2f}) "
                f"SSIM {entry['ssim']:.4f} | {entry['seconds']}s"
            )
            preview(model, epoch)
            save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                config=config,
            )
            recorder.log_metrics(
                {k: v for k, v in entry.items() if k != "epoch"}
                | _flatten(report, f"{config.eval_split}_"),
                step=epoch,
            )
        real = None
        if config.eval_split == "test":
            real = evaluate_real_smoke(model, processed, "test", device)
        if real is not None:
            recorder.log_metrics(
                {f"real_{k}": v["mean"] for k, v in real.items()}, step=config.epochs
            )

    final = {
        "disclaimer": DISCLAIMER,
        "config": asdict(config),
        "config_hash": config.hash(),
        "device": device,
        "videos": videos,
        "history": history,
        "final_report": report,
        "real_smoke": real,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"report_{config.hash()}.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    return final


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--epochs", type=int, default=DesmokeConfig.epochs)
    p.add_argument("--batch-size", type=int, default=DesmokeConfig.batch_size)
    p.add_argument("--lr", type=float, default=DesmokeConfig.lr)
    p.add_argument("--seed", type=int, default=DesmokeConfig.seed)
    p.add_argument("--crop", type=int, default=DesmokeConfig.crop, help="0 trains on full frames")
    p.add_argument(
        "--architecture",
        default=DesmokeConfig.architecture,
        choices=[name for name, a in REGISTRY.items() if a.task == TASK],
    )
    p.add_argument("--base-channels", type=int, default=None, help="default 32 (U-Net), 64 (GANs)")
    p.add_argument("--vgg-weights", default=None, help="srgan: local VGG19 weights file")
    p.add_argument(
        "--pretrain-epochs", type=int, default=None, help="srgan: SRResNet MSE epochs, default 5"
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data", default=str(DEFAULT_PROCESSED_DIR))
    p.add_argument("--out", default="runs/m5_desmoke")
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--eval-limit", type=int, default=None)
    p.add_argument("--eval-split", default="val", choices=["val", "test"])
    p.add_argument("--num-workers", type=int, default=DesmokeConfig.num_workers)
    p.add_argument(
        "--preview-count",
        type=int,
        default=DesmokeConfig.preview_count,
        help="frames per preview sheet (train, val, real); 0 turns previews off",
    )
    p.add_argument("--preview-every", type=int, default=DesmokeConfig.preview_every)
    p.add_argument(
        "--preview-scale", type=float, default=DesmokeConfig.preview_scale, help="tile size"
    )
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-quota-check", action="store_true", help="CPU runs only (R-24).")
    p.add_argument("--no-mlflow", action="store_true", help="CPU runs only (R-24).")
    p.add_argument("--experiment", default=None, help="default per architecture family")
    args = p.parse_args(argv)
    if args.eval_split == "test":
        print(
            "[osv.train] WARNING: evaluating on the held-out test split, including real smoky "
            "frames. Under R-25 every touch belongs in the counter in docs/BENCHMARK.md."
        )
    values: dict[str, Any] = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "architecture": args.architecture,
        "crop": args.crop or None,
        "train_limit": args.train_limit,
        "eval_limit": args.eval_limit,
        "eval_split": args.eval_split,
        "num_workers": args.num_workers,
        "processed_dir": args.data,
        "preview_count": args.preview_count,
        "preview_every": args.preview_every,
        "preview_scale": args.preview_scale,
    }
    if args.base_channels is not None:
        values["base_channels"] = args.base_channels
    if args.experiment is not None:
        values["experiment"] = args.experiment
    run = train
    if lookup(args.architecture, task=TASK).training == "supervised":
        config: DesmokeConfig = DesmokeConfig(**values, amp=not args.no_amp)
    else:
        from osv.train import desmoke_gan

        config = desmoke_gan.GanConfig(
            **values, vgg_weights=args.vgg_weights, pretrain_epochs=args.pretrain_epochs
        )
        run = desmoke_gan.train  # type: ignore[assignment]
    run(
        config,
        device=args.device,
        out_dir=Path(args.out),
        resume=args.resume,
        skip_quota=args.skip_quota_check,
        use_mlflow=not args.no_mlflow,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DesmokeConfig",
    "evaluate",
    "evaluate_real_smoke",
    "loss_terms",
    "split_videos",
    "train",
]
