"""M5 GAN baselines: Pix2Pix, SRGAN (scale 1) and CycleGAN desmoking.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

    python -m osv.train.desmoke --architecture pix2pix --batch-size 1 --crop 256
    python -m osv.train.desmoke --architecture srgan --vgg-weights /models/vgg19.safetensors
    python -m osv.train.desmoke --architecture cyclegan --batch-size 1 --crop 256

Started from `osv.train.desmoke`, which looks the architecture up in
`osv/arch/registry.py` (R-25) and hands GANs here. The run discipline is the
supervised baseline's: R-24 quota gate, `config_hash` resume refusal, pinned
seeds, MLflow, and the same evaluation - the smoky-to-clean generator scored on
the fixed synthetic pairs next to the no-op baseline, plus the real-smoke haze
drop on `--eval-split test` once real frames are labelled as smoke.

* Pix2Pix and SRGAN train on the synthetic pairs (new smoke every epoch).
  Pix2Pix: conditional PatchGAN, vanilla GAN loss + 100 * L1. SRGAN: first
  `pretrain_epochs` (default 5) of SRResNet on pixel MSE alone, the paper's
  initialisation of the GAN, then VGG19 relu5_4 content loss + 1e-3 *
  adversarial loss; the VGG weights are a local file (`--vgg-weights`), never a
  download (R-07). relu5_4 barely sees colour and brightness: the first
  full-frame run, without the MSE stage, drifted to a solid red frame.
* CycleGAN trains on unpaired real smoky and clean train frames - the only one
  of the three that sees real smoke in training. LSGAN, cycle 10, identity 5,
  a 50-image history per discriminator. Its smoky domain is the real frames a
  person labelled as smoke (`osv.datasets.desmoke.confirmed_smoke`); until there
  are some it refuses to start.

All three use Adam (2e-4, betas 0.5 / 0.999), constant for the first half of the
epochs and linearly decayed to zero over the second, as pix2pix and CycleGAN do;
for SRGAN the schedule spans both stages.
Steps run in float32. A GAN can paint texture the smoke hid; the PSNR / SSIM
against the clean frame, not the look of the output, is the result.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from osv.arch.registry import lookup
from osv.datasets.desmoke import SyntheticSmokePairs, UnpairedSmoke
from osv.models.desmoke import TASK
from osv.models.desmoke_gan import VGG19_RELU5_4, ImagePool, VGGPerceptual, build_gan
from osv.train.desmoke import (
    DISCLAIMER,
    DesmokeConfig,
    _flatten,
    evaluate,
    evaluate_real_smoke,
    random_crop,
    split_videos,
)
from osv.train.segmentation import RunRecorder, pin_seed, preflight_quota

SRGAN_PRETRAIN_EPOCHS = 5


@dataclass
class GanConfig(DesmokeConfig):
    architecture: str = "pix2pix"
    base_channels: int = 64  # ngf / ndf of the papers
    amp: bool = False  # unused: GAN steps run in float32
    beta1: float = 0.5
    l1_weight: float = 100.0  # pix2pix
    adversarial_weight: float = 1e-3  # srgan
    cycle_weight: float = 10.0  # cyclegan
    identity_weight: float = 5.0  # cyclegan: 0.5 * cycle_weight
    pool_size: int = 50  # cyclegan
    vgg_weights: str | None = None  # srgan; the file's sha256 is what enters the hash
    vgg_sha256: str = ""
    vgg_layer: int = VGG19_RELU5_4
    # srgan: SRResNet epochs on pixel MSE before the GAN. A fixed number, not a share of
    # `epochs`, so that a longer resumed run keeps its config_hash.
    pretrain_epochs: int | None = None
    experiment: str = "osv-m5-desmoke-gan"

    HASH_SKIP = DesmokeConfig.HASH_SKIP | {"vgg_weights"}

    def __post_init__(self) -> None:
        if self.pretrain_epochs is None:
            self.pretrain_epochs = SRGAN_PRETRAIN_EPOCHS if self.architecture == "srgan" else 0
        if self.pretrain_epochs and self.architecture != "srgan":
            raise ValueError(
                f"pretrain_epochs is SRGAN's SRResNet stage, not {self.architecture}'s"
            )


def _signed(image: torch.Tensor) -> torch.Tensor:
    return image * 2 - 1  # discriminators see the generators' [-1, 1] range


def _adversarial(logits: torch.Tensor, real: bool, *, lsgan: bool) -> torch.Tensor:
    target = torch.full_like(logits, 1.0 if real else 0.0)
    if lsgan:
        return F.mse_loss(logits, target)
    return F.binary_cross_entropy_with_logits(logits, target)


def _requires_grad(nets: list[nn.Module], flag: bool) -> None:
    for net in nets:
        for parameter in net.parameters():
            parameter.requires_grad_(flag)


def _detached(terms: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach()) for name, value in terms.items()}


def pretrain_step(nets, optimizers, batch) -> dict[str, float]:
    """One SRResNet update on pixel MSE; the discriminator waits for the GAN stage."""
    optimizers["generator"].zero_grad(set_to_none=True)
    content = F.mse_loss(nets["generator"](batch["smoky"])["clean"], batch["clean"])
    content.backward()
    optimizers["generator"].step()
    return _detached({"loss": content, "content": content})


def paired_step(nets, optimizers, batch, config: GanConfig, perceptual) -> dict[str, float]:
    """One Pix2Pix or SRGAN update: discriminator, then generator."""
    generator, discriminator = nets["generator"], nets["discriminator"]
    smoky, clean = batch["smoky"], batch["clean"]
    conditional = config.architecture == "pix2pix"

    def judge(image: torch.Tensor) -> torch.Tensor:
        if conditional:
            return discriminator(torch.cat([_signed(smoky), _signed(image)], dim=1))
        return discriminator(_signed(image))

    fake = generator(smoky)["clean"]
    optimizers["discriminator"].zero_grad(set_to_none=True)
    loss_d = 0.5 * (
        _adversarial(judge(clean), True, lsgan=False)
        + _adversarial(judge(fake.detach()), False, lsgan=False)
    )
    loss_d.backward()
    optimizers["discriminator"].step()

    _requires_grad([discriminator], False)
    optimizers["generator"].zero_grad(set_to_none=True)
    adversarial = _adversarial(judge(fake), True, lsgan=False)
    if conditional:
        content = (fake - clean).abs().mean()
        loss_g = adversarial + config.l1_weight * content
    else:
        content = perceptual(fake, clean)
        loss_g = content + config.adversarial_weight * adversarial
    loss_g.backward()
    optimizers["generator"].step()
    _requires_grad([discriminator], True)
    return _detached(
        {"loss": loss_g, "loss_d": loss_d, "adversarial": adversarial, "content": content}
    )


def cycle_step(nets, optimizers, batch, config: GanConfig, pools) -> dict[str, float]:
    """One CycleGAN update: both generators, then both discriminators."""
    to_clean, to_smoky = nets["generator"], nets["generator_back"]
    judge_clean, judge_smoky = nets["discriminator"], nets["discriminator_back"]
    smoky, clean = batch["smoky"], batch["clean"]

    fake_clean, fake_smoky = to_clean(smoky)["clean"], to_smoky(clean)["clean"]
    _requires_grad([judge_clean, judge_smoky], False)
    optimizers["generator"].zero_grad(set_to_none=True)
    adversarial = _adversarial(judge_clean(_signed(fake_clean)), True, lsgan=True) + _adversarial(
        judge_smoky(_signed(fake_smoky)), True, lsgan=True
    )
    cycle = (to_smoky(fake_clean)["clean"] - smoky).abs().mean() + (
        to_clean(fake_smoky)["clean"] - clean
    ).abs().mean()
    identity = (to_clean(clean)["clean"] - clean).abs().mean() + (
        to_smoky(smoky)["clean"] - smoky
    ).abs().mean()
    loss_g = adversarial + config.cycle_weight * cycle + config.identity_weight * identity
    loss_g.backward()
    optimizers["generator"].step()
    _requires_grad([judge_clean, judge_smoky], True)

    optimizers["discriminator"].zero_grad(set_to_none=True)
    loss_d = torch.zeros((), device=smoky.device)
    for judge, real, fake, pool in (
        (judge_clean, clean, fake_clean, pools["clean"]),
        (judge_smoky, smoky, fake_smoky, pools["smoky"]),
    ):
        loss_d = loss_d + 0.5 * (
            _adversarial(judge(_signed(real)), True, lsgan=True)
            + _adversarial(judge(_signed(pool.query(fake))), False, lsgan=True)
        )
    loss_d.backward()
    optimizers["discriminator"].step()
    return _detached(
        {
            "loss": loss_g,
            "loss_d": loss_d,
            "adversarial": adversarial,
            "cycle": cycle,
            "identity": identity,
        }
    )


def linear_decay(epochs: int):
    """pix2pix / CycleGAN schedule: full rate for the first half, then linearly to zero."""
    constant = epochs // 2
    decay = max(epochs - constant, 1)
    return lambda epoch: 1.0 - max(0, epoch + 1 - constant) / (decay + 1)


def build_optimizers(nets: dict[str, nn.Module], config: GanConfig):
    def adam(*names: str) -> torch.optim.Adam:
        parameters = itertools.chain(*(nets[n].parameters() for n in names if n in nets))
        return torch.optim.Adam(parameters, lr=config.lr, betas=(config.beta1, 0.999))

    optimizers = {
        "generator": adam("generator", "generator_back"),
        "discriminator": adam("discriminator", "discriminator_back"),
    }
    schedulers = {
        name: torch.optim.lr_scheduler.LambdaLR(optimizer, linear_decay(config.epochs))
        for name, optimizer in optimizers.items()
    }
    return optimizers, schedulers


def save_checkpoint(path: Path, *, nets, optimizers, schedulers, epoch: int, config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "nets": {name: net.state_dict() for name, net in nets.items()},
            "optimizers": {name: o.state_dict() for name, o in optimizers.items()},
            "schedulers": {name: s.state_dict() for name, s in schedulers.items()},
            "epoch": epoch,
            "config_hash": config.hash(),
            "config": asdict(config),
            "disclaimer": DISCLAIMER,
        },
        path,
    )


def load_checkpoint(path: Path, *, nets, optimizers, schedulers, config) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("config_hash") != config.hash():
        raise SystemExit(
            f"[osv.train] refusing to resume: checkpoint config_hash {state.get('config_hash')} != "
            f"current {config.hash()}. A changed configuration is a new run (R-24)."
        )
    for group, objects in (("nets", nets), ("optimizers", optimizers), ("schedulers", schedulers)):
        for name, obj in objects.items():
            obj.load_state_dict(state[group][name])
    return int(state["epoch"]) + 1


def _datasets(
    config: GanConfig, training: str, processed: Path
) -> tuple[SyntheticSmokePairs | UnpairedSmoke, SyntheticSmokePairs]:
    if training == "unpaired_gan":
        train_ds: SyntheticSmokePairs | UnpairedSmoke = UnpairedSmoke(
            "train", processed_dir=processed, seed=config.seed, limit=config.train_limit
        )
    else:
        train_ds = SyntheticSmokePairs(
            "train", processed_dir=processed, seed=config.seed, limit=config.train_limit
        )
    eval_ds = SyntheticSmokePairs(
        config.eval_split, processed_dir=processed, limit=config.eval_limit
    )
    return train_ds, eval_ds


def _perceptual(config: GanConfig, device: str) -> VGGPerceptual | None:
    if config.architecture != "srgan":
        return None
    if not config.vgg_weights:
        raise SystemExit(
            "[osv.train] srgan needs --vgg-weights: a local VGG19 file (timm/vgg19.tv_in1k "
            "model.safetensors or a torchvision vgg19 state dict). Nothing is downloaded (R-07)."
        )
    loss = VGGPerceptual(config.vgg_weights, layer=config.vgg_layer).to(device)
    config.vgg_sha256 = loss.weights_sha256 or ""
    return loss


def _run_epoch(nets, optimizers, loader, crops, config, device, *, perceptual, pools, pretraining):
    for net in nets.values():
        net.train()
    sums: dict[str, float] = {}
    seen = 0
    for raw in loader:
        batch = random_crop(raw, config.crop, crops)
        batch = {k: batch[k].to(device, non_blocking=True) for k in ("smoky", "clean")}
        if pretraining:
            terms = pretrain_step(nets, optimizers, batch)
        elif pools is None:
            terms = paired_step(nets, optimizers, batch, config, perceptual)
        else:
            terms = cycle_step(nets, optimizers, batch, config, pools)
        for key, value in terms.items():
            sums[key] = sums.get(key, 0.0) + value
        seen += 1
    return {k: v / max(seen, 1) for k, v in sums.items()}


def train(
    config: GanConfig,
    *,
    device: str,
    out_dir: Path,
    resume: bool,
    skip_quota: bool,
    use_mlflow: bool = True,
) -> dict[str, Any]:
    print(f"[osv.train] {DISCLAIMER}")
    entry = lookup(config.architecture, task=TASK)
    if entry.training == "supervised":
        raise ValueError(f"{config.architecture!r} is supervised: use osv.train.desmoke.train")
    preflight_quota(device=device, skip=skip_quota, experiment=config.experiment)
    pin_seed(config.seed)
    processed = Path(config.processed_dir)
    index = json.loads((processed / "index.json").read_text())
    config.data_config_sha256 = index["config_sha256"]
    perceptual = _perceptual(config, device)
    train_ds, eval_ds = _datasets(config, entry.training, processed)
    videos = split_videos(index, config.eval_split)
    print(
        f"[osv.train] {config.architecture} ({entry.training}): train {len(train_ds)} samples / "
        f"{len(videos['train'])} videos; {config.eval_split} {len(eval_ds)} pairs"
    )

    nets = {
        name: net.to(device)
        for name, net in build_gan(config.architecture, width=config.base_channels).items()
    }
    optimizers, schedulers = build_optimizers(nets, config)
    ckpt_path = out_dir / f"ckpt_{config.hash()}.pt"
    start_epoch = 0
    if resume and ckpt_path.is_file():
        start_epoch = load_checkpoint(
            ckpt_path, nets=nets, optimizers=optimizers, schedulers=schedulers, config=config
        )
        print(f"[osv.train] resumed from {ckpt_path} at epoch {start_epoch}")
    pools = None
    if entry.training == "unpaired_gan":
        pools = {
            name: ImagePool(
                config.pool_size, generator=torch.Generator().manual_seed(config.seed + i)
            )
            for i, name in enumerate(("clean", "smoky"))
        }

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
    recorder = RunRecorder(enabled=use_mlflow, device=device, experiment=config.experiment)
    with recorder:
        recorder.log_params(
            {
                **{k: v for k, v in asdict(config).items() if k != "extra"},
                "config_hash": config.hash(),
                "training": entry.training,
                "reference": entry.reference,
                "device": device,
                "gpu_count": torch.cuda.device_count() if device == "cuda" else 0,
                "train_videos": ",".join(videos["train"]),
                f"{config.eval_split}_videos": ",".join(videos[config.eval_split]),
                "train_samples": len(train_ds),
                "eval_pairs": len(eval_ds),
                "disclaimer": DISCLAIMER,
            }
        )
        for epoch in range(start_epoch, config.epochs):
            train_ds.set_epoch(epoch)
            started = time.time()
            means = _run_epoch(
                nets,
                optimizers,
                loader,
                crops,
                config,
                device,
                perceptual=perceptual,
                pools=pools,
                pretraining=epoch < (config.pretrain_epochs or 0),
            )
            for scheduler in schedulers.values():
                scheduler.step()
            report = evaluate(nets["generator"], eval_ds, device, batch_size=config.batch_size)
            entry_row = {
                "epoch": epoch,
                **means,
                "seconds": round(time.time() - started, 1),
                "psnr": report["all"]["psnr"]["mean"],
                "psnr_input": report["all"]["psnr_input"]["mean"],
                "ssim": report["all"]["ssim"]["mean"],
            }
            history.append(entry_row)
            print(
                f"[osv.train] epoch {epoch}: G {entry_row.get('loss', float('nan')):.4f} "
                f"D {entry_row.get('loss_d', float('nan')):.4f} | {config.eval_split} PSNR "
                f"{entry_row['psnr']:.2f} dB (input {entry_row['psnr_input']:.2f}) "
                f"SSIM {entry_row['ssim']:.4f} | {entry_row['seconds']}s"
            )
            save_checkpoint(
                ckpt_path,
                nets=nets,
                optimizers=optimizers,
                schedulers=schedulers,
                epoch=epoch,
                config=config,
            )
            recorder.log_metrics(
                {k: v for k, v in entry_row.items() if k != "epoch"}
                | _flatten(report, f"{config.eval_split}_"),
                step=epoch,
            )
        real = None
        if config.eval_split == "test":
            real = evaluate_real_smoke(nets["generator"], processed, "test", device)
        if real is not None:
            recorder.log_metrics(
                {f"real_{k}": v["mean"] for k, v in real.items()}, step=config.epochs
            )

    final = {
        "disclaimer": DISCLAIMER,
        "config": asdict(config),
        "config_hash": config.hash(),
        "architecture": asdict(entry),
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


__all__ = ["GanConfig", "cycle_step", "linear_decay", "paired_step", "train"]
