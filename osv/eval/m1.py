"""M1 test evaluation and the decision rule of RFC 0001.

RESEARCH USE ONLY - NOT FOR CLINICAL USE.

Two commands, both run by a person after training:

* ``test`` scores one **selected** checkpoint (``best.pt`` of `osv.train.m1`) on the
  held-out CholecSeg8k test videos - the ones absent from CholecT50 - and writes a
  report with per-video counts. Every call appends to a touch ledger, and a checkpoint
  already in the ledger is refused: each arm's final checkpoint touches the test videos
  exactly once (R-25). The ledger's line count is the counter to report.
* ``decide`` reads the test reports of arms A and B (at least 3 seeds each, R-25) and
  applies the RFC's rule to Δ = IoU(A) - IoU(B), the binary instrument IoU in points:
  a 95 % percentile interval by bootstrap over test videos (paired between the arms)
  and seeds; **supported** if the lower bound is above -δ, **rejected** if the upper
  bound is below -δ, otherwise **inconclusive** (R-25 INCOMPLETE). δ = 2.0 is fixed by
  the RFC before any run and is a constant here, not a flag. Arms C are reported as a
  curve next to B; H1 is decided on A against B alone.

Metrics per video (summed over frames, so dataset-level numbers are exact sums):
binary instrument IoU, the 3x3 confusion matrix (per-class IoU and Dice for grasper and
L-hook), and boundary F1 of the binary mask with a 3-pixel tolerance.

    python -m osv.eval.m1 test --checkpoint runs/m1/<run>/best.pt --ledger runs/m1/test_touches.jsonl
    python -m osv.eval.m1 decide runs/m1/test/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from osv.datasets import m1 as D
from osv.eval import metrics as M

DISCLAIMER = "RESEARCH USE ONLY - NOT FOR CLINICAL USE"
DELTA_POINTS = 2.0  # RFC 0001 section 3, confirmed by the owner 2026-10-22; never a flag
MIN_SEEDS = 3
BOUNDARY_TOLERANCE_PX = 3


def boundary(mask: np.ndarray) -> np.ndarray:
    """One-pixel inner boundary of a boolean mask."""
    eroded = cv2.erode(
        mask.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE
    )
    return mask & ~eroded.astype(bool)


def boundary_counts(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """[pred boundary px matched, pred boundary px, target boundary px matched, target px]."""
    disk = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * BOUNDARY_TOLERANCE_PX + 1, 2 * BOUNDARY_TOLERANCE_PX + 1)
    )
    pred_b, target_b = boundary(pred) & valid, boundary(target) & valid
    near_target = cv2.dilate(target_b.astype(np.uint8), disk).astype(bool)
    near_pred = cv2.dilate(pred_b.astype(np.uint8), disk).astype(bool)
    return np.array(
        [(pred_b & near_target).sum(), pred_b.sum(), (target_b & near_pred).sum(), target_b.sum()],
        dtype=np.int64,
    )


def binary_counts(cm: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(cm[1:, 1:].sum()),
        "fp": int(cm[0, 1:].sum()),
        "fn": int(cm[1:, 0].sum()),
    }


def iou(counts: dict[str, int]) -> float:
    denominator = counts["tp"] + counts["fp"] + counts["fn"]
    return counts["tp"] / denominator if denominator else float("nan")


def boundary_f1(counts: np.ndarray) -> float:
    precision = counts[0] / counts[1] if counts[1] else float("nan")
    recall = counts[2] / counts[3] if counts[3] else float("nan")
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def summarise(per_video: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cm = np.sum([v["confusion"] for v in per_video.values()], axis=0)
    edges = np.sum([v["boundary"] for v in per_video.values()], axis=0)
    classes = M.summarize(cm, D.CLASSES)
    instruments = [classes["per_class"][c]["iou"] for c in D.CLASSES[1:]]
    return {
        "binary_iou": iou(binary_counts(cm)),
        "boundary_f1": boundary_f1(edges),
        "instrument_mean_iou": float(np.mean([v for v in instruments if v is not None])),
        "per_class": classes["per_class"],
    }


def touched(ledger: Path) -> list[dict[str, Any]]:
    if not ledger.is_file():
        return []
    return [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]


def test(
    checkpoint: Path,
    *,
    ledger: Path,
    out: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    processed_dir: str | None,
    source: str | None,
    limit: int | None = None,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from osv.train import m1 as T

    digest = T.sha256_file(checkpoint)
    previous = [t for t in touched(ledger) if t["checkpoint_sha256"] == digest]
    if previous:
        raise SystemExit(
            f"[osv.eval.m1] {checkpoint} already touched the test videos on "
            f"{previous[0]['time']}: each selected checkpoint is scored once (R-25)"
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not state.get("selected_on"):
        raise SystemExit(f"[osv.eval.m1] {checkpoint} is not a selected checkpoint (best.pt)")
    config = state["config"]
    processed = processed_dir or config["processed_dir"]
    videos = T.real_split(processed)["test"]
    data = D.RealInstruments(
        videos, processed_dir=processed, source=source or config["source"] or None, limit=limit
    )

    # The touch is counted before the first test frame is read, so a crash still counts.
    ledger.parent.mkdir(parents=True, exist_ok=True)
    touch = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "arm": config["arm"],
        "real_fraction": config["real_fraction"],
        "seed": config["seed"],
        "config_hash": state["config_hash"],
        "test_videos": videos,
    }
    with ledger.open("a") as handle:
        handle.write(json.dumps(touch) + "\n")
    count = len(touched(ledger))
    print(f"[osv.eval.m1] {DISCLAIMER}\n[osv.eval.m1] test touch {count} in {ledger}: {touch}")

    model = T.build_model(architecture=config["architecture"], backbone_weights=None)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    per_video: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "frames": 0,
            "confusion": np.zeros((3, 3), np.int64),
            "boundary": np.zeros(4, np.int64),
        }
    )
    with torch.no_grad():
        for batch in loader:
            with torch.amp.autocast(device, enabled=device == "cuda"):
                logits = model(batch["image"].to(device))["out"]
            preds = logits.argmax(1).cpu().numpy()
            for pred, label, video in zip(
                preds, batch["label"].numpy(), batch["video_id"], strict=True
            ):
                entry = per_video[video]
                entry["frames"] += 1
                entry["confusion"] += M.confusion_matrix(pred, label, 3)
                valid = label != M.IGNORE_INDEX
                entry["boundary"] += boundary_counts(pred > 0, (label > 0) & valid, valid)
    videos_out = {
        v: {
            "frames": e["frames"],
            "confusion": e["confusion"].tolist(),
            "boundary": e["boundary"].tolist(),
            **binary_counts(e["confusion"]),
            "binary_iou": iou(binary_counts(e["confusion"])),
        }
        for v, e in sorted(per_video.items())
    }
    report = {
        "disclaimer": DISCLAIMER,
        "touch": touch,
        "touch_count": count,
        "config": config,
        "selected": {"epoch": state["epoch"], "on": state["selected_on"]},
        "test": summarise(videos_out),
        "per_video": videos_out,
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{Path(checkpoint).parent.name}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"[osv.eval.m1] binary IoU {report['test']['binary_iou']:.4f}, "
        f"boundary F1 {report['test']['boundary_f1']:.4f} -> {path}"
    )
    return report


def arm_label(config: dict[str, Any]) -> str:
    return f"C{round(config['real_fraction'] * 100)}" if config["arm"] == "C" else config["arm"]


def counts_table(reports: list[dict[str, Any]], videos: list[str]) -> np.ndarray:
    """seeds x videos x (tp, fp, fn) for one arm."""
    return np.array(
        [[[r["per_video"][v][k] for k in ("tp", "fp", "fn")] for v in videos] for r in reports],
        dtype=np.float64,
    )


def pooled_iou(table: np.ndarray, seeds: np.ndarray, videos: np.ndarray) -> float:
    """Mean over the drawn seeds of the IoU over all pixels of the drawn videos."""
    sums = table[seeds][:, videos].sum(axis=1)
    return float(np.mean(sums[:, 0] / sums.sum(axis=1)))


def decide(reports: list[dict[str, Any]], *, reps: int = 10_000, seed: int = 0) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        by_arm[arm_label(report["config"])].append(report)
    for arm in ("A", "B"):
        seeds = sorted(r["config"]["seed"] for r in by_arm.get(arm, []))
        if len(set(seeds)) < MIN_SEEDS or len(seeds) != len(set(seeds)):
            raise SystemExit(
                f"[osv.eval.m1] arm {arm} needs {MIN_SEEDS} distinct seeds, one report each "
                f"(R-25); got {seeds}"
            )
    videos = sorted(by_arm["A"][0]["per_video"])
    if any(sorted(r["per_video"]) != videos for rs in by_arm.values() for r in rs):
        raise SystemExit("[osv.eval.m1] reports cover different test videos")

    rng = np.random.default_rng(seed)
    tables = {arm: counts_table(rs, videos) for arm, rs in by_arm.items()}
    all_videos = np.arange(len(videos))
    draws: dict[str, list[float]] = {arm: [] for arm in tables}
    deltas = []
    for _ in range(reps):
        drawn = rng.integers(0, len(videos), len(videos))  # the same videos for every arm
        for arm, table in tables.items():
            draws[arm].append(pooled_iou(table, rng.integers(0, len(table), len(table)), drawn))
        deltas.append(100 * (draws["A"][-1] - draws["B"][-1]))
    low, high = np.percentile(deltas, [2.5, 97.5])
    point = {arm: pooled_iou(t, np.arange(len(t)), all_videos) for arm, t in tables.items()}
    if low > -DELTA_POINTS:
        verdict = "SUPPORTED"
    elif high < -DELTA_POINTS:
        verdict = "REJECTED"
    else:
        verdict = "INCONCLUSIVE (R-25 INCOMPLETE)"
    return {
        "disclaimer": DISCLAIMER,
        "hypothesis": "H1: arm A (synthetic only) is not worse than arm B (real) on real test videos",
        "delta_points": DELTA_POINTS,
        "test_videos": videos,
        "bootstrap": {"reps": reps, "seed": seed, "over": "test videos (paired) and seeds"},
        "delta_iou_points": {"estimate": 100 * (point["A"] - point["B"]), "ci95": [low, high]},
        "verdict": verdict,
        "arms": {
            arm: {
                "seeds": sorted(r["config"]["seed"] for r in by_arm[arm]),
                "binary_iou": point[arm],
                "ci95": list(np.percentile(draws[arm], [2.5, 97.5])),
                "boundary_f1": [r["test"]["boundary_f1"] for r in by_arm[arm]],
                "instrument_mean_iou": [r["test"]["instrument_mean_iou"] for r in by_arm[arm]],
            }
            for arm in sorted(tables)
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    t = commands.add_parser("test", help="score one selected checkpoint on the test videos, once")
    t.add_argument("--checkpoint", type=Path, required=True)
    t.add_argument("--ledger", type=Path, required=True, help="the R-25 touch ledger (JSONL)")
    t.add_argument("--out", type=Path, default=Path("runs/m1/test"))
    t.add_argument("--device", default="cuda")
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--num-workers", type=int, default=8)
    t.add_argument("--processed-dir")
    t.add_argument("--source", help="CholecSeg8k zip (default: the training run's)")
    d = commands.add_parser("decide", help="the RFC 0001 decision rule over test reports")
    d.add_argument("reports", type=Path, nargs="+")
    d.add_argument("--reps", type=int, default=10_000)
    d.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "test":
        test(
            args.checkpoint,
            ledger=args.ledger,
            out=args.out,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            processed_dir=args.processed_dir,
            source=args.source,
        )
        return 0
    result = decide([json.loads(p.read_text()) for p in args.reports], reps=args.reps)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DELTA_POINTS", "boundary_counts", "decide", "test"]
