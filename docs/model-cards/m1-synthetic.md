# Model Card: M1 instrument segmentation, synthetic against real labels (RFC 0001)

**RESEARCH USE ONLY — NOT FOR CLINICAL USE**

Status: **not released. H1 rejected (R-25: three seeds per arm, decided on the held-out test).**
- RFC 0001 tested the strong hypothesis **H1**: a model trained only on synthetic instruments is not worse than the same model trained on real labels, on real instruments. It is **rejected**: Δ = IoU(A) − IoU(B) = **−65.5 points**, 95 % interval [−75.9, −53.5], against a margin of −2.0.
- Fine-tuning the synthetic model on 10 % of the real frames reaches the real-label model. The design has no control arm, so this does not show what the synthetic pre-training contributed (section 5).
- No checkpoint is published to Hugging Face.

The session ran on 2026-10-24 to 2026-10-26 on one Runpod A40, EU-SE-1, under osv-infra ADR 0009, decision 1a, and runbook `train-on-runpod.md`. The runs are in the project MLflow server under the experiment `osv-m1-synthetic`. Every imported run carries its pod run id in `osv.import.source_run_id`. The run ids below are those on the project server.

## 1. Clinical Disclaimer

`RESEARCH USE ONLY — NOT FOR CLINICAL USE`. Instrument masks are an engineering target and make no clinical claim. These models were trained and scored on public, de-identified cholecystectomy video, and no output is meant to be shown to a surgeon. A segmenter that reaches 0.77 binary IoU on seven test videos is not evidence of safety or utility in any operation.

## 2. Model Description & Architecture

| Model | Registry row (`osv/arch/registry.py`) | Parameters | Pretrained weights |
|---|---|---:|---|
| DeepLabV3-ResNet50, 3 classes | `deeplabv3_resnet50`, supervised | 42.0 M (39.6 M without the auxiliary head) | ImageNet ResNet-50 backbone, torchvision `resnet50-0676ba61.pth`, local file (R-07) |

- **Classes.** Background (0), grasper (1) and L-hook (2). The binary instrument mask is `class > 0`.
- **Settings shared by every arm** (`osv/train/m1.py`):
  - Input is the full 854×480 frame.
  - Loss is unweighted cross-entropy plus 0.4 × the auxiliary head.
  - AdamW, lr 1e-4, weight decay 1e-4, cosine decay over 20 epochs, batch 8, mixed precision.
  - Every epoch has as many samples as real-train has frames (4 000), so all arms take the same number of optimiser steps.
- **Selection.** The checkpoint with the best val IoU of the binary mask, computed over all val pixels.
  - Arm A selects on **synthetic** val.
  - Arms B and C select on real val.
  - Arm A's real-val score was logged for information only (`info_real_val_*`).

| Arm | Training data | Starts from |
|---|---|---|
| **A** | synthetic composites only | ImageNet backbone |
| **B** | CholecSeg8k real-train, 100 % | ImageNet backbone |
| **C10 / C25 / C50 / C100** | 10 / 25 / 50 / 100 % of real-train frames, sampled within every video, fixed per seed | arm A's selected checkpoint of the same seed |

## 3. Training Data & Licensing Inheritance

**Synthetic (arms A and the start of C).** Built by `osv.datasets.m1_synth` (RFC 0001, section 1).

- **Backgrounds.** 717 train and 89 val tool-free CholecT50 frames, from the CholecT50 train and val videos.
  - Every background was reviewed by a person. Of 3 427 candidates, 70 % of those reviewed were rejected.
  - 717 is below the RFC's floor of about 1 000. The owner accepted this before any run.
  - 31 backgrounds, from VID78 and VID79, were recorded at 1920×1080 and resized to 854×480 (same 16:9 aspect).
- **Instruments.** 20 000 renders of `scripts/blender/instruments.py`: Blender 4.2 Cycles on the GPU, 64 samples per pixel, OIDN.
  - Every 10th render goes to val: 18 000 train, 2 000 val.
  - The geometry is the project's own, built from primitives.
- **Composites.** Train composites are drawn afresh every epoch. Val uses 500 fixed pairs.
  - Augmentations: flips, light matched to the frame, motion blur and defocus, tissue over a tip, the scope's vignette, sensor noise and JPEG.
  - M5 smoke layers are added on 20 % of the composites (99 of the 500 val pairs).
  - Each mask is exact by construction.

**Real (arms B and C, the real val and the test): CholecSeg8k** (`docs/data-cards/cholecseg8k.md`).
- **Classes.** The approved class mapping (R-14) is used. Grasper (5) becomes 1, L-hook electrocautery (9) becomes 2, every other class becomes 0, and the watershed boundary (255) stays ignored.
- **Real-train.** Videos 01, 12, 18, 25, 27, 35, 43 and 48: 4 000 frames.
- **Real val.** Videos 26 and 52: 1 120 frames.
- **Test.** The seven CholecSeg8k videos absent from CholecT50, so never a synthetic background: 09, 17, 20, 24, 28, 37 and 55. That is 2 960 frames in 37 clips.
  - Before the first test touch the owner reviewed every test clip for instruments that have no class of their own. Nothing was excluded (`osv/datasets/manifests/m1_test_review.yaml`).
- **Splits.** All splits are by whole video (R-08).

**Licence.** The most restrictive licence is **CC BY-NC-SA 4.0**. Both CholecT50 (Nwoye et al. 2022) and CholecSeg8k (Hong et al. 2020) carry it, so every checkpoint of every arm inherits it: non-commercial use only, share-alike, and attribution to both. The renders and the instrument geometry are the project's own work.

## 4. Quantitative Evaluation

Every selected checkpoint touched the test videos **exactly once**: 18 touches in total, all on 2026-10-26, recorded in `test_touches.jsonl` with the review manifest's hash (R-25). The primary metric is the IoU of the binary instrument mask over all pixels of all test frames. The 95 % intervals come from a bootstrap over test videos (paired between arms) and seeds, with 10 000 repetitions (`osv.eval.m1 decide`).

| Arm | Real labels | Binary IoU (test) | 95 % interval | Boundary F1 (3 px), seeds 1 / 2 / 3 | Mean IoU grasper & L-hook, seeds 1 / 2 / 3 |
|---|---:|---:|---|---|---|
| **A** | 0 % | **0.116** | 0.041 – 0.185 | 0.153 / 0.121 / 0.154 | 0.054 / 0.090 / 0.105 |
| C10 | 10 % | 0.775 | 0.658 – 0.832 | 0.699 / 0.726 / 0.706 | 0.653 / 0.680 / 0.640 |
| C25 | 25 % | 0.767 | 0.633 – 0.833 | 0.721 / 0.716 / 0.717 | 0.658 / 0.626 / 0.665 |
| C50 | 50 % | 0.778 | 0.648 – 0.841 | 0.728 / 0.733 / 0.718 | 0.663 / 0.698 / 0.674 |
| C100 | 100 % | 0.764 | 0.624 – 0.837 | 0.726 / 0.721 / 0.733 | 0.653 / 0.656 / 0.675 |
| **B** | 100 %, no synthetic | **0.772** | 0.654 – 0.837 | 0.731 / 0.675 / 0.724 | 0.718 / 0.638 / 0.638 |

- The intervals for A and B are from `h1.json`, and those for C from `m1_all.json` (see below).
- Arm A by seed, binary IoU: 0.114, 0.082 and 0.152.
- **The decision** (`h1.json`, arms A and B only): Δ = −65.5 points, 95 % interval [−75.9, −53.5]. The upper bound lies below −δ = −2.0, so the verdict is **REJECTED**.
- The same computation over all arms (`m1_all.json`) gives the same Δ with slightly different bounds, [−75.6, −53.3]. The bootstrap there draws from one random stream across arms, and adding arms shifts it. The RFC's decision rests on `h1.json`.
- Per-video counts and per-class IoU and Dice are in each run's test report `m1-<arm>-s<seed>-<config_hash>.json`, attached to the run in MLflow.

Inference latency (p50/p95 on an RTX 4090 or Jetson AGX Orin) has **not been measured**. For reference, one training epoch on the A40 took about 230 s for arm A and about 217 s for arm B, at 100 % GPU use.

### Runs on the project MLflow server

| Arm | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|
| A | `85cda493` | `4d1ea996` | `3916a837` |
| B | `8de89840` | `0b9e3b38` | `6dc1bd90` |
| C10 | `34c78cbf` | `4e1496ba` | `9a696f8b` |
| C25 | `92826e10` | `56b14103` | `2f64b71f` |
| C50 | `4d6aaac4` | `25395dac` | `9697a86f` |
| C100 | `83dad2b4` | `a71dcf3b` | `0a9e7210` |

The arm A runs each also have an earlier, FAILED part (section 7). The selected `best.pt`, `report.json` and the test report are attached to the run in the table, and so is the real-train frame list for arm C. The B runs finished before osv #56 and have no frame list file. Their training set is every real-train frame.

## 5. Robustness / Edge-Case Breakdown

### The domain gap

Arm A learned the synthetic task and did not carry it to real frames.

| Arm A, last epoch | Seed 1 | Seed 2 | Seed 3 |
|---|---:|---:|---:|
| synthetic val binary IoU (selection) | 0.963 | 0.963 | 0.963 |
| real val binary IoU (information only) | 0.447 | 0.476 | 0.403 |
| test binary IoU | 0.114 | 0.082 | 0.152 |

- **Real val did not track synthetic val.** Over 20 epochs arm A's real-val IoU moved between 0.21 and 0.56 with no early peak and no trend, while synthetic val rose steadily from 0.91 to 0.96. Selecting on real val would have broken the design, and it would still have stayed far below arm B.
- **Arm A loses much more from real val to test than arm B.**
  - Arm A falls from about 0.45 on real val to 0.12 on test.
  - Arm B falls from about 0.86–0.89 at its last epochs to 0.77.
  - One candidate cause was background memorisation: real-val videos 26 and 52 are also CholecT50 train videos, so their tissue could appear among the synthetic backgrounds. **It does not hold:** only 29 of the 717 backgrounds (VID26: 17, VID52: 12) come from those videos.
  - What makes the test videos so much harder for arm A is **not known**.
- **Candidate explanations not yet tested:**
  - the look of the rendered instruments: metal, specular highlights, blood and dirt, wear;
  - the 717 backgrounds;
  - shapes or poses of instruments that occur in real surgery and are missing from the renders.

### Synthetic pre-training with a little real data

The C arms reach arm B from 10 % of the real frames (about 400 frames) and stay flat up to 100 %. **This does not show that synthetic pre-training saves 90 % of the labels.** The design has no arm trained on 10 % of the real frames from the ImageNet backbone alone. Without that control, ImageNet plus 400 real frames may be enough by itself. A control arm (B10 / B25) is the cheapest experiment to separate the two.

### Smoke, glare, blood, defocus

These are **not evaluated** separately. The synthetic composites carry M5 smoke, blur and noise as augmentation, but the test has no per-condition breakdown. The `osv.robustness` perturbations have not been run on these models.

## 6. Intended & Prohibited Uses

### Intended

- The record of the RFC 0001 experiment and of its negative result.
- A baseline for later M1 work: a control arm, domain randomisation, better renders, other architectures through R-25.
- Arms B and C as reference instrument segmenters for research on CholecSeg8k-like video.

### Prohibited

- Any clinical or intra-operative use.
- Any claim that synthetic instruments replace real labels. This experiment shows the opposite for this renderer and these backgrounds.
- Any claim that synthetic pre-training reduces labelling cost. There is no control arm (section 5).
- Commercial use, which CC BY-NC-SA 4.0 does not allow.
- Training or evaluation on platform clinical data without a separate decision. ADR 0009 admits only CholecT50- and CholecSeg8k-derived data on Runpod.

## 7. Training History, Deviations and Negative Results

1. **717 backgrounds instead of about 1 000.** The owner decided this before any run, and RFC 0001 records it.
2. **Steps per epoch.** The RFC had planned 10 000 composites per epoch. That would have given arm A about 2.4 times arm B's optimiser steps, so every arm uses 4 000 samples per epoch instead. This was fixed in the RFC before any run.
3. **Render speed** (osv #46). The first renders ran the OIDN denoiser on the CPU: 5.2 s per sample instead of 1.8 s. The roughly 300 samples made that way were deleted, and the full 20 000 were rendered with OIDN on the GPU.
4. **`m1_synth` build** (osv #50). It stopped at the 16th val pair because of the 1920×1080 backgrounds. Backgrounds are now resized to the render size, and the resized ones are listed in `index.json`.
5. **Arm A crashed** after 0–3 epochs in every seed. DataLoader workers hit smoke-layer PNGs that had been truncated by an interrupted copy to the pod.
   - The layers were re-synced with rsync, and all decoded afterwards.
   - The runs resumed with `--resume` under the same config hash: seed 1 from epoch 3, seed 3 from epoch 1, and seed 2 from the start.
   - The crashed parts are in MLflow with status FAILED.
   - Arm A now decodes every smoke layer before training (osv #58).
6. **Invalid C runs.** Because arm A had crashed, the first C queue started early. It fine-tuned the `best.pt` of the unfinished A runs: C10 / C25 / C50 / C100 for seed 1, and C10 and C25 for seed 3.
   - These six runs are in MLflow with the tag `osv.m1.invalid` and never touched the test.
   - Arm C now accepts only a finished A run (osv #58).
   - All 12 C runs in section 4 started from the final A checkpoints. Their `init_sha256` was checked against A's `best.pt`.
7. **No control arm.** The RFC compared C against B only (section 5).

Smoke runs with `--train-limit 64` (one per arm) are in MLflow too, because they spent GPU time (R-24).

## 8. Reproduction

```bash
python scripts/blender/instruments.py --out <renders> --count 20000 --device gpu   # bpy 4.2, Python 3.11
python -m osv.datasets.m1_synth --backgrounds <m1_backgrounds.json> --renders <renders> \
    --out data/processed/m1_synth --smoke-dir data/processed/smoke_m5
python -m osv.train.m1 --arm A --seed 1 --backbone-weights resnet50-0676ba61.pth
python -m osv.train.m1 --arm B --seed 1 --backbone-weights resnet50-0676ba61.pth
python -m osv.train.m1 --arm C --real-fraction 0.1 --seed 1 --init runs/m1/m1-A-s1-<hash>/best.pt
python -m osv.eval.m1 test --checkpoint runs/m1/<run>/best.pt --ledger runs/m1/test_touches.jsonl
python -m osv.eval.m1 decide runs/m1/test/m1-A-*.json runs/m1/test/m1-B-*.json
```

The runbook `train-on-runpod.md` (osv-infra), section "M1", has the pod commands that were used. The commit each step used is in the session's `OSV_COMMIT` file, kept with the downloaded runs.

## 9. Citation

- Nwoye et al., Rendezvous: Attention mechanisms for the recognition of surgical action triplets in endoscopic videos, Medical Image Analysis 78 (2022) 102433.
- Hong et al., CholecSeg8k: A Semantic Segmentation Dataset for Laparoscopic Cholecystectomy Based on Cholec80, arXiv:2012.12453 (2020).
- Twinanda et al., EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos, IEEE TMI 36 (2017).
- Chen et al., Rethinking Atrous Convolution for Semantic Image Segmentation, arXiv:1706.05587 (2017).
