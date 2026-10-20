# Model Card: M5 surgical desmoking baselines

**RESEARCH USE ONLY — NOT FOR CLINICAL USE**

Status: **not released, INCOMPLETE (R-25).**
- Single seed (1337), validation split only. R-25 declares an improvement only on three or more finished seeds with non-overlapping confidence intervals.
- No checkpoint is published to Hugging Face. The held-out test split has not been touched, so there is nothing to add to the R-25 test-access counter.

This card records the first training session, which ran on 2026-10-19 and 2026-10-20 on a Runpod A40, EU-SE-1 (osv-infra ADR 0009 and runbook `train-on-runpod.md`). Its runs are in the project MLflow server under the experiments `osv-m5-desmoke-baseline` and `osv-m5-desmoke-gan`. Every imported run carries its pod run id in the tag `osv.import.source_run_id`; the ids below are those pod ids.

## 1. Clinical Disclaimer

`RESEARCH USE ONLY — NOT FOR CLINICAL USE`. These are engineering baselines. They are trained and scored on synthetic smoke, which no clinician has validated against real electrocautery smoke (data card, section 3). Nothing here is evidence of clinical utility. No output of these models may be shown to a surgeon during an operation.

## 2. Model Description & Architecture

| Model | Registry row (`osv/arch/registry.py`) | Parameters | Pretrained weights |
|---|---|---:|---|
| **Residual U-Net** | `residual_unet`, supervised | 7.77 M | none |
| Pix2Pix | `pix2pix`, paired GAN | G 41.83 M, D 2.77 M | none |
| SRGAN (scale 1) | `srgan`, paired GAN | G 1.25 M, D 5.21 M | VGG19 for the loss only |

### Residual U-Net (`osv/models/desmoke.py`)

- Four-level U-Net with base width 32. It predicts a residual on the smoky frame and also the smoke alpha.
- The residual head starts at zero, so the untrained network returns its input unchanged.
- Loss: L1 on the frame + 0.2 × (1 − SSIM) + 0.1 × L1 on the alpha.
- Optimiser: AdamW, lr 2e-4, weight decay 1e-4, mixed precision.

### GANs (`osv/models/desmoke_gan.py`, `osv/train/desmoke_gan.py`)

Pix2Pix, after Isola et al. 2017:
- generator: U-Net with decoder dropout;
- discriminator: conditional 70×70 PatchGAN;
- loss: vanilla GAN loss + 100 × L1.

SRGAN, after Ledig et al. 2017:
- generator: SRResNet with 16 residual blocks at scale 1, with no pixel-shuffle upsampling, predicting a residual on its input;
- discriminator: the paper's VGG-style network;
- the GAN stage starts after 5 epochs of SRResNet trained on pixel MSE alone;
- GAN-stage loss: VGG19 relu5_4 content loss + 1e-3 × adversarial loss;
- the VGG19 weights are a local torchvision file, `vgg19-dcbb9e9d.pth`.

Both GANs use Adam (2e-4, β 0.5 / 0.999) and float32, with the learning rate constant for the first half of training and decayed linearly to zero over the second.

## 3. Training Data & Licensing Inheritance

- Data: the `smoke_m5_build` DVC stage (`docs/data-cards/smoke-m5.md`). Blender smoke is composited over clean CholecT50 frames.
- Training uses 5 882 clean frames from 35 videos, with a new smoke layer drawn each epoch. Validation uses 518 fixed pairs from 5 videos.
- The splits are by video, following the CholecT50 paper.
- Most restrictive licence: **CC BY-NC-SA 4.0** (CholecT50). Any weights inherit it: non-commercial use only, share-alike, and attribution to Nwoye et al. 2022 (ADR 0002).
- The Blender smoke renders on their own are CC BY 4.0.

Training ran on full 854×480 frames (`--crop 0`), for 15 epochs per model. Batch size was 4 for the U-Net and SRGAN and 1 for Pix2Pix.

## 4. Quantitative Evaluation

The task is image restoration, so mIoU and Dice do not apply. Each model is scored by PSNR and SSIM against the clean frame and set next to the **no-op baseline**: the smoky input passed through unchanged. Scores below come from the last epoch of each run on the validation split (518 pairs).

| Model | Pod run | config_hash | Val PSNR, dB | Input PSNR, dB | Δ, dB | Val SSIM |
|---|---|---|---:|---:|---:|---:|
| **Residual U-Net** | `70e682cc` | `1384dc6a648e4e9a` | **33.08** | 28.99 | **+4.09** | 0.979 |
| Pix2Pix | `444794b3` | `a7be56d05daee64c` | 27.85 | 28.99 | −1.14 | 0.923 |
| SRGAN | — | — | did not converge (section 7) | | | |

Per-frame 95 % intervals are in each run's `report_<config_hash>.json`, attached to the run in MLflow.

Inference latency (p50/p95 on an RTX 4090 or Jetson AGX Orin) has **not been measured**. Training throughput on an A40 with no other process on the GPU was about 4.7 min per U-Net epoch, which is not an inference figure.

## 5. Robustness / Edge-Case Breakdown

### Smoke density

The strata are by mean smoke alpha, with edges at 0.02 and 0.06. Each cell is model PSNR / input PSNR in dB, with the difference in brackets.

| Model | light | medium | dense |
|---|---|---|---|
| **Residual U-Net** | 41.05 / 38.51 (**+2.54**) | 34.47 / 29.05 (**+5.42**) | 28.27 / 24.38 (**+3.89**) |
| Pix2Pix | 32.97 / 38.51 (−5.54) | 29.00 / 29.05 (−0.05) | 24.57 / 24.38 (+0.19) |

- The U-Net improves on the input in every stratum.
- Pix2Pix barely changes medium and dense smoke, and it damages lightly smoked frames: it paints plausible texture where the frame was nearly clean.

### Glare, blood splatter, defocus

These are **not evaluated**. The `osv.robustness` perturbations have not been run on these models.

### Real smoke

**Not evaluated.** None of the real smoky CholecT50 frames is labelled as smoke yet (data card, section 5), so the real-smoke haze indicator is skipped.

## 6. Intended & Prohibited Uses

### Intended

- A baseline for research on surgical desmoking.
- A pre-processing candidate for the smoke levels of `osv.robustness`.
- A comparison point for later M5 architectures that come in through R-25.

### Prohibited

- Any clinical or intra-operative use, including showing restored frames to a surgeon.
- Any use as evidence of what a smoky frame hides: Pix2Pix in particular can invent tissue texture.
- Commercial use, which the CC BY-NC-SA 4.0 licence does not allow.
- Training or evaluation on platform clinical data without a separate decision; ADR 0009 admits only CholecT50-derived data on Runpod.

## 7. Training History and Negative Results

1. **U-Net on 256×256 crops** (`ckpt_0cd88dc865cbe9fb.pt`, stopped).
   - After 11 epochs val PSNR was 28.75 dB, still below the input's 28.99 dB, and on light smoke the model lost 1.5 dB to the input.
   - Crops dilute the smoke signal, and they never show the scope's vignette or the frame edges.
   - On full frames the same network was ahead of the input by 2.4 dB at epoch 5.
2. **SRGAN v1** (pod run `4b590b3a`). Val PSNR 6.3 dB: every output pixel saturated to (1, 0, 0).
   - The generator output was clamped during training, so saturated pixels got no gradient.
   - The GAN also started without the paper's SRResNet MSE stage.
   - Fixed in osv #30.
3. **SRGAN v2** (`d4cbd95f`). By mistake it ran the code from before osv #30.
   - Its first epochs reproduce v1 exactly; val PSNR settled near 9.5 dB.
   - It is kept in MLflow as a duplicate of v1.
4. **SRGAN v3** (`e6672d79`, osv #30 code, stopped after epoch 2 of the MSE stage).
   - Val PSNR was 25.64, then 22.71, then 25.36 dB, all below the input, and the training MSE jumped between epochs.
   - On light smoke it scored 28.13 dB against 38.51 for the input.
   - The likely cause is the optimiser: the pix2pix settings (Adam 2e-4, β1 0.5) with BatchNorm at batch size 4 break the identity the network starts from. The paper used Adam 1e-4 with β1 0.9.
   - No fourth attempt was made.

Pod run `07308ccb` is a duplicate SRGAN v2 launch that failed with an out-of-memory error before its first epoch.

## 8. Reproduction

```bash
dvc repro smoke_m5_build
python -m osv.train.desmoke --crop 0 --batch-size 4 --epochs 15
python -m osv.train.desmoke --architecture pix2pix --batch-size 1 --crop 0 --epochs 15
```

The commit each run used is recorded in the session's `OSV_COMMIT` file, kept with the downloaded runs. The quota gate (R-24) and the MLflow record apply as for any CUDA run.

## 9. Citation

- Nwoye et al., Rendezvous: Attention mechanisms for the recognition of surgical action triplets in endoscopic videos, Medical Image Analysis 78 (2022) 102433.
- Ronneberger et al., U-Net, MICCAI 2015.
- Isola et al., Image-to-Image Translation with Conditional Adversarial Networks, CVPR 2017.
- Ledig et al., Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, CVPR 2017.
