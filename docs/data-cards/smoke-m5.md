# Data Card: M5 synthetic smoke (Blender smoke over CholecT50)

**RESEARCH USE ONLY — NOT FOR CLINICAL USE**

Status: **PROPOSED**. Code and manifests are in review; nothing has been built.
`osv/datasets/manifests/{cholect50,blender_smoke,smoke_m5}.yaml` must be set to
`APPROVED` by the project owner, and `smoke_m5.yaml: clean.haze_max` must be set
from `python -m osv.datasets.smoke_m5 calibrate`, before `dvc repro smoke_m5_build`
will run.

## 1. Dataset Origin & Original Authors

- **Clean frames:** CholecT50 (CAMMA, University of Strasbourg / IHU / IRCAD), 50
  laparoscopic cholecystectomy videos at 1 fps. No public URL: released to the
  project owner by the CAMMA Dataset Department on request; archive sha256
  `57cf20784722f034accc7016136deb62340731bd39d594ac52be3da447ceae80`.
  Cite: Nwoye et al., *Rendezvous*, Medical Image Analysis 78 (2022) 102433; splits:
  Nwoye & Padoy, arXiv:2204.05235.
- **Smoke layers:** the project's own Blender (Mantaflow) smoke simulations, rendered
  as straight-alpha RGBA over a transparent background: 36 sequences, 919 kept frames,
  1280×720. Per-sequence `config.json` / `meta.json` record seed, emitter, pulses and
  shading. The generator scripts are not yet in the repository.
- **Pairs:** composited by `osv/augment/smoke.py` with the single-scattering model
  `I = J'·(1 − a) + C·a` (optical-depth gain on the render's alpha, smoke colour scaled
  to the frame's lamp level, forward-scatter blur behind dense smoke, no smoke on the
  scope's black vignette).

## 2. License & Redistribution Constraints

- Clean frames and every composited pair: **CC BY-NC-SA 4.0** (CholecT50). Non-commercial
  use only, accepted for OSV by `docs/adr/0002-accepting-noncommercial-datasets.md`;
  share-alike; attribution to the papers above.
- Smoke renders on their own: licence **to be decided by the owner** (proposal: CC BY 4.0).
  They contain no patient data.

## 3. Modality & Clinical Scope

2D RGB frames, 854×480. Synthetic smoke resembles electrocautery plumes but is not
validated against real smoke by a clinician; models trained on it are engineering
baselines for M5 and for the smoke levels of `osv.robustness`, not evidence of
clinical utility.

## 4. De-identification / PHI Audit Results

- CholecT50 is published de-identified by its provider. The OSV platform (osv-infra)
  ran its own scan on the same archive: metadata stripped, OCR and canonical scan on
  every frame; frames with detected text were excluded there. This build reads the
  original archive and does **not** re-run that scan; `osv/deid` remains the gate for
  anything published from OSV.
- Smoke renders: no patient content by construction.
- Calibration previews are written only outside the repository; the build refuses a
  preview path inside it.

## 5. Known Biases & Limitations

- **Clean is a filter, not a fact.** CholecT50 already contains real smoke. Clean frames
  are those with a dark-channel haze score at most `haze_max`; light real haze can pass
  and becomes part of the "clean" target. The threshold, its sample and its previews
  are recorded in review.
- **Smoke variety** is bounded by 36 simulated plumes (two render generations: white
  shape-in-alpha and lit/tinted). Augmentation widens pose, density and colour, not
  plume physics.
- **No temporal model:** pairs are single frames; video consistency is not trained.
- **Domain gap:** single-scattering compositing omits light emitted or reflected by the
  smoke near instruments and does not darken the scene the way dense real smoke does.

## 6. Patient Splitting Methodology

- **Frames:** the official CholecT50 `cholect50` split, by whole video (35 train / 5 val /
  10 test), from `osv/datasets/manifests/cholect50.yaml`. A video outside the table is
  an error.
- **Smoke:** by whole sequence, never by frame, stratified by render generation, salted
  hash order: 15 % val, 15 % test, the rest train.
- **Pairs:** a pair takes its clean frame and its smoke layer from the same split. Val
  and test pairs are fixed and reproducible bit-for-bit; train pairs are composited on
  the fly from train frames and train smoke only (`osv/datasets/desmoke.py`).
- `index.json` in the build records both splits, every frame's haze score (kept and
  rejected) and every pair's parameters.
