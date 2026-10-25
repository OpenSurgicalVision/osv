# RFC 0001: M1 – can synthetic instruments replace real labels?

- **Author(s):** project owner (idea, scope); drafted with Claude Code
- **Date:** 2026-10-22
- **Status:** Draft
- **Related Issues:** #36

**RESEARCH USE ONLY — NOT FOR CLINICAL USE**

## Summary

Train the M1 instrument segmenter only on synthetic data and compare it with the same model
trained on real labels.
- **Synthetic data:** Blender-rendered graspers and L-hooks composited over CholecT50 frames that
  contain no instrument, with pixel-exact masks.
- **Real labels:** CholecSeg8k.
- **Test set:** real CholecSeg8k videos that are in neither training set.

The hypothesis under test is the **strong** one:

> **H1.** A model trained only on synthetic instruments is **not worse** than the same model
> trained on all available real labels, on real instruments.

Two classes are segmented, grasper and L-hook, and a binary instrument / background mask is
derived from them.

## Motivation & Problem Statement

Pixel labels for surgical video are the bottleneck of every M-task:
- **CholecSeg8k:** 8 080 frames from 17 videos.
- **Cost:** each frame is slow to annotate, and the instrument boundaries are the least
  consistent part of the masks.

A renderer gives unlimited instruments with exact masks, varied pose, opening, light and blur.
Whether that closes the gap to real labels is an empirical question with a useful answer
either way:
- **H1 holds:** M1 and its successors (more instrument classes, other procedures) no longer need
  instrument annotation.
- **H1 fails:** the size of the gap, and how much real data closes it (arms C below), tells the
  annotation team how many real labels the project actually needs.

## Detailed Technical Design

### 1. Data

**Backgrounds: tool-free CholecT50 frames.** CholecT50 labels every frame at 1 fps with
triplet vectors whose second element is the instrument id (0–5, −1 for none). A frame is
tool-free when no vector names an instrument. `python -m osv.datasets.m1_census` counts them
per video, per official split and per phase, and as runs of consecutive frames.
- **The census decides feasibility.** Tool-free frames are expected to be rare, a few percent,
  and concentrated in preparation and in instrument changes.
- **Too few for variety:** if fewer than about 1 000 tool-free frames from at least 20 videos
  remain in train, the backgrounds lack variety. The fallback is to inpaint real instruments
  out (section 6).
- **Census result (2026-10-23).** 8 182 tool-free train frames (11.2 %, in all 35 videos), 891
  in val and 1 988 in test (not used). Every phase has some.
- **Look at 48 random tool-free train frames (2026-10-23).**

  | What the frame shows | Share |
  |---|---:|
  | clean tissue | about 40 % |
  | an object the labels do not name (trocars, suction tips, drains, a looped rod, possibly grasper and hook shafts) | 20 % |
  | black | 15 % |
  | unreadable: inside a trocar, blown out, out of focus | 15 % |

  A visible instrument in a background teaches the model that real instruments are background,
  the one error that decides H1 against arm A. So no background is used unseen:
  - `python -m osv.datasets.m1_backgrounds sheets` keeps every 3rd frame of each run of
    consecutive tool-free frames;
  - it drops what the M5 haze score calls black or blown out;
  - it lays the rest out in numbered 6×6 sheets;
  - a person lists the tiles to reject in `rejects.txt`: any object other than clips and suture
    thread, and any frame where tissue is not clearly visible;
  - `collect` writes the reviewed list (accepted frames per split, reviewer, census hash), which
    is the input of the compositing stage.

  If about 40 % of the frames pass, every 3rd frame leaves roughly 1 100 train backgrounds.
- **Review result (2026-10-24).** Of 3 427 candidates (every 3rd frame of each run, train and
  val), 743 were black or blown out and 2 684 were reviewed. The reviewer rejected 1 878 (70 %).
  **717 train and 89 val backgrounds** remain.
- **Deviation from the threshold above, decided before any run (owner, 2026-10-24).** 717 is
  below the 1 000 set as the floor for variety. The owner chose to go ahead with 717 rather than
  inpaint instruments out of ordinary frames:
  - every training composite pairs a background with an instrument layer and augmentation drawn
    afresh each epoch;
  - the risk is that the model learns the backgrounds themselves;
  - if arm A loses, this is one of the explanations the report must weigh.
- **Splits:** synthetic training uses the CholecT50 **train** videos and synthetic validation
  uses its **val** videos. CholecT50 **test** videos stay untouched: they are held out for the
  other benchmarks.

**Instruments: `scripts/blender/instruments.py`** (Blender 4.2 / `bpy`).
- **Geometry.** The models are built from primitives in the script, so they are the project's
  own work with no third-party CAD licence.
  - Grasper: 5 mm insulated shaft, collar, clevis, and two tapered fenestrated jaws opening
    0–35°.
  - L-hook: 5 mm shaft tapering to a 1.1 mm electrode with a 5 mm bend at 90°.
- **Placement.** Each sample has one or two instruments.
  - The tip is in the central 80 % of the frame, 4–14 cm from a 70° lens.
  - The shaft leaves the frame through its border towards a trocar.
  - The roll about the shaft is random.
- **Light and materials.**
  - Light sits on the lens axis with inverse-square falloff (0.8–2 W).
  - The world is tissue-red, so the metal reflects red.
  - Steel roughness is drawn from 0.25–0.40 and insulation roughness from 0.22–0.40.
- **Outputs.**
  - `_rgba.png`: the instruments with straight alpha.
  - `_class.png`: 0 background, 1 grasper, 2 L-hook, from a one-sample, pixel-centre render
    with flat emission, so every pixel has exactly one label.
  - `_instance.png`: instance ids.
  - `.json`: every parameter needed to re-render the sample.

**Compositing and augmentation** (in `osv`, to be written: `osv/datasets/m1_synth.py`, a DVC
stage). Each rendered instrument layer goes over a random tool-free background of the same
split.
- **Field of view.** The scope's black vignette is taken from the background frame. Instrument
  pixels outside the field of view become black and are labelled background, as in
  CholecSeg8k.
- **Photometric match:**
  - gain and colour balance of the instrument layer drawn around the background's statistics;
  - specular bloom;
  - sensor noise;
  - JPEG.
- **Optics:**
  - motion blur along the shaft direction;
  - defocus that grows with distance from the tip's depth.
- **Occlusion.** The tip is partly covered by a background patch (tissue over the jaws), and
  the mask is updated to match.
- **Smoke.** Layers from the M5 `smoke_m5` build are added with the M5 scattering model on a
  fraction of samples.
- **Size and ownership.**
  - Planned size: 10 000 training composites and 1 000 validation composites, re-sampled every
    epoch from about 20 000 renders.
  - Masks are exact by construction, and every augmentation that moves pixels moves the mask
    with them.

**Real labels: CholecSeg8k** (already onboarded, `docs/data-cards/cholecseg8k.md`).
- **Classes.** The approved class mapping gives grasper (5) and L-hook electrocautery (9). Every
  other class becomes background.
- **Split.** CholecSeg8k and CholecT50 both come from Cholec80 and share several videos.
  `m1_census` lists both groups from the archives:
  - **Test:** the CholecSeg8k videos **absent from CholecT50**. They were never a synthetic
    background, so they are the only admissible test videos. From the published video list
    this is expected to be 7 of 17; the census confirms it.
  - **Real training:** the overlapping videos, split by video into train and validation
    (about 8 / 2).
- **Open question: other instruments.** CholecSeg8k labels only two instrument classes. Test
  frames that show other instruments (clipper, irrigator, scissors) have no class for them. How
  those pixels are labelled must be checked on the masks before the test set is frozen. If they
  are not consistently background, such frames are marked `ignore` using Cholec80's tool
  presence annotations, or excluded and listed.
- **Review of the test masks (owner, 2026-10-25).** Every one of the 37 test clips (2 960
  frames) was checked on its first, middle and last frame using `python -m osv.eval.m1 review`.
  No instrument without its own class was seen, so nothing is excluded. The result is recorded
  in `osv/datasets/manifests/m1_test_review.yaml`; `osv.eval.m1 test` runs only on a REVIEWED
  manifest.

### 2. Model and training

- **Architecture.** The existing `deeplabv3_resnet50` row of `osv/arch/registry.py`, with a
  3-class head and the ImageNet backbone from a local weights file (R-07). The same
  architecture, input size (full 854×480 frames), optimiser, schedule and epoch budget are used
  in every arm. A stronger architecture can be added later through R-25, for all arms at once.
- **Arms.**

| Arm | Train | Validation (model selection) | Test |
|---|---|---|---|
| **A** – synthetic only | synthetic train | synthetic val | CholecSeg8k test videos |
| **B** – real only | CholecSeg8k real-train (100 %) | CholecSeg8k real-val | same |
| **C10 / C25 / C50 / C100** | A's checkpoint, fine-tuned on 10 / 25 / 50 / 100 % of real-train frames | CholecSeg8k real-val | same |

- **Model selection.** Arm A selects its checkpoint on **synthetic** val only. Picking it on real
  frames would leak real labels into the "synthetic only" arm and invalidate H1. Its real-val
  score is logged for information, never used.
- **Real fractions.** The fractions in C are of frames, sampled uniformly within every
  real-train video. The list of frames is fixed per seed and recorded in MLflow.
- **Seeds.** Three seeds per arm (R-25), 18 runs in total.
- **Training budget (`osv/train/m1.py`, fixed before any run).**
  - Every arm uses the same settings: AdamW (lr 1e-4, weight decay 1e-4), cosine decay over
    20 epochs, batch 8, unweighted cross-entropy plus 0.4 × the auxiliary head, and
    mixed precision.
  - Arm A draws as many fresh composites per epoch as real-train has frames, so A and B take
    the same number of optimiser steps. This replaces the planned 10 000 composites per epoch,
    which would have given A about 2.4 times B's steps.
  - Real-train frames get random horizontal and vertical flips. The composites already carry
    flips.
  - The checkpoint with the best val binary IoU is selected.

### 3. Metrics and the decision rule for H1

- **Primary metric:** IoU of the binary instrument mask, computed over all pixels of all test
  frames (dataset-level IoU).
- **Secondary metrics:**
  - per-class IoU and Dice for grasper and L-hook, and their mean;
  - boundary F1 with a 3-pixel tolerance;
  - the same metrics per test video.
- **Uncertainty.** A 95 % interval for the difference Δ = IoU(A) − IoU(B), by bootstrap over
  test videos (the frames of one video are not independent) and seeds.
- **Non-inferiority margin δ = 2.0 IoU points**, fixed here before any run and confirmed by the
  project owner on 2026-10-22. It may not change once a run has started.
  - **H1 supported:** the lower bound of Δ's interval is above −δ.
  - **H1 rejected:** the upper bound of Δ's interval is below −δ.
  - **Inconclusive:** anything else. With about 7 test videos the interval will be wide, and an
    inconclusive result is reported as such (R-25 `INCOMPLETE`), never as "no difference".
- **Arms C** are reported as a curve of IoU against the real fraction, next to B. They answer
  how much real data synthetic pre-training replaces; H1 is decided on A against B alone.
- **Test discipline.** Each arm's final checkpoint touches the test videos exactly once, and
  every touch is counted (R-25 counter).

### 4. Compute, data movement and the gates

- **Rendering.** Blender's Cycles renderer on CPU takes about 5 s per sample at 48 samples per
  pixel. 20 000 renders take about 28 CPU-hours. Rendering can run on the owner's machine or a
  CPU pod; it needs no patient data.
- **Training.** The estimate is at most 2 GPU-hours per run on an A40, about 36 GPU-hours for
  all 18 runs (about $21 at $0.59/h). That nearly fills one week's R-24 quota, so the runs are
  planned over two weeks.
- **Runpod scope (ADR 0009).** On 2026-10-22 the owner extended ADR 0009 decision 1 to
  CholecSeg8k and the data built from it (osv-infra ADR 0009, decision 1a): Secure Cloud in the
  EU allowlist only, the same terms as CholecT50. All arms and the test evaluation can run on a
  Runpod pod. Before the first CholecSeg8k session the owner checks that CholecSeg8k's and
  Cholec80's terms allow processing at a cloud provider (data card open item).

### 5. Milestones

1. Census on the real archives: the tool-free count, the admissible test videos, and the check
   for other instruments in the test masks. Then this RFC is updated and either approved or
   sent back.
2. `m1_synth` compositing stage, with a contact sheet over real backgrounds reviewed by a
   person before any training.
3. Arms A and B, 3 seeds each. **H1 is decided here.**
4. Arms C, 3 seeds each.
5. Model card (`docs/model-cards/`) and the experiment page on the website, including a
   negative result if that is the outcome.

## Clinical & Safety Considerations

- Research only. Instrument masks are an engineering target and make no clinical claim. No
  output is shown to a surgeon.
- No new patient data; both datasets are public and de-identified by their providers.
- Licences:
  - the renders and the instrument geometry are the project's own work;
  - the composites inherit **CC BY-NC-SA 4.0** from CholecT50;
  - the real arms inherit CC BY-NC-SA 4.0 from CholecSeg8k.
- Annotation guidelines are unchanged. The `ignore` rule for unlabelled instruments in the test
  masks (section 1) is a benchmark rule and is documented in the data card.

## Drawbacks & Alternatives

- **Inpainting instruments out of ordinary frames** gives many more backgrounds. Its cost is
  that traces of real instruments remain unlabelled. It is the fallback if the census shows too
  few tool-free frames.
- **Cut-and-paste of real instruments** (copy the CholecSeg8k instrument pixels) is not
  synthetic and would leak real labels into arm A. Rejected for A.
- **Diffusion-generated instruments** produce no exact masks. Possible later, as texture for the
  rendered geometry.
- **EndoVis 2017/2018 as the test** is a different domain (robotic instruments, porcine
  tissue). It could be a secondary test, but it is not the question asked here.
- **Domain randomisation** (unrealistic textures and colours) against photorealism: the
  renderer starts photorealistic. If A fails, a randomised arm is the cheapest next experiment.
- **Drawback: scarce, biased backgrounds.** Tool-free frames cluster in a few phases, so the
  backgrounds are biased, and the census reports it. Test frames come from all phases.
- **Drawback: no clinical review of the instruments.** The look of the synthetic instruments is
  judged by people, not measured. Milestone 2 has a reviewed contact sheet for that reason.
