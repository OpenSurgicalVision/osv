# Model Cards

Model documentation for all baseline and released checkpoints, published to the
[OpenSurgicalVision org on Hugging Face](https://huggingface.co/opensurgicalvision). The org is
reserved; no checkpoint has been released yet, and none will be until it has a Model Card
covering every section below plus the mandatory `RESEARCH USE ONLY — NOT FOR CLINICAL USE`
disclaimer (R-03).

Mandatory sections:
1. **Clinical Disclaimer:** `RESEARCH USE ONLY — NOT FOR CLINICAL USE`
2. Model Description & Architecture
3. Training Data & Licensing Inheritance
4. Quantitative Evaluation (mIoU, Dice, Latency p95)
5. Robustness / Edge-Case Breakdown (smoke, glare, blood)
6. Intended & Prohibited Uses

Cards:
- [M5 surgical desmoking baselines](m5-desmoke.md): residual U-Net, Pix2Pix, SRGAN. Not released (INCOMPLETE under R-25).
- [M1 instrument segmentation, synthetic against real labels](m1-synthetic.md): DeepLabV3-ResNet50, RFC 0001 arms A, B and C. Not released; H1 rejected.
