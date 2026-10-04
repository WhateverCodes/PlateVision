# Own CNN conservative update

The deployed own CNN is now models/character_conservative_v1/last.pt. The previous models/character_verified_100/last.pt is retained for rollback. Pretrained OCR weights and settings were not changed.

On the same 45 development crops:

| Metric | Previous own CNN | Updated own CNN |
|---|---:|---:|
| Exact full registration | 13/45 (28.9%) | 13/45 (28.9%) |
| Correct accepted | 8 | 8 |
| Incorrect accepted | 2 | 1 |
| Rejected | 35 | 36 |
| Total character edit errors | 134 | 131 |

This is a small reduction in falsely accepted readings, not a rise in complete-plate accuracy. Character edit errors include insertions/deletions/substitutions and are not character-classifier accuracy. The 45 development crops were repeatedly used for candidate selection; no independent improvement is established. Test/final photographs were not used for training or selection.

Training: 99 human-verified glyphs received sampling weight 12, synthetic glyphs weight 1, weakly aligned glyphs weight 0.25. Small shifts, scale changes and rotations were applied. Three bounded 60-step stages used batch size 24, AdamW learning rate 0.00001 and two CPU threads. None improved full-plate accuracy. We then evaluated conservative 75% original / 25% updated parameter blends; stage 1 was selected, with later stages offering no better exact/false-acceptance counts. These are updates to our own CNN, not external pretrained OCR.

Character segmentation selection now prefers the most likely format-plausible candidate when all candidates are below the acceptance threshold. This does not promote uncertain readings to accepted ones. The own-CNN threshold remains 0.75 minimum character score.

16 targeted app/pipeline checks passed, including the actual comparison example and the pretrained main app. Start OPEN_PLATEVISION.cmd, select Compare models, and run the local comparison example to use the updated CNN.

Reproduce from the project directory with the project Python environment:
python -m training.refine_verified_cnn --out models/character_weighted_trial --base models/character_verified_100/last.pt
python -m training.compare_cnn_blends
Use a clean experiment directory; the trainer refuses to overwrite an existing experiment. Original training datasets are required and are not bundled in the code ZIP. Do not claim 80% own-model accuracy or 95% OCR accuracy in the GitHub README.
