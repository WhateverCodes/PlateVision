# Combined dataset training — 4 October 2026

This is a bounded, two-CPU-thread training experiment. Source datasets were assigned to the task their labels support; duplicate archives and reserved evaluation images were not added to training.

## Reader results

The same 45 development plate crops contain 440 reference characters. These crops have been used repeatedly for model and threshold selection. Results are not independent-test accuracy, isolated-glyph classification accuracy, or full-photo recognition accuracy.

| Reader | Complete plates | Character edits | Character score (1 − CER) | Accepted correct / wrong |
|---|---:|---:|---:|---:|
| Previous own CNN, threshold .70 | 14/45 (31.1%) | 118 | 73.2% | 9 / 2 |
| New CNN, epoch 1, threshold .70 | 14/45 (31.1%) | 145 | 67.0% | 5 / 0 |
| New CNN, epoch 2, threshold .70 | 16/45 (35.6%) | 140 | 68.2% | 5 / 0 |
| Previous adapted PaddleOCR, threshold .90 | 34/45 (75.6%) | 21 | 95.2% | 29 / 3 |
| New adapted PaddleOCR, threshold .90 | 34/45 (75.6%) | 18 | 95.9% | 33 / 4 |
| **Selected PaddleOCR, threshold .91** | **34/45 (75.6%)** | **18** | **95.9%** | **31 / 3** |

The new CNN has more exact plates but worse character performance and fewer confident correct readings. It remains an experimental checkpoint; the previous CNN stays active. The new PaddleOCR model preserves the exact-match count, reduces character errors and accepts two more correct readings at the same wrong-reading count when using .91. This threshold was selected on the development set; the result needs independent validation.

PaddleOCR training updated only its final CTC projection, using 981 prepared training lines from 838 plates: 102 lines have human-reviewed labels and 879 retain unverified source labels. It ran eight epochs and used a 50% blend of the selected first-epoch projection with the prior adapted projection. Its pretrained feature extractor and text detector were unchanged. The 26-line development check improved from 16 to 18 exact lines. This is **not full-network PaddleOCR fine-tuning**.

The own CNN completed two passes over all 42,691 new glyphs: 35,500 printed glyphs, a 1,440-image handwriting sample, and 5,751 automatically aligned real-plate glyphs. Verified and synthetic examples were replayed in every batch. The combined manifest has 43,998 images. Automatic alignment and source labels can be wrong; these glyphs were not all manually reviewed.

## Dataset accounting

### Additional validation on 54 new plate crops

These plates were held out from training and were not used for the threshold sweep above. Their source labels are not all manually verified, and capture provenance is incomplete. The reserved test split remains unused.

| Reader | Exact plates | Character edits / 517 | Accepted correct / wrong |
|---|---:|---:|---:|
| Previous CNN, .70 | 17/54 | 145 | 11 / 5 |
| New CNN epoch 2, .70 | 18/54 | 154 | 9 / 1 |
| Previous OCR, .90 | 42/54 | 31 | 35 / 1 |
| Selected new OCR, .91 | 43/54 | 29 | 34 / 2 |

Both readers gained one exact plate on this additional set. The new CNN again made more character errors, so its old checkpoint remains active. The new OCR is selected for its modest raw-reading improvement across both sets, **not for uniformly better confidence filtering**: on this additional set it accepts one fewer correct plate and one more wrong plate. No further threshold adjustment was made using these results. A 95% complete-plate accuracy claim is not supported.

| Supplied archive | Use in this experiment |
|---|---|
| archive (2), archive (5) | Identical archives, counted once. Existing screened vehicle-scene detector training subset retained. Cropped images without usable text labels were not assigned invented text. |
| archive (3), archive (9) | Duplicate image collection, counted once. Existing Indian detector training split retained. |
| archive (4) | Existing reader training split, filtered against held-out registrations and usable label formats. |
| archive (6) | Of 387 previously screened candidates, 361 passed label-format checks: 251 training, 54 validation, 56 reserved test. 26 questionable labels quarantined. Training boxes and text were used. |
| archive (7) | All 35,500 printed glyphs used in both CNN training passes. Missing class O is supplied by the other sources. |
| archive (8) | 1,440 official-training EMNIST examples, 40 per class. A representative handwriting sample, **not the entire archive**. |
| archive (10) | 3,913 official training images supply plate boxes; Pakistan registration text was not used for Indian text-reader training. 700 official test images excluded. |
| 3113449.zip | File unavailable at its supplied location; not trained. |

1,079 candidate training plates yielded 981 usable text lines; 241 plate layouts were quarantined from line training. A plate can yield two lines, so these counts do not simply add up. The old detector training subset contributes 74 images; archive (6) adds 251, producing 325 Indian training images. Adding archive (10) gives 4,238 detector training images.

Reserved test partitions were not used to tune these models. Registration and exact-file checks reduce leakage, but incomplete capture provenance and possible near-duplicates prevent claims of certified independent evaluation.

## Models and evidence

- Active OCR: `models/paddle_combined_v2`, threshold .91. Its development report contains the threshold comparison and per-plate results.
- Active CNN: `models/character_conservative_v1/last.pt`, threshold .70.
- Main Recognition and normal Compare models continue using the stronger external YOLO plate detector.
- CNN trial checkpoints remain in the canonical local project's `models/combined_cnn_2026_10_03`; they are not silently substituted into the app.
- Detailed detector results and final dataset coverage are recorded in `outputs/evaluation/combined_training_summary.json`.

Restart `OPEN_PLATEVISION.cmd` to load the updated OCR weights. No additional manual character review is required to run this version.

## Completed detector run

The mixed-data detector completed one full epoch over all 4,238 training images. A separate Indian-focused candidate completed one epoch over all 325 Indian training images after initialization from the earlier partial mixed pass. Both candidates were evaluated; the custom checkpoint with higher development AP@0.5 was retained for research.

| Detector on 61 development images / 66 labelled plates | Correct detections | False detections | Recall | AP@0.5 |
|---|---:|---:|---:|---:|
| Original own detector | 10 | 60 | 15.2% | 6.3% |
| Indian-focused candidate | 20 | 23 | 30.3% | 24.1% |
| Full mixed epoch candidate | 9 | 36 | 13.6% | 6.2% |
| Retained external YOLO | 62 | 7 | 93.9% | 93.1% |

Detection uses score .5 and IoU .5. Source boxes are not fully reviewed; these numbers are not a fully annotated independent test. The strongest custom detector is saved at `models/plate_detector_real_v3/best.pt`; main Recognition and Compare continue using YOLO.
