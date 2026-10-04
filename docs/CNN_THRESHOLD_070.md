# CNN threshold comparison

The current CNN minimum-character threshold is **0.70**. OCR remains at 0.90. The five requested thresholds were evaluated by rerunning the complete segmentation-candidate selection on the same 45 development crops with format decoding enabled. No retraining or new data were used.

| Threshold | Correct accepted | Wrong accepted | Uncertain | Exact full plates |
|---|---:|---:|---:|---:|
| 0.65 | 9 | 2 | 34 | 14/45 |
| 0.70 | 9 | 2 | 34 | 14/45 |
| 0.75 | 8 | 1 | 36 | 14/45 |
| 0.80 | 7 | 1 | 37 | 14/45 |
| 0.85 | 6 | 1 | 38 | 14/45 |

Selection maximizes accepted exact-reading F1: 2×correct / (45 + correct + wrong). This balances recovery of correct plates and incorrect accepted readings. At 0.65 and 0.70 it is 32.1%, versus 29.6% at 0.75. The tie was resolved in favour of the threshold closest to the previous 0.75 setting.

This is not a universal best threshold: 0.70 accepts 9 correct and 2 wrong (81.8% correctness among accepted readings), whereas 0.75 accepts 8 correct and 1 wrong (88.9%). Prefer 0.75 manually if avoiding wrong accepted results matters more than coverage.

All five thresholds produced identical raw strings on these crops: 14/45 complete plates correct (31.1%), with 118 edits across 440 reference characters (73.2% character score). The new default changes acceptance, not underlying accuracy, on this evaluation. Development data have been reused for selection; independent-test performance is unknown.

Restart OPEN_PLATEVISION.cmd in PlateVision8. The Recognition default and Compare models now use 0.70 for the own CNN. Older trial documents intentionally retain their original experimental thresholds. Reproduce with `python -m training.evaluate_cnn_thresholds` using the existing local data manifests.
