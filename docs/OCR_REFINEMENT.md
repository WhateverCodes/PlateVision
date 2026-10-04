# OCR refinement - 2 October 2026

Restart OPEN_PLATEVISION.cmd. In Recognition choose Whole-line OCR (pretrained) and Pretrained plate detector. Keep reading confidence at 0.95. No training or further character reviews are needed for this update.

The reader now compensates for slanted text ordering and considers alternative letters/digits actually suggested by the OCR network when its first reading fails the existing registration format. Valid first readings are preserved. Candidate ranking includes spaces even when removed from the displayed registration. Selected token scores are not increased. This is bounded token-alternative decoding, not full CTC beam search or newly trained weights.

| Reviewed development crops | Previous exact | Updated exact | Updated accepted correct | Updated accepted wrong |
|---|---:|---:|---:|---:|
| Newer set, 30 | 14 | 18 | 13 | 2 |
| Difficult set, 15 | 10 | 13 | 8 | 0 |
| Total, 45 | 24 (53.3%) | 31 (68.9%) | 21 | 2 |

Exact counts include uncertain readings shown in inspection results. At the default confidence threshold, only 21/45 are both correct and accepted; 22/45 are rejected and 2/45 incorrectly accepted. A 0.95 confidence setting does not mean 95% accuracy. The previous version accepted 19 correct and 2 wrong. Two remaining confidently wrong readings are MH01AV7467 instead of MH01AV7461, and WH02BJ2456 instead of MH02BJ2456.

These small crop sets have been used for development; this is not independent final-test or full-photo accuracy. External pretrained data overlap is unknown. Reserved final photographs remain unused. No own CNN weights or training data were changed. All inference stays local with two CPU threads.

Validation: 14 targeted tests passed, including the actual app demonstration, comparison, pipeline, slanted/two-row ordering, repeated CTC characters, separator ranking, and preservation of alternative token scores.

Your next step: restart the app and try 10 new, clear vehicle photographs in Recognition. Keep the full original photo and record its correct plate text and the app result, including failures. Send these together in a ZIP for diagnosis. Use photos separate from the reserved final test set. Do not crop away vehicles for this check; it needs to expose plate detection as well as reading errors. More individual-character review is not needed for the pretrained OCR route.
