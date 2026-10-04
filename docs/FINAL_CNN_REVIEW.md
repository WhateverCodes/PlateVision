# Final bounded CNN review — 3 October 2026

The existing CNN is retained. Two learning rates (0.000003 and 0.00003), each with two 40-step stages, did not improve the baseline. Training used 299 human-verified characters plus 1,008 synthetic characters, balanced sampling, mild geometric augmentation, consistent 32×32 normalization, and frozen BatchNorm running statistics. The architecture already includes BatchNorm and 0.25 dropout; no arbitrary extra layers were added.

| Model | Character score (1 − CER) | Character error rate | Complete-plate exact match |
|---|---:|---:|---:|
| Own CNN + segmentation | 70.2% | 131/440 = 29.8% | 13/45 = 28.9% |
| Adapted PaddleOCR | 94.3% | 25/440 = 5.7% | 31/45 = 68.9% |

These are 45 development plate crops with 440 reference characters, repeatedly used for model selection. Scores include every raw reading, even rejected readings. CER is total Levenshtein edit distance divided by reference character count: substitutions, missing characters and extra characters count as errors. 1−CER is a text-reading score, not the proportion of independently classified character crops. Vehicle/plate localization on full photos is outside this evaluation.

The CNN separately classified 216/216 synthetic validation glyphs correctly. This 100% synthetic result does not measure real-photo character classification. No independent manually labelled real-glyph test has been measured. The 299 reviewed glyphs were used for training and cannot provide that test result.

Verified character coverage is uneven: I and V have zero examples; O, X and Z have one each. Balanced sampling cannot invent missing real examples. The plate failures include missing/extra characters from segmentation as well as classification errors. Existing failure inspection and the earlier segmentation trials are documented in CNN_SEGMENTATION_TRIALS.md.

| Learning rate | Stage | Training loss | Synthetic validation loss | Full plates correct | Wrong accepted |
|---|---:|---:|---:|---:|---:|
| 3e-06 | 1 | 0.0899 | 0.0060 | 13/45 | 1 |
| 3e-06 | 2 | 0.1049 | 0.0055 | 13/45 | 1 |
| 3e-05 | 1 | 0.0893 | 0.0033 | 11/45 | 2 |
| 3e-05 | 2 | 0.0971 | 0.0023 | 13/45 | 2 |

No candidate surpassed the retained 13/45 result with at most one wrong accepted reading. Stop this improvement round and retain the current weights. An independent test with complete annotations is the next evaluation task, not another tuning cycle.
