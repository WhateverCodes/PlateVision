# Additional dataset audit: archives 8, 9 and 10

- **archive (8).zip:** EMNIST handwriting. About 6.055 GB uncompressed across overlapping formats/splits; not fully extracted. Streamed 1,440 digits/uppercase glyphs (40/class) from the official byclass training partition, checked orientation and normalized to the CNN input. Test partition untouched. A two-stage, 120-step CPU trial mixed these with 1,307 existing verified/synthetic samples.
- **archive (9).zip:** 47 photos and 52 plate boxes. All 47 image CRC/size signatures match the previously supplied archive (3).zip. Annotation labels are `number_plate`, not registration transcriptions. Not added as new recognition data.
- **archive (10).zip:** 4,613 images, 5,742 `LP` boxes, 3,913 training and 700 test annotations. Labels do not contain registration text. Inspected 24 large plate crops from distinct training IDs; they show Sindh/Karachi registrations. These are a poor match for the Indian-format reader. No test image pixels were decoded. Detection training was not started. Parent-ID grouping is provisional, not a complete capture-independence audit.

## EMNIST result on the same 45 development plate crops

| Metric | Active CNN | Trial stage 1 | Trial stage 2 |
|---|---:|---:|---:|
| Exact plates | 14/45 (31.1%) | 15/45 (33.3%) | 15/45 (33.3%) |
| Character score, 1 − CER | 73.2% | 70.7% | 68.9% |
| Correct accepted at 0.70 | 9 | 7 | 6 |
| Wrong accepted | 2 | 0 | 0 |

No trial was promoted: slightly better full-plate matching came with poorer character reading and lower correct acceptance. All existing deployed weights remain unchanged, including OCR. This is a bounded development trial, not proof that handwriting pretraining could never help under a different training regime.

The best immediate next action remains checking the 24 Indian plate crops prepared from archive (6).zip in `PlateVision1/data/dataset_trial_2026_10_03/REVIEW_24_PLATES.html`. More useful future data contain real Indian plate crops plus their exact registration text, or correctly labelled printed characters from real plates. Neither new review data nor unpromoted trial checkpoints was added to the project distribution ZIP.
