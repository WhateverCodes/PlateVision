# Accuracy update — 21 September 2026

169 user reviews saved: 158 readable, 11 excluded. New immutable reader_reviewed_v2 snapshot contains 853 training images (129 human-reviewed, 724 source-unverified); 62 additional reviewed training images. Original 15 validation and 13 test records/labels preserved. Renamed project paths repaired in the new snapshot only. ID, registration and pixel-hash overlap checks passed. External final photos unused.

Character preparation accepted 324 training plates, yielding 3,114 weakly aligned real character crops plus 1,008 synthetic replay characters. These character alignments are automatic, not human-verified boxes.

The new character_adaptation_v2 experiment used two CPU threads and a 60-second training budget: two complete epochs plus 528 samples of epoch three. It is a short candidate experiment, not an equal-epoch controlled comparison. On the unchanged development validation set it achieved 4/15 raw exact readings, 2 accepted correct, 0 accepted wrong, character error rate 39.44%. Rechecked current v1: 5/15 raw exact, 2 accepted correct, 0 accepted wrong, character error rate 38.03%. No improvement demonstrated; v1 stays selected in the application. These figures describe plate-crop recognition on a small development set, not full-photo or final-test accuracy.

## Next user task
Refresh the OCR review page, choose **Accuracy check - not reviewed**, and review 30 plates. Correct registration text or mark unreadable, then Save and next. The queue contains 395 previously withheld images; larger crops and distinct source groups are prioritized. Reviews are saved centrally in PlateVision1/data/ocr_text_review. These images remain outside training. Do not run training launchers for this task.

No newly reviewed withheld images were available when this snapshot was prepared. Once reviewed, a supplemental development evaluation can be created without moving prior training images into validation. The original difficult evaluation and reserved final tests remain separate. Capture independence is not guaranteed by source grouping.


## Completed supplemental check
200 reviews now saved. Of 31 newly reviewed queue images, 30 are readable and one was excluded. The frozen supplemental set contains 30 crops with 29 distinct corrected registrations. No training/reserved overlap was found under the recorded identity, label, pixel-hash and group checks.

| Model | Exact raw readings | Accepted correct | Accepted wrong | Character error rate |
|---|---:|---:|---:|---:|
| Current v1 | 5/30 | 3 | 2 | 35.23% |
| Candidate v2 | 4/30 | 1 | 1 | 36.24% |

Neither model is reliable yet. Current v1 remains installed because v2 has not shown a clear improvement. Two wrong accepted readings in v1 demonstrate that the confidence/format filter does not verify correctness. Observed failures include missing characters, row ordering, extra components and character substitutions. Next engineering priority is segmentation/row ordering and rejection quality, followed by re-evaluation on both development sets. No further broad manual review batch is requested at this point. Supplemental images stay outside training; the final external photographs remain reserved.
