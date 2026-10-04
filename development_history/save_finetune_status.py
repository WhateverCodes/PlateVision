from pathlib import Path
import json
p=Path('outputs/PlateVision1/data/ocr_finetune_audit_v1')
report='''# OCR fine-tuning preparation status

Completed: audited all 853 historical training records. Of these, 129 have human-reviewed text and 724 remain source-unverified. The text detector generated 81 single-line candidates; visual screening quarantined five incomplete crops, leaving 76 candidates. Another 48 have multiple/no detected regions, for 53 total layout-review records. Original labels and images are preserved. No own CNN or deployed OCR weights were changed.

Split checks: training candidates were screened against existing development/test/final-manifest hashes, text identities and existing groups. Historical membership was preserved. Near-duplicate capture independence is not yet certified. The 45 development crops, 13 existing test records and 10 reserved final photographs were not used for training. No new independent split has been claimed.

Artifacts: train.json, train.txt and images/ contain screened candidates; layout_review.json and REVIEW_LAYOUT.html contain excluded layout cases. These are preparation artifacts, not an accuracy improvement. Additional completeness/duplicate checks and a separately audited validation-line manifest are required before a fine-tuning run.

Setup: official English PP-OCRv5 config retrieved at PaddleOCR revision dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf. An isolated environment was created in work/paddle_train_env. The official PaddlePaddle 3.2.2 Windows CPU wheel downloaded and passed its PyPI SHA256 check. Installation did not complete; the stalled installer was stopped. The training checkpoint download failed with DNS getaddrinfo error for paddle-model-ecology.bj.bcebos.com. No training process remains running.

Next: obtain the official compatible training checkpoint; finish isolated dependencies; resolve layout cases and prepare validation lines; run a bounded CPU pilot, then compare full-pipeline accuracy before considering deployment. The draft configuration has batch size 2, zero data-loader workers, one epoch and learning rate 0.00001; it deliberately references a pending validation manifest so it cannot accidentally validate on training data. Runtime thread/time limits still need enforcement in the launcher.

Current deployed development accuracy remains 31/45 (68.9%), with 25 correct accepted readings and 2 wrong accepted readings. No 95% claim.
'''
(p/'STATUS.md').write_text(report,encoding='utf8')
print(p/'STATUS.md')
