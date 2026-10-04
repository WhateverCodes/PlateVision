# CNN trial after 300 character checks — 3 October 2026

All 200 new reviews were saved. The full review collection now has 300 checks: 299 usable verified character crops and one previously excluded crop. Two labels were corrected in the new batch. Originals, reviews, and earlier datasets are preserved.

Two CPU-limited trials used the new snapshot. The first updated the full CNN with verified-character weighting, weak-label downweighting and synthetic replay. Its three stages scored 11, 12 and 12 exact plates out of 45. The second kept feature extraction fixed and used only verified characters plus synthetic replay; its stages scored 11, 11 and 12. Both were worse than the current 13/45. No candidate was deployed. No OCR weights changed.

Current own-CNN result remains 13/45 exact (28.9%), 8 correct accepted, 1 wrong accepted and 36 rejected. The same 45 development crops were used for selection; these are not independent test results.

Twenty of the 32 wrong predictions have a different character count from their reference. This suggests improving plate-to-character separation is a higher-priority next experiment than more epochs on these crops. It does not prove that classification is otherwise correct. Do not request more manual character reviews solely because these trials failed.

You have completed this review batch. No additional reviews or repeated training clicks are required. The app still uses models/character_conservative_v1/last.pt. Detailed measurements are in outputs/evaluation/cnn_300_review_report.json. Training snapshots are kept locally in data/character_verified_300; experimental checkpoints remain in separate trial folders.
