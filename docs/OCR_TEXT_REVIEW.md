# Check registrations for the whole-plate reader

Double-click REVIEW_OCR_TEXT.cmd in the PlateVision folder. The screen opens locally on port 8505.

1. Keep the default Flagged - not reviewed view.
2. Read the image and correct Registration if needed. Omit IND, spaces and slogans; read two-line plates top row first.
3. If the plate cannot be read confidently, tick Cannot read reliably / unsupported script.
4. Click Save and next. Every save is written to disk immediately.
5. Start with 30 images, then report back. Do not attempt the entire collection in one sitting.

Use Saved reviews to correct an earlier decision. The original source text is retained. Source placeholders are set aside but visible in a separate view.

The local import has 1,690 images, with two placeholder labels set aside. 569 images were flagged by simple format and small-image checks. These are review priorities, not 569 proven errors. Unflagged labels have not been human-verified either.

Reviews are in data/ocr_text_review/reviews. Earlier plate/character annotations remain in data/kaggle_indian_review. No train/test split or training is started by this tool. Images sharing original text have provisional groups; corrected text, image similarity and related captures must also be considered before final splitting.

Next coding work: the compact whole-plate reader and a short CPU training benchmark. The current TRAIN_MODELS.cmd remains the synthetic practice launcher. Do not use it to train this new collection.
