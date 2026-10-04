# Next task: check 50 training characters

Double-click **REVIEW_TRAINING_CHARACTERS.cmd**. This opens a new page on local port 8506.

1. Look at the whole registration on the left and the enlarged character crop on the right.
2. Select the actual visible character. The proposed label may be wrong if automatic alignment failed.
3. If strokes are cut off, multiple characters are joined, or the character cannot be read reliably, tick **Unusable crop / cannot read reliably**.
4. Click **Save and next**. Stop at **50 characters checked**, then tell the assistant.

No boxes to draw, no new photo downloads, and no training launcher to run. Checks are saved immediately. The queue is shared by all three local project copies in PlateVision1/data/glyph_review. It contains only training images; evaluation plates remain held out.

## Why this task

Whole-plate text reviews do not confirm that individual character crops are correct. Our CNN learns from individual glyphs. The queue uses 501 existing automatically aligned glyphs from 54 human-text-reviewed training plates, prioritizing difficult examples across character classes. Position and label suggestions are provisional; judge the visible crop.

Two additional geometry experiments were tested and rejected: glyph deskewing increased accepted wrong readings from 2 to 3 on the 30-crop development set; reducing the minimum character height reduced exact readings from 6 to 5. Neither change is active. The better existing model and tilt-aware row ordering remain installed.

After the checks, training/prepare_verified_characters.py creates a separate snapshot that applies your corrections and removes unusable crops. The next candidate will be evaluated on both held-out development sets before promotion. Fifty checks are a focused first batch, not a promise of a particular accuracy.
