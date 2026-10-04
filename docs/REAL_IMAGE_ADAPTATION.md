# Real-image adaptation results

## Character recognition
Prepared 2,889 provisional real character crops from 300 images in reader_v1/train.json. Candidates were accepted only when the number of segmented shapes matched the supplied text and at least 70% of the old classifier predictions agreed with it. These are weak automatic alignments, not human-verified character annotations. No validation or test labels were used to assign training characters. Added 1,008 original synthetic training examples to reduce forgetting.

Adapted the existing own CNN for three epochs at learning rate 0.00003, using two CPU threads and batch size 24. All epochs completed in approximately 40 seconds. The original model is unchanged. The new checkpoint is models/character_adaptation_v1/last.pt.

On the same difficult 15-crop development validation set:
- Conservative original model: 3 exact raw readings, 1 correct result accepted, 0 wrong results accepted.
- Adapted model: 5 exact raw readings, 2 correct results accepted, 0 wrong results accepted.
- Character edit error decreased from 48.59% to 38.03%.
- Acceptance used the existing 0.75 minimum character score and format check. A correct raw reading can still be rejected.

This is preliminary development evidence, not final independent accuracy. The validation set was used to choose methods. The main app now selects Real-image character model (pilot) by default if its checkpoint exists, while retaining the original and experimental whole-plate choices.

## Localization
Prepared the 47-photo DataCluster sample after EXIF orientation correction, retaining source XML labels and resizing the longest edge to 1280. Development split: 33 training images, 7 validation, 7 internal test. Grouping uses supplied text and image hashes and does not establish independent capture origins. This dataset was already used for exploratory detector analysis.

The seven validation images contain 12 annotated plates. At score 0.5 and matching IoU 0.5:
- Original detector: 2 true positives, 4 false positives.
- Five-epoch real-image adaptation: 2 true positives, 5 false positives.
- Original weights at 640-pixel inference input: 1 true positive, 10 false positives.

Neither detector experiment was promoted. The app retains the original 320-pixel detector. The separate adaptation checkpoint remains available for research, not as an improved default.

## Current status
Character recognition improved, but automatic whole-vehicle recognition remains limited chiefly by localization and difficult character crops. The project is not ready for a broad reliability claim. Existing annotations, original weights and the final test photos remain intact. The 10 user test photos and internal held-out test splits were not evaluated in these comparisons. Training sessions have ended.

## Use the current local app
Open OPEN_PLATEVISION.cmd, then check Processing settings. Real-image character model (pilot) is the preferred current reader. Keep Trained plate detector for vehicle photos. Already-cropped plate image is only for an image containing a single plate; it explicitly bypasses localization. No training starts when opening the app.
