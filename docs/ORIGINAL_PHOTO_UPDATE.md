# Original-photo OCR update

Restart OPEN_PLATEVISION.cmd. No training or downloads required.

Changes: four overlapping detector views on images larger than 640 pixels; duplicate boxes suppressed. Uncertain small crops receive one enlarged OCR retry, used only when the text agrees. The OCR format check now also permits two letters followed by six digits, addressing HP 88 5801. This is a format plausibility check, not official verification. Own CNN behavior/weights are unchanged. CPU work stays sequential with two inference threads; large images take additional time.

The 45 development crops still give 31 exact readings (68.9%). Correct accepted readings increased from 21 to 25; wrong accepted readings remain 2; 18 are rejected. Confidence threshold remains 0.95. This is not a 95% accuracy result. No independent final-test result is claimed.

Original-photo diagnosis: image 6 now accepts HP885801; image 8 now accepts WB34CG4538. Image 9 now detects all four main plates, but three still fail reading/format checks; additional partial candidates also appear. Images 3 and 10 read JK02BA5065 and KL02BP7403 but remain below threshold, visible as uncertain readings. Image 4 still predicts DL3SCW6192, counted wrong against the user's confirmed DL35CW6192. Vehicle classification (image 5 car/truck) was not changed. A complete all-plate ground-truth inventory is not available for the ten photos, so no full-photo accuracy percentage is asserted.

16 targeted tests passed, including actual application inference, tiled coordinate mapping/duplicate suppression and rejection of disagreeing retry text. Reserved final test photos remain unused. User photos were used for diagnosis, so are now development examples.
