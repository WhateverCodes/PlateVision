# Simplified recognition and plate detection

Recognition now always uses pretrained YOLOv9 plate detection plus locally adapted PaddleOCR, with vehicle classification enabled, reading threshold 0.90 and conservative CPU/video defaults. Model selectors, processing settings and the legacy training guide were removed from the main page. Main recognized results remain accepted readings only; rejected readings and intermediate stages remain explicitly diagnostic.

Compare models now uses the same external pretrained plate detector for both readers by default, rather than OpenCV geometric proposals for the own-CNN route. The custom recognizer remains our from-scratch CNN, threshold 0.70. This changes localization, not recognition weights. Already-cropped mode bypasses localization for both readers. The synthetic-trained detector checkbox has been removed from the comparison UI; experimental source remains available for research.

All returned plate candidates, including low-confidence/rejected readings, appear in descending reading-confidence order in each model's table and comparison CSV. Boxes use the same rank numbers as their table. Reading confidence and plate-detection score are shown separately; neither score is measured accuracy. Detector threshold 0.50 still limits which regions become candidates. Sorting does not guarantee every actual plate is found.

This replaces the source of many arbitrary shape proposals with a trained plate-specific detector. A full-photo before/after detection benchmark has not been measured, so no numerical detection gain is claimed. To improve localization further, use reviewed Indian plate bounding boxes, background/non-plate examples and a separate held-out detection evaluation. Avoid requiring a detected vehicle for every plate, as that can miss close-up plates and vehicles the vehicle detector fails to find.

Existing 45-crop reader metrics remain historical reading-only measurements; they are not full-photo accuracy for the new shared-localization comparison. Ten targeted checks passed, including real model/app examples, sorted rejected-candidate retention, shared detector routing and removal of main-workspace choices.

Restart OPEN_PLATEVISION.cmd in PlateVision8. No model downloads or retraining are required.
