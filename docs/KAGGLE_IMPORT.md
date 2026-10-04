# Imported Kaggle dataset: review before training
## Local import
The provided archive was inspected and imported into data/kaggle_indian_review.
Its original ZIP and the existing synthetic models were not modified.
No training or automatic train/test split was started.

Source: https://www.kaggle.com/datasets/kedarsai/indian-license-plates-with-labels
The hosting listing declares CC0, but notes web-collected photos and model-generated
annotations. No license file was included in this ZIP. Preserve the source/version
information; upstream photograph rights have not been independently established.

## Audit findings
- 2,083 image files; 2,021 label files.
- 62 original images have no matching label file.
- 951 exact duplicate copies were removed by comparing decoded image pixels.
- One image, images/00000044.jpg, was excluded because EXIF rotation could misalign its boxes.
- 1,131 unique decoded images retained locally.
- 1,101 unique images have structurally usable boxes, totaling 1,248 boxes.
- 30 unique images are quarantined because no usable box was supplied.
- A geometric heuristic flags 908 labeled images as plate close-ups and 193 as scenes/collages.
  These counts are not a manually verified classification of all photos.
- Four visually similar candidate pairs were grouped provisionally. Other related images
  can remain; true registration/capture grouping still requires review.

Some duplicates have no label but their identical counterpart does; its annotation is used
only when available labels agree. Missing/empty labels are NOT treated as negative examples.
The full audit is data/kaggle_indian_review/audit.json.
Original filenames and duplicate membership are preserved in catalog.json.

## What this collection can and cannot do
The sampled PNGs are primarily already-cropped plates. They are useful for studying real
fonts and segmentation, but do not by themselves teach localization on whole-vehicle scenes.
The smaller JPG set includes plate product pictures, collages, article graphics and some
vehicles. Inspect these rather than counting all JPGs as road-scene training data.

The labels contain class 0 and normalized center/size coordinates only.
They do not supply registration text or individual character labels.
We convert them to pixel boxes for our custom architecture; no YOLO model is used.

## Review with no PowerShell
1. Double-click REVIEW_DATASET.cmd inside the PlateVision folder.
2. Select an image on the left.
3. Check the supplied green boxes; type each registration into the Registration column.
4. Do not include the small IND marking. Read two-line plates top row first.
5. If characters cannot be read, mark Unreadable instead of guessing.
6. Choose Full vehicle scene, Plate close-up, Collage / article graphic, or Exclude.
7. Check the related-image group. The same vehicle/capture should use the same group.
8. Tick the confirmation box and select Save review.
Start with 10-20 images. You do not need to finish the whole collection at once.

The page shows both the original image and plate crops.
Coordinates can be edited when a supplied box is incorrect; add/remove rows only to
correct missing/incorrect plates. Quarantined images are available through the filter.
If you are unsure how to correct a box, leave it unreviewed and ask for help.

Reviews save to data/kaggle_indian_review/reviews/<image-id>.json.
They persist when you close/reopen the app. The original catalog remains unchanged.
The export button produces a source manifest with reviewed boxes/text, not a ready-made
character dataset. It references local images in this collection.

## Next stages
After a first batch is reviewed, inspect segmentation on its real plate crops and correct
individual character boxes/labels. Split related source images together BEFORE extracting
character datasets. Keep crops and their original photographs in the same split.
Use full vehicle scenes to evaluate detector performance, and keep an untouched external
test set such as permission-cleared photos collected independently.
Do not use dataset2.jpeg as a training image and then claim an independent test result on it.

## Reproduce the import
Install the existing project requirements, then run:
    python -m training.import_kaggle_yolo --archive "path/to/archive.zip" --out data/new_review_collection

The importer refuses to overwrite an existing collection and rejects unsafe archive paths.
It streams images without running archive contents. It requires no network access.
The review app currently opens data/kaggle_indian_review.

## Validation
Importer tests cover normalized coordinate conversion, multiple boxes, malformed/nonfinite
labels, invalid classes, unsafe ZIP paths, duplicates and missing labels.
Review tests cover confirmation, text requirements, disk persistence and the Streamlit form.
