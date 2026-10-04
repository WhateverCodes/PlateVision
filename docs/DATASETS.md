# Dataset decisions
Reviewed 19 September 2026. No real dataset is bundled.

## Indian Vehicle License Plate Dataset
Source: https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset
The listing describes approximately 1.5k Indian images and XML plate annotations.
An explicit usable license was not verified from the accessible page in this review.
Do not automatically treat its images as permitted for redistribution or commercial use.
It is a candidate for a college experiment after the downloaded version's terms are checked.
It is small for detector training from random weights and lacks a guaranteed complete
character-level ground-truth set. Use the VOC adapter after inspecting its exact format.

## Indian License Plates with Labels (Kedar)
Source: https://www.kaggle.com/datasets/kedarsai/indian-license-plates-with-labels/data
The hosting listing declares CC0: Public Domain and references another Kaggle source.
The exact image count and annotation format of the downloadable version were not audited.
CC0 is permissive in its intended scope, but the uploader's label alone does not establish
all upstream photograph rights. Save the version and upstream terms before using it.
Do not silently download this candidate, infer character labels, or report a guessed count.

## UFPR-ALPR (optional research reference, outside Indian-only training scope)
Source: https://github.com/raysonlaroca/ufpr-alpr-dataset
4,500 images from 150 vehicles; more than 30,000 annotated plate characters.
PNG images; per-image text annotations describe vehicle, plate and character boxes/text.
Brazilian plates, not Indian. Official split: 40% training, 20% validation, 40% testing.
Academic research only, for researchers at educational/research institutions, noncommercial,
under its agreement. Access conditions and redistribution rules must be followed.
This is not an unrestricted public portfolio/commercial dataset.
The project's user requested Indian plates, so it is not used by default.

## Indian_LPR in-the-wild research dataset
Source: https://github.com/sanchit2843/Indian_LPR
The authors describe 16,192 images and 21,683 plates but explicitly say the dataset
cannot be made public because of legalities. Do not base an automatic setup on it.
Do not download the repository's pretrained plate/OCR weights.

## Our synthetic data
Generated locally by training/generate_synthetic_data.py. Default: 300 scenes and
1,440 glyphs (40 per each of 36 classes). Counts are configurable.
Annotations are exact generated boxes/text for scenes and labels for glyphs.
Indian-style strings and one/two-line layouts; not actual registration verification.
OpenCV Hershey rendering is used, with no bundled third-party font files or scraped images.
The generated images and original project code may be used under this project's MIT terms,
subject to the third-party software notices. No real-world training result is implied.

For a college demonstration, start small, check the full pipeline, then obtain permission-cleared
real Indian photos and labels. Keep vehicle/capture groups together and evaluate on unseen vehicles.
