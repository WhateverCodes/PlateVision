# Platevision

An educational Indian license-plate recognition project with a local Streamlit interface. It compares our character CNN, trained from scratch, with an adapted pretrained PaddleOCR reader.

This is the **dataset-free final distribution**. Source code, trained weights, experimental checkpoints, training/evaluation scripts, documentation and aggregate results are included. Dataset archives, training/validation/test images, labels, per-image evaluation records, review records and sample photographs are not included.

## Run

On the configured laptop, extract this folder directly into the project's `outputs/` directory, close any older PLATEVISION command window, and double-click **OPEN_PLATEVISION.cmd**. Keep the window open and upload your own vehicle photograph.

For a fresh Windows setup, install Python 3.12 and run these commands from this folder:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

The supplied ZIP includes actual model binaries. If cloning a Git LFS repository instead, retrieve the model files with `git lfs pull` before running the app. Datasets are unnecessary for inference on your own uploaded images. Training, dataset-dependent evaluation and manual-review tools require separately imported data. Built-in dataset photographs are intentionally absent.

## Two readers

- **Recognition:** pretrained YOLO plate localization followed by PP-OCRv5 mobile text detection and English whole-line recognition. The final CTC projection was adapted locally; the pretrained feature network was retained. OCR acceptance threshold: **0.91**.
- **Compare models:** the same pretrained plate proposals go to both the whole-line reader and our character CNN. The CNN uses grayscale/threshold preprocessing and character segmentation. Its recognition weights originate from our own training, not a downloaded OCR checkpoint. CNN acceptance threshold: **0.70**.
- A separate pretrained COCO SSDLite detector supplies vehicle categories. Category errors and plate-reading errors are different failures.
- Registration-format correction and confidence filtering do not verify registration ownership or official database validity.

## Current results

| Evaluation | Reader | Complete-plate exact match | Character score (1 − CER) |
|---|---|---:|---:|
| 45 development crops / 440 characters | Active own CNN | 31.1% (14/45) | 73.2% |
| 45 development crops / 440 characters | Active PaddleOCR | 75.6% (34/45) | 95.9% |
| Additional 54 validation crops / 517 characters | Active own CNN | 31.5% (17/54) | 72.0% |
| Additional 54 validation crops / 517 characters | Active PaddleOCR | 79.6% (43/54) | 94.4% |

These crop-reading measurements include rejected predictions. Character score counts edit errors, including insertions and deletions; it is not isolated-glyph classification accuracy. The first set was repeatedly used for tuning; the additional set has source labels that were not all manually verified. These are not certified independent-test or full-photo end-to-end accuracy claims. The project has not demonstrated 95% complete-plate accuracy.

The expanded-data CNN gained some exact plates but made more character errors, so the earlier CNN remains active. The updated OCR improved raw reading modestly; confidence filtering did not improve uniformly. See [combined training results](docs/COMBINED_DATASET_TRAINING.md). Historical documents describe earlier versions and sometimes refer to local files deliberately omitted from this distribution.

## Dataset sources — download separately

The links below identify known sources. Downloading an upstream dataset does not recreate the project's exact reviewed labels, splits and filtered subsets automatically; the full local backup preserves those. Check each dataset's stated terms at its source.

| Local archive reference | Dataset / source | Role |
|---|---|---|
| archive (2), duplicate archive (5) | [Indian License Plates with Labels — kedarsai](https://www.kaggle.com/datasets/kedarsai/indian-license-plates-with-labels) | Plate localization data; screened subset used. |
| archive (3), duplicate archive (9) | [Indian Number Plates Dataset — Datacluster Labs](https://www.kaggle.com/datasets/dataclusterlabs/indian-number-plates-dataset) | Vehicle photographs and plate boxes. |
| archive (4) | [Indian Vehicle Number Plate OCR Labeled Dataset — kp00011](https://www.kaggle.com/datasets/kp00011/indian-vehical-number-plate-ocr-labeled-dataset) | Plate crops and registration text. User-supplied source URL. |
| archive (6) | **ADD ORIGINAL DATASET LINK** — source archive had 1,698 images and 1,697 XML annotations | Indian plate boxes and text. Exact source URL was not recorded. |
| archive (7) | **ADD ORIGINAL DATASET LINK** — 35,500 augmented printed-character images | Character training; 35 classes, class O missing. Exact source URL was not recorded. |
| archive (8) | [EMNIST — official NIST dataset page](https://www.nist.gov/itl/products-and-services/emnist-dataset) | 1,440 official-training handwriting examples sampled. Original downloaded mirror URL was not recorded. |
| archive (10) | **ADD ORIGINAL DATASET LINK** — 4,613 images with LP bounding boxes | Pakistan plate detection data; 3,913 official train images used, 700 test images withheld. Exact source URL was not recorded. |
| 3113449.zip | **Source unknown / file unavailable** | Not used in training; do not list it as a trained-on dataset. |
| Own synthetic data | Generated by `training/generate_synthetic_data.py` | Practice data and synthetic replay, not real-world test evidence. |

Replace the three **ADD ORIGINAL DATASET LINK** entries with your actual source pages before publishing. No source URL has been guessed for unidentified archives. Other dataset links in historical research notes are suggestions, not proof that those datasets were used.

## Manual verification

The separate [verification_confirmation](verification_confirmation/README.md) folder keeps the review tools and summary: 300 checked characters (299 usable, one excluded), 200 plate-text checks and 11 dataset reviews. Actual review images and labels are excluded alongside the datasets.

## Repository layout

```text
app.py, config.py          Final application and settings
src/                      Detection, preprocessing, reading and UI logic
training/                 Data preparation, training and evaluation code
models/                   Active models and preserved experiment checkpoints
tests/                    Unit and application checks
docs/                     Technical notes and historical experiment reports
outputs/                  Aggregate reports without per-image datasets
verification_confirmation/ Review tools and aggregate review counts
development_history/      Historical scripts and model experiments
data/, datasets/          Documentation placeholders; no datasets
```

Historical scripts may use original machine paths or recreate earlier configurations. They are preserved as development records; do not run old deployment scripts over the finalized app. No new training is required to test recognition.

## Publishing on GitHub

Extract the ZIP and publish its contents, not the ZIP itself. Model checkpoints are retained, so this remains a substantial package even without datasets. `.gitattributes` configures `.pt`, `.pth` and `.onnx` files for Git LFS. Install Git LFS and run `git lfs install` in your repository before adding the files; commit `.gitattributes` with the project. People cloning the repository need `git lfs pull` to retrieve model weights.

GitHub's browser uploader limits individual files to 25 MiB, so use Git with LFS for the included checkpoints. See [GitHub's large-file guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) and [Git LFS configuration](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage). Dataset folders are ignored by `.gitignore` except their README placeholders.

## Attribution

Original project code uses the included LICENSE. Dependencies and pretrained/adapted weights retain their upstream terms; see THIRD_PARTY_NOTICES.md and the licences preserved beside the models. PaddleOCR and pretrained plate/vehicle detection are external components; the independent character CNN is the from-scratch route.
