# Historical documentation — superseded by the root README

# PLATEVISION
## AI-Based License Plate Recognition System

A local, educational Python computer-vision project for **Indian registration plates**.
Black and bright-white Streamlit UI with neon cyan accents. Built for an Intel Core Ultra 7
155U / 32 GB Windows laptop using a conservative CPU profile.

**Delivery status:** complete source, dataset tools, training, inference, UI and tests.
The two custom models must be trained before recognition is available. No pretrained
plate detector, OCR engine, invented results, or real-world accuracy claim is included.
Small training checks verified the machinery; their test-only checkpoints are deliberately
excluded from the distribution. The UI can be opened before training.

## Opening after setup (no PowerShell needed)
Double-click **OPEN_PLATEVISION.cmd** inside this folder. It opens the browser and
starts the app if needed. Keep the launch window open while using the app.
If the app is already running, the launcher simply opens its browser page.
The launcher uses the existing tested environment on this laptop, or the project's
.venv when the folder is moved or extracted on another computer.
Installation and model training are separate; this launcher does neither.

## Reviewing the imported Kaggle dataset
Double-click **REVIEW_DATASET.cmd** to check supplied plate boxes and enter registration
text. Start with 10-20 images; reviews are saved between sessions. This does not train
or change any model. See [the import and review guide](docs/KAGGLE_IMPORT.md) for the
audit results and the remaining steps before real-image training.

After checking plate registrations, double-click **REVIEW_CHARACTERS.cmd** to check individual letters and digits. See [the character review guide](docs/CHARACTER_REVIEW.md).

## Quick start on Windows
Install Python **3.11 or 3.12 (64-bit)** from https://www.python.org/downloads/windows/.
Extract the ZIP, open its PlateVision folder in VS Code or another editor, and open a
PowerShell terminal in that folder. Do not run from inside the ZIP.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Use `py -3.11` in the first line if you installed 3.11. No activation or PowerShell
execution-policy change is required. The app listens only on **127.0.0.1:8501**.
Open http://127.0.0.1:8501 in your browser. Stop it with Ctrl+C in its terminal.

Commands below use `python` for readability; on Windows you can always substitute
`.\.venv\Scripts\python.exe`.

## Laptop resource limits
- Inference and training: **2 PyTorch CPU threads**, OpenCV: 1 thread.
- Training data workers: **0**. Images are read on demand, not cached in RAM.
- Detector default input: **320 x 320**, batch 2; character batches: 32.
- Default training time budget: **10 minutes per invocation**, then save and exit.
- Optional lower load: `--threads 1 --batch-size 1 --max-minutes 2 --cooldown 0.2`.
- Budget is checked between batches; a current batch/checkpoint save can extend it.
- No background training, startup dataset download, or automatic GPU use.
- Video inference samples about 2 frames/second and retains only compressed event frames.
- 100 MB uploads, 24 MP still images, 4K video decode ceiling, 9,000 decoded frames,
  100 events, and 30 seconds per video by default.
- These settings reduce load but do not guarantee a temperature or fan-speed limit.
  Close the task if the laptop becomes uncomfortable. Training real models can take
  many short sessions. Keep the same batch size when resuming a checkpoint.
- The Intel integrated GPU/NPU is not used. CUDA is optional for a different NVIDIA system;
  install the matching PyTorch build from https://pytorch.org/get-started/locally/.

## Features
- Own multi-scale, multi-plate detector trained from random initialization.
- Own 36-class character CNN trained from random initialization.
- OpenCV grayscale, thresholding, contour segmentation, row ordering and optional rectification.
- Indian-format plausibility including common state registrations and BH-style plates.
- Separate pretrained general vehicle detector for car, motorcycle, bus, truck.
- Image and short-video input; session-only serials, event frames, filters and CSV.
- Reproducible manifests, grouped splits, training histories and evaluation plots.
- No OCR service, database, login, face recognition or permanent vehicle tracking.

## Architecture and computer-vision pipeline
```text
Input image / sampled video frame
 -> grayscale + optional configurable enhancement
 -> aspect-preserving resize/padding
 -> OUR PlateDetector
 -> plate boxes + detector scores
 -> crop original image and cautiously rectify a visible quadrilateral
 -> Otsu/adaptive threshold and foreground polarity candidates
 -> connected components, size filtering and row grouping
 -> OUR CharacterClassifier on centered 32x32 masks
 -> row-major ordering, text reconstruction and conservative normalization
 -> minimum-character-score gate and Indian format plausibility
 -> session registry, serials, numbered boxes, event frames, table
```

Vehicle detection operates separately on the original color frame. It never proposes
plate text or replaces our plate detector. A plate is associated with the smallest vehicle
box that contains at least 80% of its area; a small boundary tolerance is allowed.
If no credible association exists, vehicle type is **Unknown**.

## Model architecture
### Plate detector
Five backbone stages, each a stride-2 3x3 convolution then stride-1 3x3 convolution,
BatchNorm and SiLU. Channels: 1 -> 32 -> 64 -> 128 -> 256 -> 256.
A top-down 128-channel feature pyramid uses strides 8, 16 and 32. Shared two-layer
classification/regression towers predict one plate logit, four positive box distances,
and centerness. Focal loss, generalized IoU loss, and centerness BCE train the heads.
Targets use center sampling and scale ranges [0,64), [64,128), [128,infinity).
Non-maximum suppression removes overlapping candidates.

The default 320 input is a deliberate laptop setting; `--image-size 640` is available
when training a new run. Inference reads the image size from the checkpoint.
Small/distant plates can be missed at 320. Do not change input size mid-resume.

### Character model
1x32x32 -> Conv32/BN/SiLU/Pool -> Conv64/BN/SiLU/Pool ->
Conv128/BN/SiLU/Pool -> Flatten -> Linear256/ReLU/Dropout0.25 -> Linear36.
Cross-entropy loss; softmax for character scores. The alphabet is stored in each checkpoint
and dataset's alphabet.json. Both O/I and 0/1 are real classes.

### General vehicle model
Torchvision SSDLite320 MobileNet V3 Large, pretrained on COCO. This is explicitly a
third-party general detector, **not our plate or character model**.
Its download is an explicit setup command:
```powershell
python -m training.download_vehicle_weights
```
The approximately 14 MB file is verified against the publisher's SHA-256 filename prefix.
No model is downloaded during app inference. See THIRD_PARTY_NOTICES.md.

## Dataset and licensing
Read **docs/DATASETS.md** before using real data. The default runnable dataset generator
produces our own synthetic scenes and Hershey-font glyphs. No real photographs or
third-party fonts are bundled. Synthetic samples are visibly synthetic and must not be
used to claim real-road accuracy.

No real Indian dataset is bundled or downloaded automatically. A public hosting page is
not by itself proof that the uploader owns every photograph. Record the exact dataset
version, source and permissions in the source manifest. This is a college learning project.

### Generate a small starter dataset
```powershell
python -m training.generate_synthetic_data --scenes 300 --chars-per-class 40
```
Creates data/synthetic/plates and data/synthetic/characters, each with train.json, val.json,
test.json, images/ and a dataset card. This is enough to exercise training, not a guarantee
of a usable recognizer. For meaningful results, train on more varied real annotated images.

### Prepare real plate data
Create a source JSON manifest as described in docs/DATA_FORMAT.md:
```powershell
python -m training.prepare_plate_dataset --manifest data/raw/source.json --out data/indian
```
This validates images and boxes, copies images, and writes group-based splits.
It does not resize the stored originals: resizing occurs in the shared data loader,
with bounding boxes transformed consistently.

For Pascal VOC XML:
```powershell
python -m training.import_voc --annotations data/raw/annotations --images data/raw/images --groups data/raw/groups.csv --source "Dataset name and version" --license "Verified terms" --class-name number_plate --out data/raw/source.json
```
Inspect class names and the converted annotations before training. The converter assumes
standard one-based inclusive VOC coordinates. A source-specific zero-based export needs
its coordinates adjusted accordingly. Include a reviewed vehicle/capture group mapping;
do not label every neighboring video frame as a different group.

### Prepare real character data
The source manifest can contain full-image character boxes and labels. These pass through
plate preparation. Extract the glyphs **after** splitting the parent photographs:
```powershell
python -m training.prepare_character_dataset --data data/indian --out data/indian_characters
```
Do not automatically label noisy crops using predicted text; manually verify labels.
For datasets with only plate boxes, character labels must be annotated separately.

## Double-click practice training
Double-click **TRAIN_MODELS.cmd** in this folder. It creates a small synthetic dataset
if needed, then trains the plate detector and character classifier sequentially.
Each model gets a two-minute budget with two CPU threads. A current batch or save can
extend this slightly. The first dataset-generation step is additional.
Double-click it again to resume saved last.pt checkpoints. It targets 30 total epochs.
A best.pt file is produced after a full training epoch and validation.
The launcher does not download the separate pretrained vehicle model.

This is a practice workflow, not validated real-photo recognition.
No training starts until you run the launcher. Close the app before training to reduce load.
For real Indian data, use the dataset and training instructions below.

## Training the plate detector
```powershell
python -m training.train_plate_detector --data data/synthetic/plates --epochs 30 --max-minutes 10
```
Random initialization, AdamW, configurable mild augmentation, validation and atomic
checkpoints. Logs are printed per 10 batches and completed epoch.
`models/plate_detector/last.pt` is the resumable checkpoint.
`best.pt` is created only after an entire validation pass completes.

Resume with the same data, batch size, image size, seed and augmentation configuration:
```powershell
python -m training.train_plate_detector --data data/synthetic/plates --epochs 30 --max-minutes 10 --resume models/plate_detector/last.pt
```
Epochs means total requested epochs, not extra epochs. A paused validation pass is rerun.
Ctrl+C saves interrupted.pt where possible; because interruption can occur inside an update,
prefer the last.pt checkpoint for reliable continuation.

For real data, start a new run and output folder, e.g. `--data data/indian --out models/plate_real`.
Then update detector_path in config.py to that run's best.pt.
Resume refuses a different manifest. This project does not silently warm-start from
third-party models or switch datasets behind the user's back.

## Training the character classifier
```powershell
python -m training.train_character_classifier --data data/synthetic/characters --epochs 20 --max-minutes 10
python -m training.train_character_classifier --data data/synthetic/characters --epochs 20 --max-minutes 10 --resume models/character_classifier/last.pt
```
Use data/indian_characters and a new output directory for a real-data run.
Keep its alphabet consistent with the intended validation formats.

The checkpoint records architecture type, alphabet, input size, optimizer state, trained
steps, data provenance, manifest fingerprint, configuration, random states and history.
The loader rejects missing, untrained and incompatible checkpoints. Only load checkpoints
from trusted sources; the app uses PyTorch's restricted weights-only loading.

## Evaluation
Do not tune on the test set. Pick thresholds on validation data, then evaluate once.
```powershell
python -m training.evaluate_plate_detector --data data/indian --weights models/plate_real/best.pt
python -m training.evaluate_character_classifier --data data/indian_characters --weights models/character_real/best.pt
python -m training.evaluate_pipeline --data data/indian --detector models/plate_real/best.pt --characters models/character_real/best.pt
```
Outputs:
- Detector: precision/recall at the declared operating score, AP50, 101-point mAP50:95,
  mean matched IoU, precision-recall plot, sample prediction images.
- Character: accuracy, per-class and macro precision/recall/F1, labeled confusion matrix.
- End to end: complete-plate exact-match rate over **all annotated plates**, including
  missed detections and failed segmentation; false positives and failure examples.
- Loss histories and loss.png live alongside each training run.
- Evaluation reports live in outputs/evaluation by default; each evaluation supports --out.

These are model scores, not calibrated probabilities. The UI uses the **minimum**
character score to avoid concealing a weak character behind a high average.
The sample 94.2% in the project idea is not a measured result of this project.

## Running PlateVision and image workflow
1. Open the Streamlit app.
2. Load our trained best.pt files in the configured directories and download the separate vehicle weights.
3. Upload an image and press **Analyze media**. Processing is never automatic.
4. See numbered boxes and the table. Gray unlabeled boxes indicate unrecognized candidates.
5. Expand the processing view for crops, threshold masks, glyphs and raw uncertain output.
6. Export the displayed table as CSV.

Results table columns are exactly Sr No., Vehicle Type, License Plate.
For videos, Timestamp is added. Confidence belongs in the inspection view, not the table.

## Video workflow, events and filtering
The file is first decoded in a bounded validation pass. Overlong files are rejected,
not trimmed. A second streaming pass runs inference on sampled frames.
MP4, AVI, MOV and MKV decoding depends on your local OpenCV codecs.
Browser playback may require MP4/H.264 even when OpenCV can decode the file.

One event is created when a processed frame contains at least one new accepted plate.
The event includes **all confidently recognized plates visible in that frame**, including
existing serials. Repeated plates alone do not create events.
Example: frame 100 -> A and B -> one event; frame 120 -> A and B -> no event;
frame 250 -> C -> one event.

Each event stores exact seconds, frame number, FPS and one compressed annotated image.
Timestamp buttons display that actual event image and filter the table. They do **not**
claim to seek the browser player. Choose All timestamps to return to the original video.
Two events inside the same second have distinct internal IDs and selector labels.

All timestamps displays event memberships, so an old serial can appear in a later event
alongside a new plate. Unique plate count remains separate. CSV exports the current filter.
Loading new media or clearing results resets the registry and serials. No tracking algorithm
or persistent vehicle database is involved. Temporary video files are deleted after processing.

## Configuration and preprocessing experiments
config.py holds limits, paths and model thresholds. The UI exposes common inference controls.
Changing a threshold does not silently recompute existing results; press Analyze again.
Detector preprocessing can enable CLAHE, bilateral filtering and sharpening.
Keep the detector's inference preprocessing consistent with its training data:
the provided training recipe uses plain grayscale, so leave optional enhancements off
unless you train/evaluate that variant deliberately.
The experimentation notebook compares OpenCV preprocessing operations.

Indian validation is a configurable set of regular expressions in src/validator.py.
It is a plausibility test, not legal verification. Special, diplomatic, military, temporary
or unusual registrations may be rejected. No blind O->0 or I->1 substitution is performed.

Fuzzy matching is off by default. When enabled it requires the same length, first four
characters and numeric sequence, and rejects ambiguous matches. Distinct registrations
can still be merged; demonstrate this limitation in your viva.

## Project structure
- app.py: UI orchestration, safe temporary media handling, session state.
- config.py: centralized conservative defaults.
- src/architectures.py: both custom neural networks and coordinate helpers.
- src/preprocessing.py, character_segmenter.py: image processing.
- src/plate_detector.py, character_classifier.py: checkpoint-backed inference.
- src/vehicle_detector.py, association.py: separate vehicle model and association.
- src/pipeline.py: complete image recognition.
- src/timestamp_manager.py: registry, serials, events, filtering.
- src/video_processor.py: bounded streaming processing.
- src/visualization.py, ui_style.py: overlays and UI theme.
- training/: synthetic generator, real-data importers, losses, engine and evaluators.
- tests/: preprocessing, segmentation, ordering, validation, events, models, data, video and UI.
- docs/: dataset decisions, manifest format, viva notes, verification report.
- models/: model destinations and instructions; no downloaded plate/OCR weights.
- notebooks/: preprocessing experimentation notebook.
- samples/: clearly labeled generated scene for inspection.

## Tests
```powershell
python -m pytest -q
```
Tests include multiple new plates -> one event, repeats -> no event, stable serials,
reset behavior, fuzzy safety, event filtering, actual video decoding/duration rejection,
gradient flow for both models, AP accounting, and the Streamlit initial state.
Tests use generated local data; none require a real vehicle database.

## Troubleshooting
- **py/python not found:** install 64-bit Python 3.11/3.12; reopen the terminal.
- **Missing model:** finish a full training+validation epoch so best.pt exists.
- **Time budget stops before best.pt exists:** resume last.pt with identical settings.
- **No recognition:** check crops/segmentation, training provenance and held-out metrics.
  A synthetic starter model is not sufficient for arbitrary real images.
- **App runs, vehicle weights missing:** run the explicit download command, or temporarily
  disable vehicle detection in Processing settings; type will be Unknown.
- **File cannot play:** convert to constant-frame-rate H.264 MP4. Damaged clips may fail validation.
- **High fan noise:** stop training; use one thread and longer cooldown on the next run.
- **Download stalls with a cache issue:** retry pip with --no-cache-dir. Do not disable TLS checks.
- **Port already used:** add --server.port 8502 to the Streamlit command.
- **BatchNorm error:** keep detector image size at least 64; use the default 320 profile.
- **Different dataset on resume:** begin a new run/output directory.
- **Unsupported special Indian plate:** extend patterns and supply labeled examples.

## Limitations and future improvements
The delivered project is an educational training system, not a validated deployment.
Classical segmentation struggles with touching glyphs, logos, shadows and tiny or heavily
tilted plates. Rectification only works when a credible border can be found.
Frame sampling can miss brief appearances. Vehicle types can be wrong or Unknown.
Bounding boxes are axis-aligned; the detector does not learn plate corners.
The synthetic generator uses procedural scenes and a few Hershey font families; those
families overlap between splits. Synthetic metrics therefore do not measure unseen real fonts.
Group splitting uses exact decoded-image hashes, capture groups and known text identities;
near-duplicate photographs require manual review/grouping.
Possible extensions: licensed real data, corner supervision, segmentation-quality rejection,
better confidence calibration, broader fonts, and additional Indian layouts.

## Ethical / privacy considerations
Use your own or permission-cleared material for college learning and testing.
Media is processed locally. No owner lookup, person identification, surveillance feed,
permanent history database or cross-session vehicle tracking is implemented.
CSV export is an explicit user action; you control exported files.

## References
- Original learning inspiration: https://medium.com/devcenter/developing-a-license-plate-recognition-system-with-machine-learning-in-python-787833569ccd
- Torchvision vehicle implementation: https://docs.pytorch.org/vision/main/_modules/torchvision/models/detection/ssdlite.html
- Dataset sources and terms: docs/DATASETS.md

## Whole-plate text review
Double-click **REVIEW_OCR_TEXT.cmd** to check imported OCR labels. See [instructions](docs/OCR_TEXT_REVIEW.md). This prepares annotations only; the new recognizer is not yet integrated.
