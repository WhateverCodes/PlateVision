# Verification report
**Final test result: 27 passed in 5.80 seconds.**

## What was exercised
- Python 3.12 on Windows, CPU PyTorch 2.7.1 and torchvision 0.22.1.
- Unit/integration tests for actual grayscale conversion, segmentation of rendered glyphs,
  multi-row ordering, normalization, plausibility, duplicate safety, serials, reset,
  event grouping/filtering, bounding-box association and CSV table structure.
- Both custom networks: tensor shapes, finite losses and gradient propagation.
- Actual short AVI decoding, over-duration rejection and compressed event-frame creation.
- Streamlit AppTest: initial UI, settings, event selector and timestamp button filtering.
- Real manifest preparation and character extraction with inherited parent splits.
- Browser inspection of the rendered dark/neon Streamlit interface.
- Synthetic generator, both training commands, checkpoint reload and resumable training.
- Time-budget exit after a batch, then resume to complete the epoch.
- Explicit general vehicle-model download with publisher hash-prefix verification,
  then local model loading and forward inference.

## Training smoke checks are NOT model accuracy claims
Plate smoke test: 4 training images, 2 validation images, 64px input, 2 CPU threads.
Character smoke test: 72 training glyphs and 36 validation glyphs, one epoch.
These tiny checks verify code execution and saved model formats, not recognition quality.

The smoke character checkpoint scored 1/43 correct characters (about 2.33%) on its tiny
synthetic test split. On a separately generated smoke set with 9 annotated test plates,
the smoke detector's AP50 and end-to-end exact-match rate were 0.0.
The first very small test split contained only negative images; the detector evaluator
correctly returned null AP rather than an invented score.

These poor scores are expected from a few optimization steps and are recorded openly.
None of those checkpoints is included in the release or presented as a usable recognizer.
No real Indian image dataset was trained or benchmarked in this build.
You must train and evaluate the custom models before claiming recognition accuracy.

## Resource handling
Tests and tiny training checks used limited CPU threads. No long training job was run.
Testing dependencies live outside the ZIP. No background training remains.
The app itself does not train, download models or start monitoring when launched.

## Reproduce the checks
From the extracted project:
    python -m pytest -q
Read README.md for the full training/evaluation commands.
Automated tests use controlled generated fixtures; stubbed predictions appear only inside
unit tests and are never used in the application.

A successful test suite proves the tested behavior, not robustness to every camera,
font, license-plate layout, video codec or lighting condition.
