# Real-data whole-plate reader: development pilot

The new compact CNN and bidirectional GRU predict a sequence of characters using CTC loss. Weights start randomly; no pretrained OCR model is loaded. Images are converted to grayscale and resized to 32 x 192. This initial version does not rearrange two-line plates; that remains an evaluation concern.

## Dataset snapshot
reader_v1 has 791 training images (67 human-reviewed and 724 source-labeled but unverified), 15 human-reviewed validation images and 13 human-reviewed internal test images. Related original labels, corrected registration identities, exact image hashes and four close visual pairs are grouped before splitting. Capture independence is not guaranteed. 469 flagged examples await review; 7 excluded examples are unused. 395 unverified examples assigned to evaluation groups are withheld, not used for scoring or training. The separate 10 user test photos remain reserved and were not evaluated.

## Run a bounded session
Double-click TRAIN_REAL_READER.cmd. Each run uses 2 CPU threads, batch size 8, no data-loader workers, and a two-minute training budget, with a short rest between batches. The current batch and checkpoint save may extend that slightly. It resumes automatically from models/plate_reader_pilot/last.pt. Ctrl+C saves at interruption where possible. Only run one training session at a time. The original TRAIN_MODELS.cmd remains synthetic practice training.

best.pt is saved only after a completed validation pass. Reaching 30 epochs is a training limit, not proof of recognition accuracy. Later changes to the data require a new snapshot/output directory.

## Status
The standalone reader and resumable trainer are implemented and tested. The first real-data pilot has begun. The default app settings still use the previous detector and character classifier. To try the new route, use Processing settings > Plate reader > Whole-plate reader (experimental). Choose Already-cropped plate image only when uploading an actual single-plate crop. For whole-vehicle scenes, image-processing detection is experimental: development checks covered only 17 of 52 annotated plates at IoU 0.5, so it is not a reliable substitute yet. The app now offers the whole-plate reader and image-processing proposals as experimental settings, plus an explicit already-cropped-image mode. End-to-end reliability and final packaging remain. Existing model files were not overwritten.

Current measured results are in PROGRESS_2026_09_20.md. The reader has not achieved usable validation accuracy; additional epochs alone are not a verified solution.
