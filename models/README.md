# Model files
Train your own plate_detector/best.pt and character_classifier/best.pt using the README.
The application does not initialize random inference models when files are missing.
Test-only weights are excluded because they are not usable trained recognition models.

The separate vehicle_detector/ssdlite320.pth is downloaded explicitly from PyTorch:
python -m training.download_vehicle_weights

Do not replace either custom model with downloaded ALPR/OCR weights.
A complete training epoch plus validation is needed before best.pt exists.
last.pt contains a resumable training state. Model files can be large because the optimizer
is included; keep them outside version control and record their dataset provenance.
