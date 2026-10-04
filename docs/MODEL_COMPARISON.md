# My model and pretrained comparison

Restart OPEN_PLATEVISION.cmd and select **Compare models** at the top. Tick **Use comparison example** or upload a still image, then click **Compare both models**. Models run sequentially using two CPU threads. Photos stay local and no training occurs in the app.

Choose **One cropped plate** to compare readers on the same unmodified crop. Choose the vehicle option for separate full pipelines. The own route defaults to OpenCV localization plus our own character CNN; its trained plate detector can be selected explicitly. The external route uses YOLOv9-t 384 plus FastPlateOCR CCT-S v2 global. The comparison does not run vehicle classification; the Recognition workspace retains vehicle types and video processing.

Optional correct text only scores results; it never guides inference. Separate multiple registrations with commas. Counts refer to distinct accepted registration strings, not matched vehicle identities. Outputs from the models are never merged. Rejected raw guesses remain visible and CSV export includes their status. Confidence is not an accuracy percentage.

## Measured development results

Same confidence threshold 0.75. No final-test photos used. Potential overlap between pretrained training data and these public datasets is unknown.

| Plate crop set | Own exact | Pretrained exact | Own accepted correct / wrong | Pretrained accepted correct / wrong |
|---|---:|---:|---:|---:|
| Newly reviewed 30 | 6/30 | 13/30 | 4 / 2 | 7 / 3 |
| Original difficult 15 | 6/15 | 4/15 | 2 / 0 | 2 / 1 |

Reader inference averaged roughly 16–24 milliseconds per crop on this laptop, excluding model loading. These small development sets do not establish 95% accuracy.

On seven full development photos (12 source plate boxes), the own OpenCV route matched four boxes and the pretrained detector matched seven, using one-to-one IoU >= 0.5. Own accepted readings: one verified correct and one without a source transcription. External accepted readings: zero verified correct, one incorrect and one without a source transcription. Source annotations are incomplete. Finding and reading plates are separate problems; the external pipeline is not yet reliable on Indian plates.

## Own model improvement experiment

Kept the tilt-aware reading-order fix. Also trained a separate own CNN candidate on 1,509 glyphs: human-reviewed training registrations (54 accepted plate records with automatic glyph alignment), synthetic replay and mild rotation/blur augmentation. Four complete epochs took about 20 seconds with two CPU threads. This used our own initial CNN weights, no external pretrained OCR weights or evaluation images.

The candidate read 4/30 exactly versus 6/30 for the current own checkpoint, so it was not promoted. The current model remains intact. Training code and the experiment report are included. Further work should focus on missing character segmentation and representative, verified training glyphs; use the existing development sets and leave final tests reserved.

## External models

- OCR: https://github.com/ankandrew/fast-plate-ocr — CCT-S v2 global ONNX. Maximum 10 characters, colour input, no independent character segmentation. Its supplied region list does not include India; India-specific training coverage is not asserted. Region outputs are unused.
- Detector: https://github.com/ankandrew/open-image-models — YOLOv9-t 384 end-to-end plate detector.
- Runtime: CPU ONNX Runtime 1.23.2. Selected external models are about 13 MB together. Their source URLs and SHA-256 hashes are in models/pretrained/manifest.json; upstream MIT notices accompany them.
- Older nine-slot OCR models were considered but not integrated because common Indian registrations need ten characters.

Model files are bundled in the comparison ZIP; Python dependencies are not. On another machine install requirements using README.md. SETUP_PRETRAINED.cmd installs the CPU runtime and retrieves missing model files. Local setup is complete. No training launchers are needed to compare models.
