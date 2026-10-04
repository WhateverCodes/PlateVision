> Historical first-upgrade results below. For current OCR results and instructions, see OCR_REFINEMENT.md.

# Whole-line OCR is now the main recognition route

Restart OPEN_PLATEVISION.cmd and refresh the browser. Processing settings now default to **Whole-line OCR (pretrained)** and **Pretrained plate detector** when their files are installed. The existing Recognition workspace supports images, sampled videos and vehicle types. **Compare models** runs your own CNN separately alongside the new reader. To use your own model alone, select **Real-image character model (pilot)** and **Image processing (experimental)**.

The main route is now: pretrained plate detection -> colour plate crop -> pretrained text-line detection -> line straightening -> whole-line CTC recognition -> registration format and score checks. It does not segment or classify individual character crops. Two-row text is read top-to-bottom and left-to-right. A separate IND marking is omitted. Punctuation separators and spaces are removed without guessing replacement letters. If no lines are detected, the reader processes the full plate crop.

All inference runs locally, sequentially, with two CPU threads. No paid API, photo uploads or training are required to launch. Model files are bundled in the updated ZIP. Python dependencies still need installation on another machine. SETUP_PRETRAINED.cmd retrieves missing external models and installs their CPU runtime.

## Measured development comparison

| Crop set | Own CNN exact | Whole-line OCR exact | Own accepted correct / wrong | Whole-line accepted correct / wrong |
|---|---:|---:|---:|---:|
| 30 newer crops | 6/30 | 14/30 | 4 / 2 | 12 / 2 |
| 15 difficult crops | 6/15 | 10/15 | 2 / 0 | 7 / 0 |

Exact means every registration character matches, regardless of confidence rejection. The own CNN retains its minimum-character threshold of 0.75. Whole-line OCR uses mean emitted-token confidence (as in the upstream OCR convention) with a conservative default threshold of 0.95. These differently defined scores are not comparable probabilities or measured accuracy. The threshold is adjustable. Results are development measurements; pretrained training-set overlap is unknown. They do not establish 95% accuracy or 99% detection. Final reserved photographs remain unused.

The default app recognized the full scooter development example KL07BX7197, with vehicle detection enabled. Ten targeted tests passed, including actual app inference, comparison mode, line ordering, confidence/cleanup handling and a check that the whole-line route never invokes character segmentation. The example is a development demonstration, not an independent accuracy claim. The reader averaged approximately 0.05–0.09 seconds per crop in the paired checks, excluding model loading.

## Models and sources

External weights: PaddleOCR PP-OCRv5 mobile text detector and English recognizer, using RapidAI/RapidOCR v3.9.2 ONNX exports. Downloads and SHA-256 hashes are recorded in models/paddle_candidate/manifest.json. Apache license notices accompany the models. Plate localization retains the existing YOLOv9-t 384 model from ankandrew/open-image-models.

- https://github.com/PaddlePaddle/PaddleOCR
- https://github.com/RapidAI/RapidOCR
- https://github.com/ankandrew/open-image-models

Your own CNN weights and training history are unchanged. Improving that model remains a separate task using verified training glyphs. This upgrade must be described as integration of pretrained models, not training the OCR models from scratch.
