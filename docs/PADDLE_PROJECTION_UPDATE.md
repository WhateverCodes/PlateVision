# PaddleOCR final-layer adaptation — 3 October 2026

The app now prefers models/paddle_plate_adapted_v1 when installed. Original models/paddle_candidate weights remain available for rollback. Our own CNN was not changed.

This bypasses the unavailable training-checkpoint download by extracting the final CTC projection from the existing pretrained ONNX graph. Its input features stay frozen. We verified that the extracted linear layer reproduces the original model probabilities, then trained that layer with PyTorch CTC loss on 76 screened human-labelled training lines. This is final-layer adaptation of pretrained PaddleOCR, not full-network fine-tuning and not training our own CNN.

The separate development line subset contains 26 crops. The baseline read 15 exactly; the adapted candidate read 16. Eight short epochs were evaluated; epoch 1 was the first best candidate. The unblended model increased wrong accepted readings in the complete pipeline, so a conservative 50% original / 50% adapted parameter update was evaluated and selected.

| Same 45 development plate crops | Before | Selected update |
|---|---:|---:|
| Complete plate exact | 31 (68.9%) | 31 (68.9%) |
| Correct accepted | 25 | 27 |
| Incorrect accepted | 2 | 2 |
| Rejected | 18 | 16 |

The confidence threshold remains 0.95; it was not lowered. This is improved acceptance on development data, not increased exact recognition accuracy or an independent 95% claim. The 26-line subset is part of these development images. Reserved final photos were not used. Pretrained overlap and capture independence remain uncertain.

16 targeted app tests passed, including real-model inference and comparison. The user's photos 6 and 8 remain accepted correctly; photos 3 and 10 still read correctly but remain uncertain. Image 4 still disagrees with the user's confirmed label. The vehicle classifier is unchanged.

Restart OPEN_PLATEVISION.cmd in your existing project folder. No download, training clicks or new character reviews are needed on this laptop.

Developer reproduction: install ONNX tooling in a separate training environment alongside the project's PyTorch, NumPy and OpenCV dependencies, then run python -m training.adapt_paddle_projection --out models/new_projection_trial. Requires the prepared local data/ocr_finetune_audit_v1 training and validation files, which are not included in the project ZIP. The script refuses to overwrite a trial directory, uses two CPU threads and a 120-second gradient-update budget, and exports a conservative blended recognizer only if line-level validation improves. Full pipeline evaluation is still required before deployment. Source hashes, model provenance and detailed results accompany the adapted model.
