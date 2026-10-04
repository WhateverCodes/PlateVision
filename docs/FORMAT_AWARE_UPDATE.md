# Format-aware decoding update

Both readers now use a bounded Indian-registration prior to resolve ambiguous model-supported readings. Model weights, CPU limits and confidence thresholds remain unchanged. This is decoding, not retraining or registration-database verification.

## Rules and safeguards

- Known state/UT prefixes, including historical aliases, are preferred; regular layouts allow 1–2 district digits, 0–3 series letters and 1–4 serial digits. BH layouts are handled separately.
- Existing supported readings are preserved within each token sequence. Multiple CNN segmentation candidates can still compete as before.
- Repairs require the model's alternative probability, at least 0.02 and 5% of the best token probability. At most two tokens are changed by the constrained decoder. State positions can use alternative letters; other positions allow only defined confusion groups (0/O/Q, 1/I/L, 2/Z, 5/S, 6/G, 8/B).
- No missing characters are inserted, no characters deleted, no ground-truth labels used during inference, and probabilities are not boosted after correction.
- Existing OCR alternative-beam handling remains; known-state filtering improves its candidate selection. Unsupported/special registrations fall back to the previous reading rather than being forced into a common layout. The state list is a reading prior, not an exhaustive legal validator.
- A valid-looking ambiguity such as DL3SCW6192 versus DL35CW6192 cannot be settled by the format alone. Both fit the supported patterns; no hard-coded correction was added for that example.

## Paired comparison

Same 45 development crops and model weights, 440 reference characters. CNN threshold 0.75; OCR threshold 0.90. All predictions, including rejected readings, count toward character and full-plate scores.

| Metric | CNN before | CNN after | OCR before | OCR after |
|---|---:|---:|---:|---:|
| Exact whole plates | 13/45 (28.9%) | 14/45 (31.1%) | 31/45 (68.9%) | 34/45 (75.6%) |
| Character score (1 − CER) | 70.2% | 73.2% | 94.3% | 95.2% |
| Correct accepted | 8 | 8 | 29 | 29 |
| Incorrect accepted | 1 | 1 | 7 | 3 |
| Rejected/uncertain | 36 | 36 | 9 | 13 |

These are repeatedly used development data, not independent test performance. The OCR's 95.2% character score does not mean 95.2% of complete plates are correct. Some corrected text remains uncertain because the actual alternative probability is low. No previously exact plate became incorrect in this comparison, although some already-wrong CNN strings changed unfavourably.

Examples: CNN 6J15CD0564 → GJ15CD0564; OCR KH01CY6333 → MH01CY6333 and WH01JA8939 → MH01JA8939. The report records every changed reading, including remaining errors.

## Verification and use

21 targeted tests passed, covering alternative probability preservation, unsupported formats, valid ambiguity preservation, BH/no-series layouts, OCR line ordering, pipeline integration and app examples. Rerun the paired benchmark from the project directory with `python -m training.evaluate_format_decoding`; it requires the existing local development manifests and images. Reserved test images were not used.

Restart OPEN_PLATEVISION.cmd in your existing PlateVision8 folder. If its terminal is running, stop it with Ctrl+C and launch again. No new model download or training is needed.

## Sources for the format prior

- [MoRTH registration-mark appendix](https://morth.gov.in/sites/default/files/Appendices%20.pdf): state/UT prefixes (historical list).
- [West Bengal government format advisory](https://comtax.wb.gov.in/GST/GST_e-Waybill/Advisory_on_Vehicle_No.pdf): state, district, series and serial components, including omitted district zeros.
- [Government BH-series announcement](https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=1749764&lang=2&reg=48): separate BH format.
- [Delhi Transport registration guidance](https://transport.delhi.gov.in/transport/vehicle-registration): Delhi registration examples.

This heuristic is deliberately permissive for older plates and does not certify legal validity.
