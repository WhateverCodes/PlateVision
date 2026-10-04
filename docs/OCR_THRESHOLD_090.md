# OCR acceptance threshold: 0.90

User-requested setting applied to Recognition and Compare models. Own CNN remains at minimum character score 0.75. OCR model weights and image processing are unchanged.

Rescoring the same 45 saved adapted-OCR development predictions:

| Metric | 0.95 | 0.90 |
|---|---:|---:|
| Exact full plate readings | 31/45 (68.9%) | 31/45 (68.9%) |
| Correct accepted | 27 | 29 |
| Wrong accepted | 2 | 7 |
| Rejected | 16 | 9 |
| Correctness among accepted readings | 93.1% | 80.6% |

Lowering the threshold adds two correct accepted plates and five wrong accepted plates. It does not improve the OCR's actual exact-match accuracy. These are development results, not independent final-test accuracy.

Restart OPEN_PLATEVISION.cmd in the existing project folder. Check Minimum reading confidence is 0.90. You can return it to 0.95 in Recognition if you prefer fewer incorrect accepted readings. The original 0.95 references in earlier update documents describe historical results.
