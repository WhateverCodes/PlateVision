# Character-separation experiments — 3 October 2026

Target: 18/45 exact plate readings (40%) on the existing development set. Result: not achieved.

| Experiment | Exact / 45 | Correct accepted | Wrong accepted |
|---|---:|---:|---:|
| Existing app | 13 | 8 | 1 |
| More threshold candidates | 14 | 10 | 2 |
| Thresholds + CNN alternatives | 14 | 10 | 2 |
| Character stroke straightening | 14 | 10 | 4 |
| Whole-plate tilt correction | 14 | 10 | 2 |
| Preserve edge-touching components | 12 | 9 | 2 |
| Reconnect broken vertical strokes | 14 | 10 | 4 |

Previously saved CNN checkpoints were also evaluated with revised crops; none exceeded 14/45. No pretrained OCR was substituted into the own-CNN route. No evaluation labels were used for training or hard-coded corrections. These are repeatedly tuned development results, not independent final-test results.

No experimental pipeline was deployed because the small exact-match gain came with more false acceptances. The working app's original character segmenter and classifier were restored. CNN weights and all 300 character reviews remain intact. Existing checkpoints and review data were not overwritten. Experimental scripts and outputs remain available locally for analysis.

The next structural step is to measure character recognition separately on trustworthy, independently cropped glyphs, then address merged/split components using actual segmentation annotations. More thresholds and epochs alone did not reach the target. No claim that 40% or a particular number of additional reviews will guarantee improvement.
