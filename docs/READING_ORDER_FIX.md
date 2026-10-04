# Reading-order fix — 21 September 2026

Character rows now account for plate tilt before sorting. This prevents sloping single-line registrations from being read in fragments from right to left. The method estimates tilt from character positions and does not use expected registration text.

Same model, confidence threshold and evaluation labels:

| Development set | Exact before | Exact after | Accepted correct before/after | Accepted wrong before/after |
|---|---:|---:|---:|---:|
| Newly reviewed crops | 5/30 | 6/30 | 3 / 4 | 2 / 2 |
| Original difficult crops | 5/15 | 6/15 | 2 / 2 | 0 / 0 |

This is a small improvement; recognition is still unreliable. These are cropped-plate development results, not final-test or full-photo accuracy. Missing characters and character substitutions remain unresolved. Extra threshold candidates were tested and rejected because they worsened results.

Eight tests passed, including left/right tilt, two-line ordering, confidence-based candidate selection, pipeline checks and actual application inference on the development example. All three local project copies have the fix. Existing model weights remain unchanged. No new user labelling is needed for this fix. Restart the application to load updated Python code.
