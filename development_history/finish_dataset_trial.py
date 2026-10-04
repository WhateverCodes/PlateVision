from pathlib import Path
import json,shutil
p=Path('outputs/PlateVision1');data=p/'data/dataset_trial_2026_10_03';audit=json.loads((data/'audit.json').read_text());trial=json.loads((p/'models/external_glyph_trial_2026_10_03/report.json').read_text());best=max(trial['trials'],key=lambda r:(r['exact'],-r['character_errors']))
report={'audit':audit,'character_trial':trial,'decision':'Keep deployed CNN: external-glyph trial increased best exact count but reduced character score and correct accepted coverage. No model promoted. OCR unchanged.','archive_5':'SHA256-identical to previous archive (2).zip; not reimported.','archive_3113449':'Initially listed training.tsv and testing.tsv; archive disappeared before content inspection. Awaiting current location.'}
(p/'outputs/evaluation/new_dataset_trial.json').write_text(json.dumps(report,indent=2))
doc='''# New dataset audit and bounded trial

## Files

| Archive | Contents | Outcome |
|---|---|---|
| archive (5).zip | 2,083 plate images and YOLO box labels | Entire ZIP SHA256 matches previously supplied archive (2).zip. Not new training data. |
| archive (6).zip | 1,698 images, 1,697 XML annotations containing plate text and boxes | Prepared 387 candidate crops; source text requires review. Quarantined 1,310 annotation records: 1,304 known registrations/related video frames, 3 unmatched images, 3 questionable labels. |
| archive (7).zip | 35,500 labelled augmented character images, 35 classes | Missing O. Prepared 100 examples per present class (3,500 total), inspected normalization and ran a bounded trial. |
| 3113449.zip | Initially listed training.tsv and testing.tsv | Disappeared from Downloads before contents could be inspected; not imported or trained. |

Archive 6 overlaps known registrations even when image bytes differ. The exclusion screens known train/evaluation registration labels and related video groups. This does not certify all near-duplicates or capture independence. Source-provided text is not human-verified. Archive 7 lacks original plate/group provenance; a random split of augmented copies would not demonstrate independent test accuracy. No such test claim is made.

## CNN trial

3,500 new characters were mixed with the existing 1,307 verified/synthetic characters. Equal dataset mixture, class-balanced sampling, two learning rates, three 60-step stages each, two CPU threads. Existing architecture and normalization retained, BatchNorm statistics frozen. Existing model weights were preserved.

| Metric | Deployed baseline | Candidate with most exact plates |
|---|---:|---:|
| Exact plates | 14/45 (31.1%) | 16/45 (35.6%) |
| Character score (1 − CER) | 73.2% | 68.2% |
| Correct accepted at 0.70 | 9 | 6 |
| Wrong accepted at 0.70 | 2 | 0 |

The external data did not deliver a clear overall improvement. **The candidate was not promoted.** All six trial checkpoints and the full report were preserved locally. OCR weights are unchanged. These measurements use the existing development crops, not an independent test set.

## Next useful manual check

Open `data/dataset_trial_2026_10_03/REVIEW_24_PLATES.html` in the canonical PlateVision1 folder. Check each displayed crop against its supplied registration. Reply with `number → corrected text`, mark unreadable examples, or confirm the displayed labels. The page is read-only and does not save answers or start training. The 24 crops are drawn from the 387 candidates, not existing test images.

Dataset source links/licenses have not been supplied for these archives. Imported data and trial weights are local only, not added to the distributable project ZIP. Source provenance and usage terms should be established before redistribution.
'''
(p/'docs/NEW_DATASET_TRIAL.md').write_text(doc,encoding='utf8')
for target in Path('outputs').glob('PlateVision*'):
 if target==p or not (target/'app.py').exists():continue
 for name in ['docs/NEW_DATASET_TRIAL.md','outputs/evaluation/new_dataset_trial.json','training/trial_external_characters.py']:
  (target/name).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p/name,target/name)
print('Audit saved; deployed models and project ZIP unchanged.')
