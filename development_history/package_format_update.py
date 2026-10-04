from pathlib import Path
import json,shutil,zipfile
p=Path('outputs/PlateVision1');report=json.loads((p/'outputs/evaluation/format_aware_comparison.json').read_text())
doc='''# Format-aware decoding update

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
'''
(p/'docs/FORMAT_AWARE_UPDATE.md').write_text(doc,encoding='utf8')
f=p/'README.md';s=f.read_text(encoding='utf8');s=s.replace('**70.2%** | **94.3%**','**73.2%** | **95.2%**').replace('| 131 | 25 |','| 118 | 21 |').replace('**28.9% (13/45)** | **68.9% (31/45)**','**31.1% (14/45)** | **75.6% (34/45)**')
start=s.index('The current OCR acceptance threshold');end=s.index('\n## Demonstration',start)
s=s[:start]+'''The current OCR acceptance threshold is 0.90. After format-aware decoding, it accepts 29 correct and 3 incorrect readings on these crops (90.6% correctness among accepted readings); 13 remain uncertain. Confidence is not increased by format correction. See [before/after decoding results](docs/FORMAT_AWARE_UPDATE.md). Earlier threshold-only results in the historical notes predate this change.
'''+s[end:]
s=s.replace('**The stable CNN is frozen for this project version.**','**The stable CNN weights are frozen for this project version.** A subsequent format-aware decoding change improved full-plate readings from 13 to 14/45 for the CNN and 31 to 34/45 for OCR, without retraining.')
s=s.replace('See [the final review](docs/FINAL_CNN_REVIEW.md)','See [the current decoding review](docs/FORMAT_AWARE_UPDATE.md) and [the earlier training review](docs/FINAL_CNN_REVIEW.md)')
f.write_text(s,encoding='utf8')
f=p/'START_HERE.md';f.write_text('# Latest: format-aware reading update\n\nRestart OPEN_PLATEVISION.cmd. No training or downloads are needed. Both reading routes now use supported Indian plate formats to resolve model ambiguities. See docs/FORMAT_AWARE_UPDATE.md for measured results and limitations.\n\n'+f.read_text(encoding='utf8'),encoding='utf8')
f=p/'outputs/evaluation/character_metrics.json';old=json.loads(f.read_text());old['previous_metrics']=old['metrics'];old['metrics']={}
for name,key in [('cnn','own_cnn'),('paddleocr','paddleocr')]:
 a=report['models'][name]['after'];old['metrics'][key]={'plates':45,'reference_characters':440,'edit_errors':a['character_errors'],'character_error_rate':a['character_errors']/440,'character_score_1_minus_CER':a['character_score'],'exact_plates':a['exact']}
old['decoding_update']='Format-aware decoding enabled; weights unchanged. See format_aware_comparison.json for paired results.';f.write_text(json.dumps(old,indent=2))
files=['src/format_decoder.py','src/character_classifier.py','src/paddle_candidate.py','src/whole_line_reader.py','tests/test_format_decoder.py','training/evaluate_format_decoding.py','docs/FORMAT_AWARE_UPDATE.md','README.md','START_HERE.md','outputs/evaluation/format_aware_comparison.json','outputs/evaluation/character_metrics.json']
for target in Path('outputs').glob('PlateVision*'):
 if target==p or not (target/'app.py').exists():continue
 for file in files:
  (target/file).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p/file,target/file)
base=Path('outputs/PLATEVISION_Final_Review_2026-10-03.zip');out=Path('outputs/PLATEVISION_Format_Update_2026-10-03.zip')
with zipfile.ZipFile(base) as oldzip:
 prefix=next(n[:-6] for n in oldzip.namelist() if n.endswith('/app.py') and n.count('/')==1)
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as new:
  for item in oldzip.infolist():
   if item.filename not in {prefix+f for f in files}:new.writestr(item,oldzip.read(item.filename))
  for file in files:new.write(p/file,prefix+file)
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 assert len(z.namelist())==len(set(z.namelist()))
print('Updated PlateVision1–8 and verified ZIP:',out)
