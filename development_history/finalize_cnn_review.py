from pathlib import Path
import json, shutil, zipfile, sys
p=Path('outputs/PlateVision1');sys.path.insert(0,str(p))
from training.evaluate_plate_reader import edit_distance
r=json.loads((p/'outputs/evaluation/final_cnn_round/report.json').read_text())
o=json.loads((p/'models/paddle_plate_adapted_v1/development_evaluation.json').read_text())
metrics={}
for name,rows in [('own_cnn',r['baseline']['plate']['results']),('paddleocr',o['results'])]:
 n=sum(len(x['expected']) for x in rows);e=sum(edit_distance(x['expected'],x['predicted']) for x in rows)
 metrics[name]={'plates':len(rows),'reference_characters':n,'edit_errors':e,'character_error_rate':e/n,'character_score_1_minus_CER':1-e/n,'exact_plates':sum(x['expected']==x['predicted'] for x in rows)}
assert metrics['own_cnn']['edit_errors']==131 and metrics['paddleocr']['edit_errors']==25
report={'scope':'45 repeatedly used development plate crops, all predictions including rejected readings; not independent test accuracy or full-image detection accuracy. Character score is 1-CER, not isolated-glyph classification accuracy.','metrics':metrics,'synthetic_glyph_validation':r['baseline']['synthetic_characters'],'decision':'Freeze existing CNN: neither final learning-rate trial improved the baseline.'}
(p/'outputs/evaluation/character_metrics.json').write_text(json.dumps(report,indent=2))
doc='''# Final bounded CNN review — 3 October 2026

The existing CNN is retained. Two learning rates (0.000003 and 0.00003), each with two 40-step stages, did not improve the baseline. Training used 299 human-verified characters plus 1,008 synthetic characters, balanced sampling, mild geometric augmentation, consistent 32×32 normalization, and frozen BatchNorm running statistics. The architecture already includes BatchNorm and 0.25 dropout; no arbitrary extra layers were added.

| Model | Character score (1 − CER) | Character error rate | Complete-plate exact match |
|---|---:|---:|---:|
| Own CNN + segmentation | 70.2% | 131/440 = 29.8% | 13/45 = 28.9% |
| Adapted PaddleOCR | 94.3% | 25/440 = 5.7% | 31/45 = 68.9% |

These are 45 development plate crops with 440 reference characters, repeatedly used for model selection. Scores include every raw reading, even rejected readings. CER is total Levenshtein edit distance divided by reference character count: substitutions, missing characters and extra characters count as errors. 1−CER is a text-reading score, not the proportion of independently classified character crops. Vehicle/plate localization on full photos is outside this evaluation.

The CNN separately classified 216/216 synthetic validation glyphs correctly. This 100% synthetic result does not measure real-photo character classification. No independent manually labelled real-glyph test has been measured. The 299 reviewed glyphs were used for training and cannot provide that test result.

Verified character coverage is uneven: I and V have zero examples; O, X and Z have one each. Balanced sampling cannot invent missing real examples. The plate failures include missing/extra characters from segmentation as well as classification errors. Existing failure inspection and the earlier segmentation trials are documented in CNN_SEGMENTATION_TRIALS.md.

| Learning rate | Stage | Training loss | Synthetic validation loss | Full plates correct | Wrong accepted |
|---|---:|---:|---:|---:|---:|
'''
for t in r['trials']:
 for c in t['curve']:
  doc+=f"| {t['lr']} | {c['stage']} | {c['train_loss']:.4f} | {c['synthetic_validation']['loss']:.4f} | {c['plate']['exact']}/45 | {c['plate']['accepted_wrong']} |\n"
doc+='\nNo candidate surpassed the retained 13/45 result with at most one wrong accepted reading. Stop this improvement round and retain the current weights. An independent test with complete annotations is the next evaluation task, not another tuning cycle.\n'
(p/'docs/FINAL_CNN_REVIEW.md').write_text(doc,encoding='utf8')
legacy=p/'docs/LEGACY_README.md'
if not legacy.exists():legacy.write_text('# Historical documentation — superseded by the root README\n\n'+(p/'README.md').read_text(encoding='utf8'),encoding='utf8')
(p/'README.md').write_text('''# PLATEVISION

A local educational Indian license-plate recognition project with a black, white and neon Streamlit interface. It compares a character CNN trained from scratch with a pretrained whole-line OCR pipeline.

## Run

On the configured laptop, double-click **OPEN_PLATEVISION.cmd** in this folder and keep its window open. Use **Whole-line OCR (pretrained)** with **Pretrained plate detector** for the main demonstration. Compare models to inspect the own-CNN route.

For a fresh Windows setup, install Python 3.12, open a terminal in this folder, then run:

```powershell
py -3.12 -m venv .venv
.venv\\Scripts\\python.exe -m pip install --upgrade pip
.venv\\Scripts\\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.venv\\Scripts\\python.exe -m streamlit run app.py
```

Model weights are included in the prepared project ZIP; historical notes saying there is no pretrained OCR are superseded. Private local review data and the Python environment are not a portable training dataset.

## How it works

- **Main OCR:** pretrained YOLO plate localization, PP-OCRv5 text detection and English recognition, with a locally adapted final CTC projection. It reads whole lines from colour crops; the displayed black-and-white image is for inspection.
- **Own CNN:** OpenCV plate proposals and grayscale/threshold segmentation, followed by our CNN classifying individual characters. Its recognition weights were trained from scratch. An experimental learned plate detector is also available.
- **Vehicle types:** separate pretrained COCO vehicle detector. Vehicle categories can be wrong even when plate text is correct.
- Both routes include registration-format checks, confidence filtering, results and CSV export. Format checking is not registration-database verification.

## Measured results

| Metric on 45 development plate crops | Own CNN route | Adapted PaddleOCR |
|---|---:|---:|
| Character score, 1 − character error rate | **70.2%** | **94.3%** |
| Character edits / 440 reference characters | 131 | 25 |
| Complete plate exact match | **28.9% (13/45)** | **68.9% (31/45)** |

These repeatedly used development crops are not an independent final test. Reading scores include rejected predictions. They do not measure full-photo vehicle or plate detection. See [the final review](docs/FINAL_CNN_REVIEW.md) and the machine-readable [character metrics](outputs/evaluation/character_metrics.json).

Real isolated-character classification accuracy, vehicle detection precision/recall, plate detection precision/recall, and complete full-photo recognition accuracy are **not yet measured** on fully annotated independent test data. Do not substitute the CNN's 100% synthetic-glyph validation result for real-image performance.

The current OCR acceptance threshold is 0.90. It accepts 29 correct and 7 incorrect readings on these crops (80.6% correctness among accepted readings). At 0.95 it accepts 27 correct and 2 incorrect (93.1%). Lowering a threshold does not improve raw recognition accuracy. See [threshold comparison](docs/OCR_THRESHOLD_090.md).

## Demonstration successes and failures

| Local example | Observed outcome |
|---|---|
| dataset6.jpg | Reads HP885801 and KL43C5336 on two cars |
| dataset8.jpg | Reads WB34CG4538 on a tilted motorcycle plate |
| dataset4.jpg | Outputs DL3SCW6192; user-confirmed label is DL35CW6192 |
| dataset10.jpg | Reads KL02BP7403 but rejects it at 0.90 because its score is about 0.887 |
| dataset9.jpg | Finds four main plate candidates but accepts only one; decorative text causes failures |
| dataset5.jpg | Plate reading succeeds but the SUV is classified as a truck |

These are development examples, not a benchmark. Example photographs are not embedded here because redistribution rights have not been established.

## Final improvement decision

A bounded class-balanced CNN trial with two learning rates did not improve the retained model. Existing normalization, BatchNorm and dropout were inspected; mild augmentation was tested. **The stable CNN is frozen for this project version.** See [trial details](docs/FINAL_CNN_REVIEW.md), [segmentation attempts](docs/CNN_SEGMENTATION_TRIALS.md), and [Paddle adaptation](docs/PADDLE_PROJECTION_UPDATE.md).

Further evaluation should use independent, fully annotated photos and report missed plates as well as incorrect text. A 95% deployment claim is not supported by these development results. This project is intended for learning and demonstration.

## Training and provenance

Training and review tools are included under `training/` and in the Windows launchers. Dataset-dependent trials require the corresponding local manifests and reviewed images. `training/final_cnn_round.py` reproduces the bounded experiment when those inputs exist; it writes an audit report and does not replace deployed weights. Model provenance is shown in the app. Retain bundled model licenses and attribution when sharing.

The [historical guide](docs/LEGACY_README.md) preserves earlier setup and synthetic training instructions; its old model availability and performance descriptions are not current.
''',encoding='utf8')
files=['README.md','docs/LEGACY_README.md','docs/FINAL_CNN_REVIEW.md','docs/CNN_SEGMENTATION_TRIALS.md','docs/CNN_300_REVIEW_RESULTS.md','training/final_cnn_round.py','outputs/evaluation/character_metrics.json','outputs/evaluation/final_cnn_round/report.json']
for target in Path('outputs').glob('PlateVision*'):
 if target==p or not (target/'app.py').exists():continue
 for f in files:
  (target/f).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p/f,target/f)
base=Path('outputs/PLATEVISION_OCR_090_2026-10-03.zip');out=Path('outputs/PLATEVISION_Final_Review_2026-10-03.zip')
with zipfile.ZipFile(base) as old:
 prefix=next(n[:-6] for n in old.namelist() if n.endswith('/app.py') and n.count('/')==1)
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as new:
  for item in old.infolist():
   if item.filename not in {prefix+f for f in files}:new.writestr(item,old.read(item.filename))
  for f in files:new.write(p/f,prefix+f)
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 assert len(z.namelist())==len(set(z.namelist()))
print(json.dumps(report,indent=2));print('Documentation synced and final ZIP verified:',out)
