from pathlib import Path
import json,sys,shutil,zipfile
p=Path('outputs/PlateVision1');sys.path.insert(0,str(p));from src.validator import plausible_ocr
rows=json.loads((p/'models/paddle_plate_adapted_v1/development_evaluation.json').read_text())['results'];report=[]
for threshold in (.95,.90):
 accepted=[r for r in rows if r['score']>=threshold and plausible_ocr(r['predicted'])];correct=sum(r['expected']==r['predicted'] for r in accepted)
 report.append({'threshold':threshold,'images':45,'exact':31,'correct_accepted':correct,'wrong_accepted':len(accepted)-correct,'rejected':45-len(accepted),'accepted_precision':correct/len(accepted)})
(p/'outputs/evaluation/ocr_threshold_comparison.json').write_text(json.dumps({'method':'Rescored saved adapted-OCR predictions; recognition weights and predictions unchanged. Development data only.','results':report},indent=2))
doc='''# OCR acceptance threshold: 0.90

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
'''
(p/'docs/OCR_THRESHOLD_090.md').write_text(doc,encoding='utf8');start=p/'START_HERE.md';start.write_text('# Latest: OCR threshold 0.90\n\nRestart OPEN_PLATEVISION.cmd. The lower threshold accepts more correct and more incorrect readings; see docs/OCR_THRESHOLD_090.md for the measured comparison.\n\n'+start.read_text(encoding='utf8'),encoding='utf8')
files=['config.py','app.py','src/comparison.py','src/comparison_ui.py','training/evaluate_comparison.py','docs/OCR_THRESHOLD_090.md','outputs/evaluation/ocr_threshold_comparison.json','START_HERE.md']
for t in Path('outputs').glob('PlateVision*'):
 if t==p or not (t/'app.py').exists():continue
 for f in files:(t/f).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p/f,t/f)
base=Path('outputs/PLATEVISION_Paddle_Update_2026-10-03.zip');out=Path('outputs/PLATEVISION_OCR_090_2026-10-03.zip')
with zipfile.ZipFile(base) as old:
 prefix=next(n[:-6] for n in old.namelist() if n.endswith('/app.py') and n.count('/')==1)
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as new:
  for item in old.infolist():
   if item.filename not in {prefix+f for f in files}:new.writestr(item,old.read(item.filename))
  for f in files:new.write(p/f,prefix+f)
with zipfile.ZipFile(out) as z:assert z.testzip() is None
print('Updated all existing project copies; ZIP verified.')
