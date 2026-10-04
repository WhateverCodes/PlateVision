from pathlib import Path
import json,shutil,zipfile
p=Path('outputs/PlateVision1');r=json.loads((p/'outputs/evaluation/cnn_threshold_comparison.json').read_text())
s='''# CNN threshold comparison

The current CNN minimum-character threshold is **0.70**. OCR remains at 0.90. The five requested thresholds were evaluated by rerunning the complete segmentation-candidate selection on the same 45 development crops with format decoding enabled. No retraining or new data were used.

| Threshold | Correct accepted | Wrong accepted | Uncertain | Exact full plates |
|---|---:|---:|---:|---:|
'''
for v in r['comparisons']:s+=f"| {v['threshold']:.2f} | {v['correct_accepted']} | {v['wrong_accepted']} | {v['rejected']} | {v['exact']}/45 |\n"
s+='''
Selection maximizes accepted exact-reading F1: 2×correct / (45 + correct + wrong). This balances recovery of correct plates and incorrect accepted readings. At 0.65 and 0.70 it is 32.1%, versus 29.6% at 0.75. The tie was resolved in favour of the threshold closest to the previous 0.75 setting.

This is not a universal best threshold: 0.70 accepts 9 correct and 2 wrong (81.8% correctness among accepted readings), whereas 0.75 accepts 8 correct and 1 wrong (88.9%). Prefer 0.75 manually if avoiding wrong accepted results matters more than coverage.

All five thresholds produced identical raw strings on these crops: 14/45 complete plates correct (31.1%), with 118 edits across 440 reference characters (73.2% character score). The new default changes acceptance, not underlying accuracy, on this evaluation. Development data have been reused for selection; independent-test performance is unknown.

Restart OPEN_PLATEVISION.cmd in PlateVision8. The Recognition default and Compare models now use 0.70 for the own CNN. Older trial documents intentionally retain their original experimental thresholds. Reproduce with `python -m training.evaluate_cnn_thresholds` using the existing local data manifests.
'''
(p/'docs/CNN_THRESHOLD_070.md').write_text(s,encoding='utf8')
f=p/'README.md';s=f.read_text(encoding='utf8');marker='## Demonstration successes and failures';s=s.replace(marker,'## CNN acceptance threshold\n\nThe own-CNN default is now **0.70**, selected from five thresholds using accepted-reading F1 on the development set. It accepts 9 correct and 2 incorrect plates, with 34 uncertain. At the previous 0.75 it accepted 8 correct and 1 incorrect. Raw full-plate and character scores are unchanged. See [the threshold comparison](docs/CNN_THRESHOLD_070.md).\n\n'+marker);f.write_text(s,encoding='utf8')
f=p/'START_HERE.md';f.write_text('# Latest: own-CNN threshold 0.70\n\nRestart OPEN_PLATEVISION.cmd. Own CNN now uses 0.70; OCR remains at 0.90. No training is required. See docs/CNN_THRESHOLD_070.md for the measured trade-off.\n\n'+f.read_text(encoding='utf8'),encoding='utf8')
files=['config.py','src/comparison_ui.py','training/evaluate_comparison.py','training/evaluate_cnn_thresholds.py','outputs/evaluation/cnn_threshold_comparison.json','docs/CNN_THRESHOLD_070.md','README.md','START_HERE.md']
for target in Path('outputs').glob('PlateVision*'):
 if target==p or not (target/'app.py').exists():continue
 for f in files:
  (target/f).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p/f,target/f)
base=Path('outputs/PLATEVISION_Format_Update_2026-10-03.zip');out=Path('outputs/PLATEVISION_CNN_070_2026-10-03.zip')
with zipfile.ZipFile(base) as old:
 prefix=next(n[:-6] for n in old.namelist() if n.endswith('/app.py') and n.count('/')==1)
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as new:
  for item in old.infolist():
   if item.filename not in {prefix+f for f in files}:new.writestr(item,old.read(item.filename))
  for f in files:new.write(p/f,prefix+f)
with zipfile.ZipFile(out) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
print('Synced all project copies; ZIP integrity verified.')
