from pathlib import Path
import shutil,zipfile
p=Path('outputs/PlateVision1')
f=p/'README.md';s=f.read_text(encoding='utf8');s=s.replace('Use **Whole-line OCR (pretrained)** with **Pretrained plate detector** for the main demonstration. Compare models to inspect the own-CNN route.','Recognition automatically uses the pretrained plate detector and adapted PaddleOCR, with no model or processing choices. Compare models shows both readers on the same pretrained plate proposals, sorted by reading confidence, including rejected candidates.');s=s.replace('**Own CNN:** OpenCV plate proposals and grayscale/threshold segmentation','**Own CNN:** shared external pretrained plate proposals and custom grayscale/threshold segmentation');s=s.replace('An experimental learned plate detector is also available.','Older OpenCV and learned detector implementations remain in the source for research, but are not exposed in the main interface.');f.write_text(s,encoding='utf8')
doc='''# Simplified recognition and plate detection

Recognition now always uses pretrained YOLOv9 plate detection plus locally adapted PaddleOCR, with vehicle classification enabled, reading threshold 0.90 and conservative CPU/video defaults. Model selectors, processing settings and the legacy training guide were removed from the main page. Main recognized results remain accepted readings only; rejected readings and intermediate stages remain explicitly diagnostic.

Compare models now uses the same external pretrained plate detector for both readers by default, rather than OpenCV geometric proposals for the own-CNN route. The custom recognizer remains our from-scratch CNN, threshold 0.70. This changes localization, not recognition weights. Already-cropped mode bypasses localization for both readers. The synthetic-trained detector checkbox has been removed from the comparison UI; experimental source remains available for research.

All returned plate candidates, including low-confidence/rejected readings, appear in descending reading-confidence order in each model's table and comparison CSV. Boxes use the same rank numbers as their table. Reading confidence and plate-detection score are shown separately; neither score is measured accuracy. Detector threshold 0.50 still limits which regions become candidates. Sorting does not guarantee every actual plate is found.

This replaces the source of many arbitrary shape proposals with a trained plate-specific detector. A full-photo before/after detection benchmark has not been measured, so no numerical detection gain is claimed. To improve localization further, use reviewed Indian plate bounding boxes, background/non-plate examples and a separate held-out detection evaluation. Avoid requiring a detected vehicle for every plate, as that can miss close-up plates and vehicles the vehicle detector fails to find.

Existing 45-crop reader metrics remain historical reading-only measurements; they are not full-photo accuracy for the new shared-localization comparison. Ten targeted checks passed, including real model/app examples, sorted rejected-candidate retention, shared detector routing and removal of main-workspace choices.

Restart OPEN_PLATEVISION.cmd in PlateVision8. No model downloads or retraining are required.
'''
(p/'docs/SIMPLIFIED_WORKSPACE.md').write_text(doc,encoding='utf8')
f=p/'START_HERE.md';f.write_text('# Current app: automatic Recognition\n\nRestart OPEN_PLATEVISION.cmd. Recognition automatically uses the retained PaddleOCR pipeline. Compare models lists all candidates highest-confidence first, with shared pretrained plate localization. See docs/SIMPLIFIED_WORKSPACE.md.\n',encoding='utf8')
files=['app.py','src/comparison.py','src/comparison_ui.py','tests/test_simple_workspace.py','README.md','START_HERE.md','docs/SIMPLIFIED_WORKSPACE.md']
for target in Path('outputs').glob('PlateVision*'):
 if target==p or not (target/'app.py').exists():continue
 for f in files:
  (target/f).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p/f,target/f)
base=Path('outputs/PLATEVISION_CNN_070_2026-10-03.zip');out=Path('outputs/PLATEVISION_Simplified_2026-10-03.zip')
with zipfile.ZipFile(base) as old:
 prefix=next(n[:-6] for n in old.namelist() if n.endswith('/app.py') and n.count('/')==1)
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as new:
  for item in old.infolist():
   if item.filename not in {prefix+f for f in files}:new.writestr(item,old.read(item.filename))
  for f in files:new.write(p/f,prefix+f)
with zipfile.ZipFile(out) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
print('All project copies synchronized and ZIP verified.')
