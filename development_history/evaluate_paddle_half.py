import sys,json,shutil,time
from pathlib import Path
from dataclasses import replace
root=Path.cwd();p=root/'outputs/PlateVision1';sys.path.insert(0,str(p))
from config import DEFAULTS
from src.runtime import configure
from src.pipeline import Pipeline
from src.classical_detector import PlateCropDetector
from src.whole_line_reader import WholeLineReader
from training.evaluate_plate_reader import edit_distance
configure(2);folder=p/'models/paddle_head_half_trial_v1';shutil.copy2(p/'models/paddle_candidate/text_detector.onnx',folder/'text_detector.onnx')
reader=WholeLineReader(folder);cfg=replace(DEFAULTS,reader_mode='whole_line',localization_mode='crop',rectify=False,character_threshold=.95);pipe=Pipeline(cfg,PlateCropDetector(),reader)
rows=json.loads((p/'data/accuracy_check_2026_09_21/val.json').read_text())+json.loads((p/'data/reader_reviewed_v2/val.json').read_text())
import cv2
results=[]
for r in rows:
 d=pipe(cv2.imread(r['image']))[0];results.append({'id':r['id'],'expected':r['text'],'predicted':d['raw_text'],'accepted':d['recognized'],'score':d['confidence']})
report={'images':len(rows),'exact':sum(r['expected']==r['predicted'] for r in results),'accepted_correct':sum(r['accepted'] and r['expected']==r['predicted'] for r in results),'accepted_wrong':sum(r['accepted'] and r['expected']!=r['predicted'] for r in results),'character_errors':sum(edit_distance(r['expected'],r['predicted']) for r in results),'results':results}
(folder/'full_pipeline_evaluation.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='results'})
