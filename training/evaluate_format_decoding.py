"""Compare identical model weights with and without format-aware decoding."""
import json
from dataclasses import replace
from pathlib import Path
import cv2
from config import DEFAULTS
from src.runtime import configure
from src.character_classifier import Classifier
from src.whole_line_reader import WholeLineReader
from src.pipeline import Pipeline
from src.classical_detector import PlateCropDetector
from training.evaluate_plate_reader import edit_distance

def main():
 configure(2)
 rows=json.loads(Path('data/accuracy_check_2026_09_21/val.json').read_text())+json.loads(Path('data/reader_reviewed_v2/val.json').read_text())
 report={'scope':'45 existing development plate crops, 440 reference characters; no independent test or full-photo detection evaluation. Identical weights and thresholds before/after.','models':{}}
 for name,reader,mode,threshold in [('cnn',Classifier(DEFAULTS.adapted_character_path),'characters',.75),('paddleocr',WholeLineReader(DEFAULTS.whole_line_folder),'whole_line',.90)]:
  versions={}
  for enabled in (False,True):
   reader.format_aware=enabled
   pipe=Pipeline(replace(DEFAULTS,reader_mode=mode,localization_mode='crop',rectify=False,character_threshold=threshold),PlateCropDetector(),reader)
   results=[]
   for r in rows:
    d=pipe(cv2.imread(r['image']))[0]
    results.append({'id':r['id'],'expected':r['text'],'predicted':d['raw_text'],'accepted':d['recognized'],'score':d['confidence']})
   errors=sum(edit_distance(r['expected'],r['predicted']) for r in results)
   versions['after' if enabled else 'before']={'exact':sum(r['expected']==r['predicted'] for r in results),'accepted_correct':sum(r['accepted'] and r['expected']==r['predicted'] for r in results),'accepted_wrong':sum(r['accepted'] and r['expected']!=r['predicted'] for r in results),'character_errors':errors,'character_score':1-errors/sum(len(r['expected']) for r in results),'results':results}
   print(name,enabled,{k:v for k,v in versions['after' if enabled else 'before'].items() if k!='results'},flush=True)
  versions['changes']=[{'id':a['id'],'expected':a['expected'],'before':b['predicted'],'after':a['predicted']} for b,a in zip(versions['before']['results'],versions['after']['results']) if b['predicted']!=a['predicted']]
  report['models'][name]=versions
 out=Path('outputs/evaluation/format_aware_comparison.json');out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
