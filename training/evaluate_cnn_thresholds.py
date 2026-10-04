"""Bounded development-only threshold comparison; reruns candidate selection."""
import json
from pathlib import Path
from dataclasses import replace
import cv2
from config import DEFAULTS
from src.runtime import configure
from src.character_classifier import Classifier
from src.pipeline import Pipeline
from src.classical_detector import PlateCropDetector
from training.evaluate_plate_reader import edit_distance

def main():
 configure(2)
 records=json.loads(Path('data/accuracy_check_2026_09_21/val.json').read_text())+json.loads(Path('data/reader_reviewed_v2/val.json').read_text())
 reader=Classifier(DEFAULTS.adapted_character_path);reports=[]
 for threshold in (.65,.70,.75,.80,.85):
  pipe=Pipeline(replace(DEFAULTS,reader_mode='characters',localization_mode='crop',rectify=False,character_threshold=threshold),PlateCropDetector(),reader)
  rows=[]
  for r in records:
   d=pipe(cv2.imread(r['image']))[0]
   rows.append(dict(id=r['id'],expected=r['text'],predicted=d['raw_text'],accepted=d['recognized'],score=d['confidence']))
  correct=sum(r['accepted'] and r['expected']==r['predicted'] for r in rows);wrong=sum(r['accepted'] and r['expected']!=r['predicted'] for r in rows)
  # Correct accepted readings are true positives; rejected/incorrect references are false negatives.
  f1=2*correct/(len(rows)+correct+wrong)
  errors=sum(edit_distance(r['expected'],r['predicted']) for r in rows)
  result=dict(threshold=threshold,exact=sum(r['expected']==r['predicted'] for r in rows),correct_accepted=correct,wrong_accepted=wrong,rejected=len(rows)-correct-wrong,accepted_precision=correct/(correct+wrong) if correct+wrong else 0,recognition_f1=f1,character_errors=errors,character_score=1-errors/sum(len(r['expected']) for r in rows),results=rows)
  reports.append(result);print({k:v for k,v in result.items() if k!='results'},flush=True)
 # Maximize recognition F1; tie-break fewer wrong accepts, more exact, then closeness to prior .75.
 best=max(reports,key=lambda r:(r['recognition_f1'],-r['wrong_accepted'],r['exact'],-abs(r['threshold']-.75)))
 report=dict(scope='45 repeatedly used development crops; not independent test performance. Full segmentation selection rerun at each threshold. Format decoding enabled; weights unchanged.',selection_rule='Maximize accepted exact-reading F1 (2*correct_accepted/(45+correct_accepted+wrong_accepted)); ties: fewer wrong accepts, more raw exact plates, closest to previous 0.75.',selected_threshold=best['threshold'],comparisons=reports)
 Path('outputs/evaluation/cnn_threshold_comparison.json').write_text(json.dumps(report,indent=2))
 print('Selected:',best['threshold'])
if __name__=='__main__':main()
