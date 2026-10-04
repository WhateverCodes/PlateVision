"""Benchmark the replacement candidate without changing app defaults."""
import json,time
from pathlib import Path
from config import ROOT
from src.runtime import configure
from src.paddle_candidate import PaddleCandidate
from src.preprocessing import read_image
from src.validator import normalize,plausible
from training.evaluate_plate_reader import edit_distance

configure(2)
reader=PaddleCandidate(ROOT/'models/paddle_candidate/english.onnx')
report={}
for dataset in ['accuracy_check_2026_09_21','reader_reviewed_v2']:
    rows=json.loads((ROOT/'data'/dataset/'val.json').read_text());results=[];start=time.perf_counter()
    for r in rows:
        text,scores=reader.read(read_image(Path(r['image'])));text=normalize(text)
        results.append({'id':r['id'],'expected':r['text'],'predicted':text,
          'accepted':bool(scores) and min(scores)>=.75 and plausible(text)})
    report[dataset]={'images':len(rows),'exact':sum(r['expected']==r['predicted'] for r in results),
      'accepted_correct':sum(r['accepted'] and r['expected']==r['predicted'] for r in results),
      'accepted_wrong':sum(r['accepted'] and r['expected']!=r['predicted'] for r in results),
      'character_error_rate':sum(edit_distance(r['expected'],r['predicted']) for r in results)/sum(len(r['expected']) for r in results),
      'seconds':time.perf_counter()-start,'results':results}
print(json.dumps({k:{a:b for a,b in v.items() if a!='results'} for k,v in report.items()},indent=2))
report['note']='Cropped-plate development evaluation only; one-line recognizer, no two-row handling yet. Pretrained data overlap unknown. Original app unchanged.'
(ROOT/'outputs/evaluation/paddle_candidate_2026_10_02.json').write_text(json.dumps(report,indent=2))
