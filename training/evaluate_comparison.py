"""Paired cropped-plate development evaluation, never a claim of final accuracy."""
import argparse
import json
import time
from pathlib import Path
from config import DEFAULTS
from src.runtime import configure
from src.preprocessing import read_image
from src.comparison import build_pipelines,compare
from training.evaluate_plate_reader import edit_distance
from training.data_tools import save_json


def evaluate(data,out):
    configure(2)
    rows=json.loads(Path(data).read_text())
    models=build_pipelines(DEFAULTS,cropped=True)
    results={name:[] for name in models}
    for row in rows:
        paired=compare(read_image(Path(row['image'])),models)
        for name,value in paired.items():
            d=value['detections'][0]
            results[name].append({'id':row['id'],'expected':row['text'],'predicted':d['raw_text'],
                                  'accepted':d['recognized'],'score':d['confidence'],'seconds':value['seconds']})
    report={'dataset':str(data),'images':len(rows),'models':{}}
    for name,values in results.items():
        report['models'][name]={'exact':sum(v['expected']==v['predicted'] for v in values),
          'accepted_correct':sum(v['accepted'] and v['expected']==v['predicted'] for v in values),
          'accepted_wrong':sum(v['accepted'] and v['expected']!=v['predicted'] for v in values),
          'character_error_rate':sum(edit_distance(v['expected'],v['predicted']) for v in values)/sum(len(v['expected']) for v in values),
          'mean_seconds':sum(v['seconds'] for v in values)/len(values),'results':values}
    report['note']=f'Same input crops; own CNN minimum-glyph gate {DEFAULTS.character_threshold}, whole-line OCR mean-token gate {DEFAULTS.ocr_threshold}. Models run sequentially with two CPU threads. Development evaluation only; pretrained training-data overlap is unknown. Not full-photo or final-test accuracy.'
    save_json(Path(out),report)
    print(json.dumps({**report,'models':{n:{k:v for k,v in m.items() if k!='results'} for n,m in report['models'].items()}},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();evaluate(a.data,a.out)
