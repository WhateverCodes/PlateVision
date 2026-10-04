"""Compare a candidate against the active reader on the same development crops."""
import argparse,json,shutil
from pathlib import Path
from dataclasses import replace
from config import DEFAULTS
from src.runtime import configure
from src.pipeline import Pipeline
from src.classical_detector import PlateCropDetector
from src.whole_line_reader import WholeLineReader
from src.preprocessing import read_image
from training.evaluate_plate_reader import edit_distance

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folder',type=Path,required=True);args=parser.parse_args()
    configure(2)
    shutil.copy2(DEFAULTS.whole_line_folder/'text_detector.onnx',args.folder/'text_detector.onnx')
    rows=json.loads(Path('data/accuracy_check_2026_09_21/val.json').read_text())+json.loads(Path('data/reader_reviewed_v2/val.json').read_text())
    report={'scope':'45 development plate crops, repeatedly used for selection; not independent-test or full-photo accuracy.','models':{}}
    for name,folder in [('baseline',DEFAULTS.whole_line_folder),('candidate',args.folder)]:
        pipe=Pipeline(replace(DEFAULTS,reader_mode='whole_line',localization_mode='crop',rectify=False,character_threshold=.90),PlateCropDetector(),WholeLineReader(folder))
        results=[]
        for r in rows:
            d=pipe(read_image(Path(r['image'])))[0]
            results.append(dict(id=r['id'],expected=r['text'],predicted=d['raw_text'],accepted=d['recognized'],score=d['confidence']))
        metric=dict(images=len(rows),characters=sum(len(r['expected']) for r in results),exact=sum(r['expected']==r['predicted'] for r in results),accepted_correct=sum(r['accepted'] and r['expected']==r['predicted'] for r in results),accepted_wrong=sum(r['accepted'] and r['expected']!=r['predicted'] for r in results),character_errors=sum(edit_distance(r['expected'],r['predicted']) for r in results),results=results)
        report['models'][name]=metric
        print(name,{k:v for k,v in metric.items() if k!='results'},flush=True)
    (args.folder/'full_pipeline_evaluation.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
