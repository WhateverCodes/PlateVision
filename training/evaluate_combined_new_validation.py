"""Additional held-out source-labelled validation; never used for gradient updates."""
import json
from pathlib import Path
from dataclasses import replace
from config import DEFAULTS
from src.runtime import configure
from src.pipeline import Pipeline
from src.classical_detector import PlateCropDetector
from src.whole_line_reader import WholeLineReader
from src.character_classifier import Classifier
from src.preprocessing import read_image
from training.evaluate_plate_reader import edit_distance

def main():
    configure(2)
    rows=json.loads(Path('data/combined_training_2026_10_03/new_plate_splits.json').read_text())['val']
    runs=[('cnn_previous',False,DEFAULTS.adapted_character_path,.70),('cnn_candidate',False,Path('models/combined_cnn_2026_10_03/epoch2.pt'),.70),('ocr_previous',True,Path('models/paddle_plate_adapted_v1'),.90),('ocr_candidate',True,Path('models/paddle_combined_v2'),.91)]
    report={'scope':'54 newly grouped validation plate crops. Source labels not fully manually verified; near-duplicate capture provenance incomplete. Reserved test remains unused. No further threshold search on these results.','models':{}}
    for name,ocr,path,threshold in runs:
        reader=WholeLineReader(path) if ocr else Classifier(path)
        cfg=replace(DEFAULTS,reader_mode='whole_line' if ocr else 'characters',localization_mode='crop',rectify=False,character_threshold=threshold)
        pipe=Pipeline(cfg,PlateCropDetector(),reader);results=[]
        for r in rows:
            d=pipe(read_image(Path(r['crop'])))[0]
            results.append(dict(expected=r['text'],predicted=d['raw_text'],accepted=d['recognized']))
        m=dict(images=len(rows),reference_characters=sum(len(r['expected']) for r in results),exact=sum(r['expected']==r['predicted'] for r in results),character_errors=sum(edit_distance(r['expected'],r['predicted']) for r in results),accepted_correct=sum(r['accepted'] and r['expected']==r['predicted'] for r in results),accepted_wrong=sum(r['accepted'] and r['expected']!=r['predicted'] for r in results),results=results)
        report['models'][name]=m;print(name,{k:v for k,v in m.items() if k!='results'},flush=True)
    Path('outputs/evaluation/combined_new_validation.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
