import json
from pathlib import Path
from config import DEFAULTS
from src.runtime import configure
from src.character_classifier import Classifier
from training.refine_verified_cnn import evaluate
configure(2);rows=json.loads(Path('data/accuracy_check_2026_09_21/val.json').read_text())+json.loads(Path('data/reader_reviewed_v2/val.json').read_text());reports={}
for name in ['character_verified_100/last.pt','character_verified_300_head_trial/stage_3.pt','character_adaptation_v1/last.pt']:
 r=evaluate(Classifier(Path('models')/name),rows);reports[name]=r;print(name,{k:v for k,v in r.items() if k!='results'},flush=True)
Path('outputs/evaluation/cnn_segment_checkpoint_comparison.json').write_text(json.dumps(reports,indent=2))
