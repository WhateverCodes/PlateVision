import json,torch
from pathlib import Path
from src.runtime import configure
from config import DEFAULTS,ALPHABET
from src.character_classifier import Classifier
from training.refine_verified_cnn import evaluate
configure(2);rows=json.loads(Path('data/accuracy_check_2026_09_21/val.json').read_text())+json.loads(Path('data/reader_reviewed_v2/val.json').read_text())
base=Classifier(Path('models/character_verified_100/last.pt'));state={k:v.clone() for k,v in base.model.state_dict().items()};reports=[]
for stage in (1,2,3):
 candidate=Classifier(Path(f'models/character_weighted_trial/stage_{stage}.pt'))
 merged={k:(state[k]*.75+v*.25 if v.is_floating_point() else state[k]) for k,v in candidate.model.state_dict().items()};candidate.model.load_state_dict(merged)
 r=evaluate(candidate,rows);r['stage']=stage;reports.append(r);print({k:v for k,v in r.items() if k!='results'},flush=True)
 torch.save(dict(kind='character_classifier',format_version=1,trained_steps=stage*60,alphabet=ALPHABET,model=merged,provenance='Own CNN conservative update: 75% verified-100 checkpoint and 25% verified-weighted adaptation stage '+str(stage)+'. Development selected.'),f'models/character_weighted_trial/blend_{stage}.pt')
Path('models/character_weighted_trial/blend_evaluation.json').write_text(json.dumps(reports,indent=2))

