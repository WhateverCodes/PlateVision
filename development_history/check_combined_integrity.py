from pathlib import Path
import json
root=Path('outputs/PlateVision1/data/combined_training_2026_10_03');splits=json.loads((root/'new_plate_splits.json').read_text())
keys={k:{r['text'] for r in rows} for k,rows in splits.items()}
assert not keys['train']&keys['val'];assert not keys['train']&keys['test'];assert not keys['val']&keys['test']
det={k:json.loads((root/'detector'/f'{k}.json').read_text()) for k in splits}
assert all(Path(r['image']).exists() for rows in det.values() for r in rows)
assert not {r['image'] for r in det['train']}&{r['image'] for r in det['val']+det['test']}
chars=json.loads((root/'characters/train.json').read_text());assert set(r['label'] for r in chars)==set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
report={'new_plate_identity_splits_disjoint':True,'detector_train_eval_paths_disjoint':True,'detector_paths_exist':True,'character_classes':36,'limits':'Capture and near-duplicate provenance is incomplete; checks do not establish independent final-test performance.'};(root/'integrity_checks.json').write_text(json.dumps(report,indent=2));print(report)
