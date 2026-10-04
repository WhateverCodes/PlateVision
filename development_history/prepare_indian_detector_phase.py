from pathlib import Path
import json
root=Path('outputs/PlateVision1/data/combined_training_2026_10_03');out=root/'detector_indian';out.mkdir(exist_ok=True)
rows=json.loads((root/'detector/train.json').read_text());rows=[r for r in rows if r['dataset']!='archive10_pakistan'];(out/'train.json').write_text(json.dumps(rows,indent=2))
for split in ['val','test']:(out/(split+'.json')).write_text((root/'detector'/(split+'.json')).read_text())
(out/'dataset_card.json').write_text(json.dumps({'provenance':'Indian-only adaptation after multi-source plate detection training; source boxes, grouped validation.','counts':{'train':len(rows)}},indent=2));print('Indian-only adaptation images:',len(rows))
