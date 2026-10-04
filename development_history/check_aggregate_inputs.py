from pathlib import Path
import json
p=Path('outputs/PlateVision1');r=json.loads((p/'data/dataset_trial_2026_10_03/plate_candidates.json').read_text());print(r[0]);print('groups',len(set(x.get('group') for x in r)));print('sample names',[(x['image'],x['text']) for x in r[:3]])
for f in ['data/detector_real_v2/val.json','data/reader_reviewed_v2/train.json']:
 rs=json.loads((p/f).read_text());print(f,len(rs),sum(Path(x['image']).exists() for x in rs))
