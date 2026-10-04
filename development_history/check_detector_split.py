import json
from pathlib import Path
p=Path('outputs/PlateVision1/data/combined_training_2026_10_03/detector');tr=json.loads((p/'train.json').read_text());va=json.loads((p/'val.json').read_text());te=json.loads((p/'test.json').read_text());held={t for r in va+te for t in r.get('texts',[]) if t};over=[r for r in tr if any(t in held for t in r.get('texts',[]) if t)];print('Cross split text overlaps:',len(over));print('File overlap',len({r['image'] for r in tr}&{r['image'] for r in va+te}))
