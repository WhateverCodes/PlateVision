from pathlib import Path
from PIL import Image,ImageDraw
import json,numpy as np
p=Path('outputs/PlateVision1/data/combined_training_2026_10_03');rows=json.loads((p/'new_plate_splits.json').read_text())['train'];rng=np.random.default_rng(42);chosen=[rows[i] for i in rng.choice(len(rows),24,replace=False)];sheet=Image.new('RGB',(1200,800),'#ddd');d=ImageDraw.Draw(sheet)
for i,r in enumerate(chosen):
 im=Image.open(r['crop']);im.thumbnail((190,150));x=i%6*200;y=i//6*200;sheet.paste(im,(x,y));d.text((x,y+153),str(i+1)+' '+r['text'],fill='black')
sheet.save('work/new_dataset_audit/combined_train_spotcheck.jpg');(p/'spotcheck24.json').write_text(json.dumps(chosen,indent=2))
