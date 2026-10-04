import sys,zipfile,json,time
from pathlib import Path
import cv2
from dataclasses import replace
root=Path.cwd();p=root/'outputs/PlateVision1';sys.path.insert(0,str(p))
from config import DEFAULTS
from src.pipeline import Pipeline
folder=root/'work/user_photo_batch';folder.mkdir(exist_ok=True)
with zipfile.ZipFile(root/'outputs/Indian_Dataset_Starter/images.zip') as z:
 for n in z.namelist():
  if n.lower().endswith('.jpg'):(folder/Path(n).name).write_bytes(z.read(n))
pipe=Pipeline(replace(DEFAULTS,reader_mode='whole_line',localization_mode='pretrained',rectify=False,character_threshold=.95))
report={}
for f in sorted(folder.glob('*.jpg')):
 image=cv2.imread(str(f));rows=pipe(image)
 report[f.name]=[{k:d[k] for k in ('box','raw_text','confidence','recognized')} for d in rows]
 print(f.name,image.shape,report[f.name],flush=True)
(root/'work/photo_baseline.json').write_text(json.dumps(report,indent=2))
