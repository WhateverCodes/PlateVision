from PIL import Image,ImageDraw
from pathlib import Path
import json
p=Path('outputs/PlateVision1/data/ocr_finetune_audit_v1');rows=json.loads((p/'train.json').read_text())
for start in range(0,len(rows),30):
 canvas=Image.new('RGB',(1000,900),'#dddddd');draw=ImageDraw.Draw(canvas)
 for i,r in enumerate(rows[start:start+30]):
  x=(i%4)*250;y=(i//4)*110
  im=Image.open(r['image']).convert('RGB');im.thumbnail((240,75));canvas.paste(im,(x,y+25));draw.text((x+3,y+3),f"{start+i}: {r['text']}",fill='black')
 canvas.save(f'work/ocr_audit_{start}.jpg')
