from pathlib import Path
import sys,json,cv2
root=Path.cwd();p=root/'outputs/PlateVision1';sys.path.insert(0,str(p))
from src.whole_line_reader import WholeLineReader
reader=WholeLineReader(p/'models/paddle_candidate');out=p/'data/ocr_finetune_audit_v1'
rows=json.loads((p/'data/reader_reviewed_v2/val.json').read_text())+json.loads((p/'data/accuracy_check_2026_09_21/val.json').read_text())
kept=[];excluded=[]
for r in rows:
 im=cv2.imread(r['image']);lines=reader.line_crops(im)
 if len(lines)!=1:excluded.append({'id':r['id'],'reason':'not one detected line'});continue
 crop=lines[0]['crop'];dest=out/'validation_images'/f"{r['id']}.png";dest.parent.mkdir(exist_ok=True);cv2.imwrite(str(dest),crop)
 kept.append({**r,'image':str(dest),'original_image':r['image']})
(out/'validation_candidates.json').write_text(json.dumps(kept,indent=2));(out/'validation_excluded.json').write_text(json.dumps(excluded,indent=2))
from PIL import Image,ImageDraw
canvas=Image.new('RGB',(1000,((len(kept)+3)//4)*110),'#dddddd');d=ImageDraw.Draw(canvas)
for i,r in enumerate(kept):
 x=(i%4)*250;y=(i//4)*110;im=Image.open(r['image']).convert('RGB');im.thumbnail((240,75));canvas.paste(im,(x,y+25));d.text((x+3,y+3),f"{i}: {r['text']}",fill='black')
canvas.save(root/'work/validation_audit.jpg');print('Validation candidates',len(kept),'excluded',len(excluded))
