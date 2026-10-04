import json
from pathlib import Path
from PIL import Image,ImageDraw
p=Path('outputs/PlateVision1');rows=json.loads((p/'data/accuracy_check_2026_09_21/val.json').read_text())+json.loads((p/'data/reader_reviewed_v2/val.json').read_text());pred=json.loads((p/'outputs/evaluation/cnn_segmentation_decoding.json').read_text())['results'];wrong=[(r,d) for r,d in zip(rows,pred) if r['text']!=d['predicted']];out=Image.new('RGB',(1000,((len(wrong)+2)//3)*120),'#ddd');draw=ImageDraw.Draw(out)
for i,(r,d) in enumerate(wrong):
 x=i%3*333;y=i//3*120;im=Image.open(r['image']).convert('RGB');im.thumbnail((320,70));out.paste(im,(x,y+45));draw.text((x,y),f"{i}: {r['text']}\n{d['predicted']}",fill='black')
out.save('work/cnn_failed_crops.jpg')
