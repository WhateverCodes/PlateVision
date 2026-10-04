from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import struct,json,sys
import cv2,numpy as np
from PIL import Image,ImageDraw
p=Path('outputs/PlateVision1').resolve();sys.path.insert(0,str(p));from src.preprocessing import normalize_character
out=p/'data/emnist_trial_2026_10_03';out.mkdir(exist_ok=True);(out/'characters').mkdir(exist_ok=True)
rows=[];counts=Counter()
with ZipFile('C:/Users/Grace/Downloads/archive (8).zip') as z:
 mapping={int(a):chr(int(b)) for a,b in (line.split() for line in z.read('emnist-byclass-mapping.txt').decode().splitlines() if line.strip())}
 with z.open('emnist_source_files/emnist-byclass-train-labels-idx1-ubyte') as labels,z.open('emnist_source_files/emnist-byclass-train-images-idx3-ubyte') as images:
  magic,total=struct.unpack('>II',labels.read(8));imagic,itotal,h,w=struct.unpack('>IIII',images.read(16));assert magic==2049 and imagic==2051 and total==itotal and (h,w)==(28,28)
  for i in range(total):
   label=mapping[labels.read(1)[0]];raw=images.read(h*w)
   if label not in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' or counts[label]>=40:continue
   a=np.frombuffer(raw,np.uint8).reshape(h,w).T.copy();_,mask=cv2.threshold(a,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU);glyph=normalize_character(mask);dest=out/'characters'/f'{label}_{i}.png';cv2.imwrite(str(dest),glyph);counts[label]+=1;rows.append(dict(image=str(dest),label=label,origin='EMNIST_byclass_training',source_index=i))
   if len(counts)==36 and min(counts.values())==40:break
(out/'characters.json').write_text(json.dumps(rows,indent=2));print('Prepared',len(rows),'from official train partition; scanned',i+1,'rows; test untouched')
sheet=Image.new('RGB',(720,360),'#444');d=ImageDraw.Draw(sheet)
for i,label in enumerate(sorted(counts)):
 r=next(r for r in rows if r['label']==label);im=Image.open(r['image']).resize((48,48));x=i%12*60;y=i//12*110;sheet.paste(im,(x,y));d.text((x,y+52),label,fill='white')
sheet.save('work/new_dataset_audit/emnist_sample.png')
