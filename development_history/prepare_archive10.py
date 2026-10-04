from pathlib import Path
from zipfile import ZipFile
import json,xml.etree.ElementTree as ET,hashlib,re
import cv2,numpy as np
from PIL import Image,ImageDraw
p=Path('outputs/PlateVision1/data/detection_archive10_review');p.mkdir(exist_ok=True);(p/'crops').mkdir(exist_ok=True)
rows=[];groups={'train':set(),'test':set()}
with ZipFile('C:/Users/Grace/Downloads/archive (10).zip') as z:
 names=set(z.namelist())
 for n in sorted(names):
  if not n.endswith('.xml'):continue
  split=n.split('/')[0];group=Path(n).stem.split('_')[0];groups[split].add(group)
  if split!='train':continue
  e=ET.fromstring(z.read(n));w=int(e.findtext('size/width'));h=int(e.findtext('size/height'))
  for j,o in enumerate(e.findall('object')):
   box=[int(float(o.findtext('bndbox/'+k))) for k in ('xmin','ymin','xmax','ymax')];x1,y1,x2,y2=box
   if 0<=x1<x2<=w and 0<=y1<y2<=h:rows.append(dict(archive_member=n.rsplit('/',1)[0]+'/images/'+Path(n).stem+'.jpg',box=box,group=group,split=split,area=(x2-x1)*(y2-y1)))
 overlap=groups['train']&groups['test'];chosen=[];seen=set()
 for row in sorted(rows,key=lambda r:r['area'],reverse=True):
  if row['group'] in overlap or row['group'] in seen or row['archive_member'] not in names:continue
  a=cv2.imdecode(np.frombuffer(z.read(row['archive_member']),np.uint8),1)
  if a is None:continue
  x1,y1,x2,y2=row['box'];crop=a[y1:y2,x1:x2];name=hashlib.sha256((row['archive_member']+str(row['box'])).encode()).hexdigest()[:20]+'.png';cv2.imwrite(str(p/'crops'/name),crop);chosen.append(dict(**row,crop=name,text=None));seen.add(row['group'])
  if len(chosen)>=24:break
(p/'candidates.json').write_text(json.dumps(chosen,indent=2));(p/'audit.json').write_text(json.dumps(dict(train_boxes_valid=len(rows),train_test_ID_overlap=len(overlap),prepared=len(chosen),note='ID-based grouping is provisional. Only train images were decoded; official test pixels untouched. Text labels missing.'),indent=2))
sheet=Image.new('RGB',(960,720),'#ddd');d=ImageDraw.Draw(sheet)
for i,r in enumerate(chosen):
 im=Image.open(p/'crops'/r['crop']);im.thumbnail((150,145));x=i%6*160;y=i//6*180;sheet.paste(im,(x,y));d.text((x,y+150),str(i+1),fill='black')
sheet.save('work/new_dataset_audit/archive10_crops.jpg');print('Prepared',len(chosen),'train-only crops; ID overlap',len(overlap))
