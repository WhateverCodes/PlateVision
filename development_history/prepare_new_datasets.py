from pathlib import Path
from zipfile import ZipFile
from collections import Counter,defaultdict
import json,hashlib,re,sys,xml.etree.ElementTree as ET
import cv2,numpy as np
from PIL import Image,ImageDraw
p=Path('outputs/PlateVision1').resolve();sys.path.insert(0,str(p));from src.preprocessing import normalize_character
out=p/'data/dataset_trial_2026_10_03';out.mkdir(exist_ok=True);(out/'characters').mkdir(exist_ok=True);(out/'plates').mkdir(exist_ok=True)
# Conservatively screen all known plate labels, including train and held-out sets.
known=set()
for folder in ['reader_v1','reader_reviewed_v2','accuracy_check_2026_09_21']:
 for f in (p/'data'/folder).glob('*.json'):
  value=json.loads(f.read_text());rows=value if isinstance(value,list) else value.get('records',[])
  for r in rows:
   if isinstance(r,dict):known.update(re.sub('[^A-Z0-9]','',v.upper()) for v in [r.get('text',''),*r.get('texts',[])] if isinstance(v,str) and v)
f=p/'outputs/external_test/reserved_test_manifest.json'
for r in json.loads(f.read_text()).get('records',[]):known.update(r.get('texts',[]))
# Prepare a bounded sample from every character class; never pretend random augmented splits are independent.
chars=[];seen=set();rng=np.random.default_rng(42);sourcecounts=Counter();rejects=Counter()
with ZipFile('C:/Users/Grace/Downloads/archive (7).zip') as z:
 groups=defaultdict(list)
 for n in z.namelist():
  if n.lower().endswith('.jpg'):groups[Path(n).parent.name].append(n)
 for label,names in sorted(groups.items()):
  sourcecounts[label]=len(names)
  for idx in rng.permutation(len(names)):
   n=names[idx];raw=z.read(n);gray=cv2.imdecode(np.frombuffer(raw,np.uint8),0)
   if gray is None:rejects['decode']+=1;continue
   _,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
   if np.mean(mask>0)>.65:mask=255-mask
   count,_,stats,_=cv2.connectedComponentsWithStats(mask,8)
   if count<2:rejects['empty']+=1;continue
   x,y,w,h,area=max(stats[1:],key=lambda s:s[4]);glyph=normalize_character(mask[y:y+h,x:x+w]);sha=hashlib.sha256(glyph.tobytes()).hexdigest()
   if sha in seen:rejects['duplicate_normalized']+=1;continue
   seen.add(sha);dest=out/'characters'/f'{label}_{sha[:16]}.png';cv2.imwrite(str(dest),glyph)
   chars.append(dict(image=str(dest),label=label,origin='external_augmented_label_unverified',source_archive='archive (7).zip',source_member=n,sha256=sha))
   if sum(r['label']==label for r in chars)>=100:break
(out/'characters.json').write_text(json.dumps(chars,indent=2))
plates=[];quarantine=[];parsed=[]
with ZipFile('C:/Users/Grace/Downloads/archive (6).zip') as z:
 names=set(z.namelist())
 for n in sorted(names):
  if not n.endswith('.xml'):continue
  e=ET.fromstring(z.read(n));base=n.rsplit('.',1)[0];matches=[base+ext for ext in ('.jpg','.jpeg','.png','.JPG') if base+ext in names]
  if len(matches)!=1:quarantine.append(dict(source=n,reason='missing_or_ambiguous_image'));continue
  image=matches[0];group=re.sub(r'_\d+$','',Path(image).stem) if image.startswith('video_images/') else None
  for obj in e.findall('object'):
   text=re.sub('[^A-Z0-9]','',obj.findtext('name','').upper());box=obj.find('bndbox')
   try:coords=[int(float(box.findtext(k))) for k in ('xmin','ymin','xmax','ymax')]
   except Exception:quarantine.append(dict(source=n,reason='bad_box'));continue
   parsed.append(dict(source=n,image=image,text=text,box=coords,group=group))
 excludedgroups={r['group'] for r in parsed if r['text'] in known and r['group']}
 for r in parsed:
  if r['text'] in known or r['group'] in excludedgroups:
   quarantine.append(dict(**r,reason='known_registration_or_related_video'));continue
  if not re.fullmatch('[A-Z0-9]{6,12}',r['text']):quarantine.append(dict(**r,reason='label_needs_review'));continue
  a=cv2.imdecode(np.frombuffer(z.read(r['image']),np.uint8),1)
  if a is None:continue
  x1,y1,x2,y2=r['box'];h,w=a.shape[:2]
  if not (0<=x1<x2<=w and 0<=y1<y2<=h):quarantine.append(dict(**r,reason='box_out_of_bounds'));continue
  crop=a[y1:y2,x1:x2];sha=hashlib.sha256(crop.tobytes()).hexdigest();dest=out/'plates'/f'{sha[:20]}.png';cv2.imwrite(str(dest),crop)
  plates.append(dict(**r,crop=str(dest),width=x2-x1,height=y2-y1,status='source_label_unverified_do_not_train',sha256=sha))
(out/'plate_candidates.json').write_text(json.dumps(plates,indent=2));(out/'plate_quarantine.json').write_text(json.dumps(quarantine,indent=2))
report=dict(character_source_counts=dict(sourcecounts),missing_classes=sorted(set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')-set(sourcecounts)),prepared_characters=len(chars),character_rejects=dict(rejects),plate_candidates=len(plates),plate_quarantine=len(quarantine),plate_quarantine_reasons=dict(Counter(r['reason'] for r in quarantine)),note='Characters appear augmented; parent provenance unknown. No random character test split. Plate labels unverified; crops not used for training. Known text and related video screening is not complete near-duplicate certification.')
(out/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
sheet=Image.new('RGB',(720,360),'#444444');draw=ImageDraw.Draw(sheet)
for i,label in enumerate(sorted(groups)):
 row=next(r for r in chars if r['label']==label);im=Image.open(row['image']).resize((48,48));x=(i%12)*60;y=(i//12)*110;sheet.paste(im,(x,y));draw.text((x,y+52),label,fill='white')
sheet.save('work/new_dataset_audit/normalized_characters.png')
