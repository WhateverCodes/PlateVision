"""Prepare a deduplicated multi-source trial; never edit original reviews."""
from pathlib import Path
from zipfile import ZipFile
from collections import Counter,defaultdict
import json,hashlib,sys,re,xml.etree.ElementTree as ET
import cv2,numpy as np
p=Path('outputs/PlateVision1').resolve();sys.path.insert(0,str(p))
from src.runtime import configure
from src.preprocessing import normalize_character
from src.validator import plausible_ocr
from training.data_tools import grouped_split
configure(2)
out=p/'data/combined_training_2026_10_03';out.mkdir(exist_ok=True)
for d in ['detector/images','characters/images','reader']:(out/d).mkdir(parents=True,exist_ok=True)
def save(path,value):(out/path).write_text(json.dumps(value,indent=2))
def resolve(path):
 q=Path(path)
 if q.exists():return q
 parts=q.parts
 if 'data' in parts:return p/Path(*parts[parts.index('data'):])
 return q
# Build new plate groups before training: equal identities stay in one split.
new=json.loads((p/'data/dataset_trial_2026_10_03/plate_candidates.json').read_text());valid=[];excluded=[]
for r in new:
 if not plausible_ocr(r['text']) or len(r['text'])<8:excluded.append(r);continue
 valid.append(dict(**{k:v for k,v in r.items() if k!='group'},group='registration:'+r['text'],texts=[r['text']]))
splits=grouped_split(valid,42);save('new_plate_splits.json',splits);save('excluded_plate_labels.json',excluded)
# Existing approved train partition plus new training crops; remove all held-out identities.
train=json.loads((p/'data/reader_reviewed_v2/train.json').read_text());held=set()
for folder,names in [('reader_v1',['val','test','withheld_unverified']),('reader_reviewed_v2',['val']),('accuracy_check_2026_09_21',['val'])]:
 for name in names:
  for r in json.loads((p/'data'/folder/(name+'.json')).read_text()):held.add(r.get('text',''))
for r in json.loads((p/'outputs/external_test/reserved_test_manifest.json').read_text())['records']:held.update(r.get('texts',[]))
held.update(r['text'] for split in ['val','test'] for r in splits[split]);train=[r for r in train if r['text'] not in held and plausible_ocr(r['text']) and len(r['text'])>=8 and Path(r['image']).exists()]
for r in splits['train']:train.append(dict(id='new6_'+r['sha256'][:20],image=r['crop'],text=r['text'],group=r['group'],label_origin='source_unverified'))
seen=set();reader=[]
for r in train:
 a=cv2.imread(r['image']);key=hashlib.sha256(a.tobytes()).hexdigest()
 if key in seen:continue
 seen.add(key);reader.append(r)
save('reader/plate_train.json',reader)
# Detection uses existing Indian training partition, new source groups, and official Pakistan TRAIN only.
det={k:[] for k in ['train','val','test']};seen=set();counts=Counter()
for split in det:
 for r in json.loads((p/'data/detector_real_v2'/(split+'.json')).read_text()):
  q=resolve(r['image'])
  if not q.exists():counts['missing_existing_detector_images']+=1;continue
  det[split].append({**r,'image':str(q),'dataset':'existing_indian'})
with ZipFile('C:/Users/Grace/Downloads/archive (6).zip') as z:
 for split,rows in splits.items():
  for r in rows:
   raw=z.read(r['image']);sha=hashlib.sha256(raw).hexdigest()
   if sha in seen:continue
   seen.add(sha);a=cv2.imdecode(np.frombuffer(raw,np.uint8),1);h,w=a.shape[:2];scale=min(1,640/max(h,w));dest=out/'detector/images'/('in_'+sha[:20]+'.jpg');cv2.imwrite(str(dest),cv2.resize(a,(round(w*scale),round(h*scale))));box=[v*scale for v in r['box']]
   det[split].append(dict(image=str(dest),boxes=[box],texts=[r['text']],group=r['group'],dataset='archive6_indian'))
with ZipFile('C:/Users/Grace/Downloads/archive (10).zip') as z:
 names=set(z.namelist());testids={Path(n).stem.split('_')[0] for n in names if n.startswith('test/') and n.endswith('.xml')}
 for n in sorted(names):
  if not n.startswith('train/') or not n.endswith('.xml'):continue
  group=Path(n).stem.split('_')[0]
  if group in testids:counts['pakistan_test_group_overlap']+=1;continue
  e=ET.fromstring(z.read(n));member=n.rsplit('/',1)[0]+'/images/'+Path(n).stem+'.jpg'
  if member not in names:counts['missing_pakistan_image']+=1;continue
  raw=z.read(member);sha=hashlib.sha256(raw).hexdigest()
  if sha in seen:counts['duplicate_detector_image']+=1;continue
  a=cv2.imdecode(np.frombuffer(raw,np.uint8),1)
  if a is None:continue
  h,w=a.shape[:2];boxes=[]
  for o in e.findall('object'):
   box=[float(o.findtext('bndbox/'+k)) for k in ['xmin','ymin','xmax','ymax']]
   if 0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h:boxes.append(box)
  if not boxes:counts['invalid_pakistan_boxes']+=1;continue
  seen.add(sha);scale=min(1,640/max(h,w));dest=out/'detector/images'/('pk_'+sha[:20]+'.jpg');cv2.imwrite(str(dest),cv2.resize(a,(round(w*scale),round(h*scale))))
  det['train'].append(dict(image=str(dest),boxes=[[v*scale for v in b] for b in boxes],group='pk_'+group,texts=[],dataset='archive10_pakistan'))
for split,rows in det.items():save('detector/'+split+'.json',rows)
save('detector/dataset_card.json',dict(provenance='Own detector real-image adaptation: existing Indian source training partitions + group-separated archive6 + archive10 official train plate boxes. No foreign text-reader training. Evaluation is Indian development data; incomplete background annotation and capture grouping are limitations.',counts={k:len(v) for k,v in det.items()}))
# All usable unique labelled archive7 glyphs, not just earlier 100/class.
chars=json.loads((p/'data/character_verified_300/verified_and_synthetic.json').read_text());seen=set();cc=Counter()
with ZipFile('C:/Users/Grace/Downloads/archive (7).zip') as z:
 for n in z.namelist():
  if not n.lower().endswith('.jpg'):continue
  label=Path(n).parent.name
  if label not in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' or len(label)!=1:continue
  gray=cv2.imdecode(np.frombuffer(z.read(n),np.uint8),0)
  if gray is None:continue
  _,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  if np.mean(mask>0)>.65:mask=255-mask
  count,_,stats,_=cv2.connectedComponentsWithStats(mask,8)
  if count<2:continue
  x,y,w,h,area=max(stats[1:],key=lambda s:s[4]);glyph=normalize_character(mask[y:y+h,x:x+w]);sha=hashlib.sha256(glyph.tobytes()).hexdigest()
  if sha in seen:cc['duplicates']+=1;continue
  seen.add(sha);dest=out/'characters/images'/f'{label}_{sha[:20]}.png';cv2.imwrite(str(dest),glyph);chars.append(dict(image=str(dest),label=label,origin='archive7_augmented',source=n));cc[label]+=1
chars+=json.loads((p/'data/emnist_trial_2026_10_03/characters.json').read_text());save('characters/train.json',chars)
summary={'new387':{k:len(v) for k,v in splits.items()},'new387_excluded_bad_format':len(excluded),'reader_training_plates':len(reader),'detector':{k:len(v) for k,v in det.items()},'detector_sources':dict(Counter(r['dataset'] for r in det['train'])),'character_rows':len(chars),'character_archive7':dict(cc),'warnings':dict(counts),'note':'Archive2=5,3=9 deduplicated. Archive4 already represented in existing reader training. Archive8 uses 1440 representative official training glyphs, not all 6GB duplicated representations. 3113449 absent. Split by known identity/hash/group; near-duplicate independence not fully certified.'}
save('audit.json',summary);print(json.dumps(summary,indent=2))
