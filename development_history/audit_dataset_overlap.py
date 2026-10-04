from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import json,hashlib,xml.etree.ElementTree as ET
root=Path('C:/Users/Grace/Downloads');report={}
def members(name):
 with ZipFile(root/name) as z:return {(x.CRC,x.file_size) for x in z.infolist() if x.filename.lower().endswith(('.jpg','.jpeg','.png'))}
old={n:members(n) for n in ['archive (2).zip','archive (3).zip','archive (4).zip']}
for name in ['archive (5).zip','archive (6).zip','archive (7).zip']:
 with ZipFile(root/name) as z:
  images=[x for x in z.infolist() if x.filename.lower().endswith(('.jpg','.jpeg','.png'))]
  report[name]={'images':len(images),'matching_image_content_crc_size':{n:sum((x.CRC,x.file_size) in sig for x in images) for n,sig in old.items()}}
  if name=='archive (7).zip':
   report[name]['classes']=dict(Counter(Path(x.filename).parent.name for x in images));report[name]['not_augmented_names']=[x.filename for x in images if not Path(x.filename).name.startswith('aug')][:50]
  if name=='archive (6).zip':
   counts=Counter();missing=0;labels=[]
   names=set(z.namelist())
   for n in names:
    if n.endswith('.xml'):
     e=ET.fromstring(z.read(n));objects=e.findall('object');counts['boxes']+=len(objects);labels += [o.findtext('name','') for o in objects]
   report[name]['annotations']=dict(counts);report[name]['labels_sample']=labels[:15]
Path('work/new_dataset_audit/inventory.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
