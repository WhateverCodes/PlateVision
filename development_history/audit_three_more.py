from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import json,xml.etree.ElementTree as ET
root=Path('C:/Users/Grace/Downloads');report={}
with ZipFile(root/'archive (3).zip') as z:old={(x.CRC,x.file_size) for x in z.infolist() if x.filename.lower().endswith(('.jpg','.png','.jpeg'))}
for name in ['archive (9).zip','archive (10).zip']:
 with ZipFile(root/name) as z:
  names=z.namelist();images=[x for x in z.infolist() if x.filename.lower().endswith('.jpg')];labels=Counter();splits=Counter();bad=0
  for n in names:
   if not n.endswith('.xml'):continue
   try:
    e=ET.fromstring(z.read(n));labels.update(o.findtext('name','') for o in e.findall('object'));splits[n.split('/')[0]]+=1
   except ET.ParseError:bad+=1
  report[name]={'images':len(images),'annotation_classes':dict(labels),'annotation_splits':dict(splits),'bad_xml':bad,'images_matching_archive3_crc_size':sum((f.CRC,f.file_size) in old for f in images)}
with ZipFile(root/'archive (8).zip') as z:print('EMNIST files:',z.namelist())
Path('work/new_dataset_audit/second_inventory.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
