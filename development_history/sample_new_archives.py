from zipfile import ZipFile
from collections import Counter
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
out=Path('work/new_dataset_audit');out.mkdir(exist_ok=True)
for file in ['archive (7).zip','archive (6).zip','archive (5).zip']:
 with ZipFile(Path('C:/Users/Grace/Downloads')/file) as z:
  names=[n for n in z.namelist() if n.lower().endswith(('.jpg','.png','.jpeg'))]
  print(file,'folders',Counter(str(Path(n).parent) for n in names).most_common(40))
  chosen=[names[i] for i in np.linspace(0,len(names)-1,24,dtype=int)]
  sheet=Image.new('RGB',(960,720),'#dddddd');d=ImageDraw.Draw(sheet)
  for i,n in enumerate(chosen):
   a=cv2.imdecode(np.frombuffer(z.read(n),np.uint8),cv2.IMREAD_COLOR)
   if a is None:continue
   im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((150,145));x=(i%6)*160;y=(i//6)*180;sheet.paste(im,(x,y));d.text((x,y+147),str(Path(n).parent.name)+'/'+Path(n).name[:17],fill='black')
  sheet.save(out/(file.replace('.zip','.jpg')))
