from pathlib import Path
from zipfile import ZipFile
from collections import Counter
for name in ['3113449.zip','archive (7).zip','archive (6).zip','archive (5).zip']:
 with ZipFile(Path('C:/Users/Grace/Downloads')/name) as z:
  files=[x for x in z.infolist() if not x.is_dir()]
  print('\nZIP',name,'files',len(files),'uncompressed MB',round(sum(x.file_size for x in files)/1e6,1))
  print('extensions',Counter(Path(x.filename).suffix.lower() for x in files))
  print('sample',[x.filename for x in files[:16]])
  labels=[x for x in files if Path(x.filename).suffix.lower() in ('.txt','.csv','.json','.xml','.yaml','.yml','.md')]
  for f in labels[:2]:print('LABEL',f.filename,z.read(f)[:1800].decode('utf-8',errors='replace'))
