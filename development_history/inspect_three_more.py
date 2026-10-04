from pathlib import Path
from zipfile import ZipFile
from collections import Counter
for name in ['archive (8).zip','archive (9).zip','archive (10).zip']:
 with ZipFile(Path('C:/Users/Grace/Downloads')/name) as z:
  fs=[f for f in z.infolist() if not f.is_dir()];print('\n',name,len(fs),'files MB',round(sum(f.file_size for f in fs)/1e6,1));print(Counter(Path(f.filename).suffix.lower() for f in fs));print([f.filename for f in fs[:12]])
  labels=[f for f in fs if Path(f.filename).suffix.lower() in ('.csv','.txt','.xml','.json','.yaml','.tsv')]
  for f in labels[:2]:print(f.filename,z.read(f)[:1200].decode('utf8',errors='replace'))
