import urllib.request,json,hashlib,zipfile,email
from pathlib import Path
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import parse_wheel_filename
folder=Path('work/paddle_downloads');tags=set(sys_tags());pending=['setuptools','httpx','numpy','protobuf','Pillow','opt-einsum==3.3.0','networkx','typing-extensions','safetensors'];done=set()
while pending:
 req=Requirement(pending.pop(0))
 if req.marker and not req.marker.evaluate({'extra':''}):continue
 if req.name.lower() in done:continue
 version=next((s.version for s in req.specifier if s.operator=='==' and '*' not in s.version),None)
 url=f'https://pypi.org/pypi/{req.name}/'+(version+'/' if version else '')+'json'
 meta=json.load(urllib.request.urlopen(url,timeout=20));files=[]
 for f in meta['urls']:
  if f['filename'].endswith('.whl') and parse_wheel_filename(f['filename'])[3]&tags:files.append(f)
 if not files:raise RuntimeError('No compatible wheel: '+req.name)
 f=files[0];dest=folder/f['filename']
 if not dest.exists():
  data=urllib.request.urlopen(f['url'],timeout=30).read();assert hashlib.sha256(data).hexdigest()==f['digests']['sha256'];dest.write_bytes(data)
 with zipfile.ZipFile(dest) as z:
  name=next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'));metadata=email.message_from_bytes(z.read(name))
 pending.extend(metadata.get_all('Requires-Dist',[]));done.add(req.name.lower());print('Ready',req.name,flush=True)


