import urllib.request,json,hashlib
from pathlib import Path
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import parse_wheel_filename
p=Path('work/onnx_wheels');p.mkdir(exist_ok=True);tags=set(sys_tags())
for name,version in [('onnx','1.19.1'),('protobuf','6.33.0'),('ml_dtypes','0.5.3')]:
 r=json.load(urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=20));f=next(f for f in r['urls'] if f['filename'].endswith('.whl') and parse_wheel_filename(f['filename'])[3]&tags)
 data=urllib.request.urlopen(f['url'],timeout=30).read();assert hashlib.sha256(data).hexdigest()==f['digests']['sha256'];(p/f['filename']).write_bytes(data);print(name,flush=True)
