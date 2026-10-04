import urllib.request,json,hashlib
from pathlib import Path
p=Path('work/paddle_downloads');p.mkdir(exist_ok=True)
r=json.load(urllib.request.urlopen('https://pypi.org/pypi/paddlepaddle/3.2.2/json',timeout=30))
f=next(f for f in r['urls'] if 'cp312-cp312-win_amd64.whl' in f['filename'])
print('Wheel MiB',round(f['size']/1024**2,1),flush=True)
data=urllib.request.urlopen(f['url'],timeout=60).read();assert hashlib.sha256(data).hexdigest()==f['digests']['sha256'];(p/f['filename']).write_bytes(data);print('Verified CPU wheel downloaded',flush=True)
url='https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv5_mobile_rec_pretrained.pdparams'
data=urllib.request.urlopen(url,timeout=60).read();(p/'en_PP-OCRv5_mobile_rec_pretrained.pdparams').write_bytes(data);print('Training checkpoint downloaded',len(data),flush=True)
