import urllib.request,json,hashlib,time
from pathlib import Path
p=Path('work/paddle_downloads')
url='https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv5_mobile_rec_pretrained.pdparams'
try:
 with urllib.request.urlopen(url,timeout=25) as r:data=r.read()
 (p/'en_PP-OCRv5_mobile_rec_pretrained.pdparams').write_bytes(data)
 (p/'checkpoint_source.json').write_text(json.dumps({'url':url,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)},indent=2))
 print('Checkpoint downloaded:',len(data),flush=True)
except Exception as e:print('CHECKPOINT_DOWNLOAD_FAILED:',repr(e),flush=True)
