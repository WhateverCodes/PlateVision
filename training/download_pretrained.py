"""Explicit installation step; application inference never downloads models."""
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from config import ROOT

FILES = {
    'cct_s_v2_global.onnx':'https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_s_v2_global.onnx',
    'cct_s_v2_global_plate_config.yaml':'https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_s_v2_global_plate_config.yaml',
    'yolo-v9-t-384-license-plates-end2end.onnx':'https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-384-license-plates-end2end.onnx',
    'FAST_PLATE_OCR_LICENSE.txt':'https://raw.githubusercontent.com/ankandrew/fast-plate-ocr/master/LICENSE',
    'OPEN_IMAGE_MODELS_LICENSE.txt':'https://raw.githubusercontent.com/ankandrew/open-image-models/main/LICENSE',
}

def main():
    folder=ROOT/'models/pretrained';folder.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for name,url in FILES.items():
        path=folder/name
        if not path.exists():
            print('Downloading '+name,flush=True)
            temp=path.with_suffix(path.suffix+'.partial')
            try:
                with urllib.request.urlopen(url,timeout=90) as response,temp.open('wb') as out:
                    shutil.copyfileobj(response,out)
                temp.replace(path)
            finally:
                if temp.exists():temp.unlink()
        manifest.append({'file':name,'source':url,'bytes':path.stat().st_size,
                         'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Pretrained files ready. Recognition stays local.',flush=True)

if __name__=='__main__':main()
