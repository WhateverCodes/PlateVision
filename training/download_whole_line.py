"""Explicit download of versioned local OCR models; verify upstream hashes."""
import hashlib,json,urllib.request
from config import ROOT

BASE='https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv5/'
FILES={
 'english.onnx':(BASE+'rec/en_PP-OCRv5_rec_mobile.onnx','c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8'),
 'text_detector.onnx':(BASE+'det/ch_PP-OCRv5_det_mobile.onnx','4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae'),
 'RAPIDOCR_LICENSE.txt':('https://raw.githubusercontent.com/RapidAI/RapidOCR/main/LICENSE',None),
 'PADDLEOCR_LICENSE.txt':('https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/LICENSE',None),
}

def main():
    folder=ROOT/'models/paddle_candidate';folder.mkdir(parents=True,exist_ok=True);manifest=[]
    for name,(url,expected) in FILES.items():
        target=folder/name
        if not target.exists():
            print('Downloading '+name,flush=True)
            data=urllib.request.urlopen(url,timeout=90).read()
            if expected and hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Model checksum failed')
            temp=target.with_suffix('.partial');temp.write_bytes(data);temp.replace(target)
        digest=hashlib.sha256(target.read_bytes()).hexdigest()
        if expected and digest!=expected:raise ValueError('Local model checksum failed: '+name)
        manifest.append({'file':name,'source':url,'sha256':digest,'bytes':target.stat().st_size})
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Whole-line OCR models ready. Inference is local.')

if __name__=='__main__':main()
